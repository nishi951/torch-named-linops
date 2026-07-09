"""
Verifies claims about torch advanced indexing (__getitem__) vs index_select:
  1. Forward results are identical for the single-1D-index case.
  2. Backward (gradient) results are identical -- both are scatter-add.
  3. Timing is roughly comparable for the single-index case (bandwidth bound).
  4. Multiple simultaneous index tensors: __getitem__ fuses into one gather;
     the index_select equivalent needs chained calls + an intermediate.
  5. Backward is nondeterministic on CUDA with repeated indices (atomic adds),
     and torch.use_deterministic_algorithms(True) fixes that (at a cost).

Run: python verify_indexing_claims.py
"""

import torch
import time

print(f"PyTorch version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cpu":
    print(
        "NOTE: nondeterminism claims (atomics) are CUDA-specific; "
        "timing comparisons are more meaningful on GPU too."
    )

torch.manual_seed(0)


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


def bench(fn, iters=30, warmup=5):
    """Time fn() using CUDA events on GPU (accurate, avoids launch-overhead /
    host-side timing pitfalls); falls back to time.perf_counter() on CPU."""
    for _ in range(warmup):
        out = fn()
    sync()

    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn()
        end.record()
        torch.cuda.synchronize()  # required before reading event timings
        elapsed_ms = start.elapsed_time(end)  # elapsed_time returns ms
        return (elapsed_ms / 1000.0) / iters, out
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn()
        return (time.perf_counter() - t0) / iters, out


# ---------------------------------------------------------------------------
# 1 & 2. Forward + backward equivalence, single 1D index, dim 0
# ---------------------------------------------------------------------------
print("\n=== 1&2. Forward/backward equivalence (single 1D index) ===")

N, D = 100_000, 128
K = 500_000  # number of lookups, with heavy repeats to stress backward

src_a = torch.randn(N, D, device=device, requires_grad=True)
src_b = src_a.detach().clone().requires_grad_(True)

idx = torch.randint(0, N, (K,), device=device)

out_getitem = src_a[idx]  # advanced indexing
out_index_select = src_b.index_select(0, idx)  # dedicated op

print("Forward outputs match:", torch.equal(out_getitem, out_index_select))

grad_out = torch.randn_like(out_getitem)
out_getitem.backward(grad_out)
out_index_select.backward(grad_out)

# backward uses atomics when there are repeated indices -> allow tiny tolerance
grads_match = torch.allclose(src_a.grad, src_b.grad, atol=1e-4, rtol=1e-4)
max_diff = (src_a.grad - src_b.grad).abs().max().item()
print(f"Backward grads match (allclose): {grads_match}  (max abs diff: {max_diff:.2e})")

# ---------------------------------------------------------------------------
# 3. Timing: __getitem__ vs index_select, single-index case
# ---------------------------------------------------------------------------
print("\n=== 3. Timing: __getitem__ vs index_select (single index, dim 0) ===")

N, D, K = 2_000_000, 256, 2_000_000
src = torch.randn(N, D, device=device)
idx_big = torch.randint(0, N, (K,), device=device)

t_getitem, _ = bench(lambda: src[idx_big])
t_index_select, _ = bench(lambda: src.index_select(0, idx_big))

print(f"__getitem__ avg time:    {t_getitem * 1e3:.3f} ms")
print(f"index_select avg time:   {t_index_select * 1e3:.3f} ms")
print("(Expect these to be close -- both are bandwidth-bound gathers.)")

# Same, but with a reshaped/multi-dim index flattened for index_select,
# to show the reshape itself is ~free
idx_2d = torch.randint(0, N, (1000, 2000), device=device)


def via_getitem():
    return src[idx_2d]  # advanced indexing handles the 2D index directly


def via_index_select_reshaped():
    flat = idx_2d.reshape(-1)  # metadata-only, no copy
    out = src.index_select(0, flat)
    return out.reshape(*idx_2d.shape, D)  # metadata-only, no copy


out1 = via_getitem()
out2 = via_index_select_reshaped()
print("Reshaped-index results match:", torch.equal(out1, out2))

t_a, _ = bench(via_getitem, iters=10)
t_b, _ = bench(via_index_select_reshaped, iters=10)
print(f"__getitem__ (2D idx):            {t_a * 1e3:.3f} ms")
print(f"index_select + reshape (2D idx): {t_b * 1e3:.3f} ms")

# ---------------------------------------------------------------------------
# 4. Multiple simultaneous index tensors: fused vs chained
# ---------------------------------------------------------------------------
print("\n=== 4. Multiple index tensors: fused __getitem__ vs chained index_select ===")

A, B, C = 500, 500, 128
mat = torch.randn(A, B, C, device=device)
n_pick = 20_000
idx0 = torch.randint(0, A, (n_pick,), device=device)
idx1 = torch.randint(0, B, (n_pick,), device=device)


def fused():
    return mat[idx0, idx1]  # single fused gather -> shape (n_pick, C)


def chained():
    # emulate with index_select: must select along dim0 first (intermediate!),
    # then gather the per-row idx1 -- true equivalent needs `gather`, showing
    # index_select alone can't do this in one dedicated op.
    step1 = mat.index_select(0, idx0)  # (n_pick, B, C) -- big intermediate!
    idx1_exp = idx1.view(-1, 1, 1).expand(-1, 1, C)
    step2 = torch.gather(step1, 1, idx1_exp).squeeze(1)
    return step2


out_fused = fused()
out_chained = chained()
print("Fused vs chained match:", torch.equal(out_fused, out_chained))

t_fused, _ = bench(fused, iters=10)
t_chained, _ = bench(chained, iters=5)
print(f"Fused __getitem__ time:   {t_fused * 1e3:.3f} ms")
print(
    f"Chained emulation time:   {t_chained * 1e3:.3f} ms  (extra memory for intermediate!)"
)

# ---------------------------------------------------------------------------
# 5. Nondeterminism from atomic adds in the backward pass (CUDA only)
# ---------------------------------------------------------------------------
print("\n=== 5. Backward nondeterminism with repeated indices ===")

if device == "cuda":
    N, D, K = 100, 64, 10_000_000  # tiny N, huge K -> massive index collisions
    idx_nd = torch.randint(0, N, (K,), device=device)
    grad_out_nd = torch.randn(K, D, device=device)

    def run_scatter():
        # Direct scatter_add_ uses atomic adds on CUDA
        target = torch.zeros(N, D, device=device)
        target.scatter_add_(0, idx_nd.unsqueeze(1).expand(-1, D), grad_out_nd)
        return target.clone()

    # Run multiple times to detect nondeterminism
    n_runs = 5
    results = [run_scatter() for _ in range(n_runs)]

    # Compare all pairs
    max_diff_nondet = 0.0
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            diff = (results[i] - results[j]).abs().max().item()
            max_diff_nondet = max(max_diff_nondet, diff)

    print(
        f"scatter_add_ {n_runs} runs (non-deterministic mode): max diff across all pairs = {max_diff_nondet:.3e}"
    )

    torch.use_deterministic_algorithms(True)
    try:
        results_det = [run_scatter() for _ in range(n_runs)]
        max_diff_det = 0.0
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                diff = (results_det[i] - results_det[j]).abs().max().item()
                max_diff_det = max(max_diff_det, diff)
        print(
            f"scatter_add_ {n_runs} runs (deterministic mode): max diff across all pairs = {max_diff_det:.3e}"
        )
    except RuntimeError as e:
        print("Deterministic mode raised (some ops lack a deterministic impl):", e)
    finally:
        torch.use_deterministic_algorithms(False)
else:
    print(
        "Skipped: nondeterminism is a CUDA-atomics phenomenon; "
        "CPU scatter-add is typically deterministic (serial accumulation)."
    )

print("\nDone.")
