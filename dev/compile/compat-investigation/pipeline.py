"""End-to-end pipeline check: fixed-point iteration over a composed chain.

Two chains:
  simple -- batch-polymorphic Dense @ Identity (should compile end-to-end)
  nufft  -- tutorial-shaped NUFFT @ Diagonal (realistic MRI recon cost)

For each chain: eager loop vs compiled whole loop (default, fullgraph) vs an
eager loop around individually compiled linops ("compiled-inners" pattern).
"""

from __future__ import annotations

import statistics
import time

import torch
from torchlinops import Chain, Dense, Diagonal, Dim, Identity, NUFFT
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs

from cases import RESULTS_DIR, update_results_json
from runner import TOL, capture_graph_breaks, diff_metrics

STEPS = 20
ALPHA = 0.5
REPEATS = 3


def _sync() -> None:
    torch.cuda.synchronize()


def _timed(fn, x: torch.Tensor) -> tuple[torch.Tensor, float]:
    out = fn(x)
    _sync()
    durations = []
    for _ in range(REPEATS):
        _sync()
        start = time.perf_counter()
        out = fn(x)
        _sync()
        durations.append(time.perf_counter() - start)
    return out, statistics.median(durations)


def _simple_chain(device: str):
    W = torch.randn(8, 16, device=device)
    dense = Dense(W, ("M", "N"), ("B", "N"), ("B", "M"), broadcast_dims=["B"])
    identity = Identity(ishape=("B", "M"))
    return Chain(dense, identity), torch.randn(2, 16, device=device)


def _nufft_chain(device: str):
    locs = get_valid_locs((2, 8), (16, 16), 2, 4.0, "cpu", centered=True).to(device)
    nufft = NUFFT(
        locs.clone(),
        (16, 16),
        output_shape=("R",),
        batch_shape=Dim("B"),
        width=4.0,
        oversamp=1.25,
    )
    nufft._locs_orig = locs.clone()
    dcf = Diagonal(torch.randn(2, 8, dtype=torch.complex64, device=device), ("B", "R"))
    return Chain(nufft, dcf), torch.randn(
        2, 16, 16, dtype=torch.complex64, device=device
    )


CHAINS = {"simple": _simple_chain, "nufft": _nufft_chain}


def _break_kind(locations: list[str]) -> str:
    lowered = " ".join(locations).lower()
    if "triton" in lowered or "inductor" in lowered:
        return "inner (triton/inductor)"
    if "logging" in lowered or "namedlinop" in lowered:
        return "wrapper (logging/namedlinop)"
    if locations:
        return "unattributed"
    return "none"


def _row(op: str, level: str, config: str, dtype: torch.dtype, **fields) -> dict:
    row = {
        "op": op,
        "category": "pipeline",
        "method": "pipeline",
        "level": level,
        "config": config,
        "device": "cuda",
        "dtype": str(dtype),
        "status": "ok",
        "cell_class": "works",
        "graph_breaks": 0,
        "break_locations": [],
        "recompiles": None,
        "max_abs_diff": None,
        "max_rel_diff": None,
        "eager_time": None,
        "compile_time": None,
        "speedup": None,
        "traceback_tail": None,
        "notes": "",
    }
    row.update(fields)
    return row


def _run_chain(name: str, device: str = "cuda") -> list[dict]:
    chain, x0 = CHAINS[name](device)
    dtype = x0.dtype
    b = chain(x0).detach()
    chain_H = chain.H

    def step(x: torch.Tensor) -> torch.Tensor:
        for _ in range(STEPS):
            x = x - ALPHA * chain_H(chain(x) - b)
        return x

    rows = []
    eager_x, eager_time = _timed(step, x0)
    rows.append(
        _row(
            f"pipeline.{name}",
            "loop",
            "eager",
            dtype,
            eager_time=eager_time,
            notes=f"{STEPS}-step fixed point",
        )
    )

    for config in ("default", "fullgraph"):
        torch._dynamo.reset()
        errors = None
        try:
            compiled = torch.compile(step, fullgraph=(config == "fullgraph"))
            compiled(x0)
            out, compile_time = _timed(compiled, x0)
        except Exception:  # noqa: BLE001 - recorded, not swallowed
            import traceback

            errors = traceback.format_exc().strip()[-2000:]
        if errors is not None:
            rows.append(
                _row(
                    f"pipeline.{name}",
                    "loop",
                    config,
                    dtype,
                    status="error",
                    cell_class="fails_loud",
                    traceback_tail=errors,
                )
            )
            continue

        max_abs, max_rel = diff_metrics(eager_x, out)
        breaks, locations = capture_graph_breaks(step, x0, f"pipeline-{name}-{config}")
        recompiles, recompile_note = None, "recompiles covered by the dedicated phase"
        rows.append(
            _row(
                f"pipeline.{name}",
                "loop",
                config,
                dtype,
                cell_class="fails_silent" if max_rel > TOL else "works",
                graph_breaks=breaks,
                break_locations=locations,
                recompiles=recompiles,
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                eager_time=eager_time,
                compile_time=compile_time,
                speedup=eager_time / compile_time if compile_time else None,
                notes=(f"break-kind: {_break_kind(locations)}; {recompile_note}"),
            )
        )

    torch._dynamo.reset()
    try:
        compiled_chain = torch.compile(chain)
        compiled_chain_H = torch.compile(chain_H)

        def step_compiled_inners(x: torch.Tensor) -> torch.Tensor:
            for _ in range(STEPS):
                x = x - ALPHA * compiled_chain_H(compiled_chain(x) - b)
            return x

        out, compile_time = _timed(step_compiled_inners, x0)
        max_abs, max_rel = diff_metrics(eager_x, out)
        rows.append(
            _row(
                f"pipeline.{name}",
                "compiled-inners",
                "default",
                dtype,
                cell_class="fails_silent" if max_rel > TOL else "works",
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                eager_time=eager_time,
                compile_time=compile_time,
                speedup=eager_time / compile_time if compile_time else None,
                notes="eager loop with individually compiled linops",
            )
        )
    except Exception:  # noqa: BLE001 - recorded, not swallowed
        import traceback

        rows.append(
            _row(
                f"pipeline.{name}",
                "compiled-inners",
                "default",
                dtype,
                status="error",
                cell_class="fails_loud",
                traceback_tail=traceback.format_exc().strip()[-2000:],
            )
        )

    return rows


def run_pipeline(
    chains: list[str] | None = None,
    write: bool = True,
    out_path=None,
) -> list[dict]:
    if chains is None:
        chains = list(CHAINS)
    rows: list[dict] = []
    for name in chains:
        rows.extend(_run_chain(name))
    if write:
        update_results_json(out_path or RESULTS_DIR / "results.json", "pipeline", rows)
    return rows
