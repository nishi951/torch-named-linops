"""Plain-assert verification for the cell runner (scratch code, not pytest)."""

from pathlib import Path

import torch

from runner import (
    CallCounter,
    CellSpec,
    capture_graph_breaks,
    execute_cell,
    median_time,
    run_cell,
)


def _spec(config: str = "default") -> CellSpec:
    return CellSpec(
        op="Dense",
        category="linop",
        method="fwd",
        device="cuda",
        dtype=torch.float32,
        level="wrapper",
        config=config,
    )


def test_correct_target_works():
    torch.manual_seed(0)
    W = torch.randn(7, 3, device="cuda")
    bias = torch.randn(3, device="cuda")
    x = torch.randn(7, device="cuda")

    def target(t: torch.Tensor) -> torch.Tensor:
        return t @ W + bias

    res = execute_cell(_spec(), target, x)
    assert res.status == "ok", res.traceback_tail
    assert res.cell_class == "works", res.cell_class
    assert res.max_rel_diff is not None and res.max_rel_diff < 1e-3, res.max_rel_diff


def test_silent_disagreement_detected():
    torch.manual_seed(0)
    ref = torch.randn(4, device="cuda")
    bad = ref * 1.01

    res = run_cell(
        _spec(),
        reference_out=ref,
        compiled_out=bad,
        eager_time=1.0,
        compile_time=0.5,
    )
    assert res.cell_class == "fails_silent", res.cell_class
    assert res.speedup is not None and abs(res.speedup - 2.0) < 1e-6, res.speedup


def test_timing_helper_counts_warmup_plus_iters():
    torch.manual_seed(0)
    x = torch.randn(7, device="cuda")
    counter = CallCounter()

    median_time(lambda: x + 1.0, warmup=3, iters=5, device="cuda", counter=counter)

    assert counter.count >= 3 + 5, counter.count


def test_execute_cell_reports_positive_timings():
    torch.manual_seed(0)
    W = torch.randn(7, 3, device="cuda")
    x = torch.randn(7, device="cuda")

    res = execute_cell(_spec(), lambda t: t @ W, x)

    assert res.eager_time is not None and res.eager_time > 0, res.eager_time
    assert res.compile_time is not None and res.compile_time > 0, res.compile_time


def test_capture_graph_breaks_writes_log():
    def target(t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] > 4:
            return t + 1.0
        return t - 1.0

    count, locations = capture_graph_breaks(
        target, torch.randn(8, device="cuda"), "breaking-test"
    )

    log_path = Path(__file__).parent / "logs" / "breaking-test.log"
    assert log_path.exists(), log_path
    assert isinstance(count, int)
    assert isinstance(locations, list)


def test_fullgraph_unsupported_is_fails_loud():
    def target(t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Unsupported: logging.Logger method not supported for non-export cases"
        )

    res = execute_cell(_spec(config="fullgraph"), target, torch.randn(3, device="cuda"))

    assert res.status == "error", res.status
    assert res.cell_class == "fails_loud", res.cell_class


def test_cuda_only_op_on_cpu_is_na():
    def target(t: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("requires CUDA but no GPU is available")

    spec = _spec()
    spec.device = "cpu"
    res = execute_cell(spec, target, torch.randn(3))

    assert res.status == "n_a", res.status
    assert res.cell_class == "n_a", res.cell_class


def test_cupy_blocked_out_of_scope():
    def target(t: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("cupy is untraceable: _cupy_pytorch imported")

    res = execute_cell(_spec(), target, torch.randn(3, device="cuda"))

    assert res.cell_class == "blocked_out_of_scope", res.cell_class


if __name__ == "__main__":
    test_correct_target_works()
    print("test_correct_target_works PASS")
    test_silent_disagreement_detected()
    print("test_silent_disagreement_detected PASS")
    test_timing_helper_counts_warmup_plus_iters()
    print("test_timing_helper_counts_warmup_plus_iters PASS")
    test_execute_cell_reports_positive_timings()
    print("test_execute_cell_reports_positive_timings PASS")
    test_capture_graph_breaks_writes_log()
    print("test_capture_graph_breaks_writes_log PASS")
    test_fullgraph_unsupported_is_fails_loud()
    print("test_fullgraph_unsupported_is_fails_loud PASS")
    test_cuda_only_op_on_cpu_is_na()
    print("test_cuda_only_op_on_cpu_is_na PASS")
    test_cupy_blocked_out_of_scope()
    print("test_cupy_blocked_out_of_scope PASS")
