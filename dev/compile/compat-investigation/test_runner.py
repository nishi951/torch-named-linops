"""Plain-assert verification for the cell runner (scratch code, not pytest)."""

import torch

from runner import CellSpec, execute_cell, run_cell


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


if __name__ == "__main__":
    test_correct_target_works()
    print("test_correct_target_works PASS")
    test_silent_disagreement_detected()
    print("test_silent_disagreement_detected PASS")
