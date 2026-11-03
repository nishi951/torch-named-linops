"""Plain-assert verification for the case registry / sweep wiring."""

from pathlib import Path

import torch

from cases import CASES, Case, apply_recompiles, run_phase_sweep


def test_functional_case_runs_at_inner_level():
    case = Case(
        op="functional.fake",
        category="functional",
        build=lambda device, dtype, method: (
            (lambda t: t * 2.0),
            torch.ones(2, device=device, dtype=dtype),
            "fake functional op",
        ),
        methods=["fwd"],
        devices=["cuda"],
        dtypes=[torch.float32],
    )
    out = Path("/tmp/opencode/test-functional-sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = run_phase_sweep([case], "smoke", out)

    assert len(rows) == 3, len(rows)  # eager, default, fullgraph
    assert all(row.status == "ok" for row in rows), [
        (row.status, row.traceback_tail) for row in rows
    ]


COMPLEX_NATIVE = {
    "FFT",
    "Interpolate",
    "NUFFT",
    "FastSamplingNUFFT",
    "SamplingNUFFT",
    "ToeplitzPSF",
    "ArrayToBlocks",
    "BlocksToArray",
}


def test_smoke_cases_cover_float32_and_all_methods():
    for case in CASES:
        if case.category != "linop":
            continue
        assert case.methods == ["fwd", "adj", "normal"], case.op
        if case.op not in COMPLEX_NATIVE:
            assert torch.float32 in case.dtypes, case.op


def test_apply_recompiles_joins_cells():
    cells = [
        {"op": "Dense", "level": "wrapper", "config": "default", "recompiles": None}
    ]
    records = [{"op": "Dense", "level": "wrapper", "recompiles": 2}]

    apply_recompiles(cells, records)

    assert cells[0]["recompiles"] == 2, cells[0]


if __name__ == "__main__":
    test_functional_case_runs_at_inner_level()
    print("test_functional_case_runs_at_inner_level PASS")
    test_smoke_cases_cover_float32_and_all_methods()
    print("test_smoke_cases_cover_float32_and_all_methods PASS")
    test_apply_recompiles_joins_cells()
    print("test_apply_recompiles_joins_cells PASS")
