"""Plain-assert verification for the case registry / sweep wiring."""

from pathlib import Path

import torch

from cases import Case, run_phase_sweep


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


if __name__ == "__main__":
    test_functional_case_runs_at_inner_level()
    print("test_functional_case_runs_at_inner_level PASS")
