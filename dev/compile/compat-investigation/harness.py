"""Orchestrator CLI for the torch.compile compatibility investigation.

Phases:
    smoke      -- full shallow sweep over every linop
    deep       -- deep-dive sweep over the suspicious operators
    recompile  -- shape-driven recompilation checks
    autograd   -- single-op backward checks on deep-dive ops
    pipeline   -- end-to-end iteration-loop check
    all        -- every phase above, in order
"""

import argparse
from collections import Counter

from autograd_check import run_autograd_check
from cases import CASES, RESULTS_DIR, SKIPPED, run_phase_sweep, update_results_json
from pipeline import run_pipeline
from recompile import run_recompile_check
from runner import CellSpec

PHASES = ["smoke", "deep", "recompile", "autograd", "pipeline", "all"]


def _summary(rows) -> str:
    counts = Counter(row.cell_class for row in rows)
    return f"{len(rows)} cells: " + ", ".join(
        f"{name}={count}" for name, count in sorted(counts.items())
    )


def run_smoke() -> None:
    rows = run_phase_sweep(
        [case for case in CASES if not case.deep_dive],
        "smoke",
        RESULTS_DIR / "results.json",
    )
    print(_summary(rows))
    for op, reason in SKIPPED.items():
        print(f"skipped {op}: {reason}")


def run_deep() -> None:
    rows = run_phase_sweep(
        [case for case in CASES if case.deep_dive],
        "deep",
        RESULTS_DIR / "results.json",
    )
    print(_summary(rows))


def run_recompile() -> None:
    records = []
    for case in CASES:
        if case.category == "functional":
            levels = ["inner"]
        elif case.deep_dive and case.inner_build is not None:
            levels = ["wrapper", "inner"]
        else:
            levels = ["wrapper"]
        for level in levels:
            spec = CellSpec(
                op=case.op,
                category=case.category,
                method="fwd",
                device="cuda",
                dtype=case.dtypes[0],
                level=level,
                config="default",
            )
            records.append(run_recompile_check(case, spec, "cuda", case.dtypes[0]))

    update_results_json(RESULTS_DIR / "results.json", "recompile", records)
    supported = sum(1 for record in records if record["supported"])
    recompiled = sum(
        1 for record in records if record["supported"] and record["recompiles"]
    )
    print(
        f"{len(records)} recompile checks: {supported} with a shape family, "
        f"{recompiled} recompiled at least once"
    )


def run_autograd() -> None:
    rows = []
    for case in CASES:
        if not case.deep_dive:
            continue
        if case.category == "functional":
            levels = ["inner"]
        elif case.inner_build is not None:
            levels = ["wrapper", "inner"]
        else:
            levels = ["wrapper"]
        for level in levels:
            spec = CellSpec(
                op=case.op,
                category=case.category,
                method="grad",
                device="cuda",
                dtype=case.dtypes[0],
                level=level,
                config="default",
            )
            rows.append(run_autograd_check(case, spec, "cuda", case.dtypes[0]))

    update_results_json(RESULTS_DIR / "results.json", "autograd", rows)
    counts = Counter(row["cell_class"] for row in rows)
    print(
        f"{len(rows)} autograd checks: "
        + ", ".join(f"{name}={n}" for name, n in sorted(counts.items()))
    )


def run_pipeline_phase() -> None:
    rows = run_pipeline()
    counts = Counter(row["cell_class"] for row in rows)
    print(
        f"{len(rows)} pipeline rows: "
        + ", ".join(f"{name}={n}" for name, n in sorted(counts.items()))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="smoke")
    args = parser.parse_args()
    print(args.phase)
    if args.phase in ("smoke", "all"):
        run_smoke()
    if args.phase in ("deep", "all"):
        run_deep()
    if args.phase in ("recompile", "all"):
        run_recompile()
    if args.phase in ("autograd", "all"):
        run_autograd()
    if args.phase in ("pipeline", "all"):
        run_pipeline_phase()


if __name__ == "__main__":
    main()
