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

from cases import CASES, RESULTS_DIR, SKIPPED, run_phase_sweep

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="smoke")
    args = parser.parse_args()
    print(args.phase)
    if args.phase in ("smoke", "all"):
        run_smoke()
    if args.phase in ("deep", "all"):
        run_deep()


if __name__ == "__main__":
    main()
