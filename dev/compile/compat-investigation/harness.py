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

PHASES = ["smoke", "deep", "recompile", "autograd", "pipeline", "all"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="smoke")
    args = parser.parse_args()
    print(args.phase)


if __name__ == "__main__":
    main()
