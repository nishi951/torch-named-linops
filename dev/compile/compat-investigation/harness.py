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
from dataclasses import asdict

import torch

from runner import CellSpec, execute_cell

PHASES = ["smoke", "deep", "recompile", "autograd", "pipeline", "all"]


def dense_sanity_cell() -> dict:
    torch.manual_seed(0)
    W = torch.randn(7, 3, device="cuda")
    bias = torch.randn(3, device="cuda")
    x = torch.randn(7, device="cuda")

    def target(t: torch.Tensor) -> torch.Tensor:
        return t @ W + bias

    spec = CellSpec(
        op="Dense",
        category="linop",
        method="fwd",
        device="cuda",
        dtype=torch.float32,
        level="wrapper",
        config="default",
    )
    return asdict(execute_cell(spec, target, x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="smoke")
    args = parser.parse_args()
    print(args.phase)
    if args.phase in ("all", "smoke"):
        print(dense_sanity_cell())


if __name__ == "__main__":
    main()
