"""Shape-driven recompilation checks.

For each operator, compile once, then call the compiled function with two
different input shapes that the eager operator accepts. Record how many
graphs Dynamo built (recompiles) and whether results still agree.
"""

from __future__ import annotations

import torch

from runner import TOL, diff_metrics


def measure_recompiles(
    fn, inputs: list[torch.Tensor]
) -> tuple[int, list[torch.Tensor]]:
    """Run fn compiled once over several shapes; return (recompiles, outputs).

    Counts extra unique graphs Dynamo produced beyond the first. Note:
    torch._dynamo.utils exposes unique_graphs (not recompilation_reasons)
    on this torch version.
    """
    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    before = int(counters["stats"].get("unique_graphs", 0))
    compiled = torch.compile(fn)
    outputs = [compiled(x) for x in inputs]
    after = int(counters["stats"].get("unique_graphs", 0))
    unique_graphs = after - before
    return max(0, unique_graphs - 1), outputs


def _variant(input_: torch.Tensor) -> torch.Tensor | None:
    """A second shape accepted by batch-ellipsis operators, else None."""
    if input_.ndim < 2:
        return None
    shape = list(input_.shape)
    shape[0] *= 2
    return torch.randn(shape, dtype=input_.dtype, device=input_.device)


def run_recompile_check(case, spec, device: str, dtype: torch.dtype) -> dict:
    if spec.level == "inner" and case.inner_build is not None:
        builder = case.inner_build
    else:
        builder = case.build
    record = {
        "op": spec.op,
        "level": spec.level,
        "device": device,
        "dtype": str(dtype),
        "supported": False,
        "recompiles": None,
        "agrees_after": None,
        "max_rel_diff": None,
        "shapes": [],
        "notes": "",
    }
    if builder is None:
        record["notes"] = "no inner build"
        return record

    try:
        fn, input_, _ = builder(device, dtype, "fwd")
        variant = _variant(input_)
        if variant is None:
            record["notes"] = "no batch-shaped input to vary"
            return record
        eager_first = fn(input_)
        eager_second = fn(variant)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        record["notes"] = (f"eager rejected shape family: {type(exc).__name__}: {exc}")[
            :300
        ]
        return record

    try:
        recompiles, outputs = measure_recompiles(fn, [input_, variant])
        _, max_rel = diff_metrics(eager_second, outputs[1])
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        record["supported"] = True
        record["notes"] = (
            f"compiled shape change failed: {type(exc).__name__}: {exc}"
        )[:300]
        return record

    record.update(
        supported=True,
        recompiles=recompiles,
        agrees_after=bool(max_rel <= TOL),
        max_rel_diff=max_rel,
        shapes=[list(input_.shape), list(variant.shape)],
        notes="eager accepted both shapes",
    )
    del eager_first
    return record
