"""Full-level compiled benchmark phases.

Recompile, autograd, and pipeline checks, ported from the compile
compatibility investigation. Phase axes stay as investigated: CUDA-only,
forward-only for recompile, deep-dive-only for autograd, and each case's
first dtype.
"""

from __future__ import annotations

import statistics
import time
import traceback

import torch
from torch import Tensor

from benchmarks._compiled_cases import CASES, levels_for, target_for_spec
from benchmarks._compiled_runner import (
    TOL,
    CellSpec,
    capture_graph_breaks,
    check_grad_agreement,
    diff_metrics,
    measure_recompiles,
)
from torchlinops import NUFFT, Chain, Dense, Diagonal, Dim, Identity
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs

STEPS = 20
ALPHA = 0.5
REPEATS = 3


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


# --- Recompile phase ------------------------------------------------------


def _variant(input_: Tensor) -> Tensor | None:
    """A second shape accepted by batch-ellipsis operators, else None."""
    if input_.ndim < 2:
        return None
    shape = list(input_.shape)
    shape[0] *= 2
    return torch.randn(shape, dtype=input_.dtype, device=input_.device)


def run_recompile_check(case, spec: CellSpec, device: str, dtype: torch.dtype) -> dict:
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
    try:
        fn, input_ = target_for_spec(case, spec)
        variant = _variant(input_)
        if variant is None:
            record["notes"] = "no batch-shaped input to vary"
            return record
        fn(input_)
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
    return record


def run_recompile_phase(device: str = "cuda") -> list[dict]:
    rows = []
    for case in CASES:
        for level in levels_for(case, "full"):
            spec = CellSpec(
                op=case.op,
                category=case.category,
                method="fwd",
                device=device,
                dtype=case.dtypes[0],
                level=level,
                config="default",
            )
            rows.append(run_recompile_check(case, spec, device, case.dtypes[0]))
    return rows


# --- Autograd phase -------------------------------------------------------


def run_autograd_check(case, spec: CellSpec, device: str, dtype: torch.dtype) -> dict:
    row = {
        "op": spec.op,
        "level": spec.level,
        "device": device,
        "dtype": str(dtype),
        "method": "grad",
        "config": "default",
    }
    try:
        fn, x = target_for_spec(case, spec)
        y = torch.randn_like(fn(x).detach())
    except Exception:  # noqa: BLE001 - recorded, not swallowed
        row.update(
            status="error",
            cell_class="fails_loud",
            traceback_tail=traceback.format_exc().strip()[-800:],
            notes="case build failed",
        )
        return row

    row.update(check_grad_agreement(fn, x, y))
    return row


def run_autograd_phase(device: str = "cuda") -> list[dict]:
    rows = []
    for case in CASES:
        if not case.deep_dive:
            continue
        for level in levels_for(case, "full"):
            spec = CellSpec(
                op=case.op,
                category=case.category,
                method="grad",
                device=device,
                dtype=case.dtypes[0],
                level=level,
                config="default",
            )
            rows.append(run_autograd_check(case, spec, device, case.dtypes[0]))
    return rows


# --- Pipeline phase -------------------------------------------------------


def _timed(fn, x: Tensor, device: str) -> tuple[Tensor, float]:
    out = fn(x)
    _sync(device)
    durations = []
    for _ in range(REPEATS):
        _sync(device)
        start = time.perf_counter()
        out = fn(x)
        _sync(device)
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


def pipeline_break_kind(locations: list[str]) -> str:
    lowered = " ".join(locations).lower()
    if "triton" in lowered or "inductor" in lowered:
        return "inner (triton/inductor)"
    if "logging" in lowered or "logger" in lowered or "namedlinop" in lowered:
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
        "graph_breaks": None,
        "break_locations": [],
        "recompiles": None,
        "max_abs_diff": None,
        "max_rel_diff": None,
        "eager_time": None,
        "compiled_time": None,
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

    def step(x: Tensor) -> Tensor:
        for _ in range(STEPS):
            x = x - ALPHA * chain_H(chain(x) - b)
        return x

    rows = []
    eager_x, eager_time = _timed(step, x0, device)
    rows.append(
        _row(
            f"pipeline.{name}",
            "loop",
            "eager",
            dtype,
            device=device,
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
            out, compiled_time = _timed(compiled, x0, device)
        except Exception:  # noqa: BLE001 - recorded, not swallowed
            errors = traceback.format_exc().strip()[-2000:]
        if errors is not None:
            rows.append(
                _row(
                    f"pipeline.{name}",
                    "loop",
                    config,
                    dtype,
                    device=device,
                    status="error",
                    cell_class="fails_loud",
                    traceback_tail=errors,
                )
            )
            continue

        max_abs, max_rel = diff_metrics(eager_x, out)
        if config == "default":
            breaks, locations = capture_graph_breaks(
                step, x0, f"pipeline-{name}-{config}"
            )
        else:
            breaks, locations = None, []
        rows.append(
            _row(
                f"pipeline.{name}",
                "loop",
                config,
                dtype,
                device=device,
                cell_class="fails_silent" if max_rel > TOL else "works",
                graph_breaks=breaks,
                break_locations=locations,
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                eager_time=eager_time,
                compiled_time=compiled_time,
                speedup=eager_time / compiled_time if compiled_time else None,
                notes=f"break-kind: {pipeline_break_kind(locations)}",
            )
        )

    torch._dynamo.reset()
    try:
        compiled_chain = torch.compile(chain)
        compiled_chain_H = torch.compile(chain_H)

        def step_compiled_inners(x: Tensor) -> Tensor:
            for _ in range(STEPS):
                x = x - ALPHA * compiled_chain_H(compiled_chain(x) - b)
            return x

        out, compiled_time = _timed(step_compiled_inners, x0, device)
        max_abs, max_rel = diff_metrics(eager_x, out)
        rows.append(
            _row(
                f"pipeline.{name}",
                "compiled-inners",
                "default",
                dtype,
                device=device,
                cell_class="fails_silent" if max_rel > TOL else "works",
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                eager_time=eager_time,
                compiled_time=compiled_time,
                speedup=eager_time / compiled_time if compiled_time else None,
                notes="eager loop with individually compiled linops",
            )
        )
    except Exception:  # noqa: BLE001 - recorded, not swallowed
        rows.append(
            _row(
                f"pipeline.{name}",
                "compiled-inners",
                "default",
                dtype,
                device=device,
                status="error",
                cell_class="fails_loud",
                traceback_tail=traceback.format_exc().strip()[-2000:],
            )
        )

    return rows


def run_pipeline_phase(device: str = "cuda") -> list[dict]:
    rows: list[dict] = []
    for name in CHAINS:
        try:
            rows.extend(_run_chain(name, device=device))
        except Exception:  # noqa: BLE001 - recorded, not swallowed
            rows.append(
                _row(
                    f"pipeline.{name}",
                    "loop",
                    "build",
                    torch.float32,
                    device=device,
                    status="error",
                    cell_class="fails_loud",
                    graph_breaks=None,
                    traceback_tail=traceback.format_exc().strip()[-2000:],
                    notes="pipeline construction or eager run failed",
                )
            )
    return rows
