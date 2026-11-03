"""Cell-level runner: eager vs compiled execution, numerics, classification.

A cell is one point of the investigation matrix:
(op, method, device, dtype, compile level, compile config).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import torch

TOL = 1e-3
WARMUP = 20
ITERS = 100
SLOW_CALL_SECONDS = 0.5


@dataclass
class CellSpec:
    op: str
    category: str  # "linop" | "functional"
    method: str  # "fwd" | "adj" | "normal"
    device: str  # "cuda" | "cpu"
    dtype: torch.dtype
    level: str  # "wrapper" | "inner"
    config: str  # "eager" | "default" | "fullgraph"


@dataclass
class CellResult:
    spec: CellSpec
    status: str = "ok"  # "ok" | "error" | "n_a"
    cell_class: str = "works"
    graph_breaks: int = 0
    break_locations: list[str] = field(default_factory=list)
    recompiles: int | None = None
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    eager_time: float | None = None
    compile_time: float | None = None
    speedup: float | None = None
    traceback_tail: str | None = None
    notes: str = ""


def _to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()


def diff_metrics(
    reference_out: torch.Tensor, compiled_out: torch.Tensor
) -> tuple[float, float]:
    """Return (max_abs_diff, max_rel_diff), comparing complex tensors part-wise."""
    ref = _to_cpu(reference_out)
    out = _to_cpu(compiled_out)
    if ref.is_complex() or out.is_complex():
        ref = ref.to(torch.complex128)
        out = out.to(torch.complex128)
        parts = [
            (ref.real.to(torch.float64), out.real.to(torch.float64)),
            (ref.imag.to(torch.float64), out.imag.to(torch.float64)),
        ]
    else:
        parts = [(ref.to(torch.float64), out.to(torch.float64))]

    max_abs = 0.0
    max_rel = 0.0
    for a, b in parts:
        d = (a - b).abs()
        if d.numel() == 0:
            continue
        max_abs = max(max_abs, d.max().item())
        denom = a.abs().clamp_min(1e-12)
        max_rel = max(max_rel, (d / denom).max().item())
    return max_abs, max_rel


def run_cell(
    spec: CellSpec,
    reference_out: torch.Tensor,
    compiled_out: torch.Tensor,
    eager_time: float | None = None,
    compile_time: float | None = None,
    graph_breaks: int = 0,
    break_locations: list[str] | None = None,
    recompiles: int | None = None,
    notes: str = "",
) -> CellResult:
    """Classify one completed eager-vs-compiled comparison."""
    max_abs, max_rel = diff_metrics(reference_out, compiled_out)
    cell_class = "fails_silent" if max_rel > TOL else "works"
    speedup = None
    if eager_time is not None and compile_time is not None and compile_time > 0:
        speedup = eager_time / compile_time
    return CellResult(
        spec=spec,
        status="ok",
        cell_class=cell_class,
        graph_breaks=graph_breaks,
        break_locations=list(break_locations or []),
        recompiles=recompiles,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        eager_time=eager_time,
        compile_time=compile_time,
        speedup=speedup,
        traceback_tail=None,
        notes=notes,
    )


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _timed_call(
    fn: Callable[[], torch.Tensor], device: str
) -> tuple[torch.Tensor, float]:
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - start


def _is_unavailable(spec: CellSpec, exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "not implemented" in msg or "not supported" in msg:
        return True
    return spec.device == "cpu" and any(m in msg for m in ("cuda", "gpu", "triton"))


def _error_result(spec: CellSpec, exc: BaseException, phase: str) -> CellResult:
    unavailable = _is_unavailable(spec, exc)
    tail = "\n".join(traceback.format_exc().strip().splitlines()[-8:])
    return CellResult(
        spec=spec,
        status="n_a" if unavailable else "error",
        cell_class="n_a" if unavailable else "fails_loud",
        traceback_tail=tail,
        notes=f"raised during {phase}",
    )


def explain_graph_breaks(
    target: Callable[[torch.Tensor], torch.Tensor],
    input_: torch.Tensor,
) -> tuple[int, list[str]]:
    """Count and locate graph breaks via torch._dynamo.explain (best effort)."""
    try:
        explanation = torch._dynamo.explain(target)(input_)
    except Exception:
        return 0, []
    count = int(getattr(explanation, "graph_break_count", 0) or 0)
    reasons = getattr(explanation, "break_reasons", None) or []
    locations = []
    for reason in reasons:
        line = str(reason).splitlines()[0][:300] if str(reason) else ""
        locations.append(line)
    return count, locations


def execute_cell(
    spec: CellSpec,
    target: Callable[[torch.Tensor], torch.Tensor],
    input_: torch.Tensor,
) -> CellResult:
    """Run one matrix cell: eager reference, compiled run, classification."""
    try:
        ref, eager_time = _timed_call(lambda: target(input_), spec.device)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return _error_result(spec, exc, phase="eager")

    if spec.config == "eager":
        result = run_cell(spec, ref, ref, eager_time=eager_time)
        result.notes = "eager baseline"
        return result

    torch._dynamo.reset()
    try:
        compiled_fn = torch.compile(target, fullgraph=(spec.config == "fullgraph"))
        out, compile_time = _timed_call(lambda: compiled_fn(input_), spec.device)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return _error_result(spec, exc, phase="compiled")

    graph_breaks, break_locations = explain_graph_breaks(target, input_)
    return run_cell(
        spec,
        ref,
        out,
        eager_time=eager_time,
        compile_time=compile_time,
        graph_breaks=graph_breaks,
        break_locations=break_locations,
    )
