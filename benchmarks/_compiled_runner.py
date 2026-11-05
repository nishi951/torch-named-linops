"""Cell-level runner for compiled benchmarks.

A cell is one point of the compiled benchmark matrix:
(op, method, device, dtype, compile level, compile config).
"""

from __future__ import annotations

import statistics
import time
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

TOL = 1e-3
PROTOCOLS = {
    "reduced": {"warmup": 5, "iters": 20},
    "full": {"warmup": 20, "iters": 100},
}
SLOW_CALL_SECONDS = 0.5


@dataclass(frozen=True)
class CellSpec:
    op: str
    category: str  # "linop" | "functional"
    method: str  # "fwd" | "adj" | "normal"
    device: str  # "cuda" | "cpu"
    dtype: torch.dtype
    level: str  # "wrapper" | "inner"
    config: str  # "default" | "fullgraph"


@dataclass
class CellResult:
    spec: CellSpec
    status: str = "ok"  # "ok" | "error" | "n_a"
    cell_class: str = "works"
    graph_breaks: int | None = 0
    break_locations: list[str] = field(default_factory=list)
    recompiles: int | None = None
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    eager_time: float | None = None
    compiled_time: float | None = None
    speedup: float | None = None
    degraded: bool = False
    traceback_tail: str | None = None
    notes: str = ""


class CallCounter:
    """Counts fn invocations made through median_time, for warmup proofs."""

    def __init__(self) -> None:
        self.count = 0


def _to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()


def diff_metrics(
    reference_out: torch.Tensor, compiled_out: torch.Tensor
) -> tuple[float, float]:
    """Return (max_abs_diff, max_rel_diff), comparing complex tensors part-wise.

    Shape disagreement or non-finite values are disagreement by definition
    (infinite diffs), never a pass.
    """
    ref = _to_cpu(reference_out)
    out = _to_cpu(compiled_out)
    if ref.shape != out.shape:
        return float("inf"), float("inf")
    if not (torch.isfinite(ref).all() and torch.isfinite(out).all()):
        return float("inf"), float("inf")
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
    compiled_time: float | None = None,
    graph_breaks: int | None = 0,
    break_locations: list[str] | None = None,
    recompiles: int | None = None,
    degraded: bool = False,
    notes: str = "",
) -> CellResult:
    """Classify one completed eager-vs-compiled comparison."""
    max_abs, max_rel = diff_metrics(reference_out, compiled_out)
    cell_class = "fails_silent" if max_rel > TOL else "works"
    speedup = None
    if eager_time is not None and compiled_time is not None and compiled_time > 0:
        speedup = eager_time / compiled_time
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
        compiled_time=compiled_time,
        speedup=speedup,
        degraded=degraded,
        traceback_tail=None,
        notes=notes,
    )


DEGRADATION_MARKERS = (
    "does not support code generation for complex",
    "falling back to eager",
)


def detect_degradation(messages: list[str]) -> bool:
    lowered = " ".join(messages).lower()
    return any(marker in lowered for marker in DEGRADATION_MARKERS)


def effective_class(
    status: str,
    cell_class: str,
    graph_breaks: int | None,
    speedup: float | None,
    degraded: bool,
    config: str | None = None,
) -> str:
    if status == "n_a":
        return "n_a"
    if cell_class != "works":
        return cell_class
    if degraded:
        return "silent_degradation"
    if config == "default" and graph_breaks is None:
        return "works_with_unknown_breaks"
    if graph_breaks is not None and graph_breaks > 0:
        return "works_with_breaks"
    if speedup is not None and speedup < 0.9:
        return "compiles_slower"
    return "works"


def spec_key(spec: CellSpec) -> str:
    dtype = str(spec.dtype).replace("torch.", "")
    return f"{spec.op}-{spec.method}-{spec.device}-{dtype}-{spec.level}-{spec.config}"


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def median_time(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    device: str,
    counter: CallCounter | None = None,
) -> float:
    """Median per-call wall time after warmup; adapts down for slow calls."""

    def call() -> torch.Tensor:
        if counter is not None:
            counter.count += 1
        return fn()

    _sync(device)
    start = time.perf_counter()
    call()
    _sync(device)
    if time.perf_counter() - start > SLOW_CALL_SECONDS:
        iters = min(iters, 10)

    for _ in range(max(warmup - 1, 0)):
        call()
    _sync(device)

    durations = []
    for _ in range(iters):
        _sync(device)
        start = time.perf_counter()
        call()
        _sync(device)
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)


def _format_break_reason(reason) -> str:
    reason_text = str(getattr(reason, "reason", reason)).strip().splitlines()
    text = reason_text[0][:220] if reason_text else "unknown graph break"
    stack = getattr(reason, "user_stack", None) or []
    if stack:
        frame = stack[0]
        text = (f"{frame.filename}:{frame.lineno} in {frame.name}: {text}")[:300]
    return text


def capture_graph_breaks(
    target: Callable[[torch.Tensor], torch.Tensor],
    input_: torch.Tensor,
    spec_key: str,
    log_dir: Path | None = None,
) -> tuple[int | None, list[str]]:
    """Run torch._dynamo.explain and optionally write its report to a log file.

    Returns ``(None, [error])`` when explain itself fails, so callers cannot
    mistake a failed measurement for a clean zero-break run.
    """
    count: int | None = None
    locations: list[str] = []
    try:
        explanation = torch._dynamo.explain(target)(input_)
        count = max(0, int(getattr(explanation, "graph_break_count", 0) or 0))
        reasons = getattr(explanation, "break_reasons", None) or []
        locations = [_format_break_reason(reason) for reason in reasons[:3]]
        text = str(explanation)
        if locations:
            text += "\n\nBreak locations:\n" + "\n".join(locations)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        locations = [f"explain failed: {type(exc).__name__}: {exc}"]
        text = locations[0]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{spec_key}.log").write_text(text + "\n")
    return count, locations


def _is_unavailable(spec: CellSpec, exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if spec.device == "cpu" and any(m in msg for m in ("cuda", "gpu", "triton")):
        return True
    if "not implemented for" in msg and ("cpu" in msg or "cuda" in msg):
        return True
    return "requires cuda" in msg or "requires triton" in msg


def _is_blocked_out_of_scope(tail: str) -> bool:
    lowered = tail.lower()
    return any(marker in lowered for marker in ("cupy", "sigpy", "_cupy_pytorch"))


def _error_result(spec: CellSpec, exc: BaseException, phase: str) -> CellResult:
    unavailable = _is_unavailable(spec, exc)
    tail = "\n".join(traceback.format_exc().strip().splitlines()[-8:])
    if unavailable:
        status, cell_class = "n_a", "n_a"
    elif _is_blocked_out_of_scope(tail):
        status, cell_class = "error", "blocked_out_of_scope"
    else:
        status, cell_class = "error", "fails_loud"
    return CellResult(
        spec=spec,
        status=status,
        cell_class=cell_class,
        graph_breaks=None,
        traceback_tail=tail,
        notes=f"raised during {phase}",
    )


def execute_cell(
    spec: CellSpec,
    target: Callable[[torch.Tensor], torch.Tensor],
    input_: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    log_dir: Path | None = None,
) -> CellResult:
    """Run one matrix cell: eager reference, compiled run, classification."""
    try:
        ref = target(input_)
        eager_time = median_time(
            lambda: target(input_), warmup=warmup, iters=iters, device=spec.device
        )
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return _error_result(spec, exc, phase="eager")

    if spec.config == "eager":
        result = run_cell(spec, ref, ref, eager_time=eager_time)
        result.notes = "eager baseline"
        return result

    torch._dynamo.reset()
    try:
        compiled_fn = torch.compile(target, fullgraph=(spec.config == "fullgraph"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = compiled_fn(input_)
        degraded = detect_degradation([str(w.message) for w in caught])
        compiled_time = median_time(
            lambda: compiled_fn(input_), warmup=warmup, iters=iters, device=spec.device
        )
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return _error_result(spec, exc, phase="compiled")

    # Non-default configs do not get an explain pass, so their break count is
    # unknown rather than a measured zero.
    graph_breaks, break_locations = (None, [])
    if spec.config == "default":
        graph_breaks, break_locations = capture_graph_breaks(
            target, input_, spec_key(spec), log_dir=log_dir
        )
    return run_cell(
        spec,
        ref,
        out,
        eager_time=eager_time,
        compiled_time=compiled_time,
        graph_breaks=graph_breaks,
        break_locations=break_locations,
        degraded=degraded,
    )


def measure_recompiles(
    fn: Callable[[torch.Tensor], torch.Tensor], inputs: list[torch.Tensor]
) -> tuple[int, list[torch.Tensor]]:
    """Run fn compiled once over several shapes; return (recompiles, outputs).

    Recompiles = graphs Dynamo built for calls after the first (a graph-breaking
    first call may build several fragments; those are not recompiles).
    """
    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    compiled = torch.compile(fn)
    outputs = [compiled(inputs[0])]
    after_first = int(counters["stats"].get("unique_graphs", 0))
    for x in inputs[1:]:
        outputs.append(compiled(x))
    after_all = int(counters["stats"].get("unique_graphs", 0))
    return max(0, after_all - after_first), outputs


def classify_grads(eager_grad: torch.Tensor, compiled_grad: torch.Tensor) -> dict:
    max_abs, max_rel = diff_metrics(eager_grad, compiled_grad)
    return {
        "cell_class": "fails_silent" if max_rel > TOL else "works",
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
    }


def _loss_grad(fn, x: torch.Tensor, y: torch.Tensor, compiled: bool):
    leaf = x.detach().clone().requires_grad_(True)
    target = torch.compile(fn) if compiled else fn
    out = target(leaf)
    loss = (torch.abs(out - y) ** 2).sum()
    (grad,) = torch.autograd.grad(loss, leaf, allow_unused=True)
    return None if grad is None else grad.detach()


def check_grad_agreement(fn, x: torch.Tensor, y: torch.Tensor) -> dict:
    try:
        eager_grad = _loss_grad(fn, x, y, compiled=False)
    except Exception:  # noqa: BLE001 - recorded, not swallowed
        return {
            "status": "error",
            "cell_class": "fails_loud",
            "max_abs_diff": None,
            "max_rel_diff": None,
            "traceback_tail": traceback.format_exc().strip()[-800:],
            "notes": "raised during eager backward",
        }

    try:
        torch._dynamo.reset()
        compiled_grad = _loss_grad(fn, x, y, compiled=True)
    except Exception:  # noqa: BLE001 - recorded, not swallowed
        return {
            "status": "error",
            "cell_class": "fails_loud",
            "max_abs_diff": None,
            "max_rel_diff": None,
            "traceback_tail": traceback.format_exc().strip()[-800:],
            "notes": "raised during compiled backward",
        }

    if eager_grad is None or compiled_grad is None:
        return {
            "status": "error",
            "cell_class": "fails_loud",
            "max_abs_diff": None,
            "max_rel_diff": None,
            "traceback_tail": None,
            "notes": "no gradient produced",
        }

    result = classify_grads(eager_grad, compiled_grad)
    result.update(status="ok", traceback_tail=None, notes="")
    return result
