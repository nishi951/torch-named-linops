"""Compile profiling helpers for downstream users."""

import statistics
import time
import traceback
from dataclasses import dataclass, field

import torch

_VARIANT_ALIASES = {
    "forward": "forward",
    "fwd": "forward",
    "adjoint": "adjoint",
    "adj": "adjoint",
    "normal": "normal",
    "norm": "normal",
}


@dataclass
class CompileCheckRow:
    """Eager-vs-compiled result for one operator variant.

    Attributes
    ----------
    variant : str
        ``"forward"``, ``"adjoint"``, or ``"normal"``.
    status : str
        ``"ok"``, ``"numeric_mismatch"``, or ``"error"``.
    eager_time, compiled_time : float or None
        Median per-call times in seconds.
    speedup : float or None
        ``eager_time / compiled_time``.
    max_abs_diff, max_rel_diff : float or None
        Compiled-vs-eager differences.
    graph_breaks : int or None
        Measured graph-break count, or ``None`` when not measured.
    break_locations : list of str
        Up to three formatted graph-break locations.
    input_shape, output_shape : tuple or None
        Input and eager-output shapes.
    input_bytes, output_bytes : int or None
        Input and eager-output tensor byte sizes.
    error : str or None
        Trimmed traceback for failed variants.
    """

    variant: str
    status: str = "ok"
    eager_time: float | None = None
    compiled_time: float | None = None
    speedup: float | None = None
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    graph_breaks: int | None = None
    break_locations: list[str] = field(default_factory=list)
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    input_bytes: int | None = None
    output_bytes: int | None = None
    error: str | None = None


@dataclass
class CompileCheckResult:
    """Result returned by :func:`compile_check`.

    Attributes
    ----------
    device : str
        Device taken from the sample input.
    dtype : torch.dtype
        Dtype taken from the sample input.
    input_shape, output_shape : tuple
        Sample input and eager-output shapes.
    input_bytes, output_bytes : int
        Sample input and eager-output byte sizes.
    rows : list of CompileCheckRow
        One row per requested variant.
    """

    device: str
    dtype: torch.dtype
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_bytes: int
    output_bytes: int
    rows: list[CompileCheckRow]

    def summary(self) -> str:
        lines = [
            (
                f"compile_check: device={self.device} dtype={self.dtype} "
                f"input={self.input_shape} ({self.input_bytes} B) "
                f"output={self.output_shape} ({self.output_bytes} B)"
            ),
            "",
            "| variant | status | eager | compiled | speedup | max rel diff | breaks |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in self.rows:
            lines.append(
                f"| {row.variant} | {row.status} | {_fmt_s(row.eager_time)} | "
                f"{_fmt_s(row.compiled_time)} | {_fmt_x(row.speedup)} | "
                f"{_fmt_f(row.max_rel_diff)} | "
                f"{'—' if row.graph_breaks is None else row.graph_breaks} |"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _fmt_s(value):
    return "—" if value is None else f"{value * 1e6:.1f} us"


def _fmt_x(value):
    return "—" if value is None else f"{value:.2f}x"


def _fmt_f(value):
    return "—" if value is None else f"{value:.3g}"


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _median_time(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    sync()
    fn()
    sync()
    for _ in range(max(warmup - 1, 0)):
        fn()
    sync()
    durations = []
    for _ in range(iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)


def _diff_metrics(
    reference: torch.Tensor, compiled: torch.Tensor
) -> tuple[float, float]:
    reference = reference.detach().cpu()
    compiled = compiled.detach().cpu()
    if reference.shape != compiled.shape:
        return float("inf"), float("inf")
    if not (torch.isfinite(reference).all() and torch.isfinite(compiled).all()):
        return float("inf"), float("inf")
    diff = (reference - compiled).abs()
    if diff.numel() == 0:
        return 0.0, 0.0
    denom = reference.abs().clamp_min(1e-12)
    return diff.max().item(), (diff / denom).max().item()


def _graph_breaks(target, input_, fullgraph: bool) -> tuple[int | None, list[str]]:
    if fullgraph:
        return None, []
    try:
        explanation = torch._dynamo.explain(target)(input_)
        count = max(0, int(getattr(explanation, "graph_break_count", 0) or 0))
        locations = []
        for reason in (getattr(explanation, "break_reasons", None) or [])[:3]:
            text = str(getattr(reason, "reason", reason)).strip().splitlines()
            first = text[0][:220] if text else "unknown graph break"
            stack = getattr(reason, "user_stack", None) or []
            if stack:
                frame = stack[0]
                first = f"{frame.filename}:{frame.lineno} in {frame.name}: {first}"
            locations.append(first[:300])
        return count, locations
    except Exception as exc:  # noqa: BLE001 - profiling helper records failures
        return None, [f"explain failed: {type(exc).__name__}: {exc}"]


def compile_check(
    linop,
    sample_input: torch.Tensor,
    variants: list[str] | tuple[str, ...] = ("forward", "adjoint"),
    *,
    warmup: int = 5,
    iters: int = 20,
    fullgraph: bool = False,
    explain_graph_breaks: bool = True,
    print_results: bool = True,
) -> CompileCheckResult:
    """Profile eager vs ``torch.compile`` for a user-provided linop.

    This always benchmarks the wrapped operator (``linop``, ``linop.H``, or
    ``linop.N``), which is what users call in practice. The sample input sets
    the device, dtype, and input shape. An output-shaped sample is constructed
    independently for adjoint profiling.

    Parameters
    ----------
    linop : NamedLinop
        Operator to profile.
    sample_input : torch.Tensor
        Input-shaped sample. Determines device, dtype, and shape.
    variants : sequence of str, default ("forward", "adjoint")
        Variants to run. Accepts ``"forward"``/``"fwd"``,
        ``"adjoint"``/``"adj"``, and ``"normal"``/``"norm"``.
    warmup, iters : int
        Timing protocol for both eager and compiled calls.
    fullgraph : bool, default False
        Passed to ``torch.compile``. Graph breaks are not measured when true.
    explain_graph_breaks : bool, default True
        Measure graph breaks with ``torch._dynamo.explain``.
    print_results : bool, default True
        Print a summary table before returning.

    Returns
    -------
    CompileCheckResult
        Tensor sizes, timings, speedups, numerical differences, and break
        locations for each requested variant.

    Notes
    -----
    Autograd/backward profiling is not included yet.
    """
    if not isinstance(sample_input, torch.Tensor):
        raise TypeError("sample_input must be a torch.Tensor")
    if not variants:
        raise ValueError("At least one variant must be requested")

    normalized = []
    for variant in variants:
        key = _VARIANT_ALIASES.get(str(variant).lower())
        if key is None:
            raise ValueError(
                f"Unknown variant {variant!r}; use forward, adjoint, or normal"
            )
        if key not in normalized:
            normalized.append(key)

    device = sample_input.device
    try:
        eager_output = linop(sample_input)
    except Exception as exc:
        raise RuntimeError(
            "compile_check could not run the linop on sample_input; check the "
            "input shape and dtype"
        ) from exc
    output_sample = torch.randn_like(eager_output)

    rows: list[CompileCheckRow] = []
    for variant in normalized:
        if variant == "forward":
            target = lambda t: linop(t)
            input_ = sample_input
        elif variant == "adjoint":
            target = lambda t: linop.H(t)
            input_ = output_sample
        else:
            target = lambda t: linop.N(t)
            input_ = sample_input

        row = CompileCheckRow(
            variant=variant,
            input_shape=tuple(input_.shape),
            output_shape=tuple(eager_output.shape),
            input_bytes=_tensor_bytes(input_),
            output_bytes=_tensor_bytes(eager_output),
        )
        try:
            reference = target(input_)
            row.eager_time = _median_time(
                lambda target=target, input_=input_: target(input_),
                warmup=warmup,
                iters=iters,
                device=device,
            )
        except Exception:  # noqa: BLE001 - recorded in the result row
            row.status = "error"
            row.error = traceback.format_exc().strip()[-1200:]
            rows.append(row)
            continue

        torch._dynamo.reset()
        try:
            compiled = torch.compile(target, fullgraph=fullgraph)
            compiled_out = compiled(input_)
            row.compiled_time = _median_time(
                lambda compiled=compiled, input_=input_: compiled(input_),
                warmup=warmup,
                iters=iters,
                device=device,
            )
            row.max_abs_diff, row.max_rel_diff = _diff_metrics(reference, compiled_out)
            row.status = (
                "ok" if (row.max_rel_diff or 0.0) <= 1e-3 else "numeric_mismatch"
            )
            if explain_graph_breaks:
                row.graph_breaks, row.break_locations = _graph_breaks(
                    target, input_, fullgraph
                )
        except Exception:  # noqa: BLE001 - recorded in the result row
            row.status = "error"
            row.error = traceback.format_exc().strip()[-1200:]
        if row.eager_time and row.compiled_time:
            row.speedup = row.eager_time / row.compiled_time
        rows.append(row)

    result = CompileCheckResult(
        device=str(device),
        dtype=sample_input.dtype,
        input_shape=tuple(sample_input.shape),
        output_shape=tuple(eager_output.shape),
        input_bytes=_tensor_bytes(sample_input),
        output_bytes=_tensor_bytes(eager_output),
        rows=rows,
    )
    if print_results:
        print(result.summary())
    return result


__all__ = ["CompileCheckResult", "CompileCheckRow", "compile_check"]
