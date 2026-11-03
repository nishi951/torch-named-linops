"""Operator registry for the torch.compile compatibility investigation.

Each Case builds one operator instantiation on a given (device, dtype) and
exposes a callable + reference input per evaluation method.

Instantiation shapes are copied from the corresponding tests/test_*.py
fixtures (which are the repo's canonical construction patterns).
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from torchlinops import (
    FFT,
    Add,
    ArrayToBlocks,
    BlocksToArray,
    Chain,
    Concat,
    Convolution,
    Dense,
    Diagonal,
    Dim,
    FastSamplingNUFFT,
    Identity,
    Interpolate,
    NUFFT,
    Pad,
    PadDim,
    Rearrange,
    Repeat,
    Sampling,
    SamplingNUFFT,
    Scalar,
    ShapeSpec,
    SimpleLinop,
    Sleep,
    Stack,
    SumReduce,
    ToDevice,
    Truncate,
    Zero,
)
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs
from torchlinops.linops.nufft import toeplitz_psf

from runner import CellResult, CellSpec, csv_export, execute_cell

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_METHODS = ["fwd", "adj", "normal"]
CONFIGS = ["eager", "default", "fullgraph"]


@dataclass
class Case:
    op: str
    category: str
    build: Callable[
        [str, torch.dtype, str], tuple[Callable[[Tensor], Tensor], Tensor, str | None]
    ]
    methods: list[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    devices: list[str] = field(default_factory=lambda: ["cuda", "cpu"])
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.float32])
    deep_dive: bool = False


def _rand(shape, dtype: torch.dtype, device: str) -> Tensor:
    return torch.randn(shape, dtype=dtype, device=device)


def _call_for_method(
    A, x: Tensor, method: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    if method == "fwd":
        return (lambda t: A(t)), x
    if method == "adj":
        y = A(x)
        return (lambda t: A.H(t)), y
    if method == "normal":
        return (lambda t: A.N(t)), x
    raise ValueError(f"unknown method {method!r}")


def _make_case(
    op: str,
    construct: Callable[[str, torch.dtype], tuple[object, Tensor]],
    *,
    category: str = "linop",
    devices: list[str] | None = None,
    dtypes: list[torch.dtype] | None = None,
    methods: list[str] | None = None,
    deep_dive: bool = False,
    note: str | None = None,
) -> Case:
    def build(device: str, dtype: torch.dtype, method: str):
        A, x = construct(device, dtype)
        fn, inp = _call_for_method(A, x, method)
        return fn, inp, note

    return Case(
        op=op,
        category=category,
        build=build,
        methods=list(methods or DEFAULT_METHODS),
        devices=list(devices or ["cuda", "cpu"]),
        dtypes=list(dtypes or [torch.float32]),
        deep_dive=deep_dive,
    )


# --- Constructors ---------------------------------------------------------


def _dense(device, dtype):
    W = _rand((9, 3), dtype, device)
    return Dense(W, ("M", "N"), ("N",), ("M",)), _rand((3,), dtype, device)


def _dense_simple(device, dtype):
    W = _rand((5, 3), dtype, device)
    return Dense(W), _rand((3,), dtype, device)


def _diagonal(device, dtype):
    weight = _rand((10, 1, 1), dtype, device)
    return Diagonal(weight, ("M", "N", "P")), _rand((10, 5, 7), dtype, device)


def _identity(device, dtype):
    return Identity(ishape=("N", "M")), _rand((4, 5), dtype, device)


def _zero(device, dtype):
    return Zero(ishape=("N", "M")), _rand((4, 5), dtype, device)


def _shapespec(device, dtype):
    return ShapeSpec(ishape=("N", "M")), _rand((4, 5), dtype, device)


def _scalar(device, dtype):
    weight = 2.0 + 0j if dtype.is_complex else 2.0
    return Scalar(weight=weight, ioshape=("N",)), _rand((8,), dtype, device)


def _sleep(device, dtype):
    return Sleep(duration=0.0, ioshape=("N",)), _rand((8,), dtype, device)


def _fft(device, dtype):
    return FFT(ndim=2, centered=True), _rand((8, 10), torch.complex64, device)


def _convolution(device, dtype):
    weight = _rand((5,), dtype, device)
    conv = Convolution(weight, in_grid_shape=("x",), out_grid_shape=("x",))
    return conv, _rand((3, 16), dtype, device)


def _interpolate(device, dtype):
    locs = get_valid_locs((8,), (16, 16), 2, 4.0, "cpu", centered=True).to(device)
    op = Interpolate(locs, (16, 16), width=4.0, kernel="kaiser_bessel")
    return op, _rand((16, 16), torch.complex64, device)


def _nufft(device, dtype):
    locs = get_valid_locs((8,), (16, 16), 2, 4.0, "cpu", centered=True).to(device)
    op = NUFFT(locs.clone(), (16, 16), output_shape=("R",), width=4.0, oversamp=1.25)
    op._locs_orig = locs.clone()
    return op, _rand((16, 16), torch.complex64, device)


def _sampling_nufft_locs(device, grid_size=(16, 16), npts=8, oversamp=1.25):
    padded = tuple(int(oversamp * g) for g in grid_size)
    idx = torch.stack([torch.randint(0, p, (npts,)) for p in padded], dim=-1)
    locs = (idx - torch.tensor(padded) // 2).float() / oversamp
    return locs.to(device)


def _fast_sampling_nufft(device, dtype):
    locs = _sampling_nufft_locs(device)
    op = FastSamplingNUFFT(locs.clone(), (16, 16), output_shape=("R",), oversamp=1.25)
    return op, _rand((16, 16), torch.complex64, device)


def _sampling_nufft(device, dtype):
    locs = _sampling_nufft_locs(device)
    op = SamplingNUFFT(locs.clone(), (16, 16), output_shape=("R",), oversamp=1.25)
    op._locs_orig = locs.clone()
    return op, _rand((16, 16), torch.complex64, device)


def _toeplitz_psf(device, dtype):
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
    op = toeplitz_psf(nufft, oversamp=nufft.options["oversamp"])
    padded = int(16 * nufft.options["oversamp"])
    return op, _rand((2, padded, padded), torch.complex64, device)


def _sampling(device, dtype):
    N, R, K, B = 16, 3, 4, 2
    idx = torch.randint(0, N - 1, (R, K, 2)).to(device)
    op = Sampling.from_stacked_idx(idx, (N, N), ("R", "K"))
    return op, _rand((B, N, N), dtype, device)


def _array_to_blocks(device, dtype):
    op = ArrayToBlocks((8, 8), (4, 4), (4, 4))
    return op, _rand((2, 8, 8), torch.complex64, device)


def _blocks_to_array(device, dtype):
    op = BlocksToArray((8, 8), (4, 4), (4, 4))
    return op, _rand((2, 2, 4, 4), torch.complex64, device)


def _pad(device, dtype):
    return Pad((20, 20), (10, 10), Dim("XY")), _rand((10, 10), dtype, device)


def _truncate(device, dtype):
    op = Truncate(
        dim=1, from_length=5, to_length=3, ishape=("N", "M"), oshape=("N", "M")
    )
    return op, _rand((4, 5), dtype, device)


def _pad_dim(device, dtype):
    op = PadDim(dim=1, from_length=3, to_length=5, ishape=("N", "M"), oshape=("N", "M"))
    return op, _rand((4, 3), dtype, device)


def _rearrange(device, dtype):
    op = Rearrange(
        "(A B) C",
        "A B C",
        ishape=("Ab", "C"),
        oshape=("A", "B", "C"),
        axes_lengths={"A": 2, "B": 3},
    )
    return op, _rand((6, 4), dtype, device)


def _sum_reduce(device, dtype):
    return SumReduce(("A", "B", "C"), ("A", "B")), _rand((5, 2, 3), dtype, device)


def _repeat(device, dtype):
    op = Repeat({"C": 3}, ishape=("A", "B"), oshape=("A", "B", "C"), broadcast_dims=[])
    return op, _rand((4, 5), dtype, device)


def _add(device, dtype):
    A = Dense(_rand((5, 5), dtype, device), ("M", "N"), ("N",), ("M",))
    B = Dense(_rand((5, 5), dtype, device), ("M", "N"), ("N",), ("M",))
    C = Dense(_rand((5, 5), dtype, device), ("M", "N"), ("N",), ("M",))
    add = Add(A, B, C)
    add.threaded = False
    return add, _rand((5,), dtype, device)


def _stack(device, dtype):
    Q, P = 4, 3
    A = Dense(_rand((P, Q), dtype, device), ("P", "Q"), Dim("Q"), Dim("P"))
    B = Dense(_rand((P, Q), dtype, device), ("P", "Q"), Dim("Q"), Dim("P"))
    C = Stack(A, B, odim_and_idx=("M", 1))
    return C, _rand((Q,), dtype, device)


def _concat(device, dtype):
    N, P, Q = 1, 3, 4
    A = Dense(_rand((N, P, Q), dtype, device), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    B = Dense(_rand((N, P, Q), dtype, device), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    C = Concat(A, B, idim="Q")
    return C, _rand((N, 2 * Q), dtype, device)


def _chain(device, dtype):
    M, K, N = 5, 4, 3
    A = Dense(_rand((M, K), dtype, device), ("M", "K"), ("K",), ("M",))
    B = Dense(_rand((K, N), dtype, device), ("K", "N"), ("N",), ("K",))
    return Chain(B, A), _rand((N,), dtype, device)


def _to_device(device, dtype):
    dev = torch.device("cuda:0") if device == "cuda" else torch.device("cpu")
    return ToDevice(dev, dev, ioshape=("N",)), _rand((8,), dtype, device)


def _simple_linop(device, dtype):
    op = SimpleLinop(
        forward=lambda t: 2.0 * t,
        adjoint=lambda t: 2.0 * t,
        normal=lambda t: 4.0 * t,
        ishape=("N",),
    )
    return op, _rand((8,), dtype, device)


CASES: list[Case] = [
    _make_case("Dense", _dense),
    _make_case("DenseSimple", _dense_simple),
    _make_case("Diagonal", _diagonal),
    _make_case("Identity", _identity),
    _make_case("Zero", _zero, dtypes=[torch.complex64]),
    _make_case("ShapeSpec", _shapespec),
    _make_case("Scalar", _scalar),
    _make_case("Sleep", _sleep, methods=["fwd", "adj"]),
    _make_case("FFT", _fft, dtypes=[torch.complex64], deep_dive=True),
    _make_case(
        "Convolution",
        _convolution,
        dtypes=[torch.float32, torch.complex64],
        deep_dive=True,
    ),
    _make_case("Interpolate", _interpolate, dtypes=[torch.complex64], deep_dive=True),
    _make_case("NUFFT", _nufft, dtypes=[torch.complex64], deep_dive=True),
    _make_case(
        "FastSamplingNUFFT",
        _fast_sampling_nufft,
        dtypes=[torch.complex64],
        deep_dive=True,
    ),
    _make_case(
        "SamplingNUFFT",
        _sampling_nufft,
        dtypes=[torch.complex64],
        deep_dive=True,
    ),
    _make_case("ToeplitzPSF", _toeplitz_psf, dtypes=[torch.complex64], deep_dive=True),
    _make_case("Sampling", _sampling),
    _make_case("ArrayToBlocks", _array_to_blocks, dtypes=[torch.complex64]),
    _make_case("BlocksToArray", _blocks_to_array, dtypes=[torch.complex64]),
    _make_case("Pad", _pad, dtypes=[torch.complex64]),
    _make_case("Truncate", _truncate, dtypes=[torch.complex64]),
    _make_case("PadDim", _pad_dim, dtypes=[torch.complex64]),
    _make_case("Rearrange", _rearrange, dtypes=[torch.complex64]),
    _make_case("SumReduce", _sum_reduce, dtypes=[torch.complex64]),
    _make_case("Repeat", _repeat, dtypes=[torch.complex64]),
    _make_case("Add", _add, dtypes=[torch.complex64]),
    _make_case("Stack", _stack, dtypes=[torch.complex64]),
    _make_case("Concat", _concat, dtypes=[torch.complex64]),
    _make_case("Chain", _chain, dtypes=[torch.complex64]),
    _make_case("ToDevice", _to_device),
    _make_case("SimpleLinop", _simple_linop),
]

# BreakpointLinop calls pdb in fn/adj_fn: never run it in an automated sweep.
SKIPPED = {
    "BreakpointLinop": "debugging operator invokes pdb; automated run would block"
}


def _cell_to_dict(result: CellResult) -> dict:
    data = {
        "op": result.spec.op,
        "category": result.spec.category,
        "method": result.spec.method,
        "device": result.spec.device,
        "dtype": str(result.spec.dtype),
        "level": result.spec.level,
        "config": result.spec.config,
        "status": result.status,
        "cell_class": result.cell_class,
        "graph_breaks": result.graph_breaks,
        "break_locations": result.break_locations,
        "recompiles": result.recompiles,
        "max_abs_diff": result.max_abs_diff,
        "max_rel_diff": result.max_rel_diff,
        "eager_time": result.eager_time,
        "compile_time": result.compile_time,
        "speedup": result.speedup,
        "traceback_tail": result.traceback_tail,
        "notes": result.notes,
    }
    return data


def update_results_json(path: Path, key: str, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_phase_sweep(cases: list[Case], phase: str, out_path: Path) -> list[CellResult]:
    rows: list[CellResult] = []
    for case in cases:
        for method in case.methods:
            for device in case.devices:
                for dtype in case.dtypes:
                    for config in CONFIGS:
                        spec = CellSpec(
                            op=case.op,
                            category=case.category,
                            method=method,
                            device=device,
                            dtype=dtype,
                            level="wrapper",
                            config=config,
                        )
                        try:
                            target, input_, note = case.build(device, dtype, method)
                        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                            rows.append(
                                CellResult(
                                    spec=spec,
                                    status="error",
                                    cell_class="fails_loud",
                                    traceback_tail=traceback.format_exc().strip()[
                                        -800:
                                    ],
                                    notes=f"case build failed: {type(exc).__name__}: {exc}",
                                )
                            )
                            continue
                        result = execute_cell(spec, target, input_)
                        if note:
                            result.notes = " | ".join(
                                filter(None, [note, result.notes])
                            )
                        rows.append(result)

    update_results_json(out_path, "cells", [_cell_to_dict(r) for r in rows])
    csv_export(rows, out_path.with_suffix(".csv"))
    return rows
