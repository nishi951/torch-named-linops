"""Operator registry for compiled benchmarks.

Each Case follows the same shape as the repo's ``linop_input_output`` test
paradigm: building a case returns a linop, an input tensor, and an
output-shaped tensor. The output is constructed independently rather than as
``linop(input)`` so adjoint and normal targets get dense tensors without
inheriting forward views. Functional-kernel cases return their callable in the
linop position and only support ``fwd``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from benchmarks.compile.runner import CellSpec
from torchlinops import (
    FFT,
    NUFFT,
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
    Pad,
    PadDim,
    Rearrange,
    Repeat,
    Sampling,
    SamplingNUFFT,
    Scalar,
    ShapeSpec,
    SimpleLinop,
    Stack,
    SumReduce,
    ToDevice,
    Truncate,
    Zero,
)
from torchlinops.functional._interp import grid, ungrid
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs
from torchlinops.functional._unfold import fold, unfold
from torchlinops.linops.nufft import toeplitz_psf

DEFAULT_METHODS = ["fwd", "adj", "normal"]


@dataclass
class Case:
    """One operator instantiation for the compiled benchmark matrix.

    Attributes
    ----------
    op : str
        Human-readable operator name used as the result-row key.
    category : str
        ``"linop"`` for ``NamedLinop`` cases or ``"functional"`` for bare
        functional kernels.
    build : callable
        ``build(device, dtype) -> (linop, input, output)``. ``input`` and
        ``output`` are independently constructed tensors with the operator's
        input and output shapes.
    methods : list of str
        Methods to evaluate: any of ``"fwd"``, ``"adj"``, ``"normal"``.
    devices : list of str
        Devices to evaluate, normally ``["cuda", "cpu"]``.
    dtypes : list of torch.dtype
        Dtypes to evaluate. The reduced suite uses the first entry only.
    deep_dive : bool
        If true, evaluate both wrapper and inner compile levels.
    """

    op: str
    category: str
    build: Callable[[str, torch.dtype], tuple[object, Tensor, Tensor]]
    methods: list[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    devices: list[str] = field(default_factory=lambda: ["cuda", "cpu"])
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.float32])
    deep_dive: bool = False


def target_for_spec(
    case: Case, spec: CellSpec
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Return the compiled callable and input for one benchmark cell.

    Parameters
    ----------
    case : Case
        Registry entry describing how to build the operator, input, and output.
    spec : CellSpec
        Matrix cell selecting method, device, dtype, and compile level.

    Returns
    -------
    tuple of callable and torch.Tensor
        Target to compile and the tensor to feed it.
    """
    linop, x, y = case.build(spec.device, spec.dtype)

    if case.category == "functional":
        if spec.method != "fwd":
            raise ValueError(f"{case.op} only supports the forward method")
        return linop, x

    if spec.level == "inner":
        cls = type(linop)
        target = {
            "fwd": cls.fn,
            "adj": cls.adj_fn,
            "normal": cls.normal_fn,
        }[spec.method]
        input_ = y if spec.method == "adj" else x
        return (lambda t: target(linop, t)), input_

    if spec.method == "fwd":
        return (lambda t: linop(t)), x
    if spec.method == "adj":
        return (lambda t: linop.H(t)), y
    if spec.method == "normal":
        return (lambda t: linop.N(t)), x
    raise ValueError(f"unknown method {spec.method!r}")


def _make_case(
    op: str,
    construct: Callable[[str, torch.dtype], tuple[object, Tensor]],
    *,
    category: str = "linop",
    devices: list[str] | None = None,
    dtypes: list[torch.dtype] | None = None,
    methods: list[str] | None = None,
    deep_dive: bool = False,
) -> Case:
    def build(device: str, dtype: torch.dtype):
        linop, x = construct(device, dtype)
        y = torch.randn_like(linop(x))
        return linop, x, y

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
    W = torch.randn((9, 3), dtype=dtype, device=device)
    return Dense(W, ("M", "N"), ("N",), ("M",)), torch.randn(
        (3,), dtype=dtype, device=device
    )


def _dense_simple(device, dtype):
    W = torch.randn((5, 3), dtype=dtype, device=device)
    return Dense(W), torch.randn((3,), dtype=dtype, device=device)


def _diagonal(device, dtype):
    weight = torch.randn((10, 1, 1), dtype=dtype, device=device)
    return Diagonal(weight, ("M", "N", "P")), torch.randn(
        (10, 5, 7), dtype=dtype, device=device
    )


def _identity(device, dtype):
    return Identity(ishape=("N", "M")), torch.randn((4, 5), dtype=dtype, device=device)


def _zero(device, dtype):
    return Zero(ishape=("N", "M")), torch.randn((4, 5), dtype=dtype, device=device)


def _shapespec(device, dtype):
    return ShapeSpec(ishape=("N", "M")), torch.randn((4, 5), dtype=dtype, device=device)


def _scalar(device, dtype):
    weight = 2.0 + 0j if dtype.is_complex else 2.0
    return Scalar(weight=weight, ioshape=("N",)), torch.randn(
        (8,), dtype=dtype, device=device
    )


def _fft(device, dtype):
    return FFT(ndim=2, centered=True), torch.randn(
        (8, 10), dtype=torch.complex64, device=device
    )


def _convolution(device, dtype):
    weight = torch.randn((5,), dtype=dtype, device=device)
    conv = Convolution(weight, in_grid_shape=("x",), out_grid_shape=("x",))
    return conv, torch.randn((3, 16), dtype=dtype, device=device)


def _interpolate(device, dtype):
    locs = get_valid_locs((8,), (16, 16), 2, 4.0, "cpu", centered=True).to(device)
    op = Interpolate(locs, (16, 16), width=4.0, kernel="kaiser_bessel")
    return op, torch.randn((16, 16), dtype=torch.complex64, device=device)


def _nufft(device, dtype):
    locs = get_valid_locs((8,), (16, 16), 2, 4.0, "cpu", centered=True).to(device)
    op = NUFFT(locs.clone(), (16, 16), output_shape=("R",), width=4.0, oversamp=1.25)
    op._locs_orig = locs.clone()
    return op, torch.randn((16, 16), dtype=torch.complex64, device=device)


def _sampling_nufft_locs(device, grid_size=(16, 16), npts=8, oversamp=1.25):
    padded = tuple(int(oversamp * g) for g in grid_size)
    idx = torch.stack([torch.randint(0, p, (npts,)) for p in padded], dim=-1)
    locs = (idx - torch.tensor(padded) // 2).float() / oversamp
    return locs.to(device)


def _fast_sampling_nufft(device, dtype):
    locs = _sampling_nufft_locs(device)
    op = FastSamplingNUFFT(locs.clone(), (16, 16), output_shape=("R",), oversamp=1.25)
    return op, torch.randn((16, 16), dtype=torch.complex64, device=device)


def _sampling_nufft(device, dtype):
    locs = _sampling_nufft_locs(device)
    op = SamplingNUFFT(locs.clone(), (16, 16), output_shape=("R",), oversamp=1.25)
    op._locs_orig = locs.clone()
    return op, torch.randn((16, 16), dtype=torch.complex64, device=device)


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
    return op, torch.randn((2, padded, padded), dtype=torch.complex64, device=device)


def _sampling(device, dtype):
    N, R, K, B = 16, 3, 4, 2
    idx = torch.randint(0, N - 1, (R, K, 2)).to(device)
    op = Sampling.from_stacked_idx(idx, (N, N), ("R", "K"))
    return op, torch.randn((B, N, N), dtype=dtype, device=device)


def _array_to_blocks(device, dtype):
    op = ArrayToBlocks((8, 8), (4, 4), (4, 4))
    return op, torch.randn((2, 8, 8), dtype=torch.complex64, device=device)


def _blocks_to_array(device, dtype):
    op = BlocksToArray((8, 8), (4, 4), (4, 4))
    return op, torch.randn((2, 2, 4, 4), dtype=torch.complex64, device=device)


def _pad(device, dtype):
    return Pad((20, 20), (10, 10), Dim("XY")), torch.randn(
        (10, 10), dtype=dtype, device=device
    )


def _truncate(device, dtype):
    op = Truncate(
        dim=1, from_length=5, to_length=3, ishape=("N", "M"), oshape=("N", "M")
    )
    return op, torch.randn((4, 5), dtype=dtype, device=device)


def _pad_dim(device, dtype):
    op = PadDim(dim=1, from_length=3, to_length=5, ishape=("N", "M"), oshape=("N", "M"))
    return op, torch.randn((4, 3), dtype=dtype, device=device)


def _rearrange(device, dtype):
    op = Rearrange(
        "(A B) C",
        "A B C",
        ishape=("Ab", "C"),
        oshape=("A", "B", "C"),
        axes_lengths={"A": 2, "B": 3},
    )
    return op, torch.randn((6, 4), dtype=dtype, device=device)


def _sum_reduce(device, dtype):
    return SumReduce(("A", "B", "C"), ("A", "B")), torch.randn(
        (5, 2, 3), dtype=dtype, device=device
    )


def _repeat(device, dtype):
    op = Repeat({"C": 3}, ishape=("A", "B"), oshape=("A", "B", "C"), broadcast_dims=[])
    return op, torch.randn((4, 5), dtype=dtype, device=device)


def _add(device, dtype):
    A = Dense(
        torch.randn((5, 5), dtype=dtype, device=device), ("M", "N"), ("N",), ("M",)
    )
    B = Dense(
        torch.randn((5, 5), dtype=dtype, device=device), ("M", "N"), ("N",), ("M",)
    )
    C = Dense(
        torch.randn((5, 5), dtype=dtype, device=device), ("M", "N"), ("N",), ("M",)
    )
    add = Add(A, B, C)
    add.threaded = False
    return add, torch.randn((5,), dtype=dtype, device=device)


def _stack(device, dtype):
    Q, P = 4, 3
    A = Dense(
        torch.randn((P, Q), dtype=dtype, device=device), ("P", "Q"), Dim("Q"), Dim("P")
    )
    B = Dense(
        torch.randn((P, Q), dtype=dtype, device=device), ("P", "Q"), Dim("Q"), Dim("P")
    )
    C = Stack(A, B, odim_and_idx=("M", 1))
    return C, torch.randn((Q,), dtype=dtype, device=device)


def _concat(device, dtype):
    N, P, Q = 1, 3, 4
    A = Dense(
        torch.randn((N, P, Q), dtype=dtype, device=device),
        ("N", "P", "Q"),
        ("N", "Q"),
        ("N", "P"),
    )
    B = Dense(
        torch.randn((N, P, Q), dtype=dtype, device=device),
        ("N", "P", "Q"),
        ("N", "Q"),
        ("N", "P"),
    )
    C = Concat(A, B, idim="Q")
    return C, torch.randn((N, 2 * Q), dtype=dtype, device=device)


def _chain(device, dtype):
    M, K, N = 5, 4, 3
    A = Dense(
        torch.randn((M, K), dtype=dtype, device=device), ("M", "K"), ("K",), ("M",)
    )
    B = Dense(
        torch.randn((K, N), dtype=dtype, device=device), ("K", "N"), ("N",), ("K",)
    )
    return Chain(B, A), torch.randn((N,), dtype=dtype, device=device)


def _to_device(device, dtype):
    dev = torch.device("cuda:0") if device == "cuda" else torch.device("cpu")
    return ToDevice(dev, dev, ioshape=("N",)), torch.randn(
        (8,), dtype=dtype, device=device
    )


def _simple_linop(device, dtype):
    op = SimpleLinop(
        forward=lambda t: 2.0 * t,
        adjoint=lambda t: 2.0 * t,
        normal=lambda t: 4.0 * t,
        ishape=("N",),
    )
    return op, torch.randn((8,), dtype=dtype, device=device)


CASES: list[Case] = [
    _make_case("Dense", _dense),
    _make_case("DenseSimple", _dense_simple),
    _make_case("Diagonal", _diagonal),
    _make_case("Identity", _identity),
    _make_case("Zero", _zero),
    _make_case("ShapeSpec", _shapespec),
    _make_case("Scalar", _scalar),
    _make_case("FFT", _fft, dtypes=[torch.complex64], deep_dive=True),
    _make_case(
        "Convolution",
        _convolution,
        dtypes=[torch.float32, torch.complex64],
        deep_dive=True,
    ),
    _make_case(
        "Interpolate",
        _interpolate,
        dtypes=[torch.complex64],
        deep_dive=True,
    ),
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
    _make_case(
        "ToeplitzPSF",
        _toeplitz_psf,
        dtypes=[torch.complex64],
        deep_dive=True,
    ),
    _make_case("Sampling", _sampling),
    _make_case("ArrayToBlocks", _array_to_blocks, dtypes=[torch.complex64]),
    _make_case("BlocksToArray", _blocks_to_array, dtypes=[torch.complex64]),
    _make_case("Pad", _pad),
    _make_case("Truncate", _truncate),
    _make_case("PadDim", _pad_dim),
    _make_case("Rearrange", _rearrange),
    _make_case("SumReduce", _sum_reduce),
    _make_case("Repeat", _repeat),
    _make_case("Add", _add, dtypes=[torch.float32, torch.complex64]),
    _make_case("Stack", _stack),
    _make_case("Concat", _concat),
    _make_case("Chain", _chain),
    _make_case("ToDevice", _to_device),
    _make_case("SimpleLinop", _simple_linop),
]

# BreakpointLinop calls pdb in fn/adj_fn: never run it in an automated sweep.
SKIPPED = {
    "BreakpointLinop": "debugging operator invokes pdb; automated run would block",
    "Sleep": "sleep timing is not meaningful; eager sleeps and compile has no effect",
}

# --- Functional (triton kernel) cases -------------------------------------


def _locs(device: str, npts: int = 8) -> Tensor:
    return get_valid_locs((npts,), (16, 16), 2, 4.0, "cpu", centered=True).to(device)


def _functional_build(device, dtype, x, fn):
    y = torch.randn_like(fn(x))
    return fn, x, y


def _functional_unfold(device, dtype):
    x = torch.randn((2, 8, 8), dtype=torch.complex64, device=device)
    return _functional_build(device, dtype, x, lambda t: unfold(t, (4, 4), (4, 4)))


def _functional_fold(device, dtype):
    x = torch.randn((2, 2, 2, 4, 4), dtype=torch.complex64, device=device)
    return _functional_build(
        device, dtype, x, lambda t: fold(t, (8, 8), (4, 4), (4, 4))
    )


def _functional_grid(device, dtype):
    locs = _locs(device)
    x = torch.randn((8,), dtype=torch.complex64, device=device)
    return _functional_build(device, dtype, x, lambda t: grid(t, locs, (16, 16), 4.0))


def _functional_ungrid(device, dtype):
    locs = _locs(device)
    x = torch.randn((16, 16), dtype=torch.complex64, device=device)
    return _functional_build(device, dtype, x, lambda t: ungrid(t, locs, 4.0))


CASES.extend(
    [
        Case(
            op="functional.unfold",
            category="functional",
            build=_functional_unfold,
            methods=["fwd"],
            dtypes=[torch.complex64],
            deep_dive=True,
        ),
        Case(
            op="functional.fold",
            category="functional",
            build=_functional_fold,
            methods=["fwd"],
            dtypes=[torch.complex64],
            deep_dive=True,
        ),
        Case(
            op="functional.grid",
            category="functional",
            build=_functional_grid,
            methods=["fwd"],
            dtypes=[torch.complex64],
            deep_dive=True,
        ),
        Case(
            op="functional.ungrid",
            category="functional",
            build=_functional_ungrid,
            methods=["fwd"],
            dtypes=[torch.complex64],
            deep_dive=True,
        ),
    ]
)

CASE_BY_OP = {case.op: case for case in CASES}


# --- Matrix iteration -----------------------------------------------------


def levels_for(case: Case, suite_level: str) -> list[str]:
    """Return the compile levels to evaluate for a case.

    Parameters
    ----------
    case : Case
        Registry entry.
    suite_level : str
        Either ``"reduced"`` or ``"full"``.

    Returns
    -------
    list of str
        ``"wrapper"`` and/or ``"inner"``.
    """
    if case.category == "functional":
        return ["inner"]
    if case.deep_dive:
        return ["wrapper", "inner"]
    return ["wrapper"]


def methods_for(case: Case, suite_level: str) -> list[str]:
    """Return the evaluation methods for a suite level.

    Parameters
    ----------
    case : Case
        Registry entry.
    suite_level : str
        Either ``"reduced"`` or ``"full"``.

    Returns
    -------
    list of str
        Subset of ``"fwd"``, ``"adj"``, and ``"normal"``.
    """
    if suite_level == "reduced":
        return [method for method in ("fwd", "adj") if method in case.methods]
    return list(case.methods)


def configs_for(suite_level: str) -> list[str]:
    """Return the compile configs for a suite level.

    Parameters
    ----------
    suite_level : str
        Either ``"reduced"`` or ``"full"``.

    Returns
    -------
    list of str
        ``"default"`` and optionally ``"fullgraph"``.
    """
    if suite_level == "reduced":
        return ["default"]
    return ["default", "fullgraph"]


def dtypes_for(case: Case, suite_level: str) -> list[torch.dtype]:
    """Return the dtypes to evaluate for a case.

    Parameters
    ----------
    case : Case
        Registry entry.
    suite_level : str
        Either ``"reduced"`` or ``"full"``.

    Returns
    -------
    list of torch.dtype
        Reduced runs use the first declared dtype; full runs use all.
    """
    if suite_level == "reduced":
        return list(case.dtypes[:1])
    return list(case.dtypes)


def iter_specs(suite_level: str) -> list[CellSpec]:
    """Expand every registered case into matrix cells.

    Parameters
    ----------
    suite_level : str
        Either ``"reduced"`` or ``"full"``.

    Returns
    -------
    list of CellSpec
        One entry per operator/method/device/dtype/level/config combination.
    """
    specs: list[CellSpec] = []
    for case in CASES:
        for method in methods_for(case, suite_level):
            for device in case.devices:
                for dtype in dtypes_for(case, suite_level):
                    for level in levels_for(case, suite_level):
                        for config in configs_for(suite_level):
                            specs.append(
                                CellSpec(
                                    op=case.op,
                                    category=case.category,
                                    method=method,
                                    device=device,
                                    dtype=dtype,
                                    level=level,
                                    config=config,
                                )
                            )
    return specs


def spec_id(spec: CellSpec) -> str:
    """Return a stable identifier for a matrix cell.

    Parameters
    ----------
    spec : CellSpec
        Matrix cell.

    Returns
    -------
    str
        Filename-friendly cell identifier.
    """
    dtype = str(spec.dtype).replace("torch.", "")
    return f"{spec.op}-{spec.method}-{spec.device}-{dtype}-{spec.level}-{spec.config}"
