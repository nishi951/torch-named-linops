"""Performance benchmarks for core torch-named-linops operations with sigpy comparison.

Three variants are benchmarked per operation:
- torchlinops (functional): direct calls to torchlinops.functional
- torchlinops (linop): calls through the NamedLinop abstraction
- sigpy: equivalent operations in SigPy

Each operation is benchmarked at three problem sizes (small, medium, large).
Data generation is included in the timed function to avoid GPU cache effects.
A separate data-generation-only benchmark is run for each variant for reporting.

For the linop variant, the linop is constructed once outside the timed loop.
"""

from contextlib import contextmanager
from math import prod

import numpy as np
import pytest
import torch

from torchlinops import ArrayToBlocks, BlocksToArray, Interpolate, NUFFT
from torchlinops.functional import (
    array_to_blocks,
    blocks_to_array,
    interpolate,
    interpolate_adjoint,
    nufft,
    nufft_adjoint,
    get_nblocks,
)
from torchlinops.utils import from_pytorch, device_ordinal

try:
    import sigpy as sp

    SIGPY_AVAILABLE = True
except ImportError:
    SIGPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Size presets
# ---------------------------------------------------------------------------

NUFFT_2D_SIZES = {
    "small": {"grid_size": (64, 64), "npts": 4096},
    "medium": {"grid_size": (128, 128), "npts": 16384},
    "large": {"grid_size": (256, 256), "npts": 65536},
}

NUFFT_3D_SIZES = {
    "small": {"grid_size": (32, 32, 32), "npts": 4096},
    "medium": {"grid_size": (64, 64, 64), "npts": 32768},
    "large": {"grid_size": (128, 128, 128), "npts": 262144},
}

INTERP_2D_SIZES = NUFFT_2D_SIZES
INTERP_3D_SIZES = NUFFT_3D_SIZES

ARRAY_TO_BLOCKS_SIZES = {
    "small": {"grid_size": (32, 32, 32), "block_size": (8, 8, 8), "stride": (4, 4, 4)},
    "medium": {"grid_size": (64, 64, 64), "block_size": (8, 8, 8), "stride": (4, 4, 4)},
    "large": {
        "grid_size": (128, 128, 128),
        "block_size": (8, 8, 8),
        "stride": (4, 4, 4),
    },
}

SIZE_NAMES = ["small", "medium", "large"]

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.gpu,
            pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="GPU is required but not available",
            ),
        ],
    ),
]

NDIM_3D = pytest.param(3, marks=pytest.mark.slow)


def _problem_size(grid_size):
    return prod(grid_size)


def _size_label(grid_size):
    return "x".join(str(s) for s in grid_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_valid_locs(locs_batch_size, grid_size, ndim, width, device, centered=False):
    from math import ceil

    out = []
    for d in range(ndim):
        lower = ceil(width / 2)
        upper = grid_size[d] - 1 - lower
        locs = torch.rand(*locs_batch_size, device=device)
        locs = locs * (upper - lower) + lower
        out.append(locs)
    out = torch.stack(out, dim=-1).contiguous()
    if centered:
        sz = torch.tensor(grid_size, device=device)
        out -= sz / 2
    return out


def _sigpy_device(device):
    if device == "cuda":
        return sp.Device(device_ordinal(torch.device(device)))
    return sp.Device(-1)


@contextmanager
def _sigpy_device_ctx(device):
    if device == "cuda":
        dev = _sigpy_device(device)
        with dev:
            yield dev
    else:
        yield None


def _sigpy_randn(shape, device):
    if device == "cuda":
        dev = _sigpy_device(device)
        with dev:
            xp = dev.xp
            return xp.random.randn(*shape, dtype=np.float32) + 1j * xp.random.randn(
                *shape, dtype=np.float32
            )
    return np.random.randn(*shape) + 1j * np.random.randn(*shape)


def _run_benchmarks(
    benchmark_session,
    *,
    name,
    device,
    label,
    sub_label,
    description,
    size_name,
    problem_size,
    size_label,
    gen_functional,
    fn_functional,
    gen_linop,
    fn_linop,
    gen_sigpy=None,
    fn_sigpy=None,
):
    benchmark_session.run(
        name=name,
        fn=fn_functional,
        device=device,
        label=label,
        sub_label=sub_label,
        description=description,
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=problem_size,
        size_label=size_label,
    )

    benchmark_session.run(
        name=name,
        fn=fn_linop,
        device=device,
        label=label,
        sub_label=sub_label,
        description=description,
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=problem_size,
        size_label=size_label,
    )

    if SIGPY_AVAILABLE and gen_sigpy is not None and fn_sigpy is not None:
        benchmark_session.run(
            name=name,
            fn=fn_sigpy,
            device=device,
            label=label,
            sub_label=sub_label,
            description=description,
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=problem_size,
            size_label=size_label,
        )


# ---------------------------------------------------------------------------
# NUFFT
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_nufft_forward_2d(benchmark_session, device, size_name):
    spec = NUFFT_2D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 2
    width = 4.0
    oversamp = 1.25
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        return x, locs

    def fn_functional(data):
        x, locs = data
        return nufft(x, locs, oversamp=oversamp, width=width)

    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(locs, grid_size, output_shape=("K",), width=width, oversamp=oversamp)

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    def gen_sigpy():
        x = _sigpy_randn(grid_size, device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        coord = from_pytorch(locs)
        return x, coord

    def fn_sigpy(data):
        x, coord = data
        with _sigpy_device_ctx(device):
            return sp.nufft(x, coord, oversamp=oversamp, width=width)

    _run_benchmarks(
        benchmark_session,
        name="NUFFT forward 2D",
        device=device,
        label="NUFFT 2D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
        gen_functional=gen_functional,
        fn_functional=fn_functional,
        gen_linop=gen_linop,
        fn_linop=fn_linop,
        gen_sigpy=gen_sigpy,
        fn_sigpy=fn_sigpy,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_nufft_adjoint_2d(benchmark_session, device, size_name):
    spec = NUFFT_2D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 2
    width = 4.0
    oversamp = 1.25
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        y = torch.randn(npts, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        return y, locs

    def fn_functional(data):
        y, locs = data
        return nufft_adjoint(y, locs, grid_size, oversamp=oversamp, width=width)

    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(locs, grid_size, output_shape=("K",), width=width, oversamp=oversamp)
    AH = A.H

    def gen_linop():
        return torch.randn(npts, dtype=torch.complex64, device=device)

    def fn_linop(y):
        return AH(y)

    def gen_sigpy():
        y = _sigpy_randn((npts,), device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        coord = from_pytorch(locs)
        return y, coord

    def fn_sigpy(data):
        y, coord = data
        with _sigpy_device_ctx(device):
            return sp.nufft_adjoint(y, coord, grid_size, oversamp=oversamp, width=width)

    _run_benchmarks(
        benchmark_session,
        name="NUFFT adjoint 2D",
        device=device,
        label="NUFFT 2D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
        gen_functional=gen_functional,
        fn_functional=fn_functional,
        gen_linop=gen_linop,
        fn_linop=fn_linop,
        gen_sigpy=gen_sigpy,
        fn_sigpy=fn_sigpy,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("ndim", [NDIM_3D])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_nufft_forward_3d(benchmark_session, ndim, device, size_name):
    spec = NUFFT_3D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    width = 4.0
    oversamp = 1.25
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        return x, locs

    def fn_functional(data):
        x, locs = data
        return nufft(x, locs, oversamp=oversamp, width=width)

    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(locs, grid_size, output_shape=("K",), width=width, oversamp=oversamp)

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    def gen_sigpy():
        x = _sigpy_randn(grid_size, device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        coord = from_pytorch(locs)
        return x, coord

    def fn_sigpy(data):
        x, coord = data
        with _sigpy_device_ctx(device):
            return sp.nufft(x, coord, oversamp=oversamp, width=width)

    _run_benchmarks(
        benchmark_session,
        name="NUFFT forward 3D",
        device=device,
        label="NUFFT 3D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
        gen_functional=gen_functional,
        fn_functional=fn_functional,
        gen_linop=gen_linop,
        fn_linop=fn_linop,
        gen_sigpy=gen_sigpy,
        fn_sigpy=fn_sigpy,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("ndim", [NDIM_3D])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_nufft_adjoint_3d(benchmark_session, ndim, device, size_name):
    spec = NUFFT_3D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    width = 4.0
    oversamp = 1.25
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        y = torch.randn(npts, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        return y, locs

    def fn_functional(data):
        y, locs = data
        return nufft_adjoint(y, locs, grid_size, oversamp=oversamp, width=width)

    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(locs, grid_size, output_shape=("K",), width=width, oversamp=oversamp)
    AH = A.H

    def gen_linop():
        return torch.randn(npts, dtype=torch.complex64, device=device)

    def fn_linop(y):
        return AH(y)

    def gen_sigpy():
        y = _sigpy_randn((npts,), device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        coord = from_pytorch(locs)
        return y, coord

    def fn_sigpy(data):
        y, coord = data
        with _sigpy_device_ctx(device):
            return sp.nufft_adjoint(y, coord, grid_size, oversamp=oversamp, width=width)

    _run_benchmarks(
        benchmark_session,
        name="NUFFT adjoint 3D",
        device=device,
        label="NUFFT 3D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
        gen_functional=gen_functional,
        fn_functional=fn_functional,
        gen_linop=gen_linop,
        fn_linop=fn_linop,
        gen_sigpy=gen_sigpy,
        fn_sigpy=fn_sigpy,
    )


# ---------------------------------------------------------------------------
# ArrayToBlocks / BlocksToArray 3D
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_array_to_blocks_forward(benchmark_session, device, size_name):
    spec = ARRAY_TO_BLOCKS_SIZES[size_name]
    grid_size = spec["grid_size"]
    block_size = spec["block_size"]
    stride = spec["stride"]
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_functional(x):
        return array_to_blocks(x, block_size, stride)

    A = ArrayToBlocks(grid_size, block_size, stride)

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    def gen_sigpy():
        return _sigpy_randn(grid_size, device)

    def fn_sigpy(x):
        with _sigpy_device_ctx(device):
            return sp.array_to_blocks(x, block_size, stride)

    _run_benchmarks(
        benchmark_session,
        name="ArrayToBlocks forward",
        device=device,
        label="ArrayToBlocks 3D",
        sub_label=f"forward {sl}, block 8x8x8, stride 4",
        description="forward",
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
        gen_functional=gen_functional,
        fn_functional=fn_functional,
        gen_linop=gen_linop,
        fn_linop=fn_linop,
        gen_sigpy=gen_sigpy,
        fn_sigpy=fn_sigpy,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_blocks_to_array_forward(benchmark_session, device, size_name):
    spec = ARRAY_TO_BLOCKS_SIZES[size_name]
    grid_size = spec["grid_size"]
    block_size = spec["block_size"]
    stride = spec["stride"]
    nblocks = get_nblocks(grid_size, block_size, stride)
    blocks_shape = (*nblocks, *block_size)
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        return torch.randn(*blocks_shape, dtype=torch.complex64, device=device)

    def fn_functional(x):
        return blocks_to_array(x, grid_size, block_size, stride)

    A = BlocksToArray(grid_size, block_size, stride)

    def gen_linop():
        return torch.randn(*blocks_shape, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    def gen_sigpy():
        return _sigpy_randn(blocks_shape, device)

    def fn_sigpy(x):
        with _sigpy_device_ctx(device):
            return sp.blocks_to_array(x, grid_size, block_size, stride)

    _run_benchmarks(
        benchmark_session,
        name="BlocksToArray forward",
        device=device,
        label="ArrayToBlocks 3D",
        sub_label=f"adjoint {sl}, block 8x8x8, stride 4",
        description="adjoint",
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
        gen_functional=gen_functional,
        fn_functional=fn_functional,
        gen_linop=gen_linop,
        fn_linop=fn_linop,
        gen_sigpy=gen_sigpy,
        fn_sigpy=fn_sigpy,
    )


# ---------------------------------------------------------------------------
# Interpolate
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_interpolate_forward_2d(benchmark_session, device, size_name):
    spec = INTERP_2D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 2
    width = 4.0
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
        return x, locs

    def fn_functional(data):
        x, locs = data
        return interpolate(x, locs, width=width, kernel="kaiser_bessel")

    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    def gen_sigpy():
        x = _sigpy_randn(grid_size, device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
        coord = from_pytorch(locs)
        return x, coord

    def fn_sigpy(data):
        x, coord = data
        with _sigpy_device_ctx(device):
            return sp.interp.interpolate(
                x, coord, kernel="kaiser_bessel", width=width, param=1.0
            )

    _run_benchmarks(
        benchmark_session,
        name="Interpolate forward 2D",
        device=device,
        label="Interpolate 2D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
        gen_functional=gen_functional,
        fn_functional=fn_functional,
        gen_linop=gen_linop,
        fn_linop=fn_linop,
        gen_sigpy=gen_sigpy,
        fn_sigpy=fn_sigpy,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_interpolate_adjoint_2d(benchmark_session, device, size_name):
    spec = INTERP_2D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 2
    width = 4.0
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        y = torch.randn(npts, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
        return y, locs

    def fn_functional(data):
        y, locs = data
        return interpolate_adjoint(
            y, locs, grid_size, width=width, kernel="kaiser_bessel"
        )

    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")
    AH = A.H

    def gen_linop():
        return torch.randn(npts, dtype=torch.complex64, device=device)

    def fn_linop(y):
        return AH(y)

    def gen_sigpy():
        y = _sigpy_randn((npts,), device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
        coord = from_pytorch(locs)
        return y, coord

    def fn_sigpy(data):
        y, coord = data
        with _sigpy_device_ctx(device):
            return sp.interp.gridding(
                y, coord, grid_size, kernel="kaiser_bessel", width=width, param=1.0
            )

    _run_benchmarks(
        benchmark_session,
        name="Interpolate adjoint 2D",
        device=device,
        label="Interpolate 2D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
        gen_functional=gen_functional,
        fn_functional=fn_functional,
        gen_linop=gen_linop,
        fn_linop=fn_linop,
        gen_sigpy=gen_sigpy,
        fn_sigpy=fn_sigpy,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("ndim", [NDIM_3D])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_interpolate_forward_3d(benchmark_session, ndim, device, size_name):
    spec = INTERP_3D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    width = 4.0
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
        return x, locs

    def fn_functional(data):
        x, locs = data
        return interpolate(x, locs, width=width, kernel="kaiser_bessel")

    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    def gen_sigpy():
        x = _sigpy_randn(grid_size, device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
        coord = from_pytorch(locs)
        return x, coord

    def fn_sigpy(data):
        x, coord = data
        with _sigpy_device_ctx(device):
            return sp.interp.interpolate(
                x, coord, kernel="kaiser_bessel", width=width, param=1.0
            )

    _run_benchmarks(
        benchmark_session,
        name="Interpolate forward 3D",
        device=device,
        label="Interpolate 3D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
        gen_functional=gen_functional,
        fn_functional=fn_functional,
        gen_linop=gen_linop,
        fn_linop=fn_linop,
        gen_sigpy=gen_sigpy,
        fn_sigpy=fn_sigpy,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("ndim", [NDIM_3D])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_interpolate_adjoint_3d(benchmark_session, ndim, device, size_name):
    spec = INTERP_3D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    width = 4.0
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        y = torch.randn(npts, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
        return y, locs

    def fn_functional(data):
        y, locs = data
        return interpolate_adjoint(
            y, locs, grid_size, width=width, kernel="kaiser_bessel"
        )

    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")
    AH = A.H

    def gen_linop():
        return torch.randn(npts, dtype=torch.complex64, device=device)

    def fn_linop(y):
        return AH(y)

    def gen_sigpy():
        y = _sigpy_randn((npts,), device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
        coord = from_pytorch(locs)
        return y, coord

    def fn_sigpy(data):
        y, coord = data
        with _sigpy_device_ctx(device):
            return sp.interp.gridding(
                y, coord, grid_size, kernel="kaiser_bessel", width=width, param=1.0
            )

    _run_benchmarks(
        benchmark_session,
        name="Interpolate adjoint 3D",
        device=device,
        label="Interpolate 3D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
        gen_functional=gen_functional,
        fn_functional=fn_functional,
        gen_linop=gen_linop,
        fn_linop=fn_linop,
        gen_sigpy=gen_sigpy,
        fn_sigpy=fn_sigpy,
    )
