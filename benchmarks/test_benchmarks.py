"""Performance benchmarks for core torch-named-linops operations with sigpy comparison.

Three variants are benchmarked per operation:
- torchlinops (functional): direct calls to torchlinops.functional
- torchlinops (linop): calls through the NamedLinop abstraction
- sigpy: equivalent operations in SigPy

Each operation is benchmarked at three problem sizes (small, medium, large).
Data generation is included in the timed function to avoid GPU cache effects.
A separate data-generation-only benchmark is run for each variant, and its
mean time is subtracted from the operator benchmark to isolate computation cost.

For the linop variant, the linop is reconstructed with fresh locs on every
iteration (to avoid caching), and the construction cost is included in the
data-gen benchmark and subtracted out.
"""

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

# npts for NUFFT/Interpolate: fully sampled for 2D, 1/8 for 3D (realistic undersampling)
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

# block=8^3, stride=4
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


def _problem_size(grid_size):
    return prod(grid_size)


def _size_label(grid_size):
    return "x".join(str(s) for s in grid_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_valid_locs(locs_batch_size, grid_size, ndim, width, device, centered=False):
    """Generate valid interpolation locations on the requested device."""
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


def gpu_marks(fn):
    """Decorator applying pytest gpu mark and skip-if-no-gpu mark."""
    fn = pytest.mark.gpu(fn)
    fn = pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU is required but not available"
    )(fn)
    return fn


def _sigpy_device(device):
    """Return a sigpy device context for the given torch device string."""
    if device == "cuda":
        return sp.Device(device_ordinal(torch.device(device)))
    return sp.Device(-1)


# ---------------------------------------------------------------------------
# NUFFT 2D
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_nufft_forward_2d_cpu(benchmark_session, size_name):
    spec = NUFFT_2D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 2
    width = 4.0
    oversamp = 1.25
    device = "cpu"
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        return x, locs

    def fn_functional(data):
        x, locs = data
        return nufft(x, locs, oversamp=oversamp, width=width)

    benchmark_session.run(
        name="NUFFT forward 2D",
        fn=fn_functional,
        device=device,
        label="NUFFT 2D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(locs, grid_size, output_shape=("K",), width=width, oversamp=oversamp)

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    benchmark_session.run(
        name="NUFFT forward 2D",
        fn=fn_linop,
        device=device,
        label="NUFFT 2D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            x = np.random.randn(*grid_size) + 1j * np.random.randn(*grid_size)
            locs = _get_valid_locs(
                (npts,), grid_size, ndim, width, device, centered=True
            )
            coord = from_pytorch(locs)
            return x, coord

        def fn_sigpy(data):
            x, coord = data
            return sp.nufft(x, coord, oversamp=oversamp, width=width)

        benchmark_session.run(
            name="NUFFT forward 2D",
            fn=fn_sigpy,
            device=device,
            label="NUFFT 2D",
            sub_label=f"forward {sl}, {npts} locs",
            description="forward",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@gpu_marks
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_nufft_forward_2d_gpu(benchmark_session, size_name):
    spec = NUFFT_2D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 2
    width = 4.0
    oversamp = 1.25
    device = "cuda"
    dev = _sigpy_device(device)
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        return x, locs

    def fn_functional(data):
        x, locs = data
        return nufft(x, locs, oversamp=oversamp, width=width)

    benchmark_session.run(
        name="NUFFT forward 2D",
        fn=fn_functional,
        device=device,
        label="NUFFT 2D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(locs, grid_size, output_shape=("K",), width=width, oversamp=oversamp)

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    benchmark_session.run(
        name="NUFFT forward 2D",
        fn=fn_linop,
        device=device,
        label="NUFFT 2D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            with dev:
                xp = dev.xp
                x = xp.random.randn(
                    *grid_size, dtype=np.float32
                ) + 1j * xp.random.randn(*grid_size, dtype=np.float32)
                locs = _get_valid_locs(
                    (npts,), grid_size, ndim, width, device, centered=True
                )
                coord = from_pytorch(locs)
                return x, coord

        def fn_sigpy(data):
            x, coord = data
            with dev:
                return sp.nufft(x, coord, oversamp=oversamp, width=width)

        benchmark_session.run(
            name="NUFFT forward 2D",
            fn=fn_sigpy,
            device=device,
            label="NUFFT 2D",
            sub_label=f"forward {sl}, {npts} locs",
            description="forward",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_nufft_adjoint_2d_cpu(benchmark_session, size_name):
    spec = NUFFT_2D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 2
    width = 4.0
    oversamp = 1.25
    device = "cpu"
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        y = torch.randn(npts, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        return y, locs

    def fn_functional(data):
        y, locs = data
        return nufft_adjoint(y, locs, grid_size, oversamp=oversamp, width=width)

    benchmark_session.run(
        name="NUFFT adjoint 2D",
        fn=fn_functional,
        device=device,
        label="NUFFT 2D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(locs, grid_size, output_shape=("K",), width=width, oversamp=oversamp)
    AH = A.H

    def gen_linop():
        return torch.randn(npts, dtype=torch.complex64, device=device)

    def fn_linop(y):
        return AH(y)

    benchmark_session.run(
        name="NUFFT adjoint 2D",
        fn=fn_linop,
        device=device,
        label="NUFFT 2D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            y = np.random.randn(npts) + 1j * np.random.randn(npts)
            locs = _get_valid_locs(
                (npts,), grid_size, ndim, width, device, centered=True
            )
            coord = from_pytorch(locs)
            return y, coord

        def fn_sigpy(data):
            y, coord = data
            return sp.nufft_adjoint(y, coord, grid_size, oversamp=oversamp, width=width)

        benchmark_session.run(
            name="NUFFT adjoint 2D",
            fn=fn_sigpy,
            device=device,
            label="NUFFT 2D",
            sub_label=f"adjoint {sl}, {npts} locs",
            description="adjoint",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@gpu_marks
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_nufft_adjoint_2d_gpu(benchmark_session, size_name):
    spec = NUFFT_2D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 2
    width = 4.0
    oversamp = 1.25
    device = "cuda"
    dev = _sigpy_device(device)
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        y = torch.randn(npts, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        return y, locs

    def fn_functional(data):
        y, locs = data
        return nufft_adjoint(y, locs, grid_size, oversamp=oversamp, width=width)

    benchmark_session.run(
        name="NUFFT adjoint 2D",
        fn=fn_functional,
        device=device,
        label="NUFFT 2D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(locs, grid_size, output_shape=("K",), width=width, oversamp=oversamp)
    AH = A.H  # Don't benchmark adjoint creation procedure

    def gen_linop():
        return torch.randn(npts, dtype=torch.complex64, device=device)

    def fn_linop(y):
        return AH(y)

    benchmark_session.run(
        name="NUFFT adjoint 2D",
        fn=fn_linop,
        device=device,
        label="NUFFT 2D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            with dev:
                xp = dev.xp
                y = xp.random.randn(npts, dtype=np.float32) + 1j * xp.random.randn(
                    npts, dtype=np.float32
                )
                locs = _get_valid_locs(
                    (npts,), grid_size, ndim, width, device, centered=True
                )
                coord = from_pytorch(locs)
                return y, coord

        def fn_sigpy(data):
            y, coord = data
            with dev:
                return sp.nufft_adjoint(
                    y, coord, grid_size, oversamp=oversamp, width=width
                )

        benchmark_session.run(
            name="NUFFT adjoint 2D",
            fn=fn_sigpy,
            device=device,
            label="NUFFT 2D",
            sub_label=f"adjoint {sl}, {npts} locs",
            description="adjoint",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


# ---------------------------------------------------------------------------
# NUFFT 3D
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_nufft_forward_3d_cpu(benchmark_session, size_name):
    spec = NUFFT_3D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 3
    width = 4.0
    oversamp = 1.25
    device = "cpu"
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        return x, locs

    def fn_functional(data):
        x, locs = data
        return nufft(x, locs, oversamp=oversamp, width=width)

    benchmark_session.run(
        name="NUFFT forward 3D",
        fn=fn_functional,
        device=device,
        label="NUFFT 3D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(locs, grid_size, output_shape=("K",), width=width, oversamp=oversamp)

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    benchmark_session.run(
        name="NUFFT forward 3D",
        fn=fn_linop,
        device=device,
        label="NUFFT 3D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            x = np.random.randn(*grid_size) + 1j * np.random.randn(*grid_size)
            locs = _get_valid_locs(
                (npts,), grid_size, ndim, width, device, centered=True
            )
            coord = from_pytorch(locs)
            return x, coord

        def fn_sigpy(data):
            x, coord = data
            return sp.nufft(x, coord, oversamp=oversamp, width=width)

        benchmark_session.run(
            name="NUFFT forward 3D",
            fn=fn_sigpy,
            device=device,
            label="NUFFT 3D",
            sub_label=f"forward {sl}, {npts} locs",
            description="forward",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@pytest.mark.slow
@gpu_marks
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_nufft_forward_3d_gpu(benchmark_session, size_name):
    spec = NUFFT_3D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 3
    width = 4.0
    oversamp = 1.25
    device = "cuda"
    dev = _sigpy_device(device)
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        return x, locs

    def fn_functional(data):
        x, locs = data
        return nufft(x, locs, oversamp=oversamp, width=width)

    benchmark_session.run(
        name="NUFFT forward 3D",
        fn=fn_functional,
        device=device,
        label="NUFFT 3D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(locs, grid_size, output_shape=("K",), width=width, oversamp=oversamp)

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    benchmark_session.run(
        name="NUFFT forward 3D",
        fn=fn_linop,
        device=device,
        label="NUFFT 3D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            with dev:
                xp = dev.xp
                x = xp.random.randn(
                    *grid_size, dtype=np.float32
                ) + 1j * xp.random.randn(*grid_size, dtype=np.float32)
                locs = _get_valid_locs(
                    (npts,), grid_size, ndim, width, device, centered=True
                )
                coord = from_pytorch(locs)
                return x, coord

        def fn_sigpy(data):
            x, coord = data
            with dev:
                return sp.nufft(x, coord, oversamp=oversamp, width=width)

        benchmark_session.run(
            name="NUFFT forward 3D",
            fn=fn_sigpy,
            device=device,
            label="NUFFT 3D",
            sub_label=f"forward {sl}, {npts} locs",
            description="forward",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_nufft_adjoint_3d_cpu(benchmark_session, size_name):
    spec = NUFFT_3D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 3
    width = 4.0
    oversamp = 1.25
    device = "cpu"
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        y = torch.randn(npts, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        return y, locs

    def fn_functional(data):
        y, locs = data
        return nufft_adjoint(y, locs, grid_size, oversamp=oversamp, width=width)

    benchmark_session.run(
        name="NUFFT adjoint 3D",
        fn=fn_functional,
        device=device,
        label="NUFFT 3D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(locs, grid_size, output_shape=("K",), width=width, oversamp=oversamp)
    AH = A.H

    def gen_linop():
        return torch.randn(npts, dtype=torch.complex64, device=device)

    def fn_linop(y):
        return AH(y)

    benchmark_session.run(
        name="NUFFT adjoint 3D",
        fn=fn_linop,
        device=device,
        label="NUFFT 3D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            y = np.random.randn(npts) + 1j * np.random.randn(npts)
            locs = _get_valid_locs(
                (npts,), grid_size, ndim, width, device, centered=True
            )
            coord = from_pytorch(locs)
            return y, coord

        def fn_sigpy(data):
            y, coord = data
            return sp.nufft_adjoint(y, coord, grid_size, oversamp=oversamp, width=width)

        benchmark_session.run(
            name="NUFFT adjoint 3D",
            fn=fn_sigpy,
            device=device,
            label="NUFFT 3D",
            sub_label=f"adjoint {sl}, {npts} locs",
            description="adjoint",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@pytest.mark.slow
@gpu_marks
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_nufft_adjoint_3d_gpu(benchmark_session, size_name):
    spec = NUFFT_3D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 3
    width = 4.0
    oversamp = 1.25
    device = "cuda"
    dev = _sigpy_device(device)
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        y = torch.randn(npts, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
        return y, locs

    def fn_functional(data):
        y, locs = data
        return nufft_adjoint(y, locs, grid_size, oversamp=oversamp, width=width)

    benchmark_session.run(
        name="NUFFT adjoint 3D",
        fn=fn_functional,
        device=device,
        label="NUFFT 3D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(locs, grid_size, output_shape=("K",), width=width, oversamp=oversamp)
    AH = A.H

    def gen_linop():
        return torch.randn(npts, dtype=torch.complex64, device=device)

    def fn_linop(y):
        return AH(y)

    benchmark_session.run(
        name="NUFFT adjoint 3D",
        fn=fn_linop,
        device=device,
        label="NUFFT 3D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            with dev:
                xp = dev.xp
                y = xp.random.randn(npts, dtype=np.float32) + 1j * xp.random.randn(
                    npts, dtype=np.float32
                )
                locs = _get_valid_locs(
                    (npts,), grid_size, ndim, width, device, centered=True
                )
                coord = from_pytorch(locs)
                return y, coord

        def fn_sigpy(data):
            y, coord = data
            with dev:
                return sp.nufft_adjoint(
                    y, coord, grid_size, oversamp=oversamp, width=width
                )

        benchmark_session.run(
            name="NUFFT adjoint 3D",
            fn=fn_sigpy,
            device=device,
            label="NUFFT 3D",
            sub_label=f"adjoint {sl}, {npts} locs",
            description="adjoint",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


# ---------------------------------------------------------------------------
# ArrayToBlocks / BlocksToArray 3D
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_array_to_blocks_forward_cpu(benchmark_session, size_name):
    spec = ARRAY_TO_BLOCKS_SIZES[size_name]
    grid_size = spec["grid_size"]
    block_size = spec["block_size"]
    stride = spec["stride"]
    device = "cpu"
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_functional(x):
        return array_to_blocks(x, block_size, stride)

    benchmark_session.run(
        name="ArrayToBlocks forward",
        fn=fn_functional,
        device=device,
        label="ArrayToBlocks 3D",
        sub_label=f"forward {sl}, block 8x8x8, stride 4",
        description="forward",
        library="torchlinops",
        data_gen_fn=gen,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    A = ArrayToBlocks(grid_size, block_size, stride)

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    benchmark_session.run(
        name="ArrayToBlocks forward",
        fn=fn_linop,
        device=device,
        label="ArrayToBlocks 3D",
        sub_label=f"forward {sl}, block 8x8x8, stride 4",
        description="forward",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            return np.random.randn(*grid_size) + 1j * np.random.randn(*grid_size)

        def fn_sigpy(x):
            return sp.array_to_blocks(x, block_size, stride)

        benchmark_session.run(
            name="ArrayToBlocks forward",
            fn=fn_sigpy,
            device=device,
            label="ArrayToBlocks 3D",
            sub_label=f"forward {sl}, block 8x8x8, stride 4",
            description="forward",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@gpu_marks
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_array_to_blocks_forward_gpu(benchmark_session, size_name):
    spec = ARRAY_TO_BLOCKS_SIZES[size_name]
    grid_size = spec["grid_size"]
    block_size = spec["block_size"]
    stride = spec["stride"]
    device = "cuda"
    dev = _sigpy_device(device)
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_functional(x):
        return array_to_blocks(x, block_size, stride)

    benchmark_session.run(
        name="ArrayToBlocks forward",
        fn=fn_functional,
        device=device,
        label="ArrayToBlocks 3D",
        sub_label=f"forward {sl}, block 8x8x8, stride 4",
        description="forward",
        library="torchlinops",
        data_gen_fn=gen,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    A = ArrayToBlocks(grid_size, block_size, stride)

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    benchmark_session.run(
        name="ArrayToBlocks forward",
        fn=fn_linop,
        device=device,
        label="ArrayToBlocks 3D",
        sub_label=f"forward {sl}, block 8x8x8, stride 4",
        description="forward",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            with dev:
                xp = dev.xp
                return xp.random.randn(
                    *grid_size, dtype=np.float32
                ) + 1j * xp.random.randn(*grid_size, dtype=np.float32)

        def fn_sigpy(x):
            with dev:
                return sp.array_to_blocks(x, block_size, stride)

        benchmark_session.run(
            name="ArrayToBlocks forward",
            fn=fn_sigpy,
            device=device,
            label="ArrayToBlocks 3D",
            sub_label=f"forward {sl}, block 8x8x8, stride 4",
            description="forward",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_blocks_to_array_forward_cpu(benchmark_session, size_name):
    spec = ARRAY_TO_BLOCKS_SIZES[size_name]
    grid_size = spec["grid_size"]
    block_size = spec["block_size"]
    stride = spec["stride"]
    device = "cpu"
    nblocks = get_nblocks(grid_size, block_size, stride)
    blocks_shape = (*nblocks, *block_size)
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen():
        return torch.randn(*blocks_shape, dtype=torch.complex64, device=device)

    def fn_functional(x):
        return blocks_to_array(x, grid_size, block_size, stride)

    benchmark_session.run(
        name="BlocksToArray forward",
        fn=fn_functional,
        device=device,
        label="ArrayToBlocks 3D",
        sub_label=f"adjoint {sl}, block 8x8x8, stride 4",
        description="adjoint",
        library="torchlinops",
        data_gen_fn=gen,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    A = BlocksToArray(grid_size, block_size, stride)

    def gen_linop():
        return torch.randn(*blocks_shape, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    benchmark_session.run(
        name="BlocksToArray forward",
        fn=fn_linop,
        device=device,
        label="ArrayToBlocks 3D",
        sub_label=f"adjoint {sl}, block 8x8x8, stride 4",
        description="adjoint",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            return np.random.randn(*blocks_shape) + 1j * np.random.randn(*blocks_shape)

        def fn_sigpy(x):
            return sp.blocks_to_array(x, grid_size, block_size, stride)

        benchmark_session.run(
            name="BlocksToArray forward",
            fn=fn_sigpy,
            device=device,
            label="ArrayToBlocks 3D",
            sub_label=f"adjoint {sl}, block 8x8x8, stride 4",
            description="adjoint",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@gpu_marks
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_blocks_to_array_forward_gpu(benchmark_session, size_name):
    spec = ARRAY_TO_BLOCKS_SIZES[size_name]
    grid_size = spec["grid_size"]
    block_size = spec["block_size"]
    stride = spec["stride"]
    device = "cuda"
    dev = _sigpy_device(device)
    nblocks = get_nblocks(grid_size, block_size, stride)
    blocks_shape = (*nblocks, *block_size)
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen():
        return torch.randn(*blocks_shape, dtype=torch.complex64, device=device)

    def fn_functional(x):
        return blocks_to_array(x, grid_size, block_size, stride)

    benchmark_session.run(
        name="BlocksToArray forward",
        fn=fn_functional,
        device=device,
        label="ArrayToBlocks 3D",
        sub_label=f"adjoint {sl}, block 8x8x8, stride 4",
        description="adjoint",
        library="torchlinops",
        data_gen_fn=gen,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    A = BlocksToArray(grid_size, block_size, stride)

    def gen_linop():
        return torch.randn(*blocks_shape, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    benchmark_session.run(
        name="BlocksToArray forward",
        fn=fn_linop,
        device=device,
        label="ArrayToBlocks 3D",
        sub_label=f"adjoint {sl}, block 8x8x8, stride 4",
        description="adjoint",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            with dev:
                xp = dev.xp
                return xp.random.randn(
                    *blocks_shape, dtype=np.float32
                ) + 1j * xp.random.randn(*blocks_shape, dtype=np.float32)

        def fn_sigpy(x):
            with dev:
                return sp.blocks_to_array(x, grid_size, block_size, stride)

        benchmark_session.run(
            name="BlocksToArray forward",
            fn=fn_sigpy,
            device=device,
            label="ArrayToBlocks 3D",
            sub_label=f"adjoint {sl}, block 8x8x8, stride 4",
            description="adjoint",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


# ---------------------------------------------------------------------------
# Interpolate 2D
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_interpolate_forward_2d_cpu(benchmark_session, size_name):
    spec = INTERP_2D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 2
    width = 4.0
    device = "cpu"
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
        return x, locs

    def fn_functional(data):
        x, locs = data
        return interpolate(x, locs, width=width, kernel="kaiser_bessel")

    benchmark_session.run(
        name="Interpolate forward 2D",
        fn=fn_functional,
        device=device,
        label="Interpolate 2D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    benchmark_session.run(
        name="Interpolate forward 2D",
        fn=fn_linop,
        device=device,
        label="Interpolate 2D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            x = np.random.randn(*grid_size) + 1j * np.random.randn(*grid_size)
            locs = _get_valid_locs(
                (npts,), grid_size, ndim, width, device, centered=False
            )
            coord = from_pytorch(locs)
            return x, coord

        def fn_sigpy(data):
            x, coord = data
            return sp.interp.interpolate(
                x, coord, kernel="kaiser_bessel", width=width, param=1.0
            )

        benchmark_session.run(
            name="Interpolate forward 2D",
            fn=fn_sigpy,
            device=device,
            label="Interpolate 2D",
            sub_label=f"forward {sl}, {npts} locs",
            description="forward",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@gpu_marks
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_interpolate_forward_2d_gpu(benchmark_session, size_name):
    spec = INTERP_2D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 2
    width = 4.0
    device = "cuda"
    dev = _sigpy_device(device)
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
        return x, locs

    def fn_functional(data):
        x, locs = data
        return interpolate(x, locs, width=width, kernel="kaiser_bessel")

    benchmark_session.run(
        name="Interpolate forward 2D",
        fn=fn_functional,
        device=device,
        label="Interpolate 2D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    benchmark_session.run(
        name="Interpolate forward 2D",
        fn=fn_linop,
        device=device,
        label="Interpolate 2D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            with dev:
                xp = dev.xp
                x = xp.random.randn(
                    *grid_size, dtype=np.float32
                ) + 1j * xp.random.randn(*grid_size, dtype=np.float32)
                locs = _get_valid_locs(
                    (npts,), grid_size, ndim, width, device, centered=False
                )
                coord = from_pytorch(locs)
                return x, coord

        def fn_sigpy(data):
            x, coord = data
            with dev:
                return sp.interp.interpolate(
                    x, coord, kernel="kaiser_bessel", width=width, param=1.0
                )

        benchmark_session.run(
            name="Interpolate forward 2D",
            fn=fn_sigpy,
            device=device,
            label="Interpolate 2D",
            sub_label=f"forward {sl}, {npts} locs",
            description="forward",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_interpolate_adjoint_2d_cpu(benchmark_session, size_name):
    spec = INTERP_2D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 2
    width = 4.0
    device = "cpu"
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

    benchmark_session.run(
        name="Interpolate adjoint 2D",
        fn=fn_functional,
        device=device,
        label="Interpolate 2D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")
    AH = A.H

    def gen_linop():
        return torch.randn(npts, dtype=torch.complex64, device=device)

    def fn_linop(y):
        return AH(y)

    benchmark_session.run(
        name="Interpolate adjoint 2D",
        fn=fn_linop,
        device=device,
        label="Interpolate 2D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            y = np.random.randn(npts) + 1j * np.random.randn(npts)
            locs = _get_valid_locs(
                (npts,), grid_size, ndim, width, device, centered=False
            )
            coord = from_pytorch(locs)
            return y, coord

        def fn_sigpy(data):
            y, coord = data
            return sp.interp.gridding(
                y, coord, grid_size, kernel="kaiser_bessel", width=width, param=1.0
            )

        benchmark_session.run(
            name="Interpolate adjoint 2D",
            fn=fn_sigpy,
            device=device,
            label="Interpolate 2D",
            sub_label=f"adjoint {sl}, {npts} locs",
            description="adjoint",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@gpu_marks
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_interpolate_adjoint_2d_gpu(benchmark_session, size_name):
    spec = INTERP_2D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 2
    width = 4.0
    device = "cuda"
    dev = _sigpy_device(device)
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

    benchmark_session.run(
        name="Interpolate adjoint 2D",
        fn=fn_functional,
        device=device,
        label="Interpolate 2D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")
    AH = A.H

    def gen_linop():
        return torch.randn(npts, dtype=torch.complex64, device=device)

    def fn_linop(y):
        return AH(y)

    benchmark_session.run(
        name="Interpolate adjoint 2D",
        fn=fn_linop,
        device=device,
        label="Interpolate 2D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            with dev:
                xp = dev.xp
                y = xp.random.randn(npts, dtype=np.float32) + 1j * xp.random.randn(
                    npts, dtype=np.float32
                )
                locs = _get_valid_locs(
                    (npts,), grid_size, ndim, width, device, centered=False
                )
                coord = from_pytorch(locs)
                return y, coord

        def fn_sigpy(data):
            y, coord = data
            with dev:
                return sp.interp.gridding(
                    y, coord, grid_size, kernel="kaiser_bessel", width=width, param=1.0
                )

        benchmark_session.run(
            name="Interpolate adjoint 2D",
            fn=fn_sigpy,
            device=device,
            label="Interpolate 2D",
            sub_label=f"adjoint {sl}, {npts} locs",
            description="adjoint",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


# ---------------------------------------------------------------------------
# Interpolate 3D
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_interpolate_forward_3d_cpu(benchmark_session, size_name):
    spec = INTERP_3D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 3
    width = 4.0
    device = "cpu"
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
        return x, locs

    def fn_functional(data):
        x, locs = data
        return interpolate(x, locs, width=width, kernel="kaiser_bessel")

    benchmark_session.run(
        name="Interpolate forward 3D",
        fn=fn_functional,
        device=device,
        label="Interpolate 3D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    benchmark_session.run(
        name="Interpolate forward 3D",
        fn=fn_linop,
        device=device,
        label="Interpolate 3D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            x = np.random.randn(*grid_size) + 1j * np.random.randn(*grid_size)
            locs = _get_valid_locs(
                (npts,), grid_size, ndim, width, device, centered=False
            )
            coord = from_pytorch(locs)
            return x, coord

        def fn_sigpy(data):
            x, coord = data
            return sp.interp.interpolate(
                x, coord, kernel="kaiser_bessel", width=width, param=1.0
            )

        benchmark_session.run(
            name="Interpolate forward 3D",
            fn=fn_sigpy,
            device=device,
            label="Interpolate 3D",
            sub_label=f"forward {sl}, {npts} locs",
            description="forward",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@pytest.mark.slow
@gpu_marks
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_interpolate_forward_3d_gpu(benchmark_session, size_name):
    spec = INTERP_3D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 3
    width = 4.0
    device = "cuda"
    dev = _sigpy_device(device)
    ps = _problem_size(grid_size)
    sl = _size_label(grid_size)

    def gen_functional():
        x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
        locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
        return x, locs

    def fn_functional(data):
        x, locs = data
        return interpolate(x, locs, width=width, kernel="kaiser_bessel")

    benchmark_session.run(
        name="Interpolate forward 3D",
        fn=fn_functional,
        device=device,
        label="Interpolate 3D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")

    def gen_linop():
        return torch.randn(*grid_size, dtype=torch.complex64, device=device)

    def fn_linop(x):
        return A(x)

    benchmark_session.run(
        name="Interpolate forward 3D",
        fn=fn_linop,
        device=device,
        label="Interpolate 3D",
        sub_label=f"forward {sl}, {npts} locs",
        description="forward",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            with dev:
                xp = dev.xp
                x = xp.random.randn(
                    *grid_size, dtype=np.float32
                ) + 1j * xp.random.randn(*grid_size, dtype=np.float32)
                locs = _get_valid_locs(
                    (npts,), grid_size, ndim, width, device, centered=False
                )
                coord = from_pytorch(locs)
                return x, coord

        def fn_sigpy(data):
            x, coord = data
            with dev:
                return sp.interp.interpolate(
                    x, coord, kernel="kaiser_bessel", width=width, param=1.0
                )

        benchmark_session.run(
            name="Interpolate forward 3D",
            fn=fn_sigpy,
            device=device,
            label="Interpolate 3D",
            sub_label=f"forward {sl}, {npts} locs",
            description="forward",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_interpolate_adjoint_3d_cpu(benchmark_session, size_name):
    spec = INTERP_3D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 3
    width = 4.0
    device = "cpu"
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

    benchmark_session.run(
        name="Interpolate adjoint 3D",
        fn=fn_functional,
        device=device,
        label="Interpolate 3D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")
    AH = A.H

    def gen_linop():
        return torch.randn(npts, dtype=torch.complex64, device=device)

    def fn_linop(y):
        return AH(y)

    benchmark_session.run(
        name="Interpolate adjoint 3D",
        fn=fn_linop,
        device=device,
        label="Interpolate 3D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            y = np.random.randn(npts) + 1j * np.random.randn(npts)
            locs = _get_valid_locs(
                (npts,), grid_size, ndim, width, device, centered=False
            )
            coord = from_pytorch(locs)
            return y, coord

        def fn_sigpy(data):
            y, coord = data
            return sp.interp.gridding(
                y, coord, grid_size, kernel="kaiser_bessel", width=width, param=1.0
            )

        benchmark_session.run(
            name="Interpolate adjoint 3D",
            fn=fn_sigpy,
            device=device,
            label="Interpolate 3D",
            sub_label=f"adjoint {sl}, {npts} locs",
            description="adjoint",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )


@pytest.mark.benchmark
@pytest.mark.slow
@gpu_marks
@pytest.mark.parametrize("size_name", SIZE_NAMES)
def test_interpolate_adjoint_3d_gpu(benchmark_session, size_name):
    spec = INTERP_3D_SIZES[size_name]
    grid_size = spec["grid_size"]
    npts = spec["npts"]
    ndim = 3
    width = 4.0
    device = "cuda"
    dev = _sigpy_device(device)
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

    benchmark_session.run(
        name="Interpolate adjoint 3D",
        fn=fn_functional,
        device=device,
        label="Interpolate 3D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops",
        data_gen_fn=gen_functional,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    # Construct linop once outside timed functions
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")
    AH = A.H

    def gen_linop():
        return torch.randn(npts, dtype=torch.complex64, device=device)

    def fn_linop(y):
        return AH(y)

    benchmark_session.run(
        name="Interpolate adjoint 3D",
        fn=fn_linop,
        device=device,
        label="Interpolate 3D",
        sub_label=f"adjoint {sl}, {npts} locs",
        description="adjoint",
        library="torchlinops (linop)",
        data_gen_fn=gen_linop,
        size_name=size_name,
        problem_size=ps,
        size_label=sl,
    )

    if SIGPY_AVAILABLE:

        def gen_sigpy():
            with dev:
                xp = dev.xp
                y = xp.random.randn(npts, dtype=np.float32) + 1j * xp.random.randn(
                    npts, dtype=np.float32
                )
                locs = _get_valid_locs(
                    (npts,), grid_size, ndim, width, device, centered=False
                )
                coord = from_pytorch(locs)
                return y, coord

        def fn_sigpy(data):
            y, coord = data
            with dev:
                return sp.interp.gridding(
                    y, coord, grid_size, kernel="kaiser_bessel", width=width, param=1.0
                )

        benchmark_session.run(
            name="Interpolate adjoint 3D",
            fn=fn_sigpy,
            device=device,
            label="Interpolate 3D",
            sub_label=f"adjoint {sl}, {npts} locs",
            description="adjoint",
            library="sigpy",
            data_gen_fn=gen_sigpy,
            size_name=size_name,
            problem_size=ps,
            size_label=sl,
        )
