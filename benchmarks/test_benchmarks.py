"""Performance benchmarks for core torch-named-linops operators."""

import pytest
import torch

from torchlinops import ArrayToBlocks, BlocksToArray, Interpolate, NUFFT


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


def _make_nufft(grid_size, npts, device, ndim):
    width = 4.0
    oversamp = 1.25
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=True)
    A = NUFFT(
        locs,
        grid_size,
        output_shape=("K",),
        width=width,
        oversamp=oversamp,
    )
    x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
    return A, x


def _make_interpolate(grid_size, npts, device, ndim):
    width = 4.0
    locs = _get_valid_locs((npts,), grid_size, ndim, width, device, centered=False)
    A = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")
    x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
    return A, x


def _make_array_to_blocks(grid_size, block_size, stride, device):
    A = ArrayToBlocks(grid_size, block_size, stride)
    x = torch.randn(*grid_size, dtype=torch.complex64, device=device)
    return A, x


def _make_blocks_to_array(grid_size, block_size, stride, device):
    A = BlocksToArray(grid_size, block_size, stride)
    import torchlinops.functional as F

    nblocks = F.get_nblocks(grid_size, block_size, stride)
    x = torch.randn(*nblocks, *block_size, dtype=torch.complex64, device=device)
    return A, x


# ---------------------------------------------------------------------------
# NUFFT
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_nufft_forward_2d_cpu(benchmark_session):
    A, x = _make_nufft((64, 64), 1000, "cpu", 2)
    benchmark_session.run(
        name="NUFFT forward 2D",
        fn=lambda: A(x),
        device="cpu",
        label="NUFFT",
        sub_label="forward 64x64, 1000 locs",
        description="forward",
    )


@pytest.mark.benchmark
@gpu_marks
def test_nufft_forward_2d_gpu(benchmark_session):
    A, x = _make_nufft((64, 64), 1000, "cuda", 2)
    benchmark_session.run(
        name="NUFFT forward 2D",
        fn=lambda: A(x),
        device="cuda",
        label="NUFFT",
        sub_label="forward 64x64, 1000 locs",
        description="forward",
    )


@pytest.mark.benchmark
def test_nufft_adjoint_2d_cpu(benchmark_session):
    A, x = _make_nufft((64, 64), 1000, "cpu", 2)
    y = torch.randn(1000, dtype=torch.complex64, device="cpu")
    benchmark_session.run(
        name="NUFFT adjoint 2D",
        fn=lambda: A.H(y),
        device="cpu",
        label="NUFFT",
        sub_label="adjoint 64x64, 1000 locs",
        description="adjoint",
    )


@pytest.mark.benchmark
@gpu_marks
def test_nufft_adjoint_2d_gpu(benchmark_session):
    A, x = _make_nufft((64, 64), 1000, "cuda", 2)
    y = torch.randn(1000, dtype=torch.complex64, device="cuda")
    benchmark_session.run(
        name="NUFFT adjoint 2D",
        fn=lambda: A.H(y),
        device="cuda",
        label="NUFFT",
        sub_label="adjoint 64x64, 1000 locs",
        description="adjoint",
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_nufft_forward_3d_cpu(benchmark_session):
    A, x = _make_nufft((32, 32, 32), 500, "cpu", 3)
    benchmark_session.run(
        name="NUFFT forward 3D",
        fn=lambda: A(x),
        device="cpu",
        label="NUFFT",
        sub_label="forward 32x32x32, 500 locs",
        description="forward",
    )


@pytest.mark.benchmark
@pytest.mark.slow
@gpu_marks
def test_nufft_forward_3d_gpu(benchmark_session):
    A, x = _make_nufft((32, 32, 32), 500, "cuda", 3)
    benchmark_session.run(
        name="NUFFT forward 3D",
        fn=lambda: A(x),
        device="cuda",
        label="NUFFT",
        sub_label="forward 32x32x32, 500 locs",
        description="forward",
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_nufft_adjoint_3d_cpu(benchmark_session):
    A, x = _make_nufft((32, 32, 32), 500, "cpu", 3)
    y = torch.randn(500, dtype=torch.complex64, device="cpu")
    benchmark_session.run(
        name="NUFFT adjoint 3D",
        fn=lambda: A.H(y),
        device="cpu",
        label="NUFFT",
        sub_label="adjoint 32x32x32, 500 locs",
        description="adjoint",
    )


@pytest.mark.benchmark
@pytest.mark.slow
@gpu_marks
def test_nufft_adjoint_3d_gpu(benchmark_session):
    A, x = _make_nufft((32, 32, 32), 500, "cuda", 3)
    y = torch.randn(500, dtype=torch.complex64, device="cuda")
    benchmark_session.run(
        name="NUFFT adjoint 3D",
        fn=lambda: A.H(y),
        device="cuda",
        label="NUFFT",
        sub_label="adjoint 32x32x32, 500 locs",
        description="adjoint",
    )


# ---------------------------------------------------------------------------
# ArrayToBlocks / BlocksToArray
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_array_to_blocks_forward_cpu(benchmark_session):
    A, x = _make_array_to_blocks((32, 32, 32), (4, 4, 4), (2, 2, 2), "cpu")
    benchmark_session.run(
        name="ArrayToBlocks forward",
        fn=lambda: A(x),
        device="cpu",
        label="ArrayToBlocks",
        sub_label="forward 32x32x32, block 4x4x4, stride 2",
        description="forward",
    )


@pytest.mark.benchmark
@gpu_marks
def test_array_to_blocks_forward_gpu(benchmark_session):
    A, x = _make_array_to_blocks((32, 32, 32), (4, 4, 4), (2, 2, 2), "cuda")
    benchmark_session.run(
        name="ArrayToBlocks forward",
        fn=lambda: A(x),
        device="cuda",
        label="ArrayToBlocks",
        sub_label="forward 32x32x32, block 4x4x4, stride 2",
        description="forward",
    )


@pytest.mark.benchmark
def test_blocks_to_array_forward_cpu(benchmark_session):
    A, x = _make_blocks_to_array((32, 32, 32), (4, 4, 4), (2, 2, 2), "cpu")
    benchmark_session.run(
        name="BlocksToArray forward",
        fn=lambda: A(x),
        device="cpu",
        label="ArrayToBlocks",
        sub_label="adjoint 32x32x32, block 4x4x4, stride 2",
        description="adjoint",
    )


@pytest.mark.benchmark
@gpu_marks
def test_blocks_to_array_forward_gpu(benchmark_session):
    A, x = _make_blocks_to_array((32, 32, 32), (4, 4, 4), (2, 2, 2), "cuda")
    benchmark_session.run(
        name="BlocksToArray forward",
        fn=lambda: A(x),
        device="cuda",
        label="ArrayToBlocks",
        sub_label="adjoint 32x32x32, block 4x4x4, stride 2",
        description="adjoint",
    )


# ---------------------------------------------------------------------------
# Interpolate
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_interpolate_forward_2d_cpu(benchmark_session):
    A, x = _make_interpolate((64, 64), 2000, "cpu", 2)
    benchmark_session.run(
        name="Interpolate forward 2D",
        fn=lambda: A(x),
        device="cpu",
        label="Interpolate",
        sub_label="forward 64x64, 2000 locs",
        description="forward",
    )


@pytest.mark.benchmark
@gpu_marks
def test_interpolate_forward_2d_gpu(benchmark_session):
    A, x = _make_interpolate((64, 64), 2000, "cuda", 2)
    benchmark_session.run(
        name="Interpolate forward 2D",
        fn=lambda: A(x),
        device="cuda",
        label="Interpolate",
        sub_label="forward 64x64, 2000 locs",
        description="forward",
    )


@pytest.mark.benchmark
def test_interpolate_adjoint_2d_cpu(benchmark_session):
    A, x = _make_interpolate((64, 64), 2000, "cpu", 2)
    y = torch.randn(2000, dtype=torch.complex64, device="cpu")
    benchmark_session.run(
        name="Interpolate adjoint 2D",
        fn=lambda: A.H(y),
        device="cpu",
        label="Interpolate",
        sub_label="adjoint 64x64, 2000 locs",
        description="adjoint",
    )


@pytest.mark.benchmark
@gpu_marks
def test_interpolate_adjoint_2d_gpu(benchmark_session):
    A, x = _make_interpolate((64, 64), 2000, "cuda", 2)
    y = torch.randn(2000, dtype=torch.complex64, device="cuda")
    benchmark_session.run(
        name="Interpolate adjoint 2D",
        fn=lambda: A.H(y),
        device="cuda",
        label="Interpolate",
        sub_label="adjoint 64x64, 2000 locs",
        description="adjoint",
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_interpolate_forward_3d_cpu(benchmark_session):
    A, x = _make_interpolate((32, 32, 32), 1000, "cpu", 3)
    benchmark_session.run(
        name="Interpolate forward 3D",
        fn=lambda: A(x),
        device="cpu",
        label="Interpolate",
        sub_label="forward 32x32x32, 1000 locs",
        description="forward",
    )


@pytest.mark.benchmark
@pytest.mark.slow
@gpu_marks
def test_interpolate_forward_3d_gpu(benchmark_session):
    A, x = _make_interpolate((32, 32, 32), 1000, "cuda", 3)
    benchmark_session.run(
        name="Interpolate forward 3D",
        fn=lambda: A(x),
        device="cuda",
        label="Interpolate",
        sub_label="forward 32x32x32, 1000 locs",
        description="forward",
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_interpolate_adjoint_3d_cpu(benchmark_session):
    A, x = _make_interpolate((32, 32, 32), 1000, "cpu", 3)
    y = torch.randn(1000, dtype=torch.complex64, device="cpu")
    benchmark_session.run(
        name="Interpolate adjoint 3D",
        fn=lambda: A.H(y),
        device="cpu",
        label="Interpolate",
        sub_label="adjoint 32x32x32, 1000 locs",
        description="adjoint",
    )


@pytest.mark.benchmark
@pytest.mark.slow
@gpu_marks
def test_interpolate_adjoint_3d_gpu(benchmark_session):
    A, x = _make_interpolate((32, 32, 32), 1000, "cuda", 3)
    y = torch.randn(1000, dtype=torch.complex64, device="cuda")
    benchmark_session.run(
        name="Interpolate adjoint 3D",
        fn=lambda: A.H(y),
        device="cuda",
        label="Interpolate",
        sub_label="adjoint 32x32x32, 1000 locs",
        description="adjoint",
    )
