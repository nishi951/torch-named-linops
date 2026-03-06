import numpy as np
import pytest
import torch
from sigpy.fourier import toeplitz_psf as sp_toeplitz_psf

from torchlinops import NUFFT, Dense, Diagonal, Dim
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs
from torchlinops.linops.nufft import toeplitz_psf
from torchlinops.utils import from_pytorch


@pytest.fixture
def sampling_nufft_linop():
    """A NUFFT in sampling mode (integer locs), for which toeplitz_psf is not implemented."""
    grid_size = (16, 16)
    padded_size = (20, 20)
    # Locs must be in range [0, padded_size-1] for sampling mode
    locs = torch.randint(0, 19, (10, 2))
    linop = NUFFT(
        locs.clone(),
        grid_size,
        output_shape=Dim("K"),
        mode="sampling",
        do_prep_locs=False,
    )
    return linop


@pytest.fixture
def nufft_params():
    width = 4.0
    oversamp = 1.25
    # grid_size = (120, 119, 146)
    grid_size = (64, 64, 64)
    padded_size = [int(i * oversamp) for i in grid_size]
    locs = get_valid_locs(
        (20, 500),
        grid_size,
        len(grid_size),
        width,
        "cpu",
        centered=True,
    )
    return {
        "width": width,
        "oversamp": oversamp,
        "grid_size": grid_size,
        "padded_size": padded_size,
        "locs": locs,
    }


@pytest.fixture
def nufft_linop(nufft_params):
    locs = nufft_params["locs"]
    grid_size = nufft_params["grid_size"]
    width = nufft_params["width"]
    oversamp = nufft_params["oversamp"]
    linop = NUFFT(
        locs.clone(),
        grid_size,
        output_shape=Dim("RK"),
        batch_shape=Dim("A"),
        width=width,
        oversamp=oversamp,
    )
    return linop


@pytest.fixture
def simple_inner(nufft_params):
    locs = nufft_params["locs"]
    weight = torch.randn(locs.shape[:-1])
    linop = Diagonal(weight, ioshape=Dim("ARK"), broadcast_dims=Dim("A"))
    return linop


@pytest.fixture
def dense_inner(nufft_params):
    A = 2
    weight = torch.randn(A, A, dtype=torch.complex64)
    linop = Dense(
        weight,
        weightshape=Dim("AA1"),
        ishape=Dim("ARK"),
        oshape=Dim("A1RK"),
    )
    return linop


def test_toeplitz_psf_raises_for_sampling_mode(sampling_nufft_linop):
    """toeplitz_psf should raise NotImplementedError for Sampling-mode NUFFTs."""
    with pytest.raises(NotImplementedError, match="Sampling"):
        toeplitz_psf(sampling_nufft_linop)


@pytest.fixture
def nufft_linop_toeplitz(nufft_params):
    locs = nufft_params["locs"]
    grid_size = nufft_params["grid_size"]
    width = nufft_params["width"]
    oversamp = nufft_params["oversamp"]
    linop = NUFFT(
        locs.clone(),
        grid_size,
        output_shape=Dim("RK"),
        batch_shape=Dim("A"),
        width=width,
        oversamp=oversamp,
        toeplitz=True,
    )
    return linop


@pytest.mark.parametrize("inner_type", ["simple_inner", "dense_inner", None])
def test_normal_toeplitz(inner_type, nufft_linop_toeplitz, request):
    if inner_type is not None:
        inner = request.getfixturevalue(inner_type)
    else:
        inner = None
    normal_linop = nufft_linop_toeplitz.normal(inner)
    assert not any(isinstance(m, NUFFT) for m in normal_linop.modules())


@pytest.mark.parametrize("inner_type", ["simple_inner", "dense_inner", None])
def test_toeplitz_full(inner_type, nufft_linop, nufft_params, request):
    if inner_type is not None:
        inner = request.getfixturevalue(inner_type)
    else:
        inner = None
    kernel = toeplitz_psf(nufft_linop, inner)

    # Test against sigpy for no inner only
    if inner_type is None:
        psf = kernel.weight
        coord = from_pytorch(nufft_params["locs"].clone())
        psf_sp = sp_toeplitz_psf(
            coord,
            nufft_linop.grid_size,
            oversamp=nufft_linop.oversamp,
            width=nufft_linop.width,
        )
        assert np.isclose(psf_sp, psf.numpy(), rtol=1e-1).sum() / psf_sp.size > 0.99
