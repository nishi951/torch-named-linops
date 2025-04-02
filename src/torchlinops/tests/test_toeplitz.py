import pytest

import torch
import numpy as np

from torchlinops.functional._interp.tests._valid_pts import get_valid_locs
from torchlinops import NUFFT, Diagonal, Dense, Dim
from torchlinops.utils import from_pytorch

from torchlinops.linops.nufft import toeplitz_psf

from sigpy.fourier import toeplitz_psf as sp_toeplitz_psf


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


@pytest.mark.parametrize("inner_type", ["simple_inner", "dense_inner", None])
def test_toeplitz_full(inner_type, nufft_linop, nufft_params, request):
    if inner_type is not None:
        inner = request.getfixturevalue(inner_type)
    else:
        inner = None
    kernel, pad, fft, nufft = toeplitz_psf(nufft_linop, inner)

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
