import pytest
from math import prod

import torch
import sigpy as sp
import numpy as np

from torchlinops import NUFFT, PadLast, Interpolate
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs
from torchlinops.tests.test_base import BaseNamedLinopTests
from torchlinops.utils import cifftn


class TestNUFFT(BaseNamedLinopTests):
    equality_check = "approx"

    oversamp = [1.0, 1.25]

    instances = ["small3d"]

    # Unstable numerical behavior
    isclose_kwargs: dict = {"rtol": 1e-3}

    @pytest.fixture(scope="class", params=instances)
    def linop_input_output(self, request):
        spec = request.param
        spec = request.getfixturevalue(spec)
        width = spec["width"]
        oversamp = spec["oversamp"]
        grid_size = spec["grid_size"]
        locs_batch_size = spec["locs_batch_size"]
        ndim = len(grid_size)
        npts = prod(locs_batch_size)
        batch_size = spec["N"]
        ishape = (*batch_size, *grid_size)
        oshape = (*batch_size, *locs_batch_size)
        locs = get_valid_locs(
            locs_batch_size, grid_size, ndim, width, "cpu", centered=True
        )

        linop = NUFFT(
            locs,
            grid_size,
            output_shape=("R", "K"),
            width=width,
            oversamp=oversamp,
        )
        # Limit randomness
        x = 0.5 * torch.rand(ishape, dtype=torch.complex64, device="cpu") + 1
        y = 0.5 * torch.rand(oshape, dtype=torch.complex64, device="cpu") + 1
        y /= torch.linalg.vector_norm(locs, dim=-1)

        return linop, x, y

    @pytest.fixture(scope="class")
    def small3d(self, request):
        N = (2, 1)
        # grid_size = (16, 16, 24)
        grid_size = (32, 32, 32)
        locs_batch_size = (3, 5)
        width = 4.0
        oversamp = 1.25

        spec = {
            "N": N,
            "grid_size": grid_size,
            "locs_batch_size": locs_batch_size,
            "width": width,
            "oversamp": oversamp,
        }
        return spec

    def test_nufft_sigpy(self, linop_input_output):
        A, x, y = linop_input_output
        coord = A.locs.numpy()
        # sz = np.array(A.grid_size)
        # coord = np.where(coord <= (sz / 2), coord, coord - sz)
        width = A.width
        oversamp = A.oversamp

        Ax = A(x).numpy()
        Ax_sp = sp.nufft(x.numpy(), coord, oversamp=oversamp, width=width)
        assert np.allclose(Ax, Ax_sp, **self.isclose_kwargs)

        AHy = A.H(y).numpy()
        AHy_sp = sp.nufft_adjoint(
            y.numpy(), coord, x.shape, oversamp=oversamp, width=width
        )
        assert np.allclose(AHy, AHy_sp, **self.isclose_kwargs)


@pytest.fixture
def nufft_params():
    width = 6.0
    oversamp = 2.0
    grid_size = (128, 128, 128)
    padded_size = [int(i * oversamp) for i in grid_size]
    locs = get_valid_locs((10,), grid_size, len(grid_size), width, "cpu", centered=True)
    return {
        "width": width,
        "oversamp": oversamp,
        "grid_size": grid_size,
        "padded_size": padded_size,
        "locs": locs,
    }


def test_apodize(nufft_params):
    width = nufft_params["width"]
    oversamp = nufft_params["oversamp"]
    grid_size = nufft_params["grid_size"]
    padded_size = nufft_params["padded_size"]

    beta = NUFFT.beta(width, oversamp)
    apod = NUFFT.apodize_weights(grid_size, padded_size, oversamp, width, beta)

    x = np.ones(grid_size)
    apod_sp = sp.fourier._apodize(x, len(grid_size), oversamp, width, beta)
    assert np.allclose(apod, apod_sp)


def test_nufft_os_pad(nufft_params):
    grid_size = nufft_params["grid_size"]
    padded_size = nufft_params["padded_size"]
    pad = PadLast(padded_size, grid_size)

    x = torch.randn(*grid_size)
    padx = pad(x).numpy()

    padx_sp = sp.util.resize(x.numpy(), padded_size)

    assert np.allclose(padx, padx_sp)


def test_scale_locs(nufft_params):
    grid_size = nufft_params["grid_size"]
    padded_size = nufft_params["padded_size"]
    oversamp = nufft_params["oversamp"]

    # Torch version
    locs = nufft_params["locs"]
    locs_scaled = NUFFT.prep_locs(locs, grid_size, padded_size)

    coord = locs.clone().numpy()
    sz = np.array(grid_size)
    coord_scaled = sp.fourier._scale_coord(coord, grid_size, oversamp)
    assert np.allclose(locs_scaled, coord_scaled)


def test_nufft_interp(nufft_params):
    locs = nufft_params["locs"]
    grid_size = nufft_params["grid_size"]
    padded_size = nufft_params["padded_size"]
    width = nufft_params["width"]
    oversamp = nufft_params["oversamp"]
    beta = NUFFT.beta(width, oversamp)

    locs_prepared = NUFFT.prep_locs(locs.clone(), grid_size, padded_size)
    interp = Interpolate(
        locs_prepared,
        padded_size,
        batch_shape=None,
        locs_batch_shape=None,
        grid_shape=None,
        width=width,
        kernel="kaiser_bessel",
        kernel_params=dict(beta=beta),
    )

    x = torch.randn(*padded_size, dtype=torch.complex64)
    interpx = interp(x)

    interpx_sp = sp.interp.interpolate(
        x.numpy(),
        locs_prepared.numpy(),
        kernel="kaiser_bessel",
        width=width,
        param=beta,
    )

    assert np.allclose(interpx, interpx_sp, rtol=1e-3)
