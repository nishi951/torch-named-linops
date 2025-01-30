import pytest
from math import prod

import torch

from torchlinops import NUFFT
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs
from torchlinops._core.tests.test_base import BaseNamedLinopTests


class TestNUFFT(BaseNamedLinopTests):
    equality_check = "approx"

    oversamp = [1.0, 1.25]

    instances = ["small3d"]

    # Unstable numerical behavior
    isclose_kwargs: dict = {"rtol": 1e-4}

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
        locs = get_valid_locs(locs_batch_size, grid_size, ndim, width, "cpu")

        linop = NUFFT(
            locs,
            grid_size,
            output_shape=("R", "K"),
            width=width,
            oversamp=oversamp,
        )
        x = torch.randn(ishape, dtype=torch.complex64, device="cpu")
        y = torch.randn(oshape, dtype=torch.complex64, device="cpu")

        return linop, x, y

    @pytest.fixture(scope="class")
    def small3d(self, request):
        N = (2, 1)
        grid_size = (16, 16, 24)
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
