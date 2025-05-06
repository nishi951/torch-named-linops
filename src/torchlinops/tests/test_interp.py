import pytest
from itertools import product

from math import prod

import torch

from torchlinops import Interpolate
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs

from torchlinops.utils import is_adjoint
from torchlinops.tests.test_base import BaseNamedLinopTests


PYTEST_GPU_MARKS = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU is required but not available"
    ),
]


class TestInterp(BaseNamedLinopTests):
    equality_check = "approx"

    # Fixture/test parameterization
    devices = ["cpu"]
    instances = ["small3d"]

    @pytest.fixture(
        scope="class",
        params=product(instances, devices),
    )
    def linop_input_output(self, request):
        spec, dev = request.param
        spec = request.getfixturevalue(spec)
        device = torch.device(dev)
        dtype = torch.complex64

        width = spec["width"]
        grid_size = spec["grid_size"]
        locs_batch_size = spec["locs_batch_size"]
        ndim = len(grid_size)
        npts = prod(locs_batch_size)
        batch_size = spec["N"]
        ishape = (*batch_size, *grid_size)
        oshape = (*batch_size, *locs_batch_size)
        locs = get_valid_locs(locs_batch_size, grid_size, ndim, width, device)

        linop = Interpolate(locs, grid_size, width=width, kernel="kaiser_bessel")
        x = torch.randn(ishape, dtype=dtype, device=device)
        y = torch.randn(oshape, dtype=dtype, device=device)

        return linop, x, y

    @pytest.fixture(scope="class")
    def small3d(self, request):
        N = (1, 1)
        grid_size = (10, 8, 9)
        locs_batch_size = (3, 5)
        width = 4.0

        spec = {
            "N": N,
            "grid_size": grid_size,
            "locs_batch_size": locs_batch_size,
            "width": width,
        }
        return spec


def test_interp_slc():
    locs = torch.rand(8, 6, 5, 3)
    grid_size = (5, 5, 5)
    linop = Interpolate(locs, grid_size, locs_batch_shape=("A", "B", "C"))
    ibatch = (slice(None),) * 3
    obatch = [slice(None), slice(0, 4), slice(2, 3)]
    linop_split = linop.split(linop, ibatch, obatch)
    assert linop_split.locs.shape == (8, 4, 1, 3)
    assert (linop_split.locs == linop.locs[obatch]).all()
