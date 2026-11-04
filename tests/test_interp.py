from itertools import product
from math import prod

import pytest
import torch

from torchlinops import Interpolate
from torchlinops.functional import interpolate, interpolate_adjoint
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs
from torchlinops.testing import BaseNamedLinopTests

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
    @classmethod
    def linop_input_output(cls, request):
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
    @classmethod
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

    def test_size(self, linop_input_output):
        A, x, y = linop_input_output
        for dim in A.dims:
            if dim != "...":
                size = A.size(dim)
                assert size is None or isinstance(size, int)


def test_interp_slc():
    locs = torch.rand(8, 6, 5, 3)
    grid_size = (5, 5, 5)
    locs_batch_shape = ("A", "B", "C")
    linop = Interpolate(locs, grid_size, locs_batch_shape=locs_batch_shape)
    tile = {"B": slice(0, 4), "C": slice(2, 3)}
    obatch = [tile.get(dim, slice(None)) for dim in locs_batch_shape]
    linop_split = linop.split(linop, tile)
    assert linop_split.locs.shape == (8, 4, 1, 3)
    assert (linop_split.locs == linop.locs[tuple(obatch)]).all()


def test_interpolate_requires_grad_only_for_vals():
    """locs gradients are not implemented, so only vals should propagate grad."""
    locs = get_valid_locs((8,), (16, 16), 2, 4.0, "cpu", centered=True)
    locs.requires_grad_(True)

    vals = torch.randn(16, 16, dtype=torch.complex64)
    assert not interpolate(vals, locs, 4.0).requires_grad

    vals.requires_grad_(True)
    assert interpolate(vals, locs, 4.0).requires_grad

    grad_output = torch.randn(8, dtype=torch.complex64)
    assert not interpolate_adjoint(grad_output, locs, (16, 16), 4.0).requires_grad

    grad_output.requires_grad_(True)
    assert interpolate_adjoint(grad_output, locs, (16, 16), 4.0).requires_grad
