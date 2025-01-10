import pytest
from math import prod

import torch
import torchlinops.functional as F
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs

PYTEST_GPU_MARKS = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU is required but not available"
    ),
]


@pytest.mark.parametrize("kernel_type", ["kaiser_bessel", "spline"])
@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=PYTEST_GPU_MARKS)])
@pytest.mark.parametrize("dtype", ["real", "complex"])
@pytest.mark.parametrize(
    "spec",
    [
        "small1d",
        "medium1d",
        "small2d",
        pytest.param("small3d", marks=PYTEST_GPU_MARKS),
    ],
)
def test_interp_forward(kernel_type, dev, dtype, spec, request):
    spec = request.getfixturevalue(spec)
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32

    width = spec["width"]
    grid_size = spec["grid_size"]
    locs_batch_size = spec["locs_batch_size"]
    ndim = len(grid_size)
    npts = prod(locs_batch_size)
    ishape = (*spec["N"], *grid_size)

    vals = torch.arange(prod(ishape)).reshape(*ishape).to(dtype).to(device)
    vals = vals + 1.0
    vals.requires_grad_(True)
    locs = get_valid_locs(locs_batch_size, grid_size, ndim, width, device)
    y = F.interpolate(vals, locs, width=width, kernel=kernel_type)
    y.abs().sum().backward()

    gradvals = F.interpolate_adjoint(
        torch.ones_like(y), locs, grid_size, width, kernel_type
    )
    assert torch.allclose(vals.grad, gradvals)


@pytest.mark.parametrize("kernel_type", ["kaiser_bessel", "spline"])
@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=PYTEST_GPU_MARKS)])
@pytest.mark.parametrize("dtype", ["real", "complex"])
@pytest.mark.parametrize(
    "spec",
    [
        "small1d",
        "medium1d",
        "small2d",
        pytest.param("small3d", marks=PYTEST_GPU_MARKS),
    ],
)
def test_interp_adjoint(kernel_type, dev, dtype, spec, request):
    spec = request.getfixturevalue(spec)
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32

    grid_size = spec["grid_size"]
    locs_batch_size = spec["locs_batch_size"]
    width = spec["width"]
    ndim = len(grid_size)
    npts = prod(locs_batch_size)
    ishape = (*spec["N"], *locs_batch_size)
    vals = torch.arange(prod(ishape)).reshape(*ishape).to(dtype).to(device)
    vals = vals + 1.0
    vals.requires_grad_(True)
    locs = get_valid_locs(locs_batch_size, grid_size, ndim, width, device)
    y = F.interpolate_adjoint(vals, locs, grid_size, width, kernel_type)
    y.abs().sum().backward()

    gradvals = F.interpolate(torch.ones_like(y), locs, width, kernel_type)
    assert torch.allclose(vals.grad, gradvals)
