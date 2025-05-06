import pytest

from math import prod, ceil, floor

import torch
import sigpy as sp

from torchlinops.functional import ungrid
from torchlinops.utils import to_pytorch, from_pytorch
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs

PYTEST_GPU_MARKS = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU is required but not available"
    ),
]


# TODO add kernel type
@pytest.mark.parametrize("kernel_type", ["kaiser_bessel", "spline"])
@pytest.mark.parametrize("padding_mode", ["zero", "circular"])
@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=PYTEST_GPU_MARKS)])
@pytest.mark.parametrize("dtype", ["real", "complex"])
@pytest.mark.parametrize(
    "spec",
    [
        "small1d",
        "medium1d",
        pytest.param("large1d", marks=PYTEST_GPU_MARKS),
        "tiny2d",
        "small2d",
        pytest.param("medium2d", marks=PYTEST_GPU_MARKS),
        pytest.param("large2d", marks=PYTEST_GPU_MARKS),
        "small3d",
        pytest.param("medium3d", marks=PYTEST_GPU_MARKS),
        pytest.param("large3d", marks=PYTEST_GPU_MARKS + [pytest.mark.slow]),
    ],
)
def test_ungrid(kernel_type, padding_mode, dev, dtype, spec, request):
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32
    spec_name = spec
    spec = request.getfixturevalue(spec)

    width = spec["width"]
    grid_size = spec["grid_size"]
    locs_batch_size = spec["locs_batch_size"]
    ndim = len(grid_size)
    npts = prod(locs_batch_size)
    ishape = (*spec["N"], *grid_size)
    if spec_name == "large3d":
        vals = torch.randn(ishape, dtype=dtype, device=device) + 4.0
    else:
        vals = torch.arange(prod(ishape)).reshape(ishape).to(dtype).to(device)
    # locs = tuple(
    #     torch.rand(spec["npts"], device=device) + (w / 2 - 1) for d in range(ndim)
    # )
    # locs = torch.stack(locs, dim=-1).contiguous()
    if padding_mode == "zero":
        locs = get_valid_locs(locs_batch_size, grid_size, ndim, width, device)
    elif padding_mode == "circular":
        locs = get_valid_locs(
            locs_batch_size, grid_size, ndim, width, device, valid=False
        )

    interp = ungrid(
        vals,
        locs,
        width=width,
        kernel=kernel_type,
        pad_mode=padding_mode,
    )

    # Test against sigpy
    vals = from_pytorch(vals)
    # locs = from_pytorch(locs.flip(dims=(-1,)).contiguous())
    locs = from_pytorch(locs)

    # Sigpy quirks:
    # - Circular padding
    # - weight is product of 1D kernel evals along each coordinate axis
    interp_sp = sp.interpolate(vals, locs, kernel=kernel_type, width=width)
    interp_sp = to_pytorch(interp_sp)

    assert torch.allclose(interp, interp_sp)
