import pytest

import torch
import sigpy as sp

from torchlinops.functional import ungrid, grid
from torchlinops.utils import to_pytorch, from_pytorch

PYTEST_GPU_MARKS = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU is required but not available"
    ),
]


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=PYTEST_GPU_MARKS)])
@pytest.mark.parametrize("dtype", ["real", "complex"])
def test_exact_int_loc_ungrid(dev, dtype):
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32

    vals = torch.arange(1, 9).to(device).to(dtype)
    width = 4.0
    grid_size = (len(vals),)
    locs = torch.tensor([[4.0]], device=device)
    interp = ungrid(vals, locs, width=width, kernel="kaiser_bessel")

    # Test against sigpy
    vals = from_pytorch(vals)
    # locs = from_pytorch(locs.flip(dims=(-1,)).contiguous())
    locs = from_pytorch(locs)

    # Sigpy quirks:
    # - Circular padding
    # - weight is product of 1D kernel evals along each coordinate axis
    interp_sp = sp.interpolate(vals, locs, kernel="kaiser_bessel", width=width)
    interp_sp = to_pytorch(interp_sp)

    assert interp.allclose(interp_sp)


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=PYTEST_GPU_MARKS)])
@pytest.mark.parametrize("dtype", ["real", "complex"])
def test_exact_int_loc_grid(dev, dtype):
    device = torch.device(dev)
    dtype = torch.complex64 if dtype == "complex" else torch.float32

    vals = torch.tensor([[1.0]], device=device)
    width = 4.0
    grid_size = (8,)
    locs = torch.tensor([[4.0]], device=device)
    interp = grid(vals, locs, grid_size, width=width, kernel="kaiser_bessel")

    # Test against sigpy
    vals = from_pytorch(vals)
    # locs = from_pytorch(locs.flip(dims=(-1,)).contiguous())
    locs = from_pytorch(locs)

    # Sigpy quirks:
    # - Circular padding
    # - weight is product of 1D kernel evals along each coordinate axis
    interp_sp = sp.gridding(vals, locs, grid_size, kernel="kaiser_bessel", width=width)
    interp_sp = to_pytorch(interp_sp)

    assert interp.allclose(interp_sp)
