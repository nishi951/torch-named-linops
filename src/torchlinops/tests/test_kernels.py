from functools import partial

import pytest
import torch

from torchlinops.functional._interp.kernels import (
    _apply_default_kernel_params,
    get_kernel_fn,
    kaiser_bessel_torch,
    spline_torch,
    weights_torch,
)


def test_kaiser_bessel_torch():
    x = torch.linspace(-1, 1, 50)
    result = kaiser_bessel_torch(x, beta=2.0)
    assert result.shape == x.shape


def test_spline_torch():
    x = torch.linspace(-2, 2, 50)
    result = spline_torch(x)
    assert result.shape == x.shape


def test_get_kernel_fn_invalid():
    with pytest.raises(ValueError):
        get_kernel_fn("invalid_kernel", {})


def test_weights_torch_norm2():
    locs = torch.randn(10, 2)
    grid_locs = locs.clone()
    grid_size = torch.tensor([8, 8])
    radius = torch.tensor(1.5)

    kb = partial(kaiser_bessel_torch, beta=2.0)
    result, _, _ = weights_torch(locs, grid_locs, radius, 2, kb, grid_size, "zero")
    assert result is not None


def test_weights_torch_invalid_norm():
    locs = torch.randn(10, 2)
    grid_locs = locs.clone()
    grid_size = torch.tensor([8, 8])
    radius = torch.tensor(1.5)

    kb = partial(kaiser_bessel_torch, beta=2.0)
    with pytest.raises(ValueError):
        weights_torch(locs, grid_locs, radius, 3, kb, grid_size, "zero")


def test_apply_default_kernel_params_invalid():
    with pytest.raises(ValueError):
        _apply_default_kernel_params("unknown_kernel", {})
