import pytest

import torch

from torchlinops.mri._linops.nufft.backends.fi._flatten import multi_flatten


@pytest.fixture
def x():
    x = torch.randn(1, 2, 3, 4, 5, 6)
    return x


def test_flatten_int(x):
    y, xshape = multi_flatten(x, 2)
    assert y.shape == (2, 3, 4, 5, 6)
    assert (y.reshape(xshape) == x).all()


def test_no_flattens(x):
    y, xshape = multi_flatten(x, 0)
    assert y.shape == x.shape


def test_full_flatten(x):
    y, xshape = multi_flatten(x, (2, 2, 2))
    assert y.shape == (2, 12, 30)
    assert (y.reshape(xshape) == x).all()


def test_partial_flatten(x):
    y, xshape = multi_flatten(x, (2, 3))
    assert y.shape == (2, 60, 6)
    assert (y.reshape(xshape) == x).all()


@pytest.mark.xfail
def test_too_many_flatten(x):
    y, xshape = multi_flatten(x, (3, 4, 5))
