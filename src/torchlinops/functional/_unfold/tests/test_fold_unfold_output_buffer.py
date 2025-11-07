from math import prod

import pytest
import torch

from torchlinops.functional import fold, unfold
from torchlinops.functional._unfold.nblocks import get_nblocks
from torchlinops.utils import cdata

PYTEST_GPU_MARKS = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU is required but not available"
    ),
]


@pytest.fixture
def medium2d():
    N = (2, 2)
    shape = (33, 43)
    block_size = (7, 7)
    stride = (2, 2)
    return N, shape, block_size, stride


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=PYTEST_GPU_MARKS)])
def test_fold_allocated_buffer(dev, medium2d):
    device = torch.device(dev)
    dtype = torch.float32
    N, shape, block_size, stride = medium2d

    nblocks = get_nblocks(shape, block_size, stride)
    ishape = (*N, *nblocks, *block_size)
    oshape = (*N, *shape)

    x = torch.arange(prod(ishape)).reshape(ishape)
    x = x.to(device).to(dtype)
    # TODO: why does this not fail??
    y = fold(x, shape, block_size, stride)

    # Do it again, but with output allocated
    x2 = torch.randn_like(x)
    y2 = fold(x2, shape, block_size, stride, output=y)
    y3 = fold(x2, shape, block_size, stride)  # buffer not reused

    assert cdata(y) == cdata(y2)

    assert cdata(y) == cdata(y2)
    assert cdata(y3) != cdata(y2)  # Buffer should not be shared
    assert (y3 == y2).all()  # However, results should match


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=PYTEST_GPU_MARKS)])
def test_unfold_allocated_buffer(dev, medium2d):
    device = torch.device(dev)
    dtype = torch.float32
    N, shape, block_size, stride = medium2d

    nblocks = get_nblocks(shape, block_size, stride)
    ishape = (*N, *shape)

    x = torch.arange(prod(ishape)).reshape(ishape)
    x = x.to(device).to(dtype)
    y = unfold(x, block_size, stride)

    # Do it again, but with output allocated
    x2 = torch.randn_like(x)
    y2 = unfold(x2, block_size, stride, output=y)
    y3 = unfold(x2, block_size, stride)  # Buffer not reused

    assert cdata(y) == cdata(y2)
    assert cdata(y3) != cdata(y2)  # Buffer should not be shared
    assert (y3 == y2).all()  # But results should match
