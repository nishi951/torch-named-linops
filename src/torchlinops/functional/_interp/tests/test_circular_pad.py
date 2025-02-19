import pytest

import torch

from torchlinops.functional._interp._circ_pad import circular_pad, circular_pad_adjoint
from torchlinops.utils import inner


def test_circular_pad_2d():
    ishape = (15, 20)
    x = torch.randn(*ishape)
    padding = [4, 4, 2, 2]

    y = circular_pad(x, padding)
    assert y.shape == (19, 28)


def test_circular_pad_nd():
    ishape = (5, 15, 20, 19, 18)
    x = torch.randn(*ishape)
    padding = [1, 1, 2, 2, 3, 3, 5, 5]
    y = circular_pad(x, padding)
    assert y.shape == (5, 25, 26, 23, 20)


def test_circular_pad_adjoint_1d():
    ishape = (3,)
    padding = [1, 1]
    oshape = list(ishape)
    for i, (padleft, padright) in enumerate(zip(padding[::2], padding[1::2])):
        oshape[-(i + 1)] += padleft + padright
    x = torch.randn(*ishape, dtype=torch.complex64)
    y = torch.randn(*oshape, dtype=torch.complex64)

    Ax = circular_pad(x, padding)
    AHy = circular_pad_adjoint(y, padding)

    assert torch.allclose(inner(x, AHy), inner(y, Ax).conj())


def test_circular_pad_adjoint_2d():
    ishape = (3, 3)
    padding = [2, 2, 1, 1]
    oshape = list(ishape)
    for i, (padleft, padright) in enumerate(zip(padding[::2], padding[1::2])):
        oshape[-(i + 1)] += padleft + padright
    x = torch.randn(*ishape, dtype=torch.complex64)
    y = torch.randn(*oshape, dtype=torch.complex64)

    Ax = circular_pad(x, padding)
    AHy = circular_pad_adjoint(y, padding)

    assert torch.allclose(inner(x, AHy), inner(y, Ax).conj())
