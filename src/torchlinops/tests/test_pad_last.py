import pytest

import torch

from torchlinops import PadLast

from torchlinops.utils import is_adjoint


@pytest.fixture
def P():
    P = PadLast((20, 20), (10, 10), ("A",))
    return P


@pytest.fixture
def Psplit(P):
    ibatch = [slice(0, 1), slice(None), slice(None)]
    obatch = [slice(0, 1), slice(None), slice(None)]
    Psplit = P.split(P, ibatch, obatch)
    return Psplit


@pytest.fixture
def PHsplit(P):
    ibatch = [slice(0, 1), slice(None), slice(None)]
    obatch = [slice(0, 1), slice(None), slice(None)]
    # breakpoint()
    # PHsplit = P.H.split(ibatch, obatch)
    PH = P.H
    PHsplit = PH.split(PH, ibatch, obatch)
    return PHsplit


def test_split(Psplit):
    x = torch.randn(1, 10, 10)
    y = Psplit(x)
    assert tuple(y.shape) == (1, 20, 20)


def test_split_adjoint(Psplit):
    x = torch.randn(1, 10, 10)
    y = torch.randn(1, 20, 20)
    assert is_adjoint(Psplit, x, y)


def test_adjoint_split(PHsplit):
    x = torch.randn(1, 10, 10)
    y = torch.randn(1, 20, 20)
    assert is_adjoint(PHsplit, y, x)
