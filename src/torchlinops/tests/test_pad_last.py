import pytest
import torch
from torchlinops import Dim, PadLast
from torchlinops.utils import is_adjoint


@pytest.fixture
def P():
    P = PadLast((20, 20), (10, 10), Dim("AXY"))
    return P


@pytest.fixture
def Psplit(P):
    Psplit = P.split(P, {"A": slice(0, 1)})
    return Psplit


@pytest.fixture
def PHsplit(P):
    PH = P.H
    PHsplit = PH.split(PH, {"A": slice(0, 1)})
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
