import pytest
import torch

from torchlinops import Dim, PadLast
from torchlinops.linops.pad_last import crop_slice_from_pad, pad_to_size
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


def test_pad_to_size_even_even():
    pad = pad_to_size((10, 10), (20, 20))
    assert pad == [5, 5, 5, 5]


def test_pad_to_size_odd_odd():
    pad = pad_to_size((10, 10), (21, 21))
    assert pad == [5, 6, 5, 6]


def test_pad_to_size_even_odd():
    pad = pad_to_size((10, 10), (21, 21))
    assert pad == [5, 6, 5, 6]


def test_pad_to_size_1d():
    pad = pad_to_size((10,), (20,))
    assert pad == [5, 5]


def test_pad_to_size_dimension_mismatch():
    with pytest.raises(ValueError):
        pad_to_size((10, 10), (20, 20, 20))


def test_crop_slice_from_pad():
    pad = [0, 0, 5, 5]
    crop = crop_slice_from_pad(pad)
    assert crop == [slice(5, -5, None), slice(0, None, None)]


def test_crop_slice_from_pad_no_crop():
    pad = [0, 0, 0, 0]
    crop = crop_slice_from_pad(pad)
    assert crop == [slice(0, None, None), slice(0, None, None)]


def test_crop_slice_from_pad_3d():
    pad = [1, 2, 3, 4, 5, 6]
    crop = crop_slice_from_pad(pad)
    assert crop == [slice(5, -6, None), slice(3, -4, None), slice(1, -2, None)]
