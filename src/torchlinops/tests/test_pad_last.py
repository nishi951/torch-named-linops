import pytest
import torch

from torchlinops import Dim, Pad
from torchlinops.linops.pad_last import (
    Crop,
    crop_slice_from_pad,
    pad_to_scale,
    pad_to_size,
)
from torchlinops.tests.test_base import BaseNamedLinopTests
from torchlinops.utils import is_adjoint


class TestPad(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        P = Pad((20, 20), (10, 10), Dim("XY"))
        x = torch.randn(10, 10, dtype=torch.complex64)
        y = torch.randn(20, 20, dtype=torch.complex64)
        return P, x, y


# --- Split tests ---


def test_split():
    P = Pad((20, 20), (10, 10), Dim("AXY"))
    Psplit = P.split(P, {"A": slice(0, 1)})
    x = torch.randn(1, 10, 10)
    y = Psplit(x)
    assert tuple(y.shape) == (1, 20, 20)


def test_split_adjoint():
    P = Pad((20, 20), (10, 10), Dim("AXY"))
    Psplit = P.split(P, {"A": slice(0, 1)})
    x = torch.randn(1, 10, 10)
    y = torch.randn(1, 20, 20)
    assert is_adjoint(Psplit, x, y)


def test_adjoint_split():
    P = Pad((20, 20), (10, 10), Dim("AXY"))
    PH = P.H
    PHsplit = PH.split(PH, {"A": slice(0, 1)})
    x = torch.randn(1, 10, 10)
    y = torch.randn(1, 20, 20)
    assert is_adjoint(PHsplit, y, x)


# --- Validation error tests ---


def test_pad_dimension_mismatch():
    with pytest.raises(ValueError):
        Pad((20, 20), (10,))


def test_pad_fn_shape_mismatch():
    P = Pad((20, 20), (10, 10))
    x = torch.randn(5, 5)
    with pytest.raises(ValueError):
        P(x)


def test_pad_adj_fn_shape_mismatch():
    P = Pad((20, 20), (10, 10))
    y = torch.randn(10, 10)
    with pytest.raises(ValueError):
        P.H(y)


def test_pad_split_spatial_error():
    P = Pad((20, 20), (10, 10), Dim("XY"))
    with pytest.raises(ValueError):
        P.split(P, {"X": slice(0, 5)})


def test_pad_size():
    P = Pad((20, 20), (10, 10), Dim("XY"))
    assert P.size("X") == 10
    assert P.size("X1") == 20
    assert P.size("Z") is None


def test_pad_explicit_out_shape():
    P = Pad((20, 20), (10, 10), in_shape=Dim("XY"), out_shape=("A", "B"))
    assert P.out_im_shape == ("A", "B")


def test_crop_constructor():
    C = Crop((10, 10), (20, 20), ("X", "Y"), ("A", "B"), ("...",))
    x = torch.randn(20, 20)
    y = C(x)
    assert y.shape == (10, 10)


def test_pad_to_scale():
    pad = pad_to_scale((10, 10), 2.0)
    assert pad == [5, 5, 5, 5]


# --- pad_to_size / crop_slice_from_pad utility tests ---


def test_pad_to_size_even_even():
    pad = pad_to_size((10, 10), (20, 20))
    assert pad == [5, 5, 5, 5]


def test_pad_to_size_odd_odd():
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
