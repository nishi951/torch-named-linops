"""Tests for the Scalar linop."""

import pytest
import torch

from torchlinops.linops.scalar import Scalar
from torchlinops.tests.test_base import BaseNamedLinopTests


class TestScalar(BaseNamedLinopTests):
    """Scalar satisfies the full named-linop contract."""

    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        A = Scalar(weight=2.0 + 0j, ioshape=("N",))
        x = torch.randn(8, dtype=torch.complex64)
        y = torch.randn(8, dtype=torch.complex64)
        return A, x, y

    def test_split(self, linop_input_output):
        """Scalar.split is the identity (weight is global)."""
        A, x, y = linop_input_output
        A_split = A.split(A, {"N": slice(0, 4)})
        assert isinstance(A_split, Scalar)
        assert torch.allclose(A_split.weight, A.weight)


# --- Standalone scalar tests ---


def test_scalar_float_weight_auto_converts():
    """Scalar with a plain float should auto-convert to a tensor."""
    A = Scalar(3.0)
    assert isinstance(A.weight, torch.Tensor)
    x = torch.ones(5)
    assert torch.allclose(A(x), torch.full((5,), 3.0))


def test_scalar_complex_weight():
    A = Scalar(1j)
    x = torch.ones(4, dtype=torch.complex64)
    result = A(x)
    assert torch.allclose(result, torch.full((4,), 1j, dtype=torch.complex64))


def test_scalar_size_always_none():
    """Scalar.size() always returns None — it is trivially splittable."""
    A = Scalar(2.0, ioshape=("N", "M"))
    assert A.size("N") is None
    assert A.size("M") is None
    assert A.size("X") is None


def test_scalar_split_weight_mismatch_raises():
    """Scalar.split_weight should raise AssertionError when ibatch != obatch."""
    A = Scalar(2.0, ioshape=("N",))
    with pytest.raises(AssertionError, match="identically"):
        A.split_weight([slice(0, 2)], [slice(0, 3)], A.weight)


def test_scalar_adjoint_is_conjugate():
    """Adjoint of a complex Scalar multiplies by the conjugate weight."""
    A = Scalar(2.0 + 3.0j, ioshape=("N",))
    x = torch.randn(5, dtype=torch.complex64)
    result = A.H(x)
    expected = x * (2.0 - 3.0j)
    assert torch.allclose(result, expected, rtol=1e-5)
