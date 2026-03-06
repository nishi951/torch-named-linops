import pytest
import torch

from torchlinops import Identity, Zero, ShapeSpec
from torchlinops.tests.test_base import BaseNamedLinopTests


class TestIdentity(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        return (
            Identity(ishape=("N", "M"), oshape=("N", "M")),
            torch.randn(4, 5),
            torch.randn(4, 5),
        )

    def test_forward_shape(self, linop_input_output):
        A, x, y = linop_input_output
        result = A(x)
        assert result.shape == x.shape
        assert torch.equal(result, x)

    def test_adjoint_is_self(self, linop_input_output):
        A, _, _ = linop_input_output
        assert A.H is A

    def test_normal_is_self(self, linop_input_output):
        A, _, _ = linop_input_output
        assert A.N is A

    def test_split(self, linop_input_output):
        A, _, _ = linop_input_output
        result = A.split(A, {"N": slice(0, 2)})
        assert result is A

    def test_pow(self, linop_input_output):
        A, _, _ = linop_input_output
        powered = A.__pow__(2.0)
        assert isinstance(powered, Identity)


class TestZero(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        return (
            Zero(ishape=("N", "M"), oshape=("N", "M")),
            torch.randn(4, 5, dtype=torch.complex64),
            torch.randn(4, 5, dtype=torch.complex64),
        )

    def test_forward_shape(self, linop_input_output):
        A, x, y = linop_input_output
        result = A(x)
        assert result.shape == x.shape
        assert torch.all(result == 0)

    def test_split(self, linop_input_output):
        A, _, _ = linop_input_output
        result = A.split(A, {"N": slice(0, 2)})
        assert result is A


class TestShapeSpec(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        return (
            ShapeSpec(ishape=("N", "M"), oshape=("K", "L")),
            torch.randn(4, 5, dtype=torch.complex64),
            torch.randn(4, 5, dtype=torch.complex64),
        )

    def test_forward_shape(self, linop_input_output):
        A, x, y = linop_input_output
        result = A(x)
        assert result.shape == x.shape
        assert torch.equal(result, x)

    def test_adjoint_swaps_shapes(self, linop_input_output):
        A, _, _ = linop_input_output
        adj = A.H
        assert adj.ishape == A.oshape
        assert adj.oshape == A.ishape

    def test_split(self, linop_input_output):
        pytest.skip("ShapeSpec does not support split")


class TestIdentityDefaults:
    def test_identity_default_shapes(self):
        identity = Identity()
        x = torch.randn(3, 4, 5)
        y = identity(x)
        assert torch.equal(y, x)

    def test_zero_default_shapes(self):
        zero = Zero()
        x = torch.randn(3, 4, 5)
        y = zero(x)
        assert torch.all(y == 0)

    def test_normal_with_inner(self):
        inner = Identity(ishape=("N", "M"), oshape=("N", "M"))
        identity = Identity(ishape=("N", "M"), oshape=("N", "M"))
        x = torch.randn(4, 5)
        result = identity.normal(inner)(x)
        assert torch.equal(result, x)
