import pytest
import torch

from torchlinops import Dense, Diagonal
from torchlinops.tests.test_base import BaseNamedLinopTests


class TestDense(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        M, N = 9, 3
        weight = torch.randn(M, N, dtype=torch.complex64)
        A = Dense(weight, ("M", "N"), ("N",), ("M",))
        x = torch.randn(N, dtype=torch.complex64)
        y = torch.randn(M, dtype=torch.complex64)
        return A, x, y


class TestDenseBatched(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        B, M, N = 10, 3, 7
        weight = torch.randn(B, M, N, dtype=torch.complex64)
        A = Dense(weight, ("B", "M", "N"), ("B", "N"), ("B", "M"))
        x = torch.randn(B, N, dtype=torch.complex64)
        y = torch.randn(B, M, dtype=torch.complex64)
        return A, x, y

    def test_shapes(self, linop_input_output):
        A, x, y = linop_input_output
        AN = A.N
        ANx = AN(x)
        assert AN.ishape == ("B", "N")
        assert AN.oshape == ("B", "N1")
        assert AN.weightshape == ("B", "N1", "N")

        AH = A.H
        AHy = AH(y)
        assert AH.ishape == ("B", "M")
        assert AH.oshape == ("B", "N")
        assert AH.weightshape == ("B", "M", "N")


def test_dense_normal_with_inner():
    """Dense.normal(inner) should equal A.H(inner(A(x))) numerically."""
    M, N = 5, 3
    weight = torch.randn(M, N, dtype=torch.complex64)
    A = Dense(weight, ("M", "N"), ("N",), ("M",))
    inner_weight = torch.randn(M, dtype=torch.complex64)
    inner_linop = Diagonal(inner_weight, ("M",))
    normal = A.normal(inner_linop)
    x = torch.randn(N, dtype=torch.complex64)
    expected = A.H(inner_linop(A(x)))
    assert torch.allclose(normal(x), expected, rtol=1e-4, atol=1e-4)
