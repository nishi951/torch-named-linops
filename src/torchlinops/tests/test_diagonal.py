from copy import copy

import pytest
import torch

from torchlinops import Diagonal
from torchlinops.tests.test_base import BaseNamedLinopTests


class TestDiagonal(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        M = 10
        weight = torch.randn(M, 1, 1, dtype=torch.complex64)
        ioshape = ("M", "N", "P")
        A = Diagonal(weight, ioshape)
        N, P = 5, 7
        x = torch.randn(M, N, P, dtype=torch.complex64)
        y = torch.randn(M, N, P, dtype=torch.complex64)
        return A, x, y


@pytest.mark.xfail  # Deprecated behavior: changing a non-() dim to ()
def test_diagonal_shape_renaming():
    M = 10
    weight = torch.randn(M, 1, 1, dtype=torch.complex64)
    A = Diagonal(weight, ("M", "N", "P"))
    B = copy(A)
    new_ioshape = ("()", "N1", "P1")
    B.oshape = new_ioshape
    assert B.ishape == new_ioshape

    new_ioshape = ("M1", "N1", "P1")
    B.oshape = new_ioshape
    assert B.ishape == new_ioshape


def test_diagonal_pow():
    weight = torch.randn(5, dtype=torch.complex64)
    A = Diagonal(weight, ("N",))
    A2 = A**2
    x = torch.randn(5, dtype=torch.complex64)
    assert torch.allclose(A2(x), x * weight**2)
