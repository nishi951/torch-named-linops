import pytest

import torch

from torchlinops import SumReduce, Repeat

from torchlinops.utils import is_adjoint
from torchlinops.tests.test_base import BaseNamedLinopTests


class TestSumReduce(BaseNamedLinopTests):
    equality_check = "approx"

    isclose_kwargs = {"rtol": 1e-4}

    @pytest.fixture(scope="class", params=["fullshape", "ellipses"])
    def linop_input_output(self, request):
        x = torch.randn(5, 2, 3)
        y = torch.randn(5, 2)
        if request.param == "fullshape":
            linop = SumReduce(("A", "B", "C"), ("A", "B"))
        elif request.param == "ellipses":
            linop = SumReduce(("...", "C"), ("...",))
        return linop, x, y


def test_reduce_repeat():
    M, N = 5, 7
    x = torch.randn(M, N)
    y = torch.randn(N)

    A = SumReduce(("M", "N"), ("N",))
    B = Repeat({"M": M}, ("N",), ("M", "N"))

    assert is_adjoint(A, x, y)
    assert is_adjoint(B, y, x)
