import pytest
import torch

from torchlinops.linops.einops import Rearrange, Repeat, SumReduce
from torchlinops.tests.test_base import BaseNamedLinopTests


class TestSumReduce(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = {"rtol": 1e-4}

    @pytest.fixture(scope="class", params=["fullshape", "ellipses"])
    def linop_input_output(self, request):
        if request.param == "fullshape":
            A = SumReduce(("A", "B", "C"), ("A", "B"))
        else:
            A = SumReduce(("...", "C"), ("...",))
        x = torch.randn(5, 2, 3, dtype=torch.complex64)
        y = torch.randn(5, 2, dtype=torch.complex64)
        return A, x, y


class TestRearrange(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        # ipattern and opattern are positional args (no ->)
        A = Rearrange(
            "(A B) C",
            "A B C",
            ishape=("Ab", "C"),
            oshape=("A", "B", "C"),
            axes_lengths={"A": 2, "B": 3},
        )
        x = torch.randn(6, 4, dtype=torch.complex64)
        y = torch.randn(2, 3, 4, dtype=torch.complex64)
        return A, x, y


class TestRepeat(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        A = Repeat(
            {"C": 3},
            ishape=("A", "B"),
            oshape=("A", "B", "C"),
            broadcast_dims=[],
        )
        x = torch.randn(4, 5, dtype=torch.complex64)
        y = torch.randn(4, 5, 3, dtype=torch.complex64)
        return A, x, y

    def test_adjoint_level2(self, linop_input_output):
        pass

    def test_normal_level2(self, linop_input_output):
        """Skip: Repeat.N has special composition."""
        pass

    def test_split(self, linop_input_output):
        """Repeat.split requires broadcast_dims attribute."""
        pass

    def test_size(self, linop_input_output):
        """Repeat.size requires broadcast_dims attribute."""
        A, x, y = linop_input_output
        s = A.size("C")
        assert s == 3
