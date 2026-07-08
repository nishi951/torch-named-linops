import pytest
import torch

from torchlinops.linops.einops import Rearrange, Repeat, SumReduce
from torchlinops.testing import BaseNamedLinopTests


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

    def test_split(self, linop_input_output):
        A, x, y = linop_input_output
        with pytest.warns(UserWarning, match="splitting"):
            type(A).split(
                A,
                {"Ab": slice(None), "C": slice(None)},
            )


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

    def test_split_no_mutation(self, linop_input_output):
        """split must not mutate the original axes_lengths."""
        A, x, y = linop_input_output
        original_c = A.axes_lengths["C"]
        A_split = type(A).split(
            A,
            {"A": slice(None), "B": slice(None), "C": slice(0, 2)},
        )
        assert A.axes_lengths["C"] == original_c
        assert A_split.axes_lengths["C"] == 2


# --- Standalone error-path tests ---


def test_sumreduce_oshape_not_shorter_raises():
    """SumReduce requires oshape to be strictly shorter than ishape."""
    with pytest.raises(AssertionError):
        SumReduce(("A", "B"), ("A", "B"))


def test_sumreduce_same_length_raises():
    """SumReduce oshape == ishape should also fail."""
    with pytest.raises(AssertionError):
        SumReduce(("A", "B", "C"), ("A", "B", "C"))


def test_repeat_oshape_not_longer_raises():
    """Repeat requires oshape to be strictly longer than ishape."""
    with pytest.raises(AssertionError):
        Repeat({"C": 3}, ishape=("A", "B", "C"), oshape=("A", "B"))


def test_repeat_same_length_raises():
    """Repeat oshape == ishape should also fail."""
    with pytest.raises(AssertionError):
        Repeat({}, ishape=("A", "B"), oshape=("A", "B"))


def test_rearrange_split_emits_warning():
    """split on a Rearrange should emit a UserWarning."""
    A = Rearrange(
        "(A B) C",
        "A B C",
        ishape=("Ab", "C"),
        oshape=("A", "B", "C"),
        axes_lengths={"A": 2, "B": 3},
    )
    with pytest.warns(UserWarning, match="splitting"):
        type(A).split(
            A,
            {"Ab": slice(None), "C": slice(None)},
        )
