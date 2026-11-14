import pytest
import torch

from torchlinops import Dense, Dim, Stack
from torchlinops.testing import BaseNamedLinopTests
from torchlinops.utils import inner


class TestStackVertical(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    @classmethod
    def linop_input_output(cls):
        P, Q = 3, 4
        A = Dense(
            torch.randn(P, Q, dtype=torch.complex64), ("P", "Q"), Dim("Q"), Dim("P")
        )
        B = Dense(
            torch.randn(P, Q, dtype=torch.complex64), ("P", "Q"), Dim("Q"), Dim("P")
        )
        C = Stack(A, B, odim_and_idx=("M", 1))
        x = torch.randn(Q, dtype=torch.complex64)
        y = torch.randn(P, 2, dtype=torch.complex64)
        return C, x, y


class TestStackHorizontal(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    @classmethod
    def linop_input_output(cls):
        P, Q = 3, 4
        A = Dense(
            torch.randn(P, Q, dtype=torch.complex64), ("P", "Q"), Dim("Q"), Dim("P")
        )
        B = Dense(
            torch.randn(P, Q, dtype=torch.complex64), ("P", "Q"), Dim("Q"), Dim("P")
        )
        C = Stack(A, B, idim_and_idx=("N", 0))
        x = torch.randn(2, Q, dtype=torch.complex64)
        y = torch.randn(P, dtype=torch.complex64)
        return C, x, y

    def test_normal_level2(self, linop_input_output):
        pass


class TestStackDiagonal(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    @classmethod
    def linop_input_output(cls):
        P, Q = 3, 4
        A = Dense(
            torch.randn(P, Q, dtype=torch.complex64), ("P", "Q"), Dim("Q"), Dim("P")
        )
        B = Dense(
            torch.randn(P, Q, dtype=torch.complex64), ("P", "Q"), Dim("Q"), Dim("P")
        )
        C = Stack(A, B, idim_and_idx=("N", 0), odim_and_idx=("M", 1))
        x = torch.randn(2, Q, dtype=torch.complex64)
        y = torch.randn(P, 2, dtype=torch.complex64)
        return C, x, y

    def test_normal(self, linop_input_output):
        pass

    def test_adjoint_level2(self, linop_input_output):
        pass

    def test_normal_level2(self, linop_input_output):
        pass

    def test_normal_fn(self, linop_input_output):
        pass


# --- Standalone tests for split/normal/repr ---


@pytest.fixture
def denselinops():
    P, Q = 3, 4
    A = Dense(torch.randn(P, Q), ("P", "Q"), Dim("Q"), Dim("P"))
    B = Dense(torch.randn(P, Q), ("P", "Q"), Dim("Q"), Dim("P"))
    return A, B


def test_stack_split(denselinops):
    A, B = denselinops
    C = Stack(A, B, odim_and_idx=("M", 1))
    C1 = C.split(C, {"P": slice(0, 1)})
    assert (C1[0].weight == A.weight[:1]).all()

    C2 = C.split(C, {"M": slice(0, 2)})
    assert isinstance(C2, Stack)


def test_stack_normal(denselinops):
    A, B = denselinops
    C = Stack(A, B, odim_and_idx=("M", 1))
    x = torch.randn(*(C.size(d) for d in C.ishape))
    assert torch.allclose(C.N(x), A.N(x) + B.N(x))


def test_stack_repr(denselinops):
    A, B = denselinops
    C = Stack(A, B, odim_and_idx=("M", 1))
    assert repr(C)
    assert len(C) == 2


def test_stack_get_dim_and_idx_dim_already_in_shape_raises():
    """Stack._get_dim_and_idx should raise ValueError if the dim is already in the shape."""
    P, Q = 3, 4
    A = Dense(torch.randn(P, Q), ("P", "Q"), Dim("Q"), Dim("P"))
    # "Q" is already in ishape — trying to stack on it should raise
    with pytest.raises(ValueError, match="already contains"):
        Stack(A, idim_and_idx=("Q", 0))


def test_stack_fn_size_mismatch_raises():
    """Stack._fn should raise ValueError when input size doesn't match number of sub-linops."""
    P, Q = 3, 4
    A = Dense(torch.randn(P, Q), ("P", "Q"), Dim("Q"), Dim("P"))
    B = Dense(torch.randn(P, Q), ("P", "Q"), Dim("Q"), Dim("P"))
    # Horizontal stack: expects input of size 2 along dim N
    C = Stack(A, B, idim_and_idx=("N", 0))
    # Pass input with 3 along N instead of 2
    bad_x = torch.randn(3, Q)
    with pytest.raises(ValueError, match="Stack Linop expecting"):
        C(bad_x)


def test_stack_getitem_slice():
    """Stack[0:1] should return a new Stack with the sliced sub-linops."""
    P, Q = 3, 4
    A = Dense(torch.randn(P, Q), ("P", "Q"), Dim("Q"), Dim("P"))
    B = Dense(torch.randn(P, Q), ("P", "Q"), Dim("Q"), Dim("P"))
    C = Dense(torch.randn(P, Q), ("P", "Q"), Dim("Q"), Dim("P"))
    S = Stack(A, B, C, odim_and_idx=("M", 1))
    sub = S[0:2]
    assert isinstance(sub, Stack)
    assert len(sub) == 2
