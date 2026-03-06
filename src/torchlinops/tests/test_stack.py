import pytest
import torch

from torchlinops import Dense, Dim, Stack
from torchlinops.tests.test_base import BaseNamedLinopTests
from torchlinops.utils import inner


class TestStackVertical(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
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
    def linop_input_output(self):
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
    def linop_input_output(self):
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
