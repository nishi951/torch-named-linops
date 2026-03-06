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


def test_split_data_via_nested_stack():
    """split_data on a nested Stack should recurse correctly."""
    P, Q = 3, 4
    # Use Stack sub-linops that themselves have split_data (nested Stacks)
    A = Dense(torch.randn(P, Q), ("P", "Q"), Dim("Q"), Dim("P"))
    B = Dense(torch.randn(P, Q), ("P", "Q"), Dim("Q"), Dim("P"))
    # Inner stacks (vertical) — these have split_data
    inner_A = Stack(A, odim_and_idx=("M", 1))
    inner_B = Stack(B, odim_and_idx=("M", 1))
    # Outer diagonal stack
    outer = Stack(inner_A, inner_B, idim_and_idx=("N", 0), odim_and_idx=("L", 2))

    # data_list contains one item per outer linop
    data_list = [torch.randn(1), torch.randn(1)]

    # Request a slice that gives empty linop_idxs (diagonal stack, non-overlapping slices)
    ibatch = [slice(1, 2), slice(None)]  # N=1: only inner_B
    obatch = [slice(None), slice(0, 1), slice(0, 1)]  # L=0: only inner_A
    result = outer.split_data(ibatch, obatch, data_list)
    # Intersection is empty → should return 0.0
    assert result == 0.0


def test_split_data_empty_indices(denselinops):
    """When no linops satisfy the slice, split_data should return 0.0."""
    P, Q = 3, 4
    A = Dense(torch.randn(P, Q), ("P", "Q"), Dim("Q"), Dim("P"))
    B = Dense(torch.randn(P, Q), ("P", "Q"), Dim("Q"), Dim("P"))
    # Diagonal stack: different input AND output dims
    C = Stack(A, B, idim_and_idx=("N", 0), odim_and_idx=("M", 1))

    data_list = [torch.randn(Q), torch.randn(Q)]
    # Request only index 0 on output (M dim=1) and index 1 on input (N dim=0)
    # These cannot match the same sub-linop so intersection is empty
    ibatch = [slice(1, 2), slice(None)]  # N=1, Q=all
    obatch = [slice(None), slice(0, 1)]  # P=all, M=0

    result = C.split_data(ibatch, obatch, data_list)
    assert result == 0.0
