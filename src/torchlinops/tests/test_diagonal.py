import pytest

from copy import copy

import torch

from torchlinops import Diagonal

from torchlinops.utils import inner, is_adjoint


@pytest.fixture
def A():
    M = 10
    weight = torch.randn(M, 1, 1, dtype=torch.complex64)
    # weightshape = ("M",)
    ioshape = ("M", "N", "P")
    A = Diagonal(weight, ioshape)
    return A


def test_diagonal(A):
    M = A.weight.shape[0]
    N, P = 5, 7
    x = torch.randn(M, N, P, dtype=torch.complex64)
    y = torch.randn(M, N, P, dtype=torch.complex64)
    assert is_adjoint(A, x, y)


# Specific tests
@pytest.mark.xfail  # Deprecated behavior: changing a non-() dim to ()
def test_diagonal_shape_renaming(A):
    B = copy(A)

    # Change to () first
    new_ioshape = ("()", "N1", "P1")
    B.oshape = new_ioshape
    assert B.ishape == new_ioshape

    # Change to something else after
    new_ioshape = ("M1", "N1", "P1")
    B.oshape = new_ioshape
    assert B.ishape == new_ioshape
