import pytest

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
