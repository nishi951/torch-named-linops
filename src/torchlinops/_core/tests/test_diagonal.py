import pytest

import torch

from torchlinops import Diagonal

from torchlinops.utils import inner, is_adjoint


@pytest.fixture
def A(self):
    M = 10
    N, P = 5, 7
    weight = torch.randn(M, 1, 1, dtype=torch.complex64)
    # weightshape = ("M",)
    x = torch.randn(M, N, P, dtype=torch.complex64)
    y = torch.randn(M, N, P, dtype=torch.complex64)
    ioshape = ("M", "N", "P")
    A = Diagonal(weight, ioshape)
    return A


def test_diagonal(A):
    assert is_adjoint(A, x, y)
