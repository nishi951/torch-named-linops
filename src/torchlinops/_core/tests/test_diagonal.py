import torch

from torchlinops import Diagonal

from torchlinops.utils import inner, is_adjoint


def test_diagonal():
    M = 10
    N, P = 5, 7
    weight = torch.randn(M, 1, 1, dtype=torch.complex64)
    # weightshape = ("M",)
    x = torch.randn(M, N, P, dtype=torch.complex64)
    y = torch.randn(M, N, P, dtype=torch.complex64)
    ioshape = ("M", "N", "P")
    A = Diagonal(weight, ioshape)
    assert is_adjoint(A, x, y)
