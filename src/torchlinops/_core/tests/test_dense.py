import torch

from adjoint_helpers import inner, is_adjoint

from torchlinops import Dense


def test_dense():
    M, N = 9, 3
    weight = torch.randn(M, N, dtype=torch.complex64)
    weightshape = ("M", "N")
    x = torch.randn(N, dtype=torch.complex64)
    ishape = ("N",)
    y = torch.randn(M, dtype=torch.complex64)
    oshape = ("M",)
    A = Dense(weight, weightshape, ishape, oshape)
    assert is_adjoint(A, x, y)
