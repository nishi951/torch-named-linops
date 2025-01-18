import torch

from torchlinops.utils import inner, is_adjoint

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


def test_dense_shapes():
    ishape = ("B", "N")
    oshape = ("B", "M")
    B = 10
    M, N = (3, 7)
    weight = torch.randn(B, M, N)
    weightshape = ("B", "M", "N")
    device = "cpu"
    A = Dense(weight, weightshape, ishape, oshape)

    x = torch.randn(B, N)
    y = torch.randn(B, M)
    AN = A.N
    ANx = AN(x)  # Make sure it runs
    assert AN.ishape == ("B", "N")
    assert AN.oshape == ("B", "N1")
    assert AN.weightshape == ("B", "N1", "N")

    AH = A.H
    AHy = AH(y)  # Make sure it runs
    assert AH.ishape == ("B", "M")
    assert AH.oshape == ("B", "N")
    assert AH.weightshape == ("B", "M", "N")
