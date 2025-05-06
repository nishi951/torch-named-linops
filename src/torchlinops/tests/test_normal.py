import torch

from torchlinops import NS
from torchlinops import (
    NamedLinop,
    Chain,
    Dense,
    Diagonal,
    FFT,
    Scalar,
    Identity,
    Add,
)


def test_dense():
    M, N = 9, 3
    weight = torch.randn(M, N, dtype=torch.complex64)
    weightshape = ("M", "N")
    x = torch.randn(N, dtype=torch.complex64)
    ishape = ("N",)
    # y = torch.randn(M)
    oshape = ("M",)
    A = Dense(weight, weightshape, ishape, oshape)
    assert torch.isclose(A.N(x), A.H(A(x))).all()
    # Make sure dense's normal doesn't create a chain (unnecessary)
    # If desired, just make the linop explicitly
    assert not isinstance(A.N, Chain)
    assert A.N.ishape == ("N",)
    assert A.N.oshape == ("N1",)


def test_diagonal_normal():
    M = 10
    N, P = 5, 7
    weight = torch.randn(M, 1, 1, dtype=torch.complex64)
    # weightshape = ("M",)
    x = torch.randn(M, N, P, dtype=torch.complex64)
    ioshape = ("M", "N", "P")
    A = Diagonal(weight, ioshape)
    assert torch.isclose(A.N(x), A.H(A(x))).all()
    assert not isinstance(A.N, Chain)
