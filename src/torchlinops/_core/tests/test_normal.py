import torch

from torchlinops._core._linops import (
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
    AHA = A.N
    assert torch.isclose(A.N(x), A.H(A(x))).all()
    # Make sure dense's normal doesn't create a chain (unnecessary)
    # If desired, just make the linop explicitly
    assert not isinstance(A.N, Chain)
