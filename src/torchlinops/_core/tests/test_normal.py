import torch

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
    M, N = 3, 5
    weight = torch.randn(M, N)
    weightshape = ("M", "N")
    x = torch.randn(N)
    ishape = ("N",)
    # y = torch.randn(M)
    oshape = ("M",)
    A = Dense(weight, weightshape, ishape, oshape)
    breakpoint()
    assert torch.isclose(A.N(x), A.H(A(x)))
    # Make sure dense's normal doesn't create a chain (unnecessary)
    # If desired, just make the linop explicitly
    assert not isinstance(A.N, Chain)
