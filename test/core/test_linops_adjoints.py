import torch

from torchlinops.core.linops import (
    NamedLinop,
    Chain,
    Dense,
    Diagonal,
    FFT,
    Scalar,
    Identity,
    Add,
)

def inner(x, y):
    """Complex inner product"""
    return torch.sum(x.conj() * y)

def is_adjoint(
        A: NamedLinop,
        AH: NamedLinop,
        x: torch.Tensor,
        y: torch.Tensor,
        atol: float = 1e-5,
        rtol: float = 1e-8,
):
    """
    The adjoint test states that if A and AH are adjoints, then
    inner(y, Ax) = inner(AHy, x)
    """
    return torch.isclose(inner(y, A(x)), inner(AH(y), x),
                         atol=atol, rtol=rtol)

def test_dense():
    M, N = 3, 5
    weight = torch.randn(M, N)
    weightshape = ('M', 'N')
    x = torch.randn(N)
    ishape = ('N',)
    y = torch.randn(M)
    oshape = ('M')
    A = Dense(weight, weightshape, ishape, oshape)
    assert is_adjoint(A, A.H, x, y)
