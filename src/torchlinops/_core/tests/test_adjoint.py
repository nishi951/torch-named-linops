import torch

from torchlinops import NS

from torchlinops import (
    NamedLinop,
    Chain,
    Dense,
    Diagonal,
    FFT,
    SumReduce,
    Repeat,
    Scalar,
    Identity,
    Add,
    Truncate,
    PadDim,
)


def inner(x, y):
    """Complex inner product"""
    return torch.sum(x.conj() * y)


def is_adjoint(
    A: NamedLinop,
    x: torch.Tensor,
    y: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-8,
):
    """
    The adjoint test states that if A and AH are adjoints, then
    inner(y, Ax) = inner(AHy, x)
    """
    return torch.isclose(inner(y, A(x)), inner(A.H(y), x), atol=atol, rtol=rtol).all()


def test_dense():
    M, N = 9, 3
    weight = torch.randn(M, N, dtype=torch.complex64)
    weightshape = ("M", "N")
    x = torch.randn(N, dtype=torch.complex64)
    ishape = ("N",)
    y = torch.randn(M, dtype=torch.complex64)
    oshape = ("M",)
    A = Dense(weight, weightshape, NS(ishape, oshape))
    assert is_adjoint(A, x, y)


def test_diagonal():
    M = 10
    N, P = 5, 7
    weight = torch.randn(M, 1, 1, dtype=torch.complex64)
    # weightshape = ("M",)
    x = torch.randn(M, N, P, dtype=torch.complex64)
    y = torch.randn(M, N, P, dtype=torch.complex64)
    ioshape = ("M", "N", "P")
    A = Diagonal(weight, NS(ioshape))
    assert is_adjoint(A, x, y)


def test_reduce_repeat():
    M, N = 5, 7
    x = torch.randn(M, N)
    y = torch.randn(N)

    A = SumReduce(NS(("M", "N"), ("N",)))
    B = Repeat({"M": M}, NS(("N",), ("M", "N")))

    assert is_adjoint(A, x, y)
    assert is_adjoint(B, y, x)
