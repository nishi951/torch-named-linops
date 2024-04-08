import torch

from torchlinops import NamedLinop

__all__ = [
    'inner', 'is_adjoint'
]


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
