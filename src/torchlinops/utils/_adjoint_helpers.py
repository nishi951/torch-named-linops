from typing import Callable

import torch

__all__ = ["inner", "is_adjoint"]


def inner(x, y):
    """Complex inner product"""
    return torch.sum(x.conj() * y)


def is_adjoint(
    A: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-8,
):
    """
    The adjoint test states that if A and AH are adjoints, then
    inner(y, Ax) = inner(AHy, x)
    """
    yAx = inner(y, A(x))
    xAHy = inner(A.H(y), x)
    return torch.isclose(yAx, xAHy, atol=atol, rtol=rtol).all()
