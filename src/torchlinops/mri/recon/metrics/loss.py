from typing import Optional, Callable
import logging

import torch

logger = logging.getLogger(__name__)

__all__ = [
    'inner', 'l2_loss', 'l2_grad_norm', 'ptol',
]

def inner(x: torch.Tensor, y: torch.Tensor):
    assert x.shape == y.shape
    return torch.sum(torch.conj(x) * y)

def l2_loss(
        x: torch.Tensor,
        b: Optional[torch.Tensor] = None,
        A: Optional[Callable] = None,
        AH: Optional[Callable] = None,
        AHA: Optional[Callable] = None,
        AHb: Optional[torch.Tensor] = None,
):
    """Computes ||Ax - b||^2
    Equivalently, computes x^T A^T Ax - x^T A^T b - b^T A x + b^T b
    depending on which arguments are supplied.
    Returns:
        torch.Tensor
    """
    if AHb is None:
        assert b is not None and AH is not None
        AHb = AH(b)
    if AHA is None:
        if AH is None:
            return torch.linalg.norm(A(x) - b) ** 2
        assert A is not None
        AHA = lambda x: AH(A(x))
    l2_loss = inner(x, AHA(x)) - inner(x, AHb) - inner(AHb, x)
    if b is None:
        logger.warn('Computing l2 loss without b^T b term')
    else:
        l2_loss += inner(b, b)
    if l2_loss.real <= 0:
        breakpoint()
    return l2_loss.real

def l2_grad_norm(
        gr: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        A: Optional[Callable] = None,
        AH: Optional[Callable] = None,
        AHA: Optional[Callable] = None,
        AHb: Optional[torch.Tensor] = None,
):
    """Return the norm of the gradient of the l2 loss
    gr = \nabla ||Ax - b||^2
    = 2 * (A^T Ax - real(A^T b))
    """
    if gr is not None:
        return 2 * torch.linalg.norm(gr)
    assert x is not None
    if AHb is None:
        assert b is not None and AH is not None
        AHb = AH(b)
    if AHA is None:
        AHA = lambda x: AH(A(x))
    return 2 * torch.linalg.norm(AHA(x) - AHb)

def ptol(x_old, x):
    """Computes percentage change in x compared to x from
    a previous iteration.
    """
    return 100 * torch.linalg.norm(x_old - x)/torch.linalg.norm(x)
