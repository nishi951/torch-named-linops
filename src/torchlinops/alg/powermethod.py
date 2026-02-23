from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from torchlinops.utils import default_to_dict

__all__ = ["power_method"]


def power_method(
    A: Callable[[Tensor], Tensor],
    v_init: Tensor,
    max_iters: int = 50,
    eps: float = 0.0,
    tol: float = 1e-5,
    dim: Optional[int | Tuple] = None,
    tqdm_kwargs: Optional[dict] = None,
) -> tuple[Tensor, Tensor]:
    """Estimate the largest eigenvalue (in magnitude) of $A$ via the power method.

    Repeatedly applies $v \\leftarrow A(v) / \\|A(v)\\|$ until the eigenvalue
    estimate converges or *max_iters* is reached.

    Parameters
    ----------
    A : Callable[[Tensor], Tensor]
        Function implementing the matrix-vector product $A(v)$.
        $A$ should be a square (normal) operator.
    v_init : Tensor
        Initial vector. Should be nonzero.
    max_iters : int, default 50
        Maximum number of power iterations.
    eps : float, default 0.0
        Small constant added to norms to avoid division by zero.
    tol : float, default 1e-5
        Relative convergence tolerance on the eigenvalue estimate.
    dim : int or tuple, optional
        If not ``None``, compute eigenvalues along the specified dimension(s),
        enabling a batched power method over several stacked matrices.
    tqdm_kwargs : dict, optional
        Extra keyword arguments forwarded to ``tqdm``.

    Returns
    -------
    v : Tensor
        The estimated eigenvector (unit norm).
    eigenvalue : Tensor
        The estimated eigenvalue $\\|A(v)\\|$.
    """
    # Default values
    tqdm_kwargs = default_to_dict(dict(desc="Power Method"), tqdm_kwargs)
    v = v_init.clone()

    # Initialize
    vnorm = torch.linalg.vector_norm(v, dim=dim, keepdim=True)
    v = v / (vnorm + eps)
    pbar = tqdm(range(max_iters), total=max_iters, **tqdm_kwargs)
    for _ in pbar:
        vnorm_old = vnorm.clone()
        v = A(v)
        vnorm = torch.linalg.vector_norm(v, dim=dim, keepdim=True)
        v = v / (vnorm + eps)
        rdiff = (torch.abs(vnorm_old - vnorm) / torch.abs(vnorm_old)).max()

        # Display progress
        postfix = {"rdiff": rdiff.item()}
        if dim is None:
            postfix["e_val"] = vnorm.item()
        pbar.set_postfix(postfix)
        if rdiff < tol:
            break
    return v, vnorm.squeeze()
