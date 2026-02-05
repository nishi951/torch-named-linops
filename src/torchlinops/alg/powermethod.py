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
    """Finds the maximum eigenvalue (in absolute value) of square matrix A

    Parameters
    ----------
    dim : Optional[int | Tuple]
        If not None, compute eigenvalues along only that dimension
        Enables batched power method over several stacked matrices

    Returns
    -------
    Tensor : The eigenvector
    Tensor : its associated eigenvalue

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
