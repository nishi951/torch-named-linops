from typing import Callable, Tuple, Optional
from torch import Tensor

import torch
from tqdm import tqdm

from torchlinops.utils import default_to_dict


__all__ = ["power_method"]


def power_method(
    A: Callable[[Tensor], Tensor],
    ishape: Tuple,
    v_init: Optional[Tensor] = None,
    max_iters: int = 50,
    device: torch.device = "cpu",
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
    if v_init is None:
        v = torch.randn(ishape, dtype=torch.complex64, device=device)
    else:
        v = v_init.clone()

    # Initialize
    A.to(device)
    vnorm = torch.linalg.vector_norm(v, dim=dim, keepdim=True)
    v = v / (vnorm + eps)
    pbar = tqdm(range(max_iters), total=max_iters, **tqdm_kwargs)
    for _ in pbar:
        vnorm_old = vnorm.clone()
        v = A(v)
        vnorm = torch.linalg.vector_norm(v, dim=dim, keepdim=True)
        v = v / (vnorm + eps)
        # pbar.set_postfix({"eigenval": vnorm.item()})
        if (torch.abs(vnorm_old - vnorm) / torch.abs(vnorm_old)).max() < tol:
            break
    return v, vnorm.squeeze()
