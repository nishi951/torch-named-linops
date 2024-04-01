from typing import Tuple

import torch

__all__ = ["trj_mask"]


def trj_mask(
    trj: torch.Tensor, max_k: float, ord=float("inf")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Truncate a kspace trajectory to a specific radius

    Parameters
    ----------
    trj : torch.Tensor
        The ksp trajectory to truncate
        where N is the input matrix size in that dimension
    max_k : float
        The radius of the kspace region to truncate to.

    Returns
    -------
    torch.Tensor, size [K', D] | float
        The truncated kspace trajectory. Note the readout has collapsed.
    torch.Tensor, size [K...] | bool
        A boolean tensor mask

    Note: May need to pre-scale trj in each dimension to have desired behavior
    """
    return torch.linalg.norm(trj, ord=ord, dim=-1, keepdim=False) <= max_k
