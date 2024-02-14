from typing import Tuple
from math import prod

import torch

__all__ = [
    'ravel', 'multi_grid',
]

def ravel(x: torch.Tensor, shape: Tuple, dim: int):
    """
    x: torch.LongTensor, arbitrary shape,
    shape: Shape of the array that x indexes into
    dim: dimension of x that is the "indexing" dimension

    Returns:
    torch.LongTensor of same shape as x but with indexing dimension removed
    """
    out = 0
    for s, i in zip(shape, range(x.shape[dim]-1)):
        out = s * (out + torch.select(x, dim, i))
    out += torch.select(x, dim, -1)
    return torch.remainder(out, prod(shape))

def multi_grid(x: torch.Tensor, idx: torch.Tensor, final_size: Tuple, raveled: bool = False):
    """Grid values in x to im_size with indices given in idx
    x: [N... I...]
    idx: [I... ndims] or [I...] if raveled=True
    raveled: Whether the idx still needs to be raveled or not

    Returns:
    Tensor with shape [N... final_size]

    Notes:
    Adjoint of multi_index
    Might need nonnegative indices
    """
    if not raveled:
        assert len(final_size) == idx.shape[-1], f'final_size should be of dimension {idx.shape[-1]}'
        idx = ravel(idx, final_size, dim=-1)
    ndims = len(idx.shape)
    assert x.shape[-ndims:] == idx.shape, f'x and idx should correspond in last {ndims} dimensions'
    x_flat = torch.flatten(x, start_dim=-ndims, end_dim=-1) # [N... (I...)]
    idx_flat = torch.flatten(idx)

    batch_dims = x_flat.shape[:-1]
    y = torch.zeros((*batch_dims, *final_size), dtype=x_flat.dtype, device=x_flat.device)
    y = y.reshape((*batch_dims, -1))
    y = y.index_add_(-1, idx_flat, x_flat)
    y = y.reshape(*batch_dims, *final_size)
    return y
