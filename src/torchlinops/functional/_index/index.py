from jaxtyping import Shaped, Bool, Integer
from torch import Tensor

import torch

__all__ = ["index", "index_adjoint", "mask2idx", "canonicalize_idx", "slice2range"]

IndexOrSlice = Integer[Tensor, "..."] | slice


def index(
    vals: Shaped[Tensor, "..."],
    idx: tuple[IndexOrSlice, ...],
) -> Tensor:
    """
    Parameters
    ----------
    idx : tuple of Tensor or Slice objects
        Index
    """
    if len(vals.shape) < len(idx):
        raise ValueError(
            f"Input value with shape {vals.shape} cannot be indexed with index tensors of length {len(idx)}"
        )
    batch_slc = [slice(None)] * (len(vals.shape) - len(idx))
    idx_batched = (*batch_slc, *idx)
    return vals[idx_batched]


def index_adjoint(
    vals: Shaped[Tensor, "..."],
    idx: tuple[Integer[Tensor, "..."], ...],
    grid_size: tuple[int, ...],
) -> Tensor:
    """
    Parameters
    ----------
    vals :
        Batch size of vals is used to determine batch size of output
    idx : tuple of integer-valued tensors
        Use ensure_tensor_indexing to guarantee this
    grid_size : tuple of ints
        The shape of the output tensor, excluding batch dimensions
    """
    idx_stacked = torch.stack(idx, dim=0)
    out = multi_grid(vals, idx_stacked, grid_size)
    return out


def mask2idx(mask: Bool[Tensor, "..."]) -> tuple[Integer[Tensor, "..."], ...]:
    """Converts an n-dimensional boolean tensor into an n-tuple of integer tensors
    indexing the True elements of the tensor.

    Parameters
    ----------
    mask : torch.Tensor
        A boolean tensor.

    Returns
    -------
    tuple[torch.Tensor]:
        A tuple of integer tensors indexing the True elements.
    """
    if not mask.dtype == torch.bool:
        raise ValueError(f"Input tensor must be of boolean dtype, but got {mask.dtype}")
    return torch.nonzero(mask, as_tuple=True)


def canonicalize_idx(idx: Integer[Tensor, "..."], dim: int = -1):
    """
    Parameters
    ----------
    idx : [B1... D B2...]
    dim : int
        The dimension of idx to tuple-ify

    Returns
    -------
    D-tuple of [B1... B2...] tensors

    Note: dim is usually 0 or -1

    """
    return tuple(torch.select(idx, dim, i) for i in range(idx.shape[dim]))


### Helper functions
def multi_grid(
    x: torch.Tensor,
    idx: torch.Tensor,
    final_size: tuple,
    raveled: bool = False,
    ravel_dim: int = 0,
):
    """Grid values in x to im_size with indices given in idx
    x: [N... I...]
    idx: [I... ndims] or [I...] if raveled=True
    raveled: Whether the idx still needs to be raveled or not
    ravel_dim: Dimension to ravel over

    Returns:
    Tensor with shape [N... final_size]

    Notes:
    Adjoint of multi_index
    Might need nonnegative indices
    """
    if not raveled:
        if len(final_size) != idx.shape[ravel_dim]:
            raise ValueError(
                f"final_size should be of dimension {idx.shape[-1]} but got {final_size}"
            )
        idx = ravel(idx, final_size, dim=ravel_dim)
    ndim = len(idx.shape)
    if x.shape[-ndim:] != idx.shape:
        raise ValueError(
            f"x and idx should correspond in last {ndim} dimensions but got x: {x.shape} and idx: {idx.shape}"
        )
    x_flat = torch.flatten(x, start_dim=-ndim, end_dim=-1)  # [N... (I...)]
    idx_flat = torch.flatten(idx)

    batch_dims = x_flat.shape[:-1]
    y = torch.zeros(
        (*batch_dims, *final_size), dtype=x_flat.dtype, device=x_flat.device
    )
    y = y.reshape((*batch_dims, -1))
    y = y.index_add_(-1, idx_flat, x_flat)
    y = y.reshape(*batch_dims, *final_size)
    return y


def ravel(x: torch.Tensor, shape: tuple, dim: int):
    """
    x: torch.LongTensor, arbitrary shape,
    shape: Shape of the array that x indexes into
    dim: dimension of x that is the "indexing" dimension

    Returns:
    torch.LongTensor of same shape as x but with indexing dimension removed
    """
    out = 0
    shape_shifted = tuple(shape[1:]) + (1,)
    for s, s_next, i in zip(shape, shape_shifted, range(x.shape[dim])):
        out += torch.select(x, dim, i) % s
        out *= s_next
    return out


def slice2range(slice_obj: slice, n: int):
    """Convert a slice object to a range object given the array size
    Examples
    --------
    >>> tuple(slice2range(slice(None, None, None), 4))
    (0, 1, 2, 3)
    >>> tuple(slice2range(slice(None, None, -1), 3))
    (2, 1, 0)

    """
    start = (
        slice_obj.start
        if slice_obj.start is not None
        else (0 if slice_obj.step is None or slice_obj.step > 0 else n - 1)
    )
    stop = (
        slice_obj.stop
        if slice_obj.stop is not None
        else (n if slice_obj.step is None or slice_obj.step > 0 else -1)
    )
    step = slice_obj.step if slice_obj.step is not None else 1
    return range(start, stop, step)


def _unsqueeze_last(t: Tensor, n: int):
    """Unsqueeze multiple dimensions at the end of a tensor

    Examples
    --------
    >>> t = torch.arange(3)
    >>> _unsqueeze_last(t, 2).shape
    torch.Size([3, 1, 1])
    """
    return t.view(-1, *((1,) * n))


def ensure_tensor_indexing(
    idx: tuple[IndexOrSlice, ...], tshape: tuple | torch.Size
) -> tuple[Tensor, ...]:
    """Convert any slice()-type indexes to tensor indexes.

    Also broadcasts by appending slice(None) to the front of idx

    Parameters
    ----------
    idx : tuple
        Tuple of torch.Tensor (integer-valued) index tensors or slice() objects
    tshape : torch.Size or tuple
        Target size, should have length greater than or equal to that of idx

    """
    # Prepare idx
    idx = list(idx)
    if len(tshape) < len(idx):
        raise ValueError(f"Cannot broadcast idx {idx} to tshape {tshape}")
    while len(tshape) > len(idx):
        # Insert empty slices until index length matches length of target shape
        idx.insert(0, slice(None))

    # Prepare out
    out = []
    for d, (size, i) in enumerate(zip(tshape, idx)):
        if isinstance(i, Tensor):
            out.append(i)
        elif isinstance(i, slice):
            range_tensor = torch.tensor(slice2range(i, size))
            # Unsqueeze last dimensions
            range_tensor = _unsqueeze_last(range_tensor, len(tshape) - d - 1)
            out.append(range_tensor)
        else:
            raise ValueError(
                f"idx must contain only tensors or slice() objects but got {i}"
            )
    return tuple(out)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
