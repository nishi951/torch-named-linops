from jaxtyping import Shaped, Bool, Integer
from torch import Tensor

import torch

__all__ = ["index", "index_adjoint", "mask2idx", "canonicalize_idx"]

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
    idx = ensure_tensor_indexing(idx, vals.shape)
    return vals[idx]


def index_adjoint(
    vals: Shaped[Tensor, "..."],
    idx: tuple[IndexOrSlice, ...],
    grid_size: tuple[int, ...],
) -> Tensor:
    """
    Parameters
    ----------
    grid_size : tuple of ints
        The shape of the output tensor, excluding batch dimensions
    """
    batch_ndims = len(vals.shape) - len(idx[0].shape)
    if batch_ndims < 0:
        raise ValueError(
            f"Negative number of batch dimensions from input with shape {vals.shape} and sampling index with shape {len(idx[0].shape)}"
        )
    output_size = (*vals.shape[:batch_ndims], *grid_size)
    idx = ensure_tensor_indexing(idx, output_size)
    # Check for broadcastability:
    torch.broadcast_tensors(*idx, vals)

    out = torch.zeros(output_size, dtype=vals.dtype, device=vals.device)
    out.index_put_(idx, vals, accumulate=True)
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


def canonicalize_idx(idx: Integer[Tensor, "..."]):
    """
    Parameters
    ----------
    idx : [B..., D]

    Returns
    -------
    D-tuple of [B...] tensors

    """
    return tuple(idx[..., i] for i in range(idx.shape[-1]))


### Helper functions
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
