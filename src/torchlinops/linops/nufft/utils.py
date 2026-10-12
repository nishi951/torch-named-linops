"""Helper functions"""

import torch


def scale_int(t: tuple[int, ...], scale_factor: float):
    return tuple(int(scale_factor * s) for s in t)


def fftshift_locs(locs, size):
    """fftshift the locs. Assume last dimension is spatial dimension"""
    if len(size) != locs.shape[-1]:
        raise ValueError(
            f"len(size) must match locs.shape[-1] but got "
            f"len({size}) != {locs.shape}[-1]"
        )
    size = torch.as_tensor(size)
    return torch.remainder(locs + size // 2, size)


def ifftshift_locs(locs, size):
    if len(size) != locs.shape[-1]:
        raise ValueError(
            f"len(size) must match locs.shape[-1] but got "
            f"len({size}) != {locs.shape}[-1]"
        )
    size = torch.as_tensor(size)
    return torch.remainder(locs - size // 2, size)


def clamp(x, min=None, max=None):
    """Version of torch.clamp that supports broadcasting.

    Examples
    --------
    >>> clamp(torch.tensor([0, 2, 4]), torch.tensor([1]))
    tensor([1, 2, 4])

    >>> clamp(torch.tensor([0, 2, 4]), None, torch.tensor([3]))
    tensor([0, 2, 3])

    >>> clamp(torch.tensor([[0, 2, 4], [1, 3, 5]]), torch.tensor([1, 2])[:, None])
    tensor([[1, 2, 4],
            [2, 3, 5]])

    >>> clamp(torch.tensor([[0, 2, 4], [1, 3, 5]]), 1, 4)
    tensor([[1, 2, 4],
            [1, 3, 4]])


    """
    if min is not None:
        min = torch.as_tensor(min)
        x = torch.maximum(x, min)
    if max is not None:
        max = torch.as_tensor(max)
        x = torch.minimum(x, max)
    return x
