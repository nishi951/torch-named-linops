from typing import Union, Tuple

import torch

__all__ = ["multi_flatten"]


def multi_flatten(x: torch.Tensor, partitions: Union[int, Tuple]):
    """Flattens a tensor by combining some of its dimensions together

    Parameters
    ----------
    x : torch.Tensor
        Tensor with shape [A... B... C... ...]

    partitions : int or tuple of ints
        The sizes of the dimension groups A..., B...
        i.e. the number of dimensions in each of them

    Returns
    -------
    torch.Tensor with shape [(A...)(B...)(C...)...]
    torch.Size - original size of x


    Note: Undo the effect of multi_flatten with `.reshape(orig_shape)`

    """
    x_shape = x.shape
    if len(x_shape) == 0:
        return x, x_shape
    if isinstance(partitions, int):
        if partitions == 0:
            return x, x_shape
        return torch.flatten(x, start_dim=0, end_dim=partitions - 1), x_shape

    if sum(partitions) > len(x_shape):
        raise ValueError(
            f"x (shape {x_shape}) cannot be partitioned with dim sizes {partitions}"
        )
    start_dim = 0
    for p in partitions:
        if p > 0:
            x = torch.flatten(x, start_dim=start_dim, end_dim=start_dim + p - 1)
            start_dim += 1
    return x, x_shape
