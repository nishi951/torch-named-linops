from math import ceil
from typing import Optional

import numpy as np
from torchlinops.utils import NDList, batch_iterator, dict_product

from .nameddim import ND
from .namedlinop import NamedLinop

__all__ = ["split"]

Batch = tuple[int, slice]
# Represents a single batch at index 0 over the full extent
# Could convert to a full class
DEFAULT_BATCH = (0, slice(None))
Tile = dict[ND | str, Batch]


def split(linop: NamedLinop, batch_sizes: dict[ND | str, int]):
    """Split a linop into smaller linops according to some batch sizes

    Parameters
    ----------
    linop : NamedLinop
        The NamedLinop to be split
    batch_sizes : dict[ND | str -> int]
        Dictionary mapping dims to batch sizes for those dims

    Returns
    -------

    """
    # Precompute sizes and shapes
    batch_sizes = {ND.infer(k): v for k, v in batch_sizes.items()}
    sizes = {dim: linop.size(dim) for dim in linop.dims}

    # Make list of tiles
    # Each tile is a dict mapping the dimension to an (int, slice) pair
    batch_iterators = make_batch_iterators(sizes, batch_sizes)
    tiles = list(dict_product(batch_iterators))

    # Allocate outputs
    batch_dims = list(batch_sizes.keys())
    tiled_shape = tuple(ceil(sizes[dim] / batch_sizes[dim]) for dim in batch_dims)
    linops = np.ndarray(tiled_shape, dtype=object)
    input_batches = np.ndarray(tiled_shape, dtype=object)
    output_batches = np.ndarray(tiled_shape, dtype=object)
    # linops = NDList(tiled_shape, labels=batch_dims)
    # input_batches = NDList(tiled_shape, labels=batch_dims)
    # output_batches = NDList(tiled_shape, labels=batch_dims)
    for tile in tiles:
        idx = _tile_get_idx(tile, batch_dims)
        ibatches, obatches, linop_tile = split_linop_with_tile(linop, tile)
        linops[idx] = linop_tile
        input_batches[idx] = ibatches[0]  # input batch of first linop
        output_batches[idx] = obatches[-1]  # output batch of last linop
    # if flatten:
    #     # Set max depth to avoid flattening the batches themselves (which are lists of slices)
    #     return (
    #         flatten_recursive(linops.data),
    #         flatten_recursive(input_batches.data, max_depth=len(tiled_shape) - 1),
    #         flatten_recursive(output_batches.data, max_depth=len(tiled_shape) - 1),
    #     )
    return linops, input_batches, output_batches


def split_linop_with_tile(linop: NamedLinop, tile: Tile):
    """Split a linop according to batch specified in tile"""
    ibatches = [_tile_shape2batch(tile, op.ishape) for op in linop.flatten()]
    obatches = [_tile_shape2batch(tile, op.oshape) for op in linop.flatten()]
    linop_tile = linop.split(linop, *ibatches, *obatches)
    return ibatches, obatches, linop_tile


def _tile_get_idx(tile: Tile, batch_dims) -> tuple[int]:
    """Get all indices from the tile"""
    return tuple(tile.get(dim, DEFAULT_BATCH)[0] for dim in batch_dims)


def _tile_shape2batch(tile: Tile, shape: tuple[ND | str, ...]) -> list[slice]:
    """Get all batches from the tile in a specific order according to the provided shapes"""
    return [tile.get(dim, DEFAULT_BATCH)[1] for dim in shape]


def make_batch_iterators(
    total_sizes, batch_sizes
) -> dict[str, list[tuple[int, slice]]]:
    """Construct dictionaries mapping batchable dims to lists of slices
    corresponding to the actual batches

    Also includes an int index at dim 0

    Explanation
    -----------
    If we have batch size 3 for dim D (i.e. batch_sizes = {"D": 3})
    and the total size for dim D is 7, then

    batch_iterators["D"] = [(0, slice(0, 3)), (1, slice(3, 6)), (2, slice(6, 7))]

    If "E" is some other dimension not batched, then

    batch_iterators["E"] = [(0, slice(None))]



    """
    batch_iterators = {}
    for dim, total in total_sizes.items():
        batch_iterators[dim] = (
            [
                (i, slice(a, b))
                for i, (a, b) in enumerate(batch_iterator(total, batch_sizes[dim]))
            ]
            if dim in batch_sizes
            else [(0, slice(None))]
        )
    return batch_iterators


def flatten_recursive(nested_list, max_depth: Optional[int] = None):
    """Flatten a nested list, optionally to a maximum depth

    Examples
    -------
    >>> flatten_recursive([[1, 2], [3, 4], [[5, 6]]])
    [1, 2, 3, 4, 5, 6]

    # Setting max_depth = 1 will avoid flattening the last list completely
    >>> flatten_recursive([[1, 2], [3, 4], [[5, 6]]], max_depth=1)
    [1, 2, 3, 4, [5, 6]]

    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            if max_depth is None:
                flat_list.extend(flatten_recursive(item))
            elif max_depth > 0:
                flat_list.extend(flatten_recursive(item, max_depth - 1))
            else:
                flat_list.append(item)
        else:
            flat_list.append(item)
    return flat_list


if __name__ == "__main__":
    import doctest

    doctest.testmod()
