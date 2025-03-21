from typing import Optional
from math import ceil
from .namedlinop import NamedLinop
from .nameddim import NDorStr, ND
from torchlinops.utils import batch_iterator, dict_product, NDList

__all__ = ["split"]


def split(
    linop: NamedLinop,
    batch_sizes: dict[NDorStr, int],
    flatten: bool = False,
):
    """Split a linop into smaller linops according to some batch sizes

    Parameters
    ----------
    linop : NamedLinop
        The NamedLinop to be split
    batch_sizes : dict[NDorStr -> int]
        Dictionary mapping dims to batch sizes for those dims
    flatten : bool, default False
        If True, flattens the outputs into flat lists (legacy behavior)
        If False, returns NDLists of the appropriate batch dimension

    Returns
    -------
    tuple[list, list, list] if flatten is True
    tuple[NDList, NDList, NDList] if flatten is False

    """
    # Precompute sizes and shapes
    batch_sizes = {ND.infer(k): v for k, v in batch_sizes.items()}
    sizes = {dim: linop.size(dim) for dim in linop.dims}
    ishapes = [linop.ishape for linop in linop.flatten()]
    oshapes = [linop.oshape for linop in linop.flatten()]

    # Make list of tiles
    batch_iterators = make_batch_iterators(sizes, batch_sizes)
    tiles = list(dict_product(batch_iterators))

    # Allocate outputs
    batch_dims = list(batch_sizes.keys())
    tiled_shape = tuple(ceil(sizes[dim] / batch_sizes[dim]) for dim in batch_dims)
    linops = NDList(tiled_shape, labels=batch_dims)
    input_batches = NDList(tiled_shape, labels=batch_dims)
    output_batches = NDList(tiled_shape, labels=batch_dims)
    for tile in tiles:
        idx = tuple(tile.get(dim, (0, slice(None)))[0] for dim in batch_dims)
        ibatches = [
            [tile.get(dim, (0, slice(None)))[1] for dim in ishape] for ishape in ishapes
        ]
        obatches = [
            [tile.get(dim, (0, slice(None)))[1] for dim in oshape] for oshape in oshapes
        ]
        linop_tile = linop.split(linop, *ibatches, *obatches)
        # linops.append(linop_tile)
        linops[idx] = linop_tile
        input_batches[idx] = ibatches[0]  # input batch of first linop
        output_batches[idx] = obatches[-1]  # output batch of last linop
    if flatten:
        # Set max depth to avoid flattening the batches themselves (which are lists of slices)
        return (
            flatten_recursive(linops.data),
            flatten_recursive(input_batches.data, max_depth=len(tiled_shape) - 1),
            flatten_recursive(output_batches.data, max_depth=len(tiled_shape) - 1),
        )
    return linops, input_batches, output_batches


def make_batch_iterators(total_sizes, batch_sizes):
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
