from dataclasses import dataclass
from functools import partial
from math import ceil
from typing import Literal, Optional

import numpy as np
import torch
from torchlinops.utils import NDList, batch_iterator, dict_product

from .add import Add
from .concat import Concat
from .device import ToDevice
from .nameddim import ND
from .namedlinop import NamedLinop

__all__ = ["split_linop", "create_batched_linop", "BatchSpec"]

Batch = tuple[int, slice]
# Represents a single batch at index 0 over the full extent
# Could convert to a full class
DEFAULT_BATCH = (0, slice(None))
Tile = dict[ND | str, Batch]


@dataclass
class BatchSpec:
    batch_sizes: dict[ND | str, int]
    device_matrix: Optional[np.ndarray | list] = None
    base_device: Optional[torch.device] = "cpu"


def create_batched_linop(linop, batch_specs: BatchSpec | list[BatchSpec]):
    """
    Examples
    --------
    >>> from torchlinops import Dense
    >>> congpu_batchspec = BatchSpec({"B": 2})
    >>> gpu_batchspec = BatchSpec({"C": 1}, device_matrix=[torch.device("cuda:0"), "cpu"])

    """
    if isinstance(batch_specs, BatchSpec):
        # Ensure list
        batch_specs = [batch_specs]

    if len(batch_specs) == 0:
        return linop
    batch_spec = batch_specs[0]
    linops, _, _ = split_linop(
        linop, batch_spec.batch_sizes, batch_spec.device_matrix, batch_spec.base_device
    )
    linops_shape = linops.shape
    linops = linops.reshape(-1)  # Flatten
    for i, linop in enumerate(linops):
        linops[i] = create_batched_linop(linop, batch_specs[1:])
    linops = linops.reshape(linops_shape)

    for dim in reversed(batch_spec.batch_sizes):
        # Manual axis reduction because I made Concat and Add too nice
        flat_linops = linops.reshape(-1, linops.shape[-1])
        new_linops = np.empty(flat_linops.shape[0], dtype=object)
        for i, linop_arr in enumerate(flat_linops):
            linop = linop_arr[0]
            if dim in linop.ishape and dim in linop.oshape:
                new_linop = Concat(*linop_arr, idim=dim, odim=dim)
            elif dim not in linop.ishape and dim in linop.oshape:
                new_linop = Concat(*linop_arr, odim=dim)
            elif dim in linop.ishape and dim not in linop.oshape:
                new_linop = Concat(*linop_arr, idim=dim)
            else:
                new_linop = Add(*linop_arr)
            new_linops[i] = new_linop
        linops = new_linops.reshape(linops.shape[:-1])
    return linops.item()


def split_linop(
    linop: NamedLinop,
    batch_sizes: dict[ND | str, int],
    device_matrix: Optional[list | np.ndarray] = None,
    base_device: torch.device = "cpu",
):
    """Split a linop into smaller linops according to some batch sizes

    Parameters
    ----------
    linop : NamedLinop
        The NamedLinop to be split.
    batch_sizes : dict[ND | str, int]
        Dictionary mapping dims to batch sizes for those dims.
    device_matrix : list | np.ndarray, optional
        Optional list of devices to broadcast the linop over. See notes for device
        broadcasting rules.

    Returns
    -------

    """
    # Precompute sizes and shapes
    batch_sizes = {ND.infer(k): v for k, v in batch_sizes.items()}
    sizes = {dim: linop.size(dim) for dim in linop.dims}

    # Make tiles. Each tile is a dictionary mapping a dimension to an integer
    # index of the tile and a slice over that dimension.
    batch_iterators = make_batch_iterators(sizes, batch_sizes)
    tiles: list[dict[ND, Batch]] = list(dict_product(batch_iterators))

    # Allocate outputs
    batch_dims = list(batch_sizes.keys())
    tiled_shape = tuple(ceil(sizes[dim] / batch_sizes[dim]) for dim in batch_dims)
    linops = np.ndarray(tiled_shape, dtype=object)
    input_batches = np.ndarray(tiled_shape, dtype=object)
    output_batches = np.ndarray(tiled_shape, dtype=object)
    if device_matrix is not None:
        # Create and broadcast device_matrix over requested split
        if not isinstance(device_matrix, np.ndarray):
            device_matrix = np.array(device_matrix, dtype=object)
        device_matrix = fuzzy_broadcast_to(device_matrix, tiled_shape)
        if device_matrix.shape != linops.shape:
            raise ValueError(
                f"Broadcasted device matrix with shape {device_matrix.shape} can't be broadcasted to exact shape of linop tiles with shape {tiled_shape}"
            )

    for tile in tiles:
        idx = _tile_get_idx(tile, batch_dims)
        ibatches, obatches, linop_tile = split_linop_with_tile(linop, tile)
        if device_matrix is not None:
            device = device_matrix[idx]
            linop_tile = (
                ToDevice(device, base_device, linop_tile.oshape)
                @ linop_tile.to(device)
                @ ToDevice(base_device, device, linop_tile.ishape)
            )
        linops[idx] = linop_tile
        input_batches[idx] = ibatches[0]  # input batch of first linop
        output_batches[idx] = obatches[-1]  # output batch of last linop
    return linops, input_batches, output_batches


def fuzzy_broadcast_to(arr: np.ndarray, target_shape):
    """Broadcast an array to a target shape, truncating or repeating if necessary.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3])
    >>> fuzzy_broadcast_to(arr, (4,))
    array([1, 2, 3, 1])
    >>> fuzzy_broadcast_to(arr, (3, 1))
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    >>> arr2 = np.array([1, 2])
    >>> fuzzy_broadcast_to(arr2, (3, 5))
    array([[1, 2, 1, 2, 1],
           [1, 2, 1, 2, 1],
           [1, 2, 1, 2, 1]])
    """
    # Ensure target shape and arr.shape have same length
    if len(target_shape) < arr.ndim:
        target_shape = (1,) * (arr.ndim - len(target_shape)) + tuple(target_shape)
    while len(target_shape) > arr.ndim:
        arr = np.expand_dims(arr, 0)

    if not is_broadcastable(arr.shape, target_shape):
        # Identify offending dimensions and repeat
        repeats = []
        for source_dim, target_dim in zip(arr.shape, target_shape):
            if source_dim == target_dim or source_dim == 1:
                repeats.append(1)
            elif source_dim < target_dim:
                repeats.append(int(ceil(target_dim / source_dim)))
            else:
                repeats.append(1)
        arr = tile_along_axes(arr, repeats)
        # Trim the excess
        slices = tuple(slice(0, dim) for dim in target_shape)
        arr = arr[slices]
    target_shape = np.broadcast_shapes(target_shape, arr.shape)
    return np.broadcast_to(arr, target_shape)


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
    total_sizes: dict[str, int],
    batch_sizes: dict[str, int],
) -> Tile:
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


def is_broadcastable(a, b):
    try:
        np.broadcast_shapes(a, b)
        return True
    except ValueError:
        return False


def repeat_along_axes(arr, repeats):
    """
    Repeat a numpy array along its axes.

    Parameters:
    - arr: np.ndarray
    - repeats: list or tuple of ints, with one repeat count per axis.
               If len(repeats) < arr.ndim, remaining axes are not repeated.
               If len(repeats) > arr.ndim, array is reshaped to add new axes.

    Returns:
    - Repeated array.
    """
    arr = np.asarray(arr)
    ndim = arr.ndim
    if len(repeats) > ndim:
        # Add singleton dimensions to match the number of repeats
        arr = arr.reshape(arr.shape + (1,) * (len(repeats) - ndim))
    elif len(repeats) < ndim:
        repeats = list(repeats) + [1] * (ndim - len(repeats))

    for axis, repeat in enumerate(repeats):
        if repeat > 1:
            arr = np.repeat(arr, repeat, axis=axis)
    return arr


def tile_along_axes(arr, tile_counts):
    """
    Tile a numpy array along its axes.

    Parameters:
    - arr: np.ndarray
    - tile_counts: list or tuple of ints, with one tile count per axis.
                   If len(tile_counts) < arr.ndim, missing axes default to 1.
                   If len(tile_counts) > arr.ndim, singleton dimensions are added.

    Returns:
    - Tiled array.
    """
    arr = np.asarray(arr)
    ndim = arr.ndim
    if len(tile_counts) < ndim:
        tile_counts = list(tile_counts) + [1] * (ndim - len(tile_counts))
    elif len(tile_counts) > ndim:
        arr = arr.reshape(arr.shape + (1,) * (len(tile_counts) - ndim))

    return np.tile(arr, tile_counts)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
