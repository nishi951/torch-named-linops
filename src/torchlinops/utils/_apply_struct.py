"""Recursive mapping on data with function"""

from typing import Callable, Mapping, Optional

import numpy as np
import torch

__all__ = ["apply_struct", "numpy2torch", "print_shapes"]


def apply_struct(struct, fn: Callable, condition: Callable):
    """Recursively apply a function to elements of nested data structures.

    Traverses dicts, lists, and tuples. When an element satisfies
    *condition*, *fn* is applied to it; otherwise it is returned as-is.
    The original structure is modified in-place where possible.

    Parameters
    ----------
    struct : dict, list, or object
        Nested data structure to traverse.
    fn : callable
        Function to apply to each leaf that satisfies *condition*.
    condition : callable
        Predicate that returns ``True`` for elements *fn* should be applied to.

    Returns
    -------
    object
        The structure with *fn* applied to matching leaves.
    """
    if isinstance(struct, Mapping):
        kv_pairs = struct.items()
    elif isinstance(struct, list):
        kv_pairs = enumerate(struct)
    elif condition(struct):
        return fn(struct)
    else:
        return struct
        # raise NotImplementedError(f'Struct should be a dict or a list (got {type(struct)})')
    for k, v in kv_pairs:
        struct[k] = apply_struct(v, fn, condition)
    return struct


def numpy2torch(data, device: Optional[torch.device] = "cpu"):
    """Convert numpy arrays in a nested data structure to torch tensors.

    Parameters
    ----------
    data : dict, list, or numpy.ndarray
        Nested data structure potentially containing numpy arrays.
    device : torch.device, optional
        Device to place the resulting tensors on. Default is ``"cpu"``.

    Returns
    -------
    object
        The same structure with numpy arrays replaced by torch tensors.
    """
    return apply_struct(
        data,
        lambda x: torch.from_numpy(x).to(device),
        lambda x: isinstance(x, np.ndarray),
    )


def torch2numpy(data):
    """Convert torch tensors in a nested data structure to numpy arrays.

    Parameters
    ----------
    data : dict, list, or torch.Tensor
        Nested data structure potentially containing torch tensors.

    Returns
    -------
    object
        The same structure with torch tensors replaced by numpy arrays.
    """
    return apply_struct(
        data,
        lambda x: x.detach().cpu().numpy(),
        lambda x: isinstance(x, torch.Tensor),
    )


def print_shapes(data):
    """Print the shapes of tensors/arrays in a nested data structure.

    Useful for quick debugging of dictionaries containing tensors or arrays.

    Parameters
    ----------
    data : dict
        Dictionary mapping names to tensors or arrays with a ``.shape`` attribute.
    """
    for name, obj in data.items():
        print(f"{name}: {obj.shape}")
