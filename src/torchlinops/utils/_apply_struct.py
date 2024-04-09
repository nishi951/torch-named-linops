"""Recursive mapping on data with function"""

from typing import Callable, Mapping, Optional

import torch
import numpy as np

__all__ = [
    "apply_struct",
    "numpy2torch",
    "print_shapes",
]


def apply_struct(struct, fn: Callable, condition: Callable):
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
    return apply_struct(
        data,
        lambda x: torch.from_numpy(x).to(device),
        lambda x: isinstance(x, np.ndarray),
    )


def torch2numpy(data):
    return apply_struct(
        data,
        lambda x: x.detach().cpu().numpy(),
        lambda x: isinstance(x, torch.Tensor),
    )


def print_shapes(data):
    for name, obj in data.items():
        print(f"{name}: {obj.shape}")
