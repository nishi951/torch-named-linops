"""Recursive mapping on data with function"""
from typing import Callable, Mapping, Optional

import torch
import numpy as np
import cupy as cp

__all__ = [
    'recursive_map',
    'struct2np',
    'apply_struct',
    'numpy2torch',
    'print_shapes',
]

is_array = lambda x: isinstance(x, np.ndarray) or isinstance(x, cp.ndarray) or isinstance(x, torch.Tensor)
is_numeric = lambda x: isinstance(x, int) or isinstance(x, float) or isinstance(x, complex)

def recursive_map(data, func):
    """Recursively performs func on the items of data
    Only recurses through data that is not numerical arrays
    Converts objects to dicts"""
    apply = lambda x: recursive_map(x, func)
    if is_numeric(data) or isinstance(data, str) or is_array(data):
        return func(data)
    elif isinstance(data, Mapping):
        return type(data)({k: apply(v) for k, v in data.items()})
    elif (isinstance(data, list) or isinstance(data, tuple)):
        return type(data)(apply(v) for v in data)
    return apply(data.__dict__)

def struct2np(data):
    def torch2np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    return recursive_map(data, torch2np)


def apply_struct(struct, fn: Callable, condition: Callable):
    if isinstance(struct, Mapping):
        kv_pairs = struct.items()
    elif isinstance(struct, list):
        kv_pairs = enumerate(struct)
    elif condition(struct):
        return fn(struct)
    else:
        return struct
        #raise NotImplementedError(f'Struct should be a dict or a list (got {type(struct)})')
    for k, v in kv_pairs:
        struct[k] = apply_struct(v, fn, condition)
    return struct

def numpy2torch(data, device: Optional[torch.device] = 'cpu'):
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
        print(f'{name}: {obj.shape}')
