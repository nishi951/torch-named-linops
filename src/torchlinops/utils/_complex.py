import torch
import numpy as np
import sigpy as sp

__all__ = [
    "from_pytorch",
    "to_pytorch",
]


def xp(x):
    return sp.get_device(x).xp


def from_pytorch(x: torch.Tensor):
    if torch.is_complex(x):
        x_stack = torch.stack((x.real, x.imag), dim=-1)
        y = sp.from_pytorch(x_stack, iscomplex=True)
        return y
    return sp.from_pytorch(x, iscomplex=False)


def to_pytorch(x: np.ndarray, requires_grad: bool = False):
    if np.iscomplexobj(x):
        x_contig = xp(x).ascontiguousarray(x)
        y_stack = sp.to_pytorch(x_contig, requires_grad)
        y = torch.view_as_complex(y_stack.contiguous())
        return y
    return sp.to_pytorch(x, requires_grad)
