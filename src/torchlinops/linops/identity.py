import time
from copy import copy

import torch
from torch import zeros_like
from torch import Tensor

from ..nameddim import NamedShape as NS
from .namedlinop import NamedLinop

__all__ = ["Identity", "Zero", "ShapeSpec", "Sleep"]


class Identity(NamedLinop):
    """Identity operator $I(x) = x$.

    Returns the input unchanged. The adjoint, normal, and any power of the
    identity are also the identity.
    """

    def __init__(self, ishape=("...",), oshape=None):
        super().__init__(NS(ishape, oshape))

    def adjoint(self):
        return self

    def normal(self, inner=None):
        if inner is None:
            return self
        return inner

    @staticmethod
    def fn(linop: NamedLinop, x, /):
        return x

    @staticmethod
    def adj_fn(linop: NamedLinop, x, /):
        return x

    @staticmethod
    def normal_fn(linop: NamedLinop, x, /):
        # A bit faster
        return x

    @staticmethod
    def split(linop, tile):
        return linop

    def __pow__(self, _: float | Tensor):
        return copy(self)


class Zero(NamedLinop):
    """Zero operator $0(x) = 0$.

    Always returns a zero tensor with the same shape as the input.
    """

    def __init__(self, ishape=("...",), oshape=None):
        super().__init__(NS(ishape, oshape))

    @staticmethod
    def fn(self, x, /):
        return zeros_like(x)

    @staticmethod
    def adj_fn(self, x, /):
        return zeros_like(x)

    @staticmethod
    def normal_fn(self, x, /):
        return zeros_like(x)

    @staticmethod
    def split(linop, tile):
        return linop


class ShapeSpec(Identity):
    """Identity operator that renames dimensions.

    Functionally identical to ``Identity`` but maps from one set of named
    dimensions to another, acting as a shape adapter between linops.
    """

    def adjoint(self):
        new = copy(self)
        new.shape = self.shape.adjoint()
        return new

    def normal(self, inner=None):
        if inner is None:
            # Behaves like a diagonal linop
            return ShapeSpec(self.ishape, self.ishape)
        pre = copy(self)
        post = self.adjoint()
        pre.oshape = inner.ishape
        post.ishape = inner.oshape
        normal = post @ inner @ pre
        normal._shape_updates = getattr(inner, "_shape_updates", {})
        return normal


class Sleep(NamedLinop):
    """Identity-like operator that sleeps for a specified duration.

    Returns the input unchanged but introduces a delay in ``fn()`` only.
    Useful for benchmarking and simulating computation time.

    On CUDA devices, uses ``torch.cuda._sleep()`` on the input tensor's
    default stream. On CPU, uses ``time.sleep()``.
    """

    def __init__(self, duration: float = 0.1, ioshape=("...",), oshape=None):
        super().__init__(NS(ioshape, oshape))
        self.duration = duration

    @staticmethod
    def fn(linop, x, /):
        if x.is_cuda:
            props = torch.cuda.get_device_properties(x.device)
            cycles = int(linop.duration * props.clock_rate * 1000)
            with torch.cuda.stream(torch.cuda.default_stream(x.device)):
                torch.cuda._sleep(cycles)
        else:
            time.sleep(linop.duration)
        return x

    @staticmethod
    def adj_fn(linop, x, /):
        return x

    @staticmethod
    def normal_fn(linop, x, /):
        return x

    @staticmethod
    def split(linop, tile):
        return copy(linop)
