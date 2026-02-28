from copy import copy

from torch import Tensor

from ..nameddim import NamedShape as NS
from .namedlinop import NamedLinop

__all__ = ["Identity", "Zero", "ShapeSpec"]


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

    def split_forward(self, ibatch, obatch):
        # TODO: Allow non-diagonal splitting
        assert ibatch == obatch, "Identity linop must be split identically"
        return self

    def __pow__(self, _: float | Tensor):
        return type(self)(self.ishape, self.oshape)


class Zero(NamedLinop):
    """Zero operator $0(x) = 0$.

    Always returns a zero tensor with the same shape as the input.
    """

    def __init__(self, ishape=("...",), oshape=None):
        super().__init__(NS(ishape, oshape))

    @staticmethod
    def fn(self, x, /):
        return x.zero_()

    @staticmethod
    def adj_fn(self, x, /):
        return x.zero_()

    @staticmethod
    def normal_fn(self, x, /):
        return x.zero_()

    def split_forward(self, ibatch, obatch):
        return self


class ShapeSpec(Identity):
    """Identity operator that renames dimensions.

    Functionally identical to ``Identity`` but maps from one set of named
    dimensions to another, acting as a shape adapter between linops.
    """

    def adjoint(self):
        return type(self)(self.oshape, self.ishape)

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
