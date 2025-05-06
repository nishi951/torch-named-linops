from copy import copy

from .namedlinop import NamedLinop
from .nameddim import NS

__all__ = ["Identity", "Zero", "ShapeSpec"]


class Identity(NamedLinop):
    def __init__(self, ishape=("...",), oshape=None):
        super().__init__(NS(ishape, oshape))

    def forward(self, x):
        return self.fn(self, x)

    def adjoint(self):
        return self

    def normal(self, inner=None):
        if inner is None:
            return self
        return inner

    @staticmethod
    def fn(linop, x, /):
        return x

    @staticmethod
    def adj_fn(linop, x, /):
        return x

    @staticmethod
    def normal_fn(linop, x, /):
        return x

    def split_forward(self, ibatch, obatch):
        # TODO: Allow non-diagonal splitting
        assert ibatch == obatch, "Identity linop must be split identically"
        return self

    def split_forward_fn(self, ibatch, obatch, /):
        assert ibatch == obatch, "Identity linop must be split identically"
        return None

    def __pow__(self, exponent):
        return type(self)(self.ishape, self.oshape)


class Zero(NamedLinop):
    """Simple linop that always outputs 0, but with the same shape as the input"""

    def __init__(self, ishape=("...",), oshape=None):
        super().__init__(NS(ishape, oshape))

    def forward(self, x):
        return self.fn(self, x)

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


# Alias for changing input and output shapes
class ShapeSpec(Identity):
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
