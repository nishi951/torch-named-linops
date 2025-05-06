"""Truncate/Pad
Maybe replace with more generic slicing linop later
"""

from copy import copy

from torchlinops.utils import end_pad_with_zeros
from .namedlinop import NamedLinop
from .nameddim import NS, Shape

__all__ = ["Truncate", "PadDim"]


class Truncate(NamedLinop):
    def __init__(
        self,
        dim: int,
        length: int,
        ishape: Shape,
        oshape: Shape,
    ):
        self.dim = dim
        self.length = length

        # Create the slices
        self.slc = [slice(None)] * len(ishape)
        self.slc[dim] = slice(0, self.length)
        self.slc = tuple(self.slc)

        self.end_slc = [slice(None)] * len(oshape)
        self.end_slc[dim] = slice(-self.length, None)
        self.end_slc = tuple(self.slc)
        super().__init__(NS(ishape, oshape))

    def forward(self, x):
        return self.fn(self, x)

    @staticmethod
    def fn(linop, x, /):
        return x[linop.slc]

    @staticmethod
    def adj_fn(linop, y, /):
        return end_pad_with_zeros(y, linop.dim, linop.length)

    @staticmethod
    def normal_fn(linop, x, /):
        x[linop.end_slc] = 0.0
        return x

    def split_forward(self, ibatch, obatch):
        if ibatch[self.dim] != slice(None) or obatch[self.dim] != slice(None):
            raise ValueError("Cannot slice a Truncate linop along truncation dimension")
        return type(self)(self.dim, self.length, self.ishape, self.oshape)

    def split_forward_fn(self, ibatch, obatch, /, data=None):
        if ibatch[self.dim] != slice(None) or obatch[self.dim] != slice(None):
            raise ValueError("Cannot slice a Truncate linop along truncation dimension")
        return None

    # Linop changes relative size, but can't determine the size itself
    def size(self, dim):
        return None

    def size_fn(self, dim, /, data=None):
        return None

    def adjoint(self):
        return PadDim(self.dim, self.length, self.oshape, self.ishape)

    def normal(self, inner=None):
        """Diagonal in all dims except the last one"""
        pre = copy(self)
        post = self.adjoint()
        if inner is None:
            return post @ pre
        pre.oshape = inner.ishape
        post.ishape = inner.oshape
        new_oshape = list(inner.oshape)
        new_oshape[self.dim] = post.oshape[self.dim]
        post.oshape = tuple(new_oshape)
        return post @ inner @ pre

    @staticmethod
    def is_in_slice(a_slice, idx):
        """TODO: unused"""
        if idx < a_slice.start or idx >= a_slice.stop:
            return False
        step = a_slice.step if a_slice.step else 1
        if (idx - a_slice.start) % step == 0:
            return True
        else:
            return False


class PadDim(NamedLinop):
    def __init__(self, dim, length, ishape, oshape):
        self.dim = dim
        self.length = length
        # Create the slices
        self.slc = [slice(None)] * len(ishape)
        self.slc[dim] = slice(0, self.length)
        self.slc = tuple(self.slc)

        self.end_slc = [slice(None)] * len(oshape)
        self.end_slc[dim] = slice(-self.length, 0)
        self.end_slc = tuple(self.end_slc)
        super().__init__(NS(ishape, oshape))

    def forward(self, x):
        return self.fn(self, x)

    def adjoint(self):
        return Truncate(self.dim, self.length, self.oshape, self.ishape)

    def normal(self, inner=None):
        """Diagonal in all dims except the last one"""
        pre = copy(self)
        post = copy(self).H
        if inner is None:
            return post @ pre
        pre.oshape = inner.ishape
        post.ishape = inner.oshape
        return post @ inner @ pre

    @staticmethod
    def fn(linop, x, /):
        return end_pad_with_zeros(x, linop.dim, linop.length)

    @staticmethod
    def adj_fn(linop, y, /):
        return y[linop.slc]

    @staticmethod
    def normal_fn(linop, x, /):
        x[linop.end_slc] = 0.0
        return x

    def split_forward(self, ibatch, obatch):
        if ibatch[self.dim] != slice(None) or obatch[self.dim] != slice(None):
            raise ValueError("Cannot slice a PadEnd linop along truncation dimension")
        return type(self)(self.dim, self.length, self.ishape, self.oshape)

    def split_forward_fn(self, ibatch, obatch, /, data=None):
        if ibatch[self.dim] != slice(None) or obatch[self.dim] != slice(None):
            raise ValueError("Cannot slice a PadEnd linop along truncation dimension")
        return None

    # Linop changes relative size, but can't determine the size itself
    def size(self, dim):
        return None

    def size_fn(self, dim, /, data=None):
        return None

    @staticmethod
    def is_in_slice(a_slice, idx):
        """TODO: unused"""
        if idx < a_slice.start or idx >= a_slice.stop:
            return False
        step = a_slice.step if a_slice.step else 1
        if (idx - a_slice.start) % step == 0:
            return True
        else:
            return False
