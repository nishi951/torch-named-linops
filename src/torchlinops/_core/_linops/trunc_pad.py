"""Truncate/Pad
Maybe replace with more generic slicing linop later
"""

from torchlinops.utils import end_pad_with_zeros
from torchlinops._core._linops.namedlinop import NamedLinop

__all__ = [
    "Truncate",
    "PadDim",
]


class Truncate(NamedLinop):
    def __init__(self, dim, length, ishape, oshape):
        self.dim = dim
        self.length = length

        # Create the slices
        self.slc = [slice(None)] * len(ishape)
        self.slc[dim] = slice(0, self.length)
        self.slc = tuple(self.slc)

        self.end_slc = [slice(None)] * len(oshape)
        self.end_slc[dim] = slice(-self.length, None)
        self.end_slc = tuple(self.slc)
        super().__init__(ishape, oshape)
        # self.oshape[dim] = self.oshape[dim].next_unused(self.oshape)

    def forward(self, x):
        return self.fn(x)

    def fn(self, x, /):
        return x[self.slc]

    def adj_fn(self, y, /):
        return end_pad_with_zeros(y, self.dim, self.length)

    def normal_fn(self, x, /):
        x[self.end_slc] = 0.0
        return x

    def split_forward(self, ibatch, obatch):
        if ibatch[self.dim] != slice(None) or obatch[self.dim] != slice(None):
            raise ValueError("Cannot slice a Truncate linop along truncation dimension")
        return self

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
        super().__init__(ishape, oshape)
        # self.oshape[dim] = self.oshape[dim].next_unused(self.oshape)

    def forward(self, x):
        return self.fn(x)

    def adjoint(self):
        return Truncate(self.dim, self.length, self.oshape, self.ishape)

    def fn(self, x, /):
        return end_pad_with_zeros(x, self.dim, self.length)

    def adj_fn(self, y, /):
        return y[self.slc]

    def normal_fn(self, x, /):
        x[self.end_slc] = 0.0
        return x

    def split_forward(self, ibatch, obatch):
        if ibatch[self.dim] != slice(None) or obatch[self.dim] != slice(None):
            raise ValueError("Cannot slice a PadEnd linop along truncation dimension")
        return self

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
