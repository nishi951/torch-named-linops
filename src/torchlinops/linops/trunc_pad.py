"""Truncate/Pad
Maybe replace with more generic slicing linop later
"""

from copy import copy

from torchlinops.utils import end_pad_with_zeros

from ..nameddim import NamedShape as NS, Shape
from .namedlinop import NamedLinop
from .identity import Identity

__all__ = ["Truncate", "PadDim"]


class Truncate(NamedLinop):
    """Truncation (slicing) operator along the last dimension.

    Extracts a contiguous slice from the input. The adjoint zero-pads
    back to the original size.
    """

    def __init__(
        self,
        dim: int,
        from_length: int,
        to_length: int,
        ishape: Shape,
        oshape: Shape,
    ):
        self.dim = dim
        self.from_length = from_length
        self.to_length = to_length
        if self.from_length < 0:
            raise ValueError(
                f"from_length must be nonnegative but got {self.from_length}"
            )
        if self.to_length < 0 or self.to_length > from_length:
            raise ValueError(
                f"to_length must be in [0, {self.from_length}] but got {self.to_length}"
            )
        self.slc = [slice(None)] * len(ishape)
        self.slc[dim] = slice(0, self.to_length)
        self.slc = tuple(self.slc)

        self.end_slc = [slice(None)] * len(ishape)
        self.end_slc[dim] = slice(self.to_length - self.from_length, None)
        self.end_slc = tuple(self.end_slc)
        super().__init__(NS(ishape, oshape))

    @staticmethod
    def fn(truncate, x, /):
        if x.shape[truncate.dim] != truncate.from_length:
            raise ValueError(
                f"Truncate expecting size {truncate.from_length} at x.shape[{truncate.dim}] but got {x.shape[truncate.dim]} (x.shape: {x.shape})"
            )
        return x[truncate.slc]

    @staticmethod
    def adj_fn(truncate, y, /):
        if y.shape[truncate.dim] != truncate.to_length:
            raise ValueError(
                f"Truncate (adjoint) expecting size {truncate.to_length} at x.shape[{truncate.dim}] but got {y.shape[truncate.dim]} (y.shape: {y.shape})"
            )
        return end_pad_with_zeros(
            y,
            truncate.dim,
            truncate.from_length - truncate.to_length,
        )

    @staticmethod
    def normal_fn(truncate, x, /):
        if x.shape[truncate.dim] != truncate.from_length:
            raise ValueError(
                f"Truncate (normal) expecting size {truncate.from_length} at x.shape[{truncate.dim}] but got {x.shape[truncate.dim]} (x.shape: {x.shape})"
            )
        x = x.clone()
        x[truncate.end_slc] = 0.0
        return x

    def split_forward(self, ibatch, obatch):
        if ibatch[self.dim] != slice(None) or obatch[self.dim] != slice(None):
            raise ValueError("Cannot slice a Truncate linop along truncation dimension")
        return type(self)(
            self.dim, self.from_length, self.to_length, self.ishape, self.oshape
        )

    def adjoint(self):
        return PadDim(
            self.dim,
            self.to_length,
            self.from_length,
            self.oshape,
            self.ishape,
        )

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


class PadDim(NamedLinop):
    """Zero-padding operator along a specified dimension.

    Pads the input with zeros. The adjoint truncates (slices) back to
    the original size.
    """

    def __init__(self, dim, from_length, to_length, ishape, oshape):
        self.dim = dim
        self.from_length = from_length
        self.to_length = to_length
        if self.to_length < 0:
            raise ValueError(f"to_length must be nonnegative but got {self.to_length}")

        if self.from_length < 0 or self.from_length > self.to_length:
            raise ValueError(
                f"to_length must be in [0, {self.to_length}] but got {self.from_length}"
            )

        self.slc = [slice(None)] * len(ishape)
        self.slc[dim] = slice(0, self.from_length)
        self.slc = tuple(self.slc)
        super().__init__(NS(ishape, oshape))

    def adjoint(self):
        return Truncate(
            self.dim, self.to_length, self.from_length, self.oshape, self.ishape
        )

    def normal(self, inner=None):
        """Diagonal in all dims except the last one"""
        if inner is None:
            return Identity(self.ishape)
        pre = copy(self)
        post = copy(self).H
        pre.oshape = inner.ishape
        post.ishape = inner.oshape
        return post @ inner @ pre

    @staticmethod
    def fn(padend, x, /):
        if x.shape[padend.dim] != padend.from_length:
            raise ValueError(
                f"padend expecting size {padend.from_length} at x.shape[{padend.dim}] but got {x.shape[padend.dim]} (x.shape: {x.shape})"
            )
        return end_pad_with_zeros(x, padend.dim, padend.to_length - padend.from_length)

    @staticmethod
    def adj_fn(padend, y, /):
        if y.shape[padend.dim] != padend.to_length:
            raise ValueError(
                f"PadDim (adjoint) expecting size {padend.to_length} at x.shape[{padend.dim}] but got {y.shape[truncate.dim]} (y.shape: {y.shape})"
            )
        return y[padend.slc]

    @staticmethod
    def normal_fn(padend, x, /):
        x = x.clone()
        return x

    def split_forward(self, ibatch, obatch):
        if ibatch[self.dim] != slice(None) or obatch[self.dim] != slice(None):
            raise ValueError("Cannot slice a PadEnd linop along truncation dimension")
        return type(self)(
            self.dim, self.from_length, self.to_length, self.ishape, self.oshape
        )
