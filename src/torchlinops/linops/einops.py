from warnings import warn
from copy import copy, deepcopy
from typing import Optional
from collections.abc import Mapping

import torch
from einops import rearrange, reduce, repeat

from .identity import Identity
from .namedlinop import NamedLinop
from .nameddim import ND, NS, NamedShape, Shape

__all__ = [
    "Rearrange",
    "SumReduce",
    "Repeat",
]


class Rearrange(NamedLinop):
    """Moves around dimensions."""

    def __init__(
        self,
        ipattern,
        opattern,
        ishape: Shape,
        oshape: Shape,
        axes_lengths: Optional[Mapping] = None,
    ):
        # assert len(ishape) == len(
        #     oshape
        # ), "Rearrange currently only supports pure dimension permutations"
        super().__init__(NS(ishape, oshape))
        self.ipattern = ipattern
        self.opattern = opattern
        axes_lengths = axes_lengths if axes_lengths is not None else {}
        self._shape.add("axes_lengths", axes_lengths)

    @property
    def axes_lengths(self):
        return self._shape.axes_lengths

    def forward(self, x):
        return self.fn(self, x)

    @staticmethod
    def fn(linop, x, /):
        axes_lengths = {str(k): v for k, v in linop.axes_lengths.items()}
        return rearrange(x, f"{linop.ipattern} -> {linop.opattern}", **axes_lengths)

    @staticmethod
    def adj_fn(linop, x, /):
        axes_lengths = {str(k): v for k, v in linop.axes_lengths.items()}
        return rearrange(x, f"{linop.opattern} -> {linop.ipattern}", **axes_lengths)

    def split_forward(self, ibatch, obatch):
        """TODO: Add compound shapes so splitting through rearrange can work."""
        warn(
            f"Splitting Rearrange linop with shape {self._shape} - splitting a rearrange may behave unusually."
        )
        new_axes_lengths = deepcopy(self.axes_lengths)
        for dim, slc in zip(self.ishape, ibatch):
            if dim in self.axes_lengths:
                n = self.axes_lengths[dim]
                new_axes_lengths[dim] = slicelen(n, slc)
        for dim, slc in zip(self.oshape, obatch):
            if dim in self.axes_lengths:
                n = self.axes_lengths[dim]
                new_axes_lengths[dim] = slicelen(n, slc)

        out = type(self)(
            self.ipattern, self.opattern, self.ishape, self.oshape, new_axes_lengths
        )
        return out

    def split_forward_fn(self, ibatch, obatch, /):
        """Rearranging is transparent to splitting"""
        return None

    def size(self, dim: str):
        """Rearranging does not determine any dimensions"""
        return None

    def size_fn(self, dim: str, /):
        """Rearranging does not determine any dimensions"""
        return None

    def normal(self, inner=None):
        if inner is None:
            return Identity(self.ishape)
        return super().normal(inner)


class SumReduce(NamedLinop):
    """Wrapper for einops' reduce,

    Adjoint of Repeat
    """

    def __init__(self, ishape, oshape):
        """
        ipattern : string
            Input shape spec, einops style
        opattern : string
            Output shape spec, einops style
        """
        super().__init__(NS(ishape, oshape))
        assert len(self.oshape) < len(self.ishape), (
            f"Reduce must be over at least one dimension: got {self.ishape} -> {self.oshape}"
        )

    def forward(self, x):
        return self.fn(self, x)

    @staticmethod
    def fn(linop, x, /):
        x = reduce(x, f"{linop.ipattern} -> {linop.opattern}", "sum")
        return x

    @staticmethod
    def adj_fn(linop, x, /):
        x = repeat(x, f"{linop.opattern} -> {linop.adj_ipattern}")
        return x

    def split_forward(self, ibatch, obatch):
        return self

    def split_forward_fn(self, ibatch, obatch, /):
        """Reducing is transparent to splitting"""
        return None

    def size(self, dim: str):
        """Reducing does not determine any dimensions"""
        return None

    def size_fn(self, dim: str, /, ipattern, opattern, size_spec):
        """Reducing does not determine any dimensions"""
        return None

    def adjoint(self):
        broadcast_dims = [d for d in self.ishape if d not in self.oshape]
        n_repeats = {d: 1 for d in broadcast_dims}
        return Repeat(n_repeats, self._shape.H, None, broadcast_dims)

    def normal(self, inner=None):
        pre = copy(self)
        post = self.adjoint()
        # New post output shape (post = Repeat)
        # If dimension is not summed over (i.e. it is in pre_adj_ishape) , it stays the same
        # Otherwise, if dimension is summed over, its name changes
        # This automatically updates the axes_lengths as well.
        post.oshape = tuple(
            d.next_unused(self.ishape) if d not in post.ishape else d
            for d in post.oshape
        )
        shape_updates = {
            d: new_d for d, new_d in zip(pre.ishape, post.oshape) if d != new_d
        }
        if inner is not None:
            pre.oshape = inner.ishape
            post.ishape = inner.oshape
            inner_shape_updates = getattr(inner, "_shape_updates", {})
            shape_updates.update(inner_shape_updates)
            normal = post @ inner @ pre
            normal._shape_updates = shape_updates
        else:
            normal = post @ pre
            normal._shape_updates = shape_updates
        return normal

    @property
    def adj_ishape(self):
        return self.fill_singleton_dims(self.ishape, self.oshape)

    @property
    def adj_ipattern(self):
        return " ".join(str(d) if d is not None else "()" for d in self.adj_ishape)

    @property
    def ipattern(self):
        return " ".join(str(d) for d in self.ishape)

    @property
    def opattern(self):
        return " ".join(str(d) for d in self.oshape)

    @staticmethod
    def fill_singleton_dims(ishape, oshape):
        out = []
        for idim in ishape:
            if idim in oshape:
                out.append(idim)
            else:
                out.append(None)
        return tuple(out)


class Repeat(NamedLinop):
    """Unsqueezes and expands a tensor along dim"""

    def __init__(
        self, n_repeats: Mapping, ishape, oshape, broadcast_dims: Optional[list] = None
    ):
        super().__init__(NS(ishape, oshape))
        assert len(self.oshape) > len(self.ishape), (
            f"Repeat must add at least one dimension: got {self.ishape} -> {self.oshape}"
        )
        self._shape.add("axes_lengths", n_repeats)
        # self.axes_lengths = n_repeats
        # self.axes_lengths = {ND.infer(k): v for k, v in self.axes_lengths.items()}
        broadcast_dims = broadcast_dims if broadcast_dims is not None else []
        self._shape.add("broadcast_dims", broadcast_dims)

    @property
    def axes_lengths(self):
        return self._shape.axes_lengths

    @property
    def broadcast_dims(self):
        return self._shape.broadcast_dims

    def forward(self, x):
        return self.fn(self, x)

    @staticmethod
    def fn(linop, x, /):
        x = repeat(
            x,
            f"{linop.ipattern} -> {linop.opattern}",
            **{str(k): v for k, v in linop.axes_lengths.items()},
        )
        return x

    @staticmethod
    def adj_fn(linop, x, /):
        x = reduce(x, f"{linop.opattern} -> {linop.ipattern}", "sum")
        return x

    def split_forward(self, ibatch, obatch):
        """Repeat fewer times, depending on the size of obatch"""
        new_axes_lengths = deepcopy(self.axes_lengths)
        for dim, slc in zip(self.oshape, obatch):
            if dim in self.axes_lengths and dim not in self.broadcast_dims:
                self.axes_lengths[dim] = slicelen(self.size(dim), slc)
        return type(self)(
            new_axes_lengths, self.ishape, self.oshape, self.broadcast_dims
        )

    def split_forward_fn(self, ibatch, obatch, /):
        """No data to split"""
        return None

    def size(self, dim: str):
        return self.size_fn(dim)

    def size_fn(self, dim, /):
        if dim in self.broadcast_dims:
            return None
        return self.axes_lengths.get(dim, None)

    def adjoint(self):
        return SumReduce(self._shape.H, None)

    def normal(self, inner=None):
        pre = copy(self)
        post = self.adjoint()
        if inner is not None:
            pre.oshape = inner.ishape
            post.ishape = inner.oshape
            return post @ inner @ pre
        # No updated dims because Repeat -> SumReduce
        # Gets rid of the new dimensions immediately
        # TODO: simplify this more?
        return post @ pre

    @property
    def adj_ishape(self):
        return self.fill_singleton_dims(self.oshape, self.ishape)

    @property
    def adj_ipattern(self):
        return " ".join(str(d) if d is not None else "()" for d in self.adj_ishape)

    @property
    def ipattern(self):
        return " ".join(str(d) for d in self.ishape)

    @property
    def opattern(self):
        return " ".join(str(d) for d in self.oshape)

    @staticmethod
    def fill_singleton_dims(ishape, oshape):
        out = []
        for idim in ishape:
            if idim in oshape:
                out.append(idim)
            else:
                out.append(None)
        return tuple(out)


def slicelen(n, slc):
    return len(range(*slc.indices(n)))
