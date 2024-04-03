from copy import copy
from typing import Optional, Mapping

import torch
from einops import rearrange, reduce, repeat

from .namedlinop import NamedLinop
from .nameddim import ND, NS, NamedShape

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
        shape: NamedShape,
        axes_lengths: Optional[Mapping] = None,
    ):
        # assert len(ishape) == len(
        #     oshape
        # ), "Rearrange currently only supports pure dimension permutations"
        super().__init__(shape)
        self.ipattern = ipattern
        self.opattern = opattern
        self.axes_lengths = axes_lengths if axes_lengths is not None else {}

    def forward(self, x):
        return self.fn(x)

    def normal(self, inner=None):
        pre = copy(self)
        post = copy(self).H
        if inner is not None:
            pre.oshape = inner.ishape

            new_ishape = inner.oshape
            new_oshape = list(post.oshape)
            for old_d, new_d in zip(post.ishape, new_ishape):
                if old_d != new_d and old_d in new_oshape:
                    # Replace the old dimension with the new dimension
                    new_oshape[new_oshape.index(old_d)] = new_d
            post.ishape = new_ishape
            post.oshape = tuple(new_oshape)
            return post @ inner @ pre
        return post @ pre

    def fn(self, x, /):
        return rearrange(x, f"{self.ipattern} -> {self.opattern}", **self.axes_lengths)

    def adj_fn(self, x, /):
        return rearrange(x, f"{self.opattern} -> {self.ipattern}", **self.axes_lengths)

    def split_forward(self, ibatch, obatch):
        """Rearranging is transparent to splitting"""
        return self

    def split_forward_fn(self, ibatch, obatch, /):
        """Rearranging is transparent to splitting"""
        return None

    def size(self, dim: str):
        """Rearranging does not determine any dimensions"""
        return None

    def size_fn(self, dim: str, /):
        """Rearranging does not determine any dimensions"""
        return None


class SumReduce(NamedLinop):
    """Wrapper for einops' reduce,

    Adjoint of Repeat
    """

    def __init__(self, shape: NamedShape):
        """
        ipattern : string
            Input shape spec, einops style
        opattern : string
            Output shape spec, einops style
        """
        super().__init__(shape)
        assert (
            len(self.oshape) < len(self.ishape)
        ), f"Reduce must be over at least one dimension: got {self.ishape} -> {self.oshape}"

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

    def forward(self, x):
        return self.fn(x)

    def fn(self, x, /):
        x = reduce(x, f"{self.ipattern} -> {self.opattern}", "sum")
        return x

    def adj_fn(self, x, /):
        x = repeat(x, f"{self.opattern} -> {self.adj_ipattern}")
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
        n_repeats = {d: 1 for d in self.ishape if d not in self.oshape}
        return Repeat(n_repeats, self._shape.H)

    def normal(self, inner=None):
        pre = copy(self)
        post = copy(self).H
        # New post output shape (post = Repeat)
        # If dimension is not summed over (i.e. it is in pre_adj_ishape) , it stays the same
        # Otherwise, if dimension is summed over, its name changes
        new_oshape = []
        new_axes_lengths = {}
        for d in self.ishape:
            if d in self.adj_ishape:
                new_oshape.append(d)
            else:
                new_d = d.next_unused(self.ishape)
                new_oshape.append(new_d)
                if d in post.axes_lengths:
                    # Replace old dimension with a new one
                    new_axes_lengths[new_d] = post.axes_lengths[d]
        post.oshape = tuple(new_oshape)
        post.axes_lengths = new_axes_lengths
        if inner is not None:
            pre.oshape = inner.ishape
            post.ishape = inner.oshape
            return post @ inner @ pre
        return post @ pre


class Repeat(NamedLinop):
    """Unsqueezes and expands a tensor along dim"""

    def __init__(self, n_repeats: Mapping, shape: NamedShape):
        super().__init__(shape)
        assert len(self.oshape) > len(
            self.ishape
        ), f"Repeat must add at least one dimension: got {self.ishape} -> {self.oshape}"
        self.axes_lengths = n_repeats
        self.axes_lengths = {ND.infer(k): v for k, v in self.axes_lengths.items()}

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

    def forward(self, x):
        return self.fn(x)

    def fn(self, x, /):
        x = repeat(
            x,
            f"{self.ipattern} -> {self.opattern}",
            **{str(k): v for k, v in self.axes_lengths.items()},
        )
        return x

    def adj_fn(self, x, /):
        x = reduce(x, f"{self.opattern} -> {self.ipattern}", "sum")
        return x

    def split_forward(self, ibatch, obatch):
        """Repeat fewer times, depending on the size of obatch"""
        A = copy(self)
        for dim, slc in zip(self.oshape, obatch):
            if dim in A.axes_lengths:
                A.axes_lengths[dim] = self.slice_len(slc, self.size(dim))
        return A

    def split_forward_fn(self, ibatch, obatch, /):
        """No data to split"""
        return None

    def size(self, dim: str):
        return self.size_fn(dim)

    def size_fn(self, dim, /):
        return self.axes_lengths.get(dim, None)

    def adjoint(self):
        return SumReduce(self._shape.H)

    def normal(self, inner=None):
        pre = copy(self)
        post = copy(self).H
        post.oshape = tuple(
            d if d in pre.adj_ishape else d.next_unused(pre.ishape) for d in pre.ishape
        )
        if inner is not None:
            return post @ inner @ pre
        return post @ pre

    @staticmethod
    def slice_len(slc, n):
        """
        n: length of sequence slc is being applied to
        """
        return len(range(*slc.indices(n)))
