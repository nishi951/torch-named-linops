from typing import Optional, Mapping

import torch
from einops import rearrange, reduce, repeat

from .namedlinop import NamedLinop

__all__ = [
    "Rearrange",
    "SumReduce",
    "Repeat",
]


class Rearrange(NamedLinop):
    """Moves around dimensions."""

    def __init__(
        self, istr, ostr, ishape, oshape, axes_lengths: Optional[Mapping] = None
    ):
        assert len(ishape) == len(
            oshape
        ), "Rearrange currently only supports pure dimension permutations"
        super().__init__(ishape, oshape)
        self.istr = istr
        self.ostr = ostr
        self.axes_lengths = axes_lengths if axes_lengths is not None else {}

    def forward(self, x):
        return self.fn(x, self.istr, self.ostr, self.axes_lengths)

    def fn(self, x, /, istr, ostr, axes_lengths):
        return rearrange(x, f"{istr} -> {ostr}", **axes_lengths)

    def adj_fn(self, x, /, ostr, istr, axes_lengths):
        return rearrange(x, f"{ostr} -> {istr}", **axes_lengths)

    def split_forward(self, ibatch, obatch):
        """Rearranging is transparent to splitting"""
        return self

    def split_forward_fn(self, ibatch, obatch, /, istr, ostr, axes_lengths):
        """Rearranging is transparent to splitting"""
        return (istr, ostr, axes_lengths)

    def size(self, dim: str):
        """Rearranging does not determine any dimensions"""
        return None

    def size_fn(self, dim: str, /, istr, ostr, axes_lengths):
        """Rearranging does not determine any dimensions"""
        return None


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
        super().__init__(ishape, oshape)
        assert (
            len(self.oshape) < len(self.ishape)
        ), f"Reduce must be over at least one dimension: got {self.ishape} -> {self.oshape}"
        self.adj_ipattern = self.fill_singleton_dims(self.ishape, self.oshape)
        self.ipattern = " ".join(ishape)
        self.opattern = " ".join(oshape)

    @staticmethod
    def fill_singleton_dims(ishape, oshape):
        out = []
        for idim in ishape:
            if idim in oshape:
                out.append(idim)
            else:
                out.append("()")
        return out

    def forward(self, x):
        return self.fn(x, self.ipattern, self.opattern)

    def fn(self, x, /):
        x = reduce(x, f"{self.ipattern} -> {self.opattern}", "sum")
        return x

    def adj_fn(self, x, /):
        x = repeat(x, f"{self.opattern} -> {self.adj_ipattern}")
        return x

    def split_forward_fn(self, ibatch, obatch, /):
        """Reducing is transparent to splitting"""
        return tuple()

    def size(self, dim: str):
        """Reducing does not determine any dimensions"""
        return None

    def size_fn(self, dim: str, /, ipattern, opattern, size_spec):
        """Reducing does not determine any dimensions"""
        return None


class Repeat(NamedLinop):
    """Unsqueezes and expands a tensor along dim
    TODO: Replace with einops' repeat
    """

    def __init__(self, n_repeats, dim, ishape, oshape):
        assert len(ishape) + 1 == len(
            oshape
        ), "oshape should have 1 more dim than ishape"
        super().__init__(ishape, oshape)
        self.n_repeats = n_repeats
        self.dim = dim

    def forward(self, x):
        return self.fn(x, self.n_repeats)

    def fn(self, x, /, n_repeats):
        expand_size = [-1] * len(self.oshape)
        expand_size[self.dim] = n_repeats
        x = x.unsqueeze(self.dim)
        # print(x)
        return x.expand(*expand_size)

    def adj_fn(self, x, /, n_repeats):
        return torch.sum(x, dim=self.dim, keepdim=False)

    def split_forward(self, ibatch, obatch):
        """Repeat fewer times, depending on the size of obatch"""
        assert len(ibatch) == len(
            self.ishape
        ), "length of ibatch should match length of ishape"
        assert len(obatch) == len(
            self.oshape
        ), "length of obatch should match length of oshape"
        return type(self)(
            n_repeats=self.split_forward_fn(ibatch, obatch, self.n_repeats),
            dim=self.dim,
            ishape=self.ishape,
            oshape=self.oshape,
        )

    def split_forward_fn(self, ibatch, obatch, /, n_repeats):
        return self.slice_len(obatch[self.dim], n_repeats)

    def size(self, dim: str):
        return self.size_fn(dim, self.n_repeats)

    def size_fn(self, dim, /, n_repeats):
        if dim == self.oshape[self.dim]:
            return n_repeats
        return None

    @staticmethod
    def slice_len(slc, n):
        """
        n: length of sequence slc is being applied to
        """
        return len(range(*slc.indices(n)))
