from copy import copy

import torch

from .namedlinop import NamedLinop

__all__ = ["Diagonal"]


class Diagonal(NamedLinop):
    def __init__(self, weight: torch.Tensor, ioshape):
        assert len(weight.shape) <= len(
            ioshape
        ), "All dimensions must be named or broadcastable"
        super().__init__(ioshape, ioshape)
        self.weight = weight

    def forward(self, x):
        return self.fn(x, self.weight)

    def fn(self, x, /, weight):
        return x * weight

    def adj_fn(self, x, /, weight):
        return x * torch.conj(weight)

    def normal_fn(self, x, /, weight):
        return x * torch.abs(weight) ** 2

    def adjoint(self):
        return type(self)(self.weight.conj(), self.ishape)

    def normal(self, inner=None):
        if inner is None:
            return type(self)(torch.abs(self.weight**2), self.ishape)
        # Update the shapes
        pre = copy(self)
        pre.ishape = inner.ishape
        pre.oshape = inner.ishape

        post = copy(self).H
        post.ishape = inner.oshape
        post.oshape = inner.oshape
        return post @ inner @ pre

    def split_forward(self, ibatch, obatch):
        weight = self.split_forward_fn(ibatch, obatch, self.weight)
        return type(self)(weight, self.ishape, self.oshape)

    def split_forward_fn(self, ibatch, obatch, /, weight):
        assert ibatch == obatch, "Diagonal linop must be split identically"
        return weight[ibatch]

    def size(self, dim: str):
        return self.size_fn(dim, self.weight)

    def size_fn(self, dim: str, weight):
        if dim in self.ishape:
            return weight.shape[self.ishape.index(dim)]
        return None

    def __pow__(self, exponent):
        return type(self)(weight ** exponent, self.ishape)
