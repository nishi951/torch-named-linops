from typing import List, Optional

import torch
import torch.nn as nn

from torchlinops._core._linops.nameddim import NS
from .namedlinop import NamedLinop, ND

__all__ = ["Diagonal"]


class Diagonal(NamedLinop):
    def __init__(
        self, weight: torch.Tensor, ioshape, broadcast_dims: Optional[List[str]] = None
    ):
        assert len(weight.shape) <= len(
            ioshape
        ), "All dimensions must be named or broadcastable"
        super().__init__(NS(ioshape))
        self.weight = nn.Parameter(weight, requires_grad=False)
        assert (
            len(self.ishape) >= len(self.weight.shape)
        ), f"Weight cannot have fewer dimensions than the input shape: ishape: {self.ishape}, weight: {weight.shape}"
        broadcast_dims = broadcast_dims if broadcast_dims is not None else []
        self._shape.add('broadcast_dims', broadcast_dims)

    @property
    def broadcast_dims(self):
        return self._shape.lookup('broadcast_dims')

    def forward(self, x):
        return self.fn(x, self.weight)

    def fn(self, x, /, weight):
        return x * weight

    def adj_fn(self, x, /, weight):
        return x * torch.conj(weight)

    def normal_fn(self, x, /, weight):
        return x * torch.abs(weight) ** 2

    def adjoint(self):
        return type(self)(self.weight.conj(), self.ishape, self.broadcast_dims)

    def normal(self, inner=None):
        if inner is None:
            return type(self)(
                torch.abs(self.weight) ** 2, self.ishape, self.broadcast_dims
            )
        return super().normal(inner)

    def split_forward(self, ibatch, obatch):
        weight = self.split_forward_fn(ibatch, obatch, self.weight)
        return type(self)(weight, self.ishape, self.broadcast_dims)

    def split_forward_fn(self, ibatch, obatch, /, weight):
        assert ibatch == obatch, "Diagonal linop must be split identically"
        # Filter out broadcastable dims
        ibatch = [
            slice(None) if dim in self.broadcast_dims else slc
            for slc, dim in zip(ibatch, self.ishape)
        ]
        return weight[ibatch[-len(weight.shape) :]]

    def size(self, dim: str):
        return self.size_fn(dim, self.weight)

    def size_fn(self, dim: str, weight):
        if dim in self.ishape:
            n_broadcast = len(self.ishape) - len(weight.shape)
            if self.ishape.index(dim) < n_broadcast or dim in self.broadcast_dims:
                return None
            else:
                return weight.shape[self.ishape.index(dim) - n_broadcast]
        return None

    def __pow__(self, exponent):
        return type(self)(self.weight**exponent, self.ishape)
