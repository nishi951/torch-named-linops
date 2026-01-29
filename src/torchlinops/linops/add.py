import torch
import torch.nn as nn

from ..nameddim import NamedShape as NS, isequal
from .namedlinop import NamedLinop

__all__ = ["Add"]


class Add(NamedLinop):
    """The sum of one or more linear operators."""

    def __init__(self, *linops, stream=None):
        """
        Parameters
        ----------
        *linops : list[NamedLinop]
            One or more linear operators to be added together
        stream : Optional

        """
        assert all(isequal(linop.ishape, linops[0].ishape) for linop in linops), (
            f"Add: All linops must share same ishape. Found {linops}"
        )
        assert all(isequal(linop.oshape, linops[0].oshape) for linop in linops), (
            f"Add: All linops must share same oshape. Linops: {linops}"
        )
        # TODO: specialize the ishape and oshape on most specific one
        super().__init__(NS(linops[0].ishape, linops[0].oshape), stream=stream)
        self.linops = nn.ModuleList(linops)

    @staticmethod
    def fn(add, x: torch.Tensor, /):
        return sum(linop.fn(linop, x) for linop in add.linops)

    @staticmethod
    def adj_fn(add, x: torch.Tensor, /):
        return sum(linop.adj_fn(linop, x) for linop in add.linops)

    # @staticmethod
    # def normal_fn(add, x: torch.Tensor, /):
    #     # Note: Alternatively, make every possible combination of terms? Might be faster in some cases?
    #     return add.adj_fn(add, add.fn(add, x))

    def split_forward(self, ibatch, obatch):
        """ibatch, obatch specified according to the shape of the
        forward op
        """
        linops = [linop.split_forward(ibatch, obatch) for linop in self.linops]
        return type(self)(*linops)

    def split_forward_fn(self, ibatch, obatch, data_list):
        """Split data into batches
        ibatches, obatches specified according to the shape of the
        forward op
        """
        data = [
            linop.split_forward_fn(ibatch, obatch, data)
            for linop, data in zip(self.linops, data_list)
        ]
        return data

    def adjoint(self):
        return type(self)(*(linop.adjoint() for linop in self.linops))

    def size(self, dim):
        for linop in self.linops:
            out = linop.size(dim)
            if out is not None:
                return out
        return None

    def size_fn(self, dim, data):
        for linop, data in zip(self.linops, data):
            out = linop.size_fn(dim, data)
            if out is not None:
                return out
        return None

    @property
    def dims(self):
        return set().union(*[linop.dims for linop in self.linops])

    @property
    def H(self):
        """Adjoint operator"""
        if self._adj is None:
            linops = list(linop.adjoint() for linop in self.linops)
            _adj = type(self)(*linops)
            self._adj = [_adj]  # Prevent registration as a submodule
        return self._adj[0]

    @property
    def N(self):
        """Normal operator (is this really necessary?)"""
        if self._normal is None:
            linops = list(linop.normal() for linop in self.linops)
            _normal = type(self)(*linops)
            self._normal = [_normal]  # Prevent registration as a submodule
        return self._normal[0]

    def flatten(self):
        return [self]

    def __getitem__(self, idx):
        return self.linops[idx]

    def __len__(self):
        return len(self.linops)

    def __repr__(self):
        linop_chain = " + ".join(repr(linop) for linop in self.linops)
        return linop_chain
