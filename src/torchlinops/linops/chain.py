from collections.abc import Mapping
from typing import Optional

import torch
import torch.nn as nn

from torchlinops.utils import INDENT

from ..nameddim import NamedDimension as ND, NamedShape as NS, isequal
from .namedlinop import NamedLinop


class Chain(NamedLinop):
    """A sequence or composition of linops"""

    def __init__(self, *linops, name: Optional[str] = None):
        """
        Parameters
        ----------
        *linops : list
            Linops in order of execution
            i.e. if `linops = [A, B, C]`, then mathematically, the linop in question is `CBA`

        """
        super().__init__(NS(linops[0].ishape, linops[-1].oshape), name=name)
        self.linops = nn.ModuleList(list(linops))
        self._check_inputs_outputs()

    def _check_inputs_outputs(self):
        curr_shape = self.ishape
        for i, linop in enumerate(self.linops):
            if not isequal(linop.ishape, curr_shape):
                raise ValueError(
                    f"Mismatched shape: expected {linop.ishape}, got {curr_shape} at input to {linop}. Full stack: {self}, index {i}"
                )
            curr_shape = linop.oshape

    @staticmethod
    def fn(chain, x: torch.Tensor, /):
        for linop in chain.linops:
            x = linop(x)
        return x

    @staticmethod
    def adj_fn(chain, x: torch.Tensor, /):
        for linop in reversed(chain.linops):
            x = linop.H(x)
        return x

    # @staticmethod
    # def normal_fn(chain, x: torch.Tensor):
    #     # fn does the reversing so it's unnecessary to do it here
    #     # If the normal hasn't been explicitly formed with`.N`, do things the naive way
    #     return chain.adj_fn(chain, chain.fn(chain, x))

    def split_forward(self, ibatches, obatches):
        """ibatches, obatches specified according to the shape of the
        forward op
        """
        linops = [
            linop.split_forward(ibatch, obatch)
            for linop, ibatch, obatch in zip(self.linops, ibatches, obatches)
        ]
        return type(self)(*linops, name=self._name)

    def size(self, dim):
        out = None
        for linop in self.linops:
            tmp = linop.size(dim)
            if tmp is not None:
                if out is None:
                    out = tmp
                elif out != tmp:
                    raise ValueError(
                        f"Conflicting linop sizes found: {out} and {tmp} for dim {dim} in linop {linop} out of all linops {self.linops}"
                    )
        return out

    @property
    def dims(self):
        """Get the dims that appear anywhere in this linop chain."""
        return set().union(*[linop.dims for linop in self.linops])

    def adjoint(self):
        linops = list(linop.adjoint() for linop in reversed(self.linops))
        return type(self)(*linops, name=self._name)

    def normal(self, inner=None):
        for linop in reversed(self.linops):
            inner = linop.normal(inner)
        return inner

    @staticmethod
    def split(chain, tile: Mapping[ND | str, slice]):
        """Split a linop into sub-linops.

        Parameters
        ----------
        chain : Chain
            The chain linop to split.
        tile : Mapping[ND | str, slice]
            Dictionary specifying how to slice the linop dimensions
        """
        ibatches = [
            [tile.get(dim, slice(None)) for dim in linop.ishape]
            for linop in chain.linops
        ]
        obatches = [
            [tile.get(dim, slice(None)) for dim in linop.oshape]
            for linop in chain.linops
        ]
        return chain.split_forward(ibatches, obatches)

    @staticmethod
    def adj_split(chain, tile: Mapping[ND | str, slice]):
        """Split an adjoint linop into sub-linops.

        Parameters
        ----------
        chain : Chain
            The chain linop to split.
        tile : Mapping[ND | str, slice]
            Dictionary specifying how to slice the linop dimensions
        """
        ibatches = [
            [tile.get(dim, slice(None)) for dim in linop.ishape]
            for linop in chain.linops
        ]
        obatches = [
            [tile.get(dim, slice(None)) for dim in linop.oshape]
            for linop in chain.linops
        ]
        return chain.H.split_forward(obatches, ibatches).H

    @property
    def shape(self):
        return NS(self.linops[0].ishape, self.linops[-1].oshape)

    @shape.setter
    def shape(self, val):
        self.ishape = val.ishape
        self.oshape = val.oshape

    @property
    def ishape(self):
        return self.linops[0].ishape

    @ishape.setter
    def ishape(self, val):
        self.linops[0].ishape = val

    @property
    def oshape(self):
        return self.linops[-1].oshape

    @oshape.setter
    def oshape(self, val):
        self.linops[-1].oshape = val

    def flatten(self):
        return list(self.linops)

    def __getitem__(self, idx):
        linops = self.linops[idx]
        if isinstance(linops, NamedLinop):
            return linops
        return Chain(*linops, name=self._name)

    def __len__(self):
        return len(self.linops)

    def __repr__(self):
        output = ""
        output += INDENT.indent(self.repr_name + "(\n")
        with INDENT:
            for linop in self.linops:
                output += repr(linop) + "\n"
        output += INDENT.indent(")")
        return output
