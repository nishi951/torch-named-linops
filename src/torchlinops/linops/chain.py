import logging
from collections.abc import Mapping
from copy import copy
from typing import Optional

import torch
import torch.nn as nn

import torchlinops.config as config
from torchlinops.utils import INDENT

from ..nameddim import NamedDimension as ND, NamedShape as NS, isequal
from .device import ToDevice
from .namedlinop import NamedLinop

logger = logging.getLogger("torchlinops")


def _log_transfer(msg):
    if config.log_device_transfers:
        logger.info(msg)


class Chain(NamedLinop):
    """Composition (sequential application) of named linear operators.

    If ``Chain(A, B, C)`` is created, then the forward pass applies
    $A$ first, then $B$, then $C$: mathematically the operator is $C B A$.

    Attributes
    ----------
    linops : nn.ModuleList
        The constituent linops in **execution order** (inner to outer).
    """

    is_container = True

    def __init__(self, *linops, name: Optional[str] = None):
        """
        Parameters
        ----------
        *linops : NamedLinop
            Linops in order of execution. If ``linops = (A, B, C)``, the
            mathematical operator is $C B A$.
        name : str, optional
            Display name for this chain.
        """
        super().__init__(NS(linops[0].ishape, linops[-1].oshape), name=name)
        if len(linops) == 0:
            raise ValueError(f"Chain must contain at least one linop.")
        self.linops = nn.ModuleList(list(linops))
        
        if config.shape_inference:
            self._infer_shapes()
        
        self._check_inputs_outputs()

    @property
    def linops(self):
        return self._linops

    @linops.setter
    def linops(self, new_linops):
        self._linops = new_linops

    def __setattr__(self, name, value):
        """Bypasses pytorch's setattr, just for linops"""
        if name == "linops":
            # Force descriptor lookup for this name
            type(self).linops.fset(self, value)
        else:
            super().__setattr__(name, value)

    def _check_inputs_outputs(self):
        curr_shape = self.ishape
        for i, linop in enumerate(self.linops):
            if not isequal(linop.ishape, curr_shape)[0]:
                raise ValueError(
                    f"Mismatched shape: expected {linop.ishape}, got {curr_shape} at input to {linop}. Full stack: {self}, index {i}"
                )
            curr_shape = linop.oshape

    def _infer_shapes(self):
        """Propagate shapes forward through the chain, resolving wildcards."""
        from ..nameddim import resolve_wildcards
        
        for i in range(len(self.linops) - 1):
            curr = self.linops[i]
            next_linop = self.linops[i + 1]
            
            # Resolve wildcards in next_linop.ishape using curr.oshape
            resolved_ishape = resolve_wildcards(next_linop.ishape, curr.oshape)
            if resolved_ishape != next_linop.ishape:
                next_linop.ishape = resolved_ishape

    @staticmethod
    def fn(chain, x: torch.Tensor, context=None):
        x = chain[0](x, context)  # First linop inherits context
        for linop in chain.linops[1:]:
            x = linop(x)
        return x

    @staticmethod
    def adj_fn(chain, x: torch.Tensor, context=None):
        linops = list(reversed(chain.linops))
        x = linops[0].H(x, context)
        for linop in linops[1:]:
            x = linop.H(x)
        return x

    @staticmethod
    def normal_fn(chain, x: torch.Tensor, context=None):
        # fn does the reversing so it's unnecessary to do it here
        # If the normal hasn't been explicitly formed with`.N`, do things the naive way
        # Only the inner chain.fn inherits context
        return chain.adj_fn(chain, chain.fn(chain, x, context), context=None)

    @staticmethod
    def split(chain, tile: Mapping[ND | str, slice]):
        """Split a chain linop into sub-linops.

        Distributes the tile to each constituent linop based on dimension names.

        Parameters
        ----------
        chain : Chain
            The chain linop to split.
        tile : Mapping[ND | str, slice]
            Dictionary specifying how to slice the linop dimensions
        """
        split_linops = []
        for linop in chain.linops:
            sub_tile = {dim: tile.get(dim, slice(None)) for dim in linop.dims}
            split_linops.append(type(linop).split(linop, sub_tile))
        split = copy(chain)
        split.linops = nn.ModuleList(split_linops)
        return split

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
        adj = copy(self)
        adj.linops = nn.ModuleList(linops)
        return adj

    def normal(self, inner=None):
        """Compute the normal operator by folding through the chain.

        For a chain $C B A$, the normal is computed as
        $A^H (B^H (C^H C (B (A \\cdot))))$ by iterating ``linop.normal(inner)``
        in reverse order. This enables Toeplitz embedding and other per-linop
        normal optimizations to compose correctly.

        Parameters
        ----------
        inner : NamedLinop, optional
            An inner operator seeded from an outer chain or ``None``.

        Returns
        -------
        NamedLinop
            The composed normal operator.
        """
        for linop in reversed(self.linops):
            inner = linop.normal(inner)
        return inner

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

    def __copy__(self):
        return super().__copy__()
