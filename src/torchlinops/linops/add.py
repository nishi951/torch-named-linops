import torch
import torch.nn as nn
import logging

import torchlinops.config as config
from ..nameddim import NamedShape as NS, isequal
from .namedlinop import NamedLinop
from .device import ToDevice

__all__ = ["Add"]

logger = logging.getLogger("torchlinops")


def _log_transfer(msg):
    if config.log_device_transfers:
        logger.info(msg)


class Add(NamedLinop):
    """The sum of one or more linear operators.

    Attributes
    ----------
    linops : nn.ModuleList
        The list of linops being added together.
    """

    def __init__(self, *linops, **kwargs):
        """
        Parameters
        ----------
        *linops : tuple[NamedLinop]
            The linear operators to be added together.
        """
        assert all(isequal(linop.ishape, linops[0].ishape) for linop in linops), (
            f"Add: All linops must share same ishape. Found {linops}"
        )
        assert all(isequal(linop.oshape, linops[0].oshape) for linop in linops), (
            f"Add: All linops must share same oshape. Linops: {linops}"
        )
        # TODO: specialize the ishape and oshape on most specific one
        super().__init__(NS(linops[0].ishape, linops[0].oshape), **kwargs)
        self.linops = nn.ModuleList(linops)

        self._setup_events()

    def _setup_events(self):
        """Organize ToDevice to trigger on start of linop"""
        for linop in self.linops:
            linop.input_listener = (self, "input_listener")

    @staticmethod
    def fn(add, x: torch.Tensor, /):
        return sum(linop(x) for linop in add.linops)

    @staticmethod
    def adj_fn(add, x: torch.Tensor, /):
        return sum(linop.H(x) for linop in add.linops)

    def split_forward(self, ibatch, obatch):
        linops = [linop.split_forward(ibatch, obatch) for linop in self.linops]
        return type(self)(*linops)

    def adjoint(self):
        return type(self)(*(linop.adjoint() for linop in self.linops))

    def size(self, dim):
        for linop in self.linops:
            out = linop.size(dim)
            if out is not None:
                return out
        return None

    @property
    def dims(self):
        return set().union(*[linop.dims for linop in self.linops])

    @property
    def H(self):
        if config.cache_adjoint_normal:
            config._warn_if_caching_enabled()
            if self._adjoint is None:
                linops = list(linop.adjoint() for linop in self.linops)
                _adj = type(self)(*linops)
                self._adjoint = [_adj]
            return self._adjoint[0]
        return self.adjoint()

    @property
    def N(self):
        if config.cache_adjoint_normal:
            config._warn_if_caching_enabled()
            if self._normal is None:
                linops = list(linop.normal() for linop in self.linops)
                _normal = type(self)(*linops)
                self._normal = [_normal]
            return self._normal[0]
        return self.normal()

    def flatten(self):
        return [self]

    def __getitem__(self, idx):
        return self.linops[idx]

    def __len__(self):
        return len(self.linops)

    def __repr__(self):
        linop_chain = " + ".join(repr(linop) for linop in self.linops)
        return linop_chain
