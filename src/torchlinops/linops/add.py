import logging
from copy import copy

import torch
import torch.nn as nn

import torchlinops.config as config

from ..nameddim import NamedShape as NS, isequal, max_shape, standardize_shapes
from .device import ToDevice
from .namedlinop import NamedLinop
from .threadable import Threadable

__all__ = ["Add"]

logger = logging.getLogger("torchlinops")


def _log_transfer(msg):
    if config.log_device_transfers:
        logger.info(msg)


class Add(Threadable, NamedLinop):
    """The sum of one or more linear operators.

    Attributes
    ----------
    linops : nn.ModuleList
        The list of linops being added together.
    threaded : bool
        Whether to run sub-linops in parallel. Default is True.
    num_workers : int | None
        Number of worker threads. If None, defaults to number of sub-linops.
    """

    def __init__(self, *linops):
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
        super().__init__(NS(linops[0].ishape, linops[0].oshape))
        self.linops = nn.ModuleList(linops)

    @staticmethod
    def fn(add, x: torch.Tensor, /):
        if add.threaded:
            return add.threaded_apply_sum_reduce([x] * len(add.linops), add.num_workers)
        return sum(linop(x) for linop in add.linops)

    @staticmethod
    def adj_fn(add, x: torch.Tensor, /):
        if add.threaded:
            adj_linops = [linop.H for linop in add.linops]
            return add.threaded_apply_sum_reduce([x] * len(adj_linops), add.num_workers)
        return sum(linop.H(x) for linop in add.linops)

    def split_forward(self, ibatch, obatch):
        split = copy(self)
        linops = [linop.split_forward(ibatch, obatch) for linop in self.linops]
        split._linops = nn.ModuleList(linops)
        return split

    def adjoint(self):
        adj = copy(self)
        adj._linops = nn.ModuleList([linop.adjoint() for linop in self.linops])
        adj.shape = self.shape.adjoint()
        return adj

    def normal(self, inner=None):
        if inner is None:
            max_ishape = max_shape([linop.N.ishape for linop in self.linops])
            max_oshape = max_shape([linop.N.oshape for linop in self.linops])
            new_shape = NS(max_ishape, max_oshape)
            all_combinations = []
            for left_linop in self.linops:
                for right_linop in self.linops:
                    if left_linop == right_linop:
                        all_combinations.append(left_linop.N)
                    else:
                        all_combinations.append(left_linop.H @ right_linop)
            all_combinations = standardize_shapes(all_combinations, new_shape)
            normal = copy(self)
            normal.linops = nn.ModuleList(list(all_combinations))
            normal.shape = normal.linops[0].shape
            return normal
        return super().normal(inner)

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
        try:
            if config.cache_adjoint_normal:
                config._warn_if_caching_enabled()
                if self._adjoint is None:
                    self._adjoint = [self.adjoint()]
                return self._adjoint[0]
            return self.adjoint()
        except AttributeError as e:
            raise RuntimeError(f"AttributeError in {type(self).__name__}.H: {e}") from e

    @property
    def N(self):
        try:
            if config.cache_adjoint_normal:
                config._warn_if_caching_enabled()
                if self._normal is None:
                    self._normal = [self.normal()]
                return self._normal[0]
            return self.normal()
        except AttributeError as e:
            raise RuntimeError(f"AttributeError in {type(self).__name__}.N: {e}") from e

    def flatten(self):
        return [self]

    def __getitem__(self, idx):
        linops = self.linops[idx]
        if isinstance(linops, NamedLinop):
            return linops
        new = copy(self)
        new._linops = nn.ModuleList(linops)
        return new

    def __len__(self):
        return len(self.linops)

    def __repr__(self):
        linop_chain = " + ".join(repr(linop) for linop in self.linops)
        return linop_chain
