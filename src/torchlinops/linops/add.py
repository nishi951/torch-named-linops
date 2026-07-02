import logging
from copy import copy
from typing import Optional

import torch
import torch.nn as nn

import torchlinops.config as config

from ..nameddim import NamedShape as NS, isequal, max_shape, standardize_shapes
from .device import ToDevice
from .namedlinop import NamedLinop
from .schedule import parallel_execute

__all__ = ["Add"]

logger = logging.getLogger("torchlinops")


def _log_transfer(msg):
    if config.log_device_transfers:
        logger.info(msg)


class Add(NamedLinop):
    """The sum of one or more linear operators.

    When ``threaded=True`` (default), each sub-linop is executed in parallel
    using a ThreadPoolExecutor, which is useful for I/O-bound operations or
    operations that release the GIL (e.g., PyTorch tensor operations).

    Shared linops (e.g., ``Add(A, A)``) are used directly without copying —
    each thread receives the same linop object but independent execution context.

    Attributes
    ----------
    linops : nn.ModuleList
        The list of linops being added together.
    threaded : bool
        Whether to run sub-linops in parallel. Default is True.
    num_workers : int | None
        Number of worker threads. If None, defaults to the number of sub-linops.
    """

    is_container = True

    def __init__(
        self,
        *linops,
        threaded: bool = True,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        *linops : tuple[NamedLinop]
            The linear operators to be added together.
        threaded : bool, optional
            Whether to run sub-linops in parallel. Default is True.
        num_workers : int | None, optional
            Number of worker threads. If None, defaults to the number of sub-linops.
        """
        assert all(isequal(linop.ishape, linops[0].ishape) for linop in linops), (
            f"Add: All linops must share same ishape. Found {linops}"
        )
        assert all(isequal(linop.oshape, linops[0].oshape) for linop in linops), (
            f"Add: All linops must share same oshape. Linops: {linops}"
        )
        super().__init__(NS(linops[0].ishape, linops[0].oshape), **kwargs)
        self.threaded = threaded
        self.num_workers = num_workers
        self._linops = nn.ModuleList(linops)

    @property
    def linops(self):
        return self._linops

    @linops.setter
    def linops(self, new_linops):
        self._linops = new_linops

    def __setattr__(self, name, value):
        """Bypass PyTorch's setattr for linops."""
        if name == "linops":
            type(self).linops.fset(self, value)
        else:
            super().__setattr__(name, value)

    @staticmethod
    def fn(add, x: torch.Tensor, /, context=None):
        return parallel_execute(
            add.linops,
            [x] * len(add),
            context,
            reduce_fn=sum,
            threaded=add.threaded,
            num_workers=add.num_workers,
        )

    @staticmethod
    def adj_fn(add, x: torch.Tensor, /, context=None):
        return parallel_execute(
            [linop.H for linop in add.linops],
            [x] * len(add),
            context,
            reduce_fn=sum,
            threaded=add.threaded,
            num_workers=add.num_workers,
        )

    @staticmethod
    def split(add, tile):
        split = copy(add)
        linops = [type(linop).split(linop, tile) for linop in add.linops]
        split.linops = nn.ModuleList(linops)
        return split

    def adjoint(self):
        adj = copy(self)
        adj.linops = nn.ModuleList([linop.adjoint() for linop in self.linops])
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
        new.linops = nn.ModuleList(linops)
        return new

    def __len__(self):
        return len(self.linops)

    def __repr__(self):
        linop_chain = " + ".join(repr(linop) for linop in self.linops)
        return linop_chain
