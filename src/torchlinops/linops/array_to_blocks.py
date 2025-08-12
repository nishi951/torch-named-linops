from typing import Optional
from copy import copy
from torch import Tensor

from warnings import warn

import torch.nn as nn

import torchlinops.functional as F
from torchlinops.utils import default_to

from .namedlinop import NamedLinop
from .nameddim import ELLIPSES, NS, Shape

__all__ = ["ArrayToBlocks", "BlocksToArray"]


class ArrayToBlocks(NamedLinop):
    def __init__(
        self,
        grid_size: tuple[int, ...],
        block_size: tuple[int, ...],
        stride: tuple[int, ...],
        mask: Optional[Tensor] = None,
        batch_shape: Optional[Shape] = None,
        array_shape: Optional[Shape] = None,
        blocks_shape: Optional[Shape] = None,
    ):
        """

        mask : Tensor
        """
        self.grid_size = grid_size
        self.ndim = len(self.grid_size)
        self.block_size = block_size
        self.stride = stride

        self.batch_shape = default_to(("...",), batch_shape)
        self.array_shape = default_to(("...",), array_shape)
        self.blocks_shape = default_to(("...",), blocks_shape)
        shape = NS(self.batch_shape) + NS(self.array_shape, self.blocks_shape)
        super().__init__(shape)

        if mask is not None:
            self.mask = nn.Parameter(mask, requires_grad=False)
        else:
            self.mask = mask

    @staticmethod
    def fn(arraytoblocks, x, /):
        return F.array_to_blocks(
            x,
            arraytoblocks.block_size,
            arraytoblocks.stride,
            arraytoblocks.mask,
        )

    @staticmethod
    def adj_fn(arraytoblocks, x, /):
        return F.blocks_to_array(
            x,
            arraytoblocks.grid_size,
            arraytoblocks.block_size,
            arraytoblocks.stride,
            arraytoblocks.mask,
        )

    @staticmethod
    def normal_fn(arraytoblocks, x, /):
        return arraytoblocks.adj_fn(arraytoblocks, arraytoblocks.fn(arraytoblocks, x))

    def split_forward(self, ibatch, obatch):
        return copy(self)

    def adjoint(self):
        return BlocksToArray(
            self.grid_size,
            self.block_size,
            self.stride,
            self.mask,
            self.batch_shape,
            self.blocks_shape,
            self.array_shape,
        )

    def size(self, dim):
        return self.size_fn(dim)

    def size_fn(self, dim):
        ndim = len(self.grid_size)
        if dim in self.ishape[-ndim:]:
            i = self.ishape.index(dim) - len(self.ishape)
            return self.grid_size[i]
        return None


class BlocksToArray(NamedLinop):
    def __init__(
        self,
        grid_size: tuple[int, ...],
        block_size: tuple[int, ...],
        stride: tuple[int, ...],
        mask: Optional[Tensor] = None,
        batch_shape: Optional = None,
        blocks_shape: Optional = None,
        array_shape: Optional = None,
    ):
        self.grid_size = grid_size
        self.ndim = len(self.grid_size)
        self.block_size = block_size
        self.stride = stride

        self.batch_shape = default_to(("...",), batch_shape)
        self.blocks_shape = default_to(("...",), blocks_shape)
        self.array_shape = default_to(("...",), array_shape)
        shape = NS(self.batch_shape) + NS(self.blocks_shape, self.array_shape)
        super().__init__(shape)
        if mask is not None:
            self.mask = nn.Parameter(mask, requires_grad=False)
        else:
            self.mask = mask

    @staticmethod
    def fn(blockstoarray, x, /):
        return F.blocks_to_array(
            x,
            blockstoarray.grid_size,
            blockstoarray.block_size,
            blockstoarray.stride,
            blockstoarray.mask,
        )

    @staticmethod
    def adj_fn(blockstoarray, x, /):
        if x.shape[-blockstoarray.ndim :] != blockstoarray.grid_size:
            raise RuntimeError(
                f"BlocksToArray expected input with full size {blockstoarray.grid_size} but got {x.shape}"
            )
        return F.array_to_blocks(
            x,
            blockstoarray.block_size,
            blockstoarray.stride,
            blockstoarray.mask,
        )

    def split_forward(self, ibatch, obatch):
        return copy(self)

    def adjoint(self):
        return ArrayToBlocks(
            self.grid_size,
            self.block_size,
            self.stride,
            self.mask,
            self.batch_shape,
            self.array_shape,
            self.blocks_shape,
        )

    def size(self, dim):
        return self.size_fn(dim)

    def size_fn(self, dim):
        ndim = len(self.grid_size)
        if dim in self.oshape[-ndim:]:
            i = self.oshape.index(dim) - len(self.oshape)
            return self.grid_size[i]
        return None
