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
        im_size: tuple[int, ...],
        block_size: tuple[int, ...],
        stride: tuple[int, ...],
        mask: Optional[Tensor] = None,
        batch_shape: Optional[Shape] = None,
        array_shape: Optional[Shape] = None,
        blocks_shape: Optional[Shape] = None,
    ):
        self.im_size = im_size
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

    def forward(self, x):
        return self.fn(self, x)

    @staticmethod
    def fn(linop, x):
        return F.array_to_blocks(
            x,
            linop.block_size,
            linop.stride,
            linop.mask,
        )

    @staticmethod
    def adj_fn(linop, x):
        return F.blocks_to_array(
            x,
            linop.im_size,
            linop.block_size,
            linop.stride,
            linop.mask,
        )

    @staticmethod
    def normal_fn(linop, x):
        return linop.adj_fn(linop, linop.fn(linop, x))

    def split_forward(self, ibatch, obatch):
        return copy(self)

    def adjoint(self):
        return BlocksToArray(
            self.im_size,
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
        ndim = len(self.im_size)
        if dim in self.ishape[-ndim:]:
            i = self.ishape.index(dim) - len(self.ishape)
            return self.im_size[i]
        return None


class BlocksToArray(NamedLinop):
    def __init__(
        self,
        im_size: tuple[int, ...],
        block_size: tuple[int, ...],
        stride: tuple[int, ...],
        mask: Optional[Tensor] = None,
        batch_shape: Optional = None,
        blocks_shape: Optional = None,
        array_shape: Optional = None,
    ):
        self.im_size = im_size
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

    def forward(self, x):
        return self.fn(self, x)

    @staticmethod
    def fn(linop, x):
        return F.blocks_to_array(
            x,
            linop.im_size,
            linop.block_size,
            linop.stride,
            linop.mask,
        )

    @staticmethod
    def adj_fn(linop, x):
        return F.array_to_blocks(
            x,
            linop.block_size,
            linop.stride,
            linop.mask,
        )

    @staticmethod
    def normal_fn(linop, x):
        return linop.adj_fn(linop, linop.fn(linop, x))

    def split_forward(self, ibatch, obatch):
        return copy(self)

    def adjoint(self):
        return ArrayToBlocks(
            self.im_size,
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
        ndim = len(self.im_size)
        if dim in self.oshape[-ndim:]:
            i = self.oshape.index(dim) - len(self.oshape)
            return self.im_size[i]
        return None
