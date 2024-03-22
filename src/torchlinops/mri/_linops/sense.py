from copy import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchlinops._core._shapes import get2dor3d
from torchlinops._core._linops import NamedLinop

__all__ = ["SENSE"]


class SENSE(NamedLinop):
    def __init__(
        self,
        mps: torch.Tensor,
        coil_str: str = "C",
        in_batch_shape: Optional[Tuple] = None,
    ):
        self.im_size = mps.shape[1:]
        self.D = len(self.im_size)
        self.coildim = -(self.D + 1)
        self.in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        self.out_batch_shape = self.in_batch_shape + (coil_str,)
        ishape = self.in_batch_shape + get2dor3d(self.im_size)
        oshape = self.out_batch_shape + get2dor3d(self.im_size)
        super().__init__(ishape, oshape)
        self.coil_str = coil_str
        self.mps = nn.Parameter(mps, requires_grad=False)

    def forward(self, x):
        return self.fn(x, self.mps)

    def fn(self, x, /, mps):
        return x.unsqueeze(self.coildim) * mps

    def adj_fn(self, x, /, mps):
        return torch.sum(x * torch.conj(mps), dim=self.coildim)

    def split_forward(self, ibatch, obatch):
        """Split over coil dim only"""
        for islc, oslc in zip(ibatch[-self.D :], obatch[-self.D :]):
            if islc != oslc:
                raise IndexError(
                    "SENSE currently only supports matched image input/output slicing."
                )
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.mps),
            coil_str=self.coil_str,
            in_batch_shape=self.in_batch_shape,
        )

    def split_forward_fn(self, ibatch, obatch, /, mps):
        return mps[obatch[self.coildim :]]

    def size(self, dim: str):
        return self.size_fn(dim, self.mps)

    def size_fn(self, dim: str, mps):
        mps_shape = self.oshape[self.coildim :]
        if dim in mps_shape:
            return mps.shape[mps_shape.index(dim)]
        return None

    def normal(self, inner=None):
        if inner is None:
            return super().normal(inner)
        return copy(self).H @ inner @ copy(self)
