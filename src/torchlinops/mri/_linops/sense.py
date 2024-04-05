from copy import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchlinops._core._shapes import get2dor3d
from torchlinops._core._linops import NamedLinop, NS, Diagonal

__all__ = ["SENSE"]


class SENSE(NamedLinop):
    """
    input: [... Nx Ny [Nz]]
    output: [... C Nx Ny [Nz]]
    """

    def __init__(
        self,
        mps: torch.Tensor,
        coildim: str = "C",
        in_batch_shape: Optional[Tuple] = None,
    ):
        self.im_size = mps.shape[1:]
        shape = (
            NS(in_batch_shape) + NS(tuple(), (coildim,)) + NS(get2dor3d(self.im_size))
        )
        super().__init__(shape)
        self.D = len(self.im_size)
        self.mps = nn.Parameter(mps, requires_grad=False)
        self.coildim = coildim
        self.coil_ax = -(len(self.im_size) + 1)

    def forward(self, x):
        return self.fn(x, self.mps)

    def fn(self, x, /, mps):
        return x.unsqueeze(self.coil_ax) * mps

    def adj_fn(self, x, /, mps):
        return torch.sum(x * torch.conj(mps), dim=self.coil_ax)

    def split_forward(self, ibatch, obatch):
        """Split over coil dim only"""
        for islc, oslc in zip(ibatch[-self.D :], obatch[-self.D :]):
            if islc != oslc:
                raise IndexError(
                    "SENSE currently only supports matched image input/output slicing."
                )
        split = copy(self)
        split.mps.data = self.split_forward_fn(ibatch, obatch, self.mps)
        return split

    def split_forward_fn(self, ibatch, obatch, /, mps):
        return mps[obatch[self.coil_ax :]]

    def size(self, dim: str):
        return self.size_fn(dim, self.mps)

    def size_fn(self, dim: str, mps):
        forward_oshape = (self.coildim,) + get2dor3d(self.im_size)
        mps_shape = forward_oshape[self.coil_ax :]
        if dim in mps_shape:
            return mps.shape[mps_shape.index(dim)]
        return None

    def normal(self, inner=None):
        if inner is None:
            abs_mps = torch.sum(torch.abs(self.mps) ** 2, dim=0)
            normal = Diagonal(abs_mps, get2dor3d(self.im_size))
            return normal
        pre = copy(self)
        pre.oshape = inner.ishape
        post = copy(self).H
        post.ishape = inner.oshape
        return post @ inner @ pre
