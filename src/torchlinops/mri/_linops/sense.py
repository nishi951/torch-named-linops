from copy import copy, deepcopy
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
        # self.mps = mps
        self._shape.add("coildim", coildim)
        self.coil_ax = -(len(self.im_size) + 1)

    @property
    def coildim(self):
        return self._shape.lookup("coildim")

    def forward(self, x):
        return self.fn(self, x, self.mps)

    @staticmethod
    def fn(linop, x, /, mps):
        return x.unsqueeze(linop.coil_ax) * mps

    @staticmethod
    def adj_fn(linop, x, /, mps):
        return torch.sum(x * torch.conj(mps), dim=linop.coil_ax)

    def split_forward(self, ibatch, obatch):
        """Split over coil dim only"""
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.mps),
            self.coildim,
            self.ishape[: -self.D],
        )

    def split_forward_fn(self, ibatch, obatch, /, mps):
        for islc, oslc in zip(ibatch[-self.D :], obatch[-self.D :]):
            if islc != oslc:
                raise IndexError(
                    "SENSE currently only supports matched image input/output slicing."
                )
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
        return super().normal(inner)
