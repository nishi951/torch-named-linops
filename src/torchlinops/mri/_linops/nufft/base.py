from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchlinops._core._linops import NamedLinop
from torchlinops._core._shapes import get2dor3d


class NUFFTBase(NamedLinop):
    def __init__(
        self,
        trj: torch.Tensor,
        im_size: Tuple,
        in_batch_shape: Optional[Tuple] = None,
        out_batch_shape: Optional[Tuple] = None,
        shared_batch_shape: Optional[Tuple] = None,
    ):
        """
        img (input) [S... N... Nx Ny [Nz]]
        trj: [S... K..., D] in sigpy style [-N/2, N/2]
        in_batch_shape : Tuple
            The shape of [N...] in img
        out_batch_shape : Tuple
            The shape of [K...] in trj.
        shared_batch_shape : Tuple
            The shape of [S...] in trj

        """
        self.in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        self.out_batch_shape = (
            out_batch_shape if out_batch_shape is not None else tuple()
        )
        self.shared_batch_shape = (
            shared_batch_shape if shared_batch_shape is not None else tuple()
        )
        self.shared_dims = len(self.shared_batch_shape)
        ishape = self.shared_batch_shape + self.in_batch_shape + get2dor3d(im_size)
        oshape = self.shared_batch_shape + self.in_batch_shape + self.out_batch_shape
        super().__init__(ishape, oshape)
        self.trj = nn.Parameter(trj, requires_grad=False)
        self.im_size = im_size

        # Precompute
        self.D = len(im_size)

    def split_forward(self, ibatch, obatch):
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.trj),
            im_size=self.im_size,
            in_batch_shape=self.in_batch_shape,
            out_batch_shape=self.out_batch_shape,
            shared_batch_shape=self.shared_batch_shape,
        )

    def split_forward_fn(self, ibatch, obatch, /, trj):
        shared_batch = obatch[: self.shared_dims]
        kbatch = obatch[self.shared_dims + len(self.in_batch_shape) :]
        trj_slc = tuple(shared_batch + kbatch + [slice(None)])
        # trj_slc = obatch[:-1] + [slice(None)] + obatch[-1:]
        return trj[trj_slc]

    def size(self, dim: str):
        return self.size_fn(dim, self.trj)

    def size_fn(self, dim: str, trj):
        if dim in self.shared_batch_shape:
            idx = self.shared_batch_shape.index(dim)
        elif dim in self.out_batch_shape:
            idx = len(self.shared_batch_shape) + self.out_batch_shape.index(dim)
        else:
            return None
        return trj.shape[idx]
