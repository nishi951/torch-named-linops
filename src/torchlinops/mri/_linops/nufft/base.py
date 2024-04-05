from typing import Optional, Tuple, Mapping, Iterable
from copy import copy

import torch
import torch.nn as nn

from torchlinops._core._linops import NamedLinop, ND, NS, NamedShape
from torchlinops._core._shapes import get2dor3d

from .toeplitz import toeplitz


class NamedNufftShape(NamedShape):
    def __init__(
        self,
        shared_shape: Iterable,
        batch_shape: Iterable,
        img_shape: Iterable,
        ksp_shape: Iterable,
    ):
        shared_shape = list(ND.infer(shared_batch_shape))
        batch_shape = list(ND.infer(in_batch_shape))
        im_shape = list(ND.infer(get2dor2d(im_size)))
        ksp_shape = list(ND.infer(out_batch_shape))
        diag_ishape = shared_shape + batch_shape
        dense_ishape = img_shape
        dense_oshape = ksp_shape
        super().__init__(diag_ishape, dense_ishape, dense_oshape)


class NUFFTBase(NamedLinop):
    def __init__(
        self,
        trj: torch.Tensor,
        im_size: Tuple,
        shared_batch_shape: Optional[Tuple] = None,
        in_batch_shape: Optional[Tuple] = None,
        out_batch_shape: Optional[Tuple] = None,
        extras: Optional[Mapping] = None,
        toeplitz: bool = False,
        toeplitz_oversamp: float = 2.0,
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
        extras : Mapping
            Implementation-specific Optional extra stuff
        toeplitz : bool
            Whether or not to use toeplitz embedding for the normal operator
        toeplitz_oversamp: float
            Oversampling factor for toeplitz embedding. Defaults to 2x
        """

        shape = (
            NS(shared_batch_shape)
            + NS(in_batch_shape)
            + NS(get2dor3d(im_size), out_batch_shape)
        )
        # ishape, oshape = self.setup_shapes(
        #     in_batch_shape, out_batch_shape, shared_batch_shape, im_size
        # )
        super().__init__(shape)
        self.trj = nn.Parameter(trj, requires_grad=False)
        self.im_size = im_size
        self.extras = extras if extras is not None else {}
        self.toeplitz = toeplitz
        self.toeplitz_oversamp = toeplitz_oversamp

        # Precompute
        shared_batch_shape = (
            shared_batch_shape if shared_batch_shape is not None else tuple()
        )
        self.shared_dims = len(shared_batch_shape)
        self.D = len(im_size)
        self.im_shape = get2dor3d(im_size)
        self.shared_batch_shape = self.ishape[: self.shared_dims]
        self.in_batch_shape = self.ishape[self.shared_dims : -self.D]
        self.out_batch_shape = self.oshape[
            self.shared_dims + len(self.in_batch_shape) :
        ]

    def forward(self):
        raise NotImplementedError(f"{type(self).__name__} cannot be used directly")

    def change_im_size(self, new_im_size):
        # Necessary for sigpy scaling
        for i in range(self.trj.shape[-1]):
            self.trj[..., i] *= new_im_size[i] / self.im_size[i]
        self.im_size = new_im_size
        return self

    def normal(self, inner=None):
        if self.toeplitz:
            T = toeplitz(self, inner, self.toeplitz_oversamp, self.trj.device)
            return T
        # Don't modify post.oshape
        if inner is None:
            pre = copy(self)
            post = copy(self).H
            return post @ pre

        pre = copy(self)
        pre.oshape = inner.ishape
        # Fully overwrite post's shape
        post = copy(self).H
        post.ishape = inner.oshape
        return post @ inner @ pre

    def split_forward(self, ibatch, obatch):
        split = copy(self)
        split.trj.data = self.split_forward_fn(ibatch, obatch, self.trj)
        return split

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
