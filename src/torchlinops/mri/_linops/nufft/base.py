from typing import Optional, Tuple, Mapping, Iterable
from copy import copy

import torch
import torch.nn as nn

from torchlinops._core._linops import NamedLinop, ND, NamedComboShape
from torchlinops._core._shapes import get2dor3d

from .toeplitz import toeplitz


class NamedNufftShape(NamedComboShape):
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
        in_batch_shape: Optional[Tuple] = None,
        out_batch_shape: Optional[Tuple] = None,
        shared_batch_shape: Optional[Tuple] = None,
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
        ishape, oshape = self.setup_shapes(
            in_batch_shape, out_batch_shape, shared_batch_shape, im_size
        )
        super().__init__(ishape, oshape)
        self.trj = nn.Parameter(trj, requires_grad=False)
        self.im_size = im_size
        self.extras = extras if extras is not None else {}
        self.toeplitz = toeplitz
        self.toeplitz_oversamp = toeplitz_oversamp

        # Precompute
        self.D = len(im_size)

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
        pre = copy(self)
        post = copy(self)
        # Don't modify post.oshape
        if inner is None:
            return post @ pre
        nS = self.shared_dims
        nK = len(self.out_batch_shape)
        # inner = [S... N... K...] -> [S'... N'... K'...]
        # inner.ishape[:nS] = S
        # inner.ishape[nS:-nK] = N
        # inner.ishape[-nK:] = K
        pre.shared_batch_shape = inner.ishape[:nS]  # S
        pre.out_batch_shape = inner.ishape[-nK:]  # K
        pre.oshape = ND.infer(
            pre.get_oshape(
                pre.in_batch_shape,
                pre.out_batch_shape,
                pre.shared_batch_shape,
                pre.im_size,
            )
        )

        post.shared_batch_shape = inner.oshape[:nS]  # S
        post.in_batch_shape = inner.oshape[nS:-nK]  # N
        post.ishape = ND.infer(
            post.get_ishape(
                post.in_batch_shape,
                post.out_batch_shape,
                post.shared_batch_shape,
                post.im_size,
            )
        )
        post.oshape = ND.infer(
            post.get_oshape(
                post.in_batch_shape,
                post.out_batch_shape,
                post.shared_batch_shape,
                post.im_size,
            )
        )

        return post.H @ inner @ pre

    @staticmethod
    def get_ishape(in_batch_shape, out_batch_shape, shared_batch_shape, im_size):
        return shared_batch_shape + in_batch_shape + get2dor3d(im_size)

    @staticmethod
    def get_oshape(in_batch_shape, out_batch_shape, shared_batch_shape, im_size):
        return shared_batch_shape + in_batch_shape + out_batch_shape

    def setup_shapes(
        self, in_batch_shape, out_batch_shape, shared_batch_shape, im_size
    ):
        self.in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        self.out_batch_shape = (
            out_batch_shape if out_batch_shape is not None else tuple()
        )
        self.shared_batch_shape = (
            shared_batch_shape if shared_batch_shape is not None else tuple()
        )
        self.shared_dims = len(self.shared_batch_shape)
        ishape = self.get_ishape(
            self.in_batch_shape, self.out_batch_shape, self.shared_batch_shape, im_size
        )
        oshape = self.get_oshape(
            self.in_batch_shape, self.out_batch_shape, self.shared_batch_shape, im_size
        )
        return ishape, oshape

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
