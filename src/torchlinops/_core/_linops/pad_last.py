from copy import copy

import torch.nn.functional as F

from .namedlinop import NamedLinop
from . import Identity
from .nameddim import NamedDimension as ND, get2dor3d, NS


__all__ = ["PadLast"]


class PadLast(NamedLinop):
    """Pad the last dimensions of the input volume
    ishape: [B... Nx Ny [Nz]]
    oshape: [B... Nx1 Ny1 [Nz1]]

    """

    def __init__(self, pad_im_size, im_size, batch_shape):
        assert len(pad_im_size) == len(im_size), f'Padded and unpadded dims should be the same length. padded: {pad_im_size} unpadded: {im_size}'

        im_shape = ND.infer(get2dor3d(im_size))
        pad_im_shape = tuple(d.next_unused(im_shape) for d in im_shape)
        shape = NS(batch_shape) + NS(im_shape, pad_im_shape)
        super().__init__(shape)
        self.D = len(im_size)
        self.im_size = tuple(im_size)
        self.pad_im_size = tuple(pad_im_size)
        self.in_im_size = tuple(im_size)
        self.out_im_size = tuple(pad_im_size)
        for psz in pad_im_size:
            assert not (psz % 2), "Pad sizes must be even"

        sizes = [[(psz - isz) // 2] * 2 for psz, isz in zip(self.out_im_size, self.in_im_size)]
        self.pad = sum(sizes, start=[])
        self.pad.reverse()

        self.crop_slice = [
            slice(self.pad[2 * i], -self.pad[2 * i + 1])
            for i in range(len(self.pad) // 2)
        ]
        self.crop_slice.reverse()

    def forward(self, x):
        """Pad the last n dimensions of x"""
        return self.fn(x)

    def fn(self, x, /):
        assert tuple(x.shape[-self.D :]) == self.in_im_size
        pad = self.pad + [0, 0] * (x.ndim - self.D)
        return F.pad(x, pad)

    def adj_fn(self, y, /):
        """Crop the last n dimensions of y"""
        assert tuple(y.shape[-self.D :]) == self.out_im_size
        slc = [slice(None)] * (y.ndim - self.D) + self.crop_slice
        return y[slc]

    def adjoint(self):
        adj = super().adjoint()
        adj.in_im_size, adj.out_im_size = self.out_im_size, self.in_im_size
        return adj

    def normal(self, inner=None):
        if inner is None:
            # Adjoint is exactly the inverse
            return Identity(self.ishape)
        return copy(self).H @ inner @ copy(self)

    def split_forward(self, ibatch, obatch):
        for islc, oslc in zip(ibatch[-self.D :], obatch[-self.D :]):
            raise ValueError(f"{type(self).__name__} cannot be split along image dim")
        return self

    def split_forward_fn(self, ibatch, obatch, /):
        for islc, oslc in zip(ibatch[-self.D :], obatch[-self.D :]):
            raise ValueError(f"{type(self).__name__} cannot be split along image dim")
        return None

    def size(self, dim: str):
        return self.size_fn(dim)

    def size_fn(self, dim: str, /):
        if dim in self.ishape[-self.D:]:
            return self.in_im_size[self.im_shape.index(dim)]
        elif dim in self.oshape[-self.D:]:
            return self.out_im_size[self.pad_im_shape.index(dim)]
        return None
