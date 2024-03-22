from copy import copy

import torch.nn.functional as F

from .namedlinop import NamedLinop
from . import Identity
from .nameddim import NamedDimension as ND, get2dor3d


__all__ = ["PadLast"]


class PadLast(NamedLinop):
    """Pad the last dimensions of the input volume
    ishape: [B... Nx Ny [Nz]]
    oshape: [B... Nx1 Ny1 [Nz1]]

    """

    def __init__(self, pad_im_size, im_size, batch_shape):
        batch_shape = batch_shape if batch_shape is not None else tuple()
        im_shape = ND.from_tuple(get2dor3d(im_size))
        pad_im_shape = ND.from_tuple(get2dor3d(im_size))
        pad_im_shape = tuple(d.next_unused(im_shape) for d in pad_im_shape)
        ishape = batch_shape + im_shape
        oshape = batch_shape + pad_im_shape
        self.im_shape = im_shape
        self.pad_im_shape = pad_im_shape
        super().__init__(ishape, oshape)

        assert len(pad_im_size) == len(im_size)
        self.im_dim = len(im_size)
        self.im_size = tuple(im_size)
        self.pad_im_size = tuple(pad_im_size)
        for psz in pad_im_size:
            assert not (psz % 2), "Pad sizes must be even"

        sizes = [[(psz - isz) // 2] * 2 for psz, isz in zip(pad_im_size, im_size)]
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
        assert tuple(x.shape[-self.im_dim :]) == self.im_size
        pad = self.pad + [0, 0] * (x.ndim - self.im_dim)
        return F.pad(x, pad)

    def adj_fn(self, y, /):
        """Crop the last n dimensions of y"""
        assert tuple(y.shape[-self.im_dim :]) == self.pad_im_size
        slc = [slice(None)] * (y.ndim - self.im_dim) + self.crop_slice
        return y[slc]

    def normal(self, inner=None):
        if inner is None:
            # Adjoint is exactly the inverse
            return Identity(self.ishape)
        return copy(self).H @ inner @ copy(self)

    def split_forward(self, ibatch, obatch):
        for islc, oslc in zip(ibatch[-self.im_dim :], obatch[-self.im_dim :]):
            raise ValueError(f"{type(self).__name__} cannot be split along image dim")
        return self

    def split_forward_fn(self, ibatch, obatch, /):
        for islc, oslc in zip(ibatch[-self.im_dim :], obatch[-self.im_dim :]):
            raise ValueError(f"{type(self).__name__} cannot be split along image dim")
        return None

    def size(self, dim: str):
        return self.size_fn(dim)

    def size_fn(self, dim: str, /):
        if dim in self.im_shape:
            return self.im_size[self.im_shape.index(dim)]
        elif dim in self.pad_im_shape:
            return self.pad_im_size[self.pad_im_shape.index(dim)]
        return None
