from typing import Optional
from copy import copy, deepcopy

import torch.nn.functional as F

from .namedlinop import NamedLinop
from .identity import Identity
from .nameddim import ND, get_nd_shape, NS, NDorStr, Shape
from torchlinops.utils import default_to


__all__ = ["PadLast"]


class PadLast(NamedLinop):
    """Zero Pad the last dimensions of the input volume
    Padding is centered:
    - TODO: support non-centered padding?
    ishape: [B... Nx Ny [Nz]]
    oshape: [B... Nx1 Ny1 [Nz1]]

    """

    def __init__(
        self,
        pad_im_size: tuple[int, ...],
        im_size: tuple[int, ...],
        in_shape: Optional[Shape] = None,
        out_shape: Optional[Shape] = None,
        batch_shape: Optional[Shape] = None,
    ):
        assert len(pad_im_size) == len(im_size), (
            f"Padded and unpadded dims should be the same length. padded: {pad_im_size} unpadded: {im_size}"
        )

        if in_shape is None:
            self.in_im_shape = ND.infer(get_nd_shape(im_size))
        else:
            self.in_im_shape = ND.infer(in_shape)
        if out_shape is None:
            self.out_im_shape = tuple(
                d.next_unused(self.in_im_shape) for d in self.in_im_shape
            )
        else:
            self.out_im_shape = out_shape
        batch_shape = default_to(("...",), batch_shape)

        shape = NS(batch_shape) + NS(self.in_im_shape, self.out_im_shape)
        super().__init__(shape)
        self.D = len(im_size)
        self.im_size = tuple(im_size)
        self.pad_im_size = tuple(pad_im_size)
        self.in_im_size = tuple(im_size)
        self.out_im_size = tuple(pad_im_size)
        # for psz in pad_im_size:
        #     assert not (psz % 2), "Pad sizes must be even"

        # sizes = [
        #     [(psz - isz) // 2] * 2
        #     for psz, isz in zip(self.out_im_size, self.in_im_size)
        # ]
        # self.pad = sum(sizes, start=[])
        # self.pad.reverse()

        self.pad = pad_to_size(self.im_size, self.pad_im_size)

        # Make crop slice that undoes padding
        # Need to reverse crop_slice because padding is reversed
        self.crop_slice = []
        for i in range(len(self.pad) // 2):
            start = self.pad[2 * i]
            stop = -self.pad[2 * i + 1]
            if stop == 0:
                # Don't set stop to 0, set it to the end
                stop = None
            self.crop_slice.append(slice(start, stop))
        self.crop_slice.reverse()

    def forward(self, x):
        """Pad the last n dimensions of x"""
        return self.fn(self, x)

    @staticmethod
    def fn(linop, x, /):
        assert tuple(x.shape[-linop.D :]) == linop.im_size
        pad = linop.pad + [0, 0] * (x.ndim - linop.D)
        return F.pad(x, pad)

    @staticmethod
    def adj_fn(linop, y, /):
        """Crop the last n dimensions of y"""
        assert tuple(y.shape[-linop.D :]) == linop.pad_im_size
        slc = [slice(None)] * (y.ndim - linop.D) + linop.crop_slice
        return y[slc]

    def adjoint(self):
        adj = super().adjoint()
        adj.in_im_shape, adj.out_im_shape = self.out_im_shape, self.in_im_shape
        adj.in_im_size, adj.out_im_size = self.out_im_size, self.in_im_size
        return adj

    def split_forward(self, ibatch, obatch):
        self.split_forward_fn(ibatch, obatch)
        return self

    def split_forward_fn(self, ibatch, obatch, /):
        for islc, oslc in zip(ibatch[-self.D :], obatch[-self.D :]):
            if islc != slice(None) or oslc != slice(None):
                raise ValueError(
                    f"{type(self).__name__} cannot be split along image dim"
                )
        return None

    def size(self, dim: str):
        return self.size_fn(dim)

    def size_fn(self, dim: str, /):
        if dim in self.ishape[-self.D :]:
            return self.in_im_size[self.in_im_shape.index(dim)]
        elif dim in self.oshape[-self.D :]:
            return self.out_im_size[self.out_im_shape.index(dim)]
        return None


def pad_to_scale(grid_size, scale_factor):
    """Convenience wrapper for pad_to_size but with a scale factor instead"""
    padded_size = [int(i * scale_factor) for i in grid_size]
    return pad_to_size(grid_size, padded_size)


def pad_to_size(grid_size, padded_size):
    """Construct a padding list suitable for torch.nn.functional.pad

    Pad will take an input with size `grid_size` and return an output with size `factor * grid_size`

    Padding rules:
    odd/even image + even pad -> [pad // 2, pad // 2]
    even image + odd pad -> [pad // 2, pad // 2 + 1]
    odd image + odd pad -> [pad // 2 + 1, pad // 2]

    Preserves the "center" of the image or kspace if it has been fftshifted

    Useful for e.g. padding an image to increase resolution in fourier domain
    """
    if len(grid_size) != len(padded_size):
        raise ValueError(
            f"Dimension mismatch: cannot pad from size {grid_size} to size {padded_size}."
        )
    total_padding = [p - i for p, i in zip(padded_size, grid_size)]
    pad = []
    for i, tp in zip(grid_size, total_padding):
        if tp % 2:  # pad is odd
            if i % 2:  # im is odd
                pad_left = tp // 2 + 1
                pad_right = tp // 2
            else:  # im is even
                pad_left = tp // 2
                pad_right = tp // 2 + 1
        else:
            pad_left = tp // 2
            pad_right = tp // 2
        pad.append([pad_left, pad_right])
    pad.reverse()
    return sum(pad, start=[])
