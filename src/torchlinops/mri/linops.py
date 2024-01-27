from typing import Optional
import torch
from torchkbnufft import KbNufft, KbNufftAdjoint

from ..core.base import NamedLinop

__all__ = [
    'NUFFT',
    'SENSE',
]


def get2dor3d(im_size, kspace=False):
    if len(im_size) == 2:
        im_dim = ('Kx', 'Ky') if kspace else ('Nx', 'Ny')
    elif len(im_size) == 3:
        im_dim = ('Kx', 'Ky', 'Kz') if kspace else ('Nx', 'Ny')
    else:
        raise ValueError(f'Image size {im_size} - should have length 2 or 3')
    return im_dim


class NUFFT(NamedLinop):
    def __init__(
            self,
            trj,
            im_size,
            img_batch_shape,
            trj_batch_shape,
            coil_dim: Optional[str] = 'C',
            readout_dim: str = 'K',
            norm='ortho',
            kbnufft_kwargs=None,
    ):
        """
        trj: (... d k) in -pi to pi (tkbn-style)
        """
        self.img_batch_shape = img_batch_shape
        self.trj_batch_shape = trj_batch_shape
        ishape = img_batch_shape + get2dor3d(im_size)
        if coil_dim is not None:
            oshape = trj_batch_shape + (coil_dim, readout_dim) # [R C K]
        else:
            oshape = trj_batch_shape + (readout_dim,)
        super().__init__(ishape, oshape)
        self.trj = trj
        self.im_size = im_size
        self.coil_dim = coil_dim
        self.readout_dim = readout_dim

        expected_out_dim = (len(trj.shape)-2) + (1 if self.coil_dim else 0) + 1 # (K,)
        assert len(self.oshape) == expected_out_dim, f'Output shape {self.oshape} does not match expected output dimension {expected_out_dim}'

        # KbNufft-specific
        self.norm = norm
        self.kbnufft_kwargs = kbnufft_kwargs if kbnufft_kwargs is not None else {}
        self.nufft = KbNufft(im_size, **self.kbnufft_kwargs)
        self.nufft_adj = KbNufftAdjoint(im_size, **self.kbnufft_kwargs)

    def forward(self, x: torch.Tensor):
        return self.fn(x, self.trj)

    def fn(self, x, /, trj):
        y = self.nufft(x, trj, norm=self.norm)
        return y

    def adj_fn(self, x, /, trj):
        y = self.nufft_adj(x, trj, norm=self.norm)
        return y

    def normal_fn(self, x, /, trj):
        return self.adj_fn(self.fn(x, trj), trj)

    def split_forward(self, ibatch, obatch):
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.trj),
            im_size=self.im_size,
            img_batch_shape=self.img_batch_shape,
            trj_batch_shape=self.trj_batch_shape,
            coil_dim=self.coil_dim,
            readout_dim=self.readout_dim,
            norm=self.norm,
            kbnufft_kwargs=self.kbnufft_kwargs,
        )

    def split_forward_fn(self, ibatch, obatch, trj):
        if self.coil_dim is None:
            # obatch is [... K]
            trj_slc = obatch[:-1] + [slice(None)] + obatch[-1:]
        else:
            # obatch is [... C K]
            trj_slc = obatch[:-2] + [slice(None)] + obatch[-1:]
        return trj[trj_slc]

    def size(self, dim: str):
        return self.size_fn(dim, self.trj)

    def size_fn(self, dim: str, trj):
        if dim == self.readout_dim:
            return trj.shape[-1]
        elif dim == self.oshape[0]:
            return trj.shape[0]
        return None



class SENSE(NamedLinop):
    def __init__(self, mps, coil_str: str = 'C'):
        im_size = mps.shape[1:]
        im_shape = get2dor3d(im_size, kspace=False)
        super().__init__(im_shape, (coil_str, *im_shape))
        self.coil_str = coil_str
        self.mps = mps

    def forward(self, x):
        return self.fn(x, self.mps)

    def fn(self, x, /, mps):
        return x * mps

    def adj_fn(self, x, /, mps):
        return torch.sum(x * torch.conj(mps), dim=0)

    def split_forward(self, ibatch, obatch):
        """Split over coil dim only"""
        for islc, oslc in zip(ibatch, obatch[1:]):
            if islc != oslc:
                raise IndexError(f'SENSE currently only supports matched image input/output slicing.')
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.mps)
        )

    def split_forward_fn(self, ibatch, obatch, /, weight):
        return self.mps[obatch]

    def size(self, dim: str):
        return self.size_fn(dim, self.mps)

    def size_fn(self, dim: str, mps):
        if dim in self.oshape:
            return mps.shape[self.oshape.index(dim)]
        return None
