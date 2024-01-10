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
            norm='ortho',
            kbnufft_kwargs=None,
            # Add more stuff for e.g. grog
            grog_normal=True,
            grog_config=None,
    ):
        """
        trj: (... d k) in -pi to pi (tkbn-style)
        """
        ishape = img_batch_shape + get2dor3d(im_size)
        oshape = trj_batch_shape + ('K',) # [R C K]
        super().__init__(ishape, oshape)
        self.trj = trj
        self.im_size = im_size

        # KbNufft-specific
        self.norm = norm
        self.kbnufft_kwargs = kbnufft_kwargs if kbnufft_kwargs is not None else {}
        self.nufft = KbNufft(im_size, **self.kbnufft_kwargs)
        self.nufft_adj = KbNufftAdjoint(im_size, **self.kbnufft_kwargs)

        # GROG
        self.grog_normal = grog_normal
        self.grog_config = grog_config

    def forward(self, x: torch.Tensor):
        return self.fn(x, self.trj)

    def _split(self, ibatch, obatch):
        assert obatch[-2] == slice(None), 'NUFFT cannot be sliced in spatial dim'
        return type(self)(
            self.trj[obatch],
            trj_batch_shape=self.oshape[-2:],
            norm=self.norm,
            kbnufft_kwargs=self.kbnufft_kwargs,
            grog_normal=self.grog_normal,
            grog_config=self.grog_config,
        )

    def size(self, dim: str):
        return self.size_fn(dim, self.trj)

    def fn(self, x, /, trj):
        y = self.nufft(x, trj, norm=self.norm)
        return y

    def adj_fn(self, x, /, trj):
        y = self.nufft_adj(x, trj, norm=self.norm)
        return y

    def normal_fn(self, x, /, trj):
        ...

    def split_fn(self, ibatch, obatch, trj):
        return trj[obatch]

    def size_fn(self, dim: str, trj):
        if dim == 'K':
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

    def _split(self, ibatch, obatch):
        """Split over coil dim only"""
        for islc, oslc in zip(ibatch, obatch[1:]):
            if islc != oslc:
                raise IndexError(f'SENSE currently only supports matched image input/output slicing.')
        return type(self)(self.mps[obatch])

    def size(self, dim: str):
        if dim == self.coil_str:
            return self.mps.shape[0]
