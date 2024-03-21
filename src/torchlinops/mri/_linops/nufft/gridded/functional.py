from typing import Tuple

import torch
import torch.fft as fft

from .indexing import multi_grid

__all__ = [
    'gridded_nufft',
    'gridded_nufft_adjoint',
]

def gridded_nufft(x: torch.Tensor, trj: torch.Tensor, dim: Tuple):
    """
    x: [N...  Nx Ny [Nz]] # N... may include coils
    trj: [K... D] (sigpy-style)
    output: [N... K...]
    """
    # FFT
    x = fft.ifftshift(x, dim=dim)
    Fx = fft.fftn(x, dim=dim, norm="ortho")

    # Index
    N = x.shape[: -len(dim)]
    batch_slc = (slice(None),) * len(N)
    trj_split = tuple(trj[..., i] for i in range(trj.shape[-1]))
    omega_slc = (*batch_slc, *trj_split)
    return Fx[omega_slc]

def gridded_nufft_adjoint(y: torch.Tensor, trj: torch.Tensor, dim: Tuple, im_size: Tuple):
    # Un-Index (i.e. grid)
    Fx = multi_grid(y, trj, final_size=im_size)

    # IFFT
    x = fft.ifftn(Fx, dim=dim, norm="ortho")
    x = fft.fftshift(x, dim=dim)
    return x
