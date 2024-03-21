from typing import Optional, Callable
from warnings import warn

import torch
from torchlinops._core._linops.nameddim import N2K
from torchlinops._core._linops import Dense, FFT, PadLast

__all__ = ['toeplitz']


def toeplitz(Nufft: Callable, Inner: Optional[Callable] = None, oversamp: float = 2., device: torch.device = 'cpu'):
    """
    """
    if Inner is None:
        warn('No Inner linop provided to toeplitz (not even density compensation?)')
    # Get Padding
    im_size = Nufft.im_size
    pad_im_size = tuple(int(oversamp * d) for d in im_size)
    shared_batch_shape = Nufft.shared_batch_shape
    in_batch_shape = Nufft.in_batch_shape
    batch_shape = shared_batch_shape + in_batch_shape
    Pad = PadLast(pad_im_size, im_size, batch_shape)

    # Get FFT
    dim = tuple(range(-len(im_size), 0))
    fft_ishape = Pad.oshape
    fft_oshape = batch_shape + N2K(Pad.im_shape)
    F = FFT(fft_ishape, fft_oshape, dim=dim, norm='ortho', centered=True)

    # Get Kernels
    weight_shape = tuple(Nufft.size(d) for d in Nufft.oshape)
    weight = torch.ones(*weight_shape, dtype=torch.complex64, device=device)
    # weight: [[S...] N... K...]
    if Inner is not None:
        assert len(Inner.ishape) == len(Inner.oshape), "Inner linop must have identical input and output shape lengths"
        weight = Inner(weight)
        # weight: [[S...] N1... K...]
    kernel = F(Pad(Nufft.H(weight)))
    # After Nufft.H: weight: [[S...] N1... Nx Ny [Nz]]
    # After Pad: weight: [[S...] N1... Nx1 Ny1 [Nz1]]
    # After F: weight: [[S...] N1... Kx1 Ky1 [Kz1]]
    kernel_shape = Inner.oshape[:len(batch_shape)] + F.oshape[-len(dim):]

    Kern = Dense(kernel, kernel_shape, F.oshape, F.H.ishape)

    return Pad.H @ F.H @ Kern @ F @ Pad

