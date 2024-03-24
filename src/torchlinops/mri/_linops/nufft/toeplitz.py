from copy import deepcopy
from typing import Optional, Callable
from warnings import warn
from itertools import product

import torch
from torchlinops._core._linops.nameddim import N2K, ND
from torchlinops._core._linops import Dense, FFT, PadLast

__all__ = ["toeplitz"]


def toeplitz(
    Nufft: Callable,
    Inner: Optional[Callable] = None,
    oversamp: float = 2.0,
    device: torch.device = "cpu",
):
    """Compute the toeplitz kernel for a nufft object with optional inner linop

    References:
        Baron, C.A., Dwork, N., Pauly, J.M. and Nishimura, D.G. (2018),
        Rapid compressed sensing reconstruction of 3D non-Cartesian MRI.
        Magn. Reson. Med., 79: 2685-2692. https://doi.org/10.1002/mrm.26928
    """
    if Inner is None:
        warn("No Inner linop provided to toeplitz (not even density compensation?)")
    # Get Padding
    nufft_batch_shape = Nufft.shared_batch_shape + Nufft.in_batch_shape
    Pad = _pad(Nufft.im_size, oversamp, nufft_batch_shape)

    # Get FFT
    F = _fft(
        len(Pad.im_size),
        ishape=Pad.oshape,
        oshape=nufft_batch_shape + N2K(Pad.im_shape),
    )

    # Create oversampled nufft
    Nufft_oversamp = deepcopy(Nufft)
    Nufft_oversamp.change_im_size(Pad.pad_im_size)

    if Inner is not None:
        assert len(Inner.ishape) == len(
            Inner.oshape
        ), "Inner linop must have identical input and output shape lengths"

        # Collect all combinations of input/output pairs
        kernel_changed_shape, changed = _changed_shape(Inner.ishape, Inner.oshape)
        n_changed = len(kernel_changed_shape)
        n_nufft_batch = len(nufft_batch_shape)
        nufft_changed_batch_shape = tuple(
            odim if changed[i] else idim
            for i, (idim, odim) in enumerate(
                zip(Inner.ishape[:n_nufft_batch], Inner.oshape[:n_nufft_batch])
            )
        )
        kernel_shape = (
            kernel_changed_shape + nufft_changed_batch_shape + N2K(Pad.im_shape)
        )
        kernel_size = list(_size(d, Nufft, Inner, Pad) for d in kernel_shape)

        kernel_size[-len(Pad.im_shape) :] = Pad.pad_im_size
        kernel_size = tuple(kernel_size)
        weight_shape = Inner.ishape
        weight_size = tuple(_size(d, Nufft, Inner, Pad) for d in weight_shape)

        # Nufft out batch shape should not be affected non-diagonally
        changed_ranges = tuple(
            range(s) if changed[i] else (slice(None),)
            for i, s in enumerate(weight_size)
        )
        kernel = torch.zeros(*kernel_size, dtype=torch.complex64, device=device)
        for idx in product(*changed_ranges):
            weight = torch.zeros(*weight_size, dtype=torch.complex64, device=device)
            weight[tuple(idx)].fill_(1.0)
            weight = Inner(weight)
            weight = F(Nufft_oversamp.H(weight))
            changed_idx = tuple(slc for slc, chg in zip(idx, changed) if chg)
            # full_idx = changed_idx + (slice(None),) * len(idx)
            kernel[changed_idx] = weight
        kernel *= oversamp ** Nufft.trj.shape[-1]  # Fix scaling
        kernel_oshape = Inner.oshape[: -len(Nufft.out_batch_shape)] + N2K(Pad.im_shape)
        Kern = Dense(kernel, kernel_shape, F.oshape, kernel_oshape)

    else:
        weight_shape = Nufft.oshape
        weight_size = tuple(_size(d, Nufft, Pad) for d in Nufft.oshape)
        weight = torch.ones(*weight_size, dtype=torch.complex64, device=device)

        kernel = F(Nufft_oversamp.H(weight))
        kernel *= oversamp ** Nufft.trj.shape[-1]  # Fix scaling
        kernel_shape = (
            Nufft.shared_batch_shape
            + Nufft.in_batch_shape
            + F.oshape[-len(Pad.im_shape) :]
        )
        Kern = Dense(kernel, kernel_shape, F.oshape, F.H.ishape)

    return Pad.normal(F.normal(Kern))


def _pad(im_size, oversamp, batch_shape):
    """Helper function for making a padding linop"""
    pad_im_size = tuple(int(oversamp * d) for d in im_size)
    Pad = PadLast(pad_im_size, im_size, batch_shape)
    return Pad


def _fft(ndim, ishape, oshape):
    """Helper function for making an n-dimensional FFT"""
    dim = tuple(range(-ndim, 0))
    F = FFT(ishape, oshape, dim=dim, norm="ortho", centered=True)
    return F


def _size(dim: ND, *linops):
    """Get the size of dimension, possibly from multiple linops
    Returns
    -------
    int : the size of the dimension, or 1 (i.e. broadcastable) if the linops
    cannot determine it
    """
    for linop in linops:
        size = linop.size(dim)
        if size is not None:
            return size
    return 1


def _changed_shape(ishape, oshape):
    """Helper function for getting the kernel shape and slices
    B = nufft shared + in batch shape
    K = nufft out batch shape
    Kx1, Ky1, Kz1 = output padded fourier-domain shapes

    kernel_fourier_dim: [Kx1, Ky1, [Kz1]
    B1 = nufft shared + in batch shape, different dims only. May be different length than B
    kernel shape: [B1..., B..., Kx1, Ky1, [Kz1]]
    ishape: [B..., K...]
    in_batch_shape: [B...]
    oshape: [B1..., K...]
    out_batch_shape: [B1...]
    """
    changed_shape = []
    changed = []
    for idim, odim in zip(ishape, oshape):
        if idim != odim:
            changed_shape.append(idim)
            changed.append(True)
        else:
            changed.append(False)
    changed_shape = tuple(changed_shape)
    return changed_shape, changed
