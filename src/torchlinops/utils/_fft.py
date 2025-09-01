from torch import Tensor
import torch.fft as fft

__all__ = ["cfft", "cifft", "cfft2", "cifft2", "cfftn", "cifftn"]


def cfftn(x, dim=None, norm="ortho"):
    """Compute the centered n-dimenional FFT.

    Assumes the origin lies in the middle of the array (i.e., that the array has
    been fftshifted)

    Parameters
    ----------
    dim : tuple[int, ...]
        The dimensions over which to take the ifft.
    norm : norm (str, optional)
        Normalization mode. For the forward transform (fft()), these correspond to:
        "forward" - normalize by 1/n
        "backward" - no normalization
        "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
        Calling the backward transform (cifftn()) with the same normalization
        mode will apply an overall normalization of 1/n between the two transforms.
        This is required to make ifft() the exact inverse. Default is "backward"
        (no normalization).
    """
    x = fft.ifftshift(x, dim=dim)
    x = fft.fftn(x, dim=dim, norm=norm)
    x = fft.fftshift(x, dim=dim)
    return x


def cifftn(x, dim=None, norm="ortho"):
    """Compute the centered n-dimensional inverse FFT.

    Assumes the origin lies in the middle of the array (i.e., that the array has
    been fftshifted)

    Parameters
    ----------
    dim : tuple[int, ...]
        The dimensions over which to take the ifft.
    norm : norm (str, optional)
        Normalization mode. For the backward transform (ifft()), these correspond to:
        "forward" - no normalization
        "backward" - normalize by 1/n
        "ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)
        Calling the forward transform (cfftn()) with the same normalization mode
        will apply an overall normalization of 1/n between the two transforms. This
        is required to make ifft() the exact inverse. Default is "backward"
        (normalize by 1/n).
    """
    x = fft.ifftshift(x, dim=dim)
    x = fft.ifftn(x, dim=dim, norm=norm)
    x = fft.fftshift(x, dim=dim)
    return x


# Convenience functions
def cfft(x: Tensor, **kwargs):
    return cfftn(x, dim=(-1,), **kwargs)


def cifft(x: Tensor, **kwargs):
    return cifftn(x, dim=(-1,), **kwargs)


def cfft2(x: Tensor, **kwargs):
    return cfftn(x, dim=(-2, -1), **kwargs)


def cifft2(x: Tensor, **kwargs):
    return cifftn(x, dim=(-2, -1), **kwargs)
