from torch import Tensor
import torch.fft as fft

__all__ = ["cfft", "cifft", "cfft2", "cifft2", "cfftn", "cifftn"]


def cfftn(x, dim=None, norm="ortho"):
    """Matches Sigpy's fft, but in torch
    c = centered
    """
    x = fft.ifftshift(x, dim=dim)
    x = fft.fftn(x, dim=dim, norm=norm)
    x = fft.fftshift(x, dim=dim)
    return x


def cifftn(x, dim=None, norm="ortho"):
    """Matches Sigpy's fft adjoint, but in torch"""
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
