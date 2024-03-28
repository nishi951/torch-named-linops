import torch.fft as fft

__all__ = ['cfft', 'cifft']

def cfft(x, dim=None, norm='ortho'):
    """Matches Sigpy's fft, but in torch
    c = centered
    """
    x = fft.ifftshift(x, dim=dim)
    x = fft.fftn(x, dim=dim, norm=norm)
    x = fft.fftshift(x, dim=dim)
    return x


def cifft(x, dim=None, norm='ortho'):
    """Matches Sigpy's fft adjoint, but in torch"""
    x = fft.ifftshift(x, dim=dim)
    x = fft.ifftn(x, dim=dim, norm=norm)
    x = fft.fftshift(x, dim=dim)
    return x
