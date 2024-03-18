import torch.fft as fft

from .namedlinop import NamedLinop


class FFT(NamedLinop):
    def __init__(self, ishape, oshape, dim, norm, centered: bool = False):
        """
        centered=True mimicks sigpy behavior
        """
        super().__init__(ishape, oshape)
        self.dim = dim
        self.norm = norm
        self.centered = centered

    def forward(self, x, /):
        return self.fn(x)

    def fn(self, x):
        if self.centered:
            x = fft.ifftshift(x, dim=self.dim)
        x = fft.fftn(x, dim=self.dim, norm=self.norm)
        if self.centered:
            x = fft.fftshift(x, dim=self.dim)
        return x

    def adj_fn(self, x):
        if self.centered:
            x = fft.ifftshift(x, dim=self.dim)
        x = fft.ifftn(x, dim=self.dim, norm=self.norm)
        if self.centered:
            x = fft.fftshift(x, dim=self.dim)
        return x

    def normal_fn(self, x):
        return x

    def split_forward(self, ibatch, obatch):
        return self.split_forward_fn(ibatch, obatch)

    def split_forward_fn(self, ibatch, obatch, /):
        return type(self)(self.ishape, self.oshape, self.dim, self.norm)

    def size(self, dim):
        return self.size_fn(dim)

    def size_fn(self, dim: str, /):
        """FFT doesn't determine any dimensions"""
        return None
