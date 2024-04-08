import torch.fft as fft

from .namedlinop import NamedLinop
from .nameddim import get2dor3d, NS
from .identity import Identity


class FFT(NamedLinop):
    def __init__(self, dim, batch_shape, norm, centered: bool = False):
        """
        Currently only supports 2D and 3D FFTs
        centered=True mimicks sigpy behavior
        """
        shape = NS(batch_shape) + NS(get2dor3d(dim), get2dor3d(dim, kspace=True))
        super().__init__(shape)
        self._shape.add('batch_shape', batch_shape)
        self.dim = dim
        self.norm = norm
        self.centered = centered

    @property
    def batch_shape(self):
        return self._shape.lookup('batch_shape')

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
        return type(self)(self.dim, self.batch_shape, self.norm, self.centered)

    def split_forward_fn(self, ibatch, obatch, /):
        return None

    def size(self, dim):
        return self.size_fn(dim)

    def size_fn(self, dim: str, /):
        """FFT doesn't determine any dimensions"""
        return None

    def normal(self, inner=None):
        if inner is None:
            return Identity(self.ishape)
        return super().normal(inner)
