from copy import deepcopy
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
        self._shape.add("batch_shape", batch_shape)
        self.dim = dim
        self.norm = norm
        self.centered = centered

    @property
    def batch_shape(self):
        return self._shape.lookup("batch_shape")

    def forward(self, x, /):
        return self.fn(self, x)

    @staticmethod
    def fn(linop, x):
        if linop.centered:
            x = fft.ifftshift(x, dim=linop.dim)
        x = fft.fftn(x, dim=linop.dim, norm=linop.norm)
        if linop.centered:
            x = fft.fftshift(x, dim=linop.dim)
        return x

    @staticmethod
    def adj_fn(linop, x):
        if linop.centered:
            x = fft.ifftshift(x, dim=linop.dim)
        x = fft.ifftn(x, dim=linop.dim, norm=linop.norm)
        if linop.centered:
            x = fft.fftshift(x, dim=linop.dim)
        return x

    @staticmethod
    def normal_fn(linop, x):
        return x

    def split_forward(self, ibatch, obatch):
        new = type(self)(self.dim, self.batch_shape, self.norm, self.centered)
        new._shape = deepcopy(self._shape)
        return new

    def normal(self, inner=None):
        if inner is None:
            return Identity(self.ishape)
        return super().normal(inner)
