import torch
from einops import einsum

from .base import NamedLinop

__all__ = [
    'Dense',
    'Diagonal',
    'Repeat',
]


class Broadcast(NamedLinop):
    """Return a rearrange matching batched ishape to oshape.
    Basically broadcast to each other
    TODO: Fix this class
    """
    def __init__(self, ishape, oshape):
        super().__init__(ishape, oshape)
        self.ishape_str = ' '.join(ishape)
        self.oshape_str = ' '.join(oshape)

    def forward(self, x):
        return self.fn(x, self.ishape_str, self.oshape_str)

    def fn(cls, x, ishape_str, oshape_str):
        return rearrange(x, f'... {ishape_str} -> ... {oshape_str}')

    def adj_fn(cls, x: torch.Tensor, ishape_str, oshape_str):
        return rearrange(x, f'... {oshape_str} -> ... {ishape_str}')

    def split(self, ibatch, obatch):
        return self # Literally don't change anything


class Dense(NamedLinop):
    """
    Example:
    x: [A, Nx, Ny]
    weightshape: [A, T]
    oshape: [T, Nx, Ny]
    """
    def __init__(self, weight, weightshape, ishape, oshape):
        super().__init__(ishape, oshape)
        self.weight = weight
        self.einstr = f'{" ".join(self.ishape)},{" ".join(self.weightshape)}->{" ".join(self.oshape)}'
        self.adj_einstr = f'{" ".join(self.oshape)},{" ".join(self.weightshape)}->{" ".join(self.ishape)}'

    def fn(self, x, /, weight):
        return einsum(x, weight, self.einstr)

    def adj_fn(self, x, /, weight):
        return einsum(x, weight, self.adj_einstr)

    def normal_fn(self, x, /, weight):
        return self.adj_fn(self.fn(x, weight), weight)

    def split_forward(self, ibatch, obatch):
        weight = self.split_forward_fn(ibatch, obatch, self.weight)
        return type(self)(weight, self.weightshape, self.ishape, self.oshape)

    def split_forward_fn(self, ibatch, obatch, /, weight):
        weightbatch = [slice(None)] * len(self.weightshape)
        for dim, batch in zip(self.ishape, ibatch):
            if dim in self.weightshape:
                weightbatch[self.weightshape.index(dim)] = batch
        for dim, batch in zip(self.oshape, ibatch):
            if dim in self.weightshape:
                weightbatch[self.weightshape.index(dim)] = batch
        return weight[weightbatch]

    def size(self, dim: str):
        return self.size_fn(dim, self.weight)

    def size_fn(self, dim: str, weight):
        if dim in self.weightshape:
            return weight.shape[self.weightshape.index(dim)]
        return None


class Diagonal(NamedLinop):
    def __init__(self, weight, ishape, oshape):
        assert ishape == oshape, 'Diagonal linops must have matching input and output dimensions'
        assert len(weight.shape) == len(ishape), 'All dimensions must be named'
        super().__init__(ishape, oshape)
        self.weight = weight

    def forward(self, x):
        return x * self.weight

    def fn(self, x, /, weight):
        return x * weight

    def adj_fn(self, x, /, weight):
        return x * torch.conj(weight)

    def normal_fn(self, x, /, weight):
        return x * torch.abs(weight) ** 2

    def split_forward(self, ibatch, obatch):
        weight = self.split_forward_fn(ibatch, obatch, self.weight)
        return type(self)(weight, self.ishape, self.oshape)

    def split_forward_fn(self, ibatch, obatch, /, weight):
        assert ibatch == obatch, 'Diagonal linop must be split identically'
        return weight[ibatch]

    def size(self, dim: str):
        return self.size_fn(dim, self.weight)

    def size_fn(self, dim: str, weight):
        if dim in self.ishape:
            return weight.shape[self.ishape.index(dim)]
        return None


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




def slice_len(slc, n):
    """
    n: length of sequence slc is being applied to
    """
    return len(range(*slc.indices(n)))


class Repeat(NamedLinop):
    """Unsqueezes and expands a tensor along dim
    """
    def __init__(self, n_repeats, dim, ishape, oshape):
        assert len(ishape) + 1 == len(oshape), 'oshape should have 1 more dim than ishape'
        super().__init__(ishape, oshape)
        self.n_repeats = n_repeats
        self.dim = dim

    def forward(self, x):
        return self.fn(x, self.n_repeats)

    def fn(self, x, /, n_repeats):
        expand_size = [-1] * len(self.oshape)
        expand_size[self.dim] = n_repeats
        x = x.unsqueeze(self.dim)
        # print(x)
        return x.expand(*expand_size)

    def adj_fn(self, x, /, n_repeats):
        return torch.sum(x, dim=self.dim, keepdim=False)

    def split_forward(self, ibatch, obatch):
        """Repeat fewer times, depending on the size of obatch"""
        assert len(ibatch) == len(self.ishape), 'length of ibatch should match length of ishape'
        assert len(obatch) == len(self.oshape), 'length of obatch should match length of oshape'
        return type(self)(
            n_repeats=self.split_forward_fn(ibatch, obatch, self.n_repeats),
            dim=self.dim,
            ishape=self.ishape,
            oshape=self.oshape,
        )

    def split_forward_fn(self, ibatch, obatch, /, n_repeats):
        return slice_len(obatch[self.dim], n_repeats)

    def size(self, dim: str):
        return self.size_fn(dim, self.n_repeats)

    def size_fn(self, dim, /, n_repeats):
        if dim == self.oshape[self.dim]:
            return n_repeats
        return None

