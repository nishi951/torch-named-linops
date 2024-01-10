import torch

from .base import NamedLinop

class Broadcast(NamedLinop):
    """Return a rearrange matching batched ishape to oshape.
    Basically broadcast to each other
    """
    def __init__(self, ishape, oshape):
        super().__init__(ishape, oshape)
        self.ishape_str = ' '.join(ishape)
        self.oshape_str = ' '.join(oshape)

    def forward(self, x):
        return self.fn(x, self.ishape_str, self.oshape_str)

    @classmethod
    def fn(cls, x, ishape_str, oshape_str):
        return rearrange(x, f'... {ishape_str} -> ... {oshape_str}')

    @classmethod
    def adj_fn(cls, x: torch.Tensor, ishape_str, oshape_str):
        return rearrange(x, f'... {oshape_str} -> ... {ishape_str}')

    def _split(self, ibatch, obatch):
        return self # Literally don't change anything

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
        # Derived
        self.expand_size = [-1] * len(oshape)
        self.expand_size[self.dim] = self.n_repeats

    def forward(self, x):
        return self.fn(x)

    def fn(self, x, /):
        x = x.unsqueeze(self.dim)
        # print(x)
        return x.expand(*self.expand_size)

    def adj_fn(self, x, /):
        return torch.sum(x, dim=self.dim, keepdim=False)

    def _split(self, ibatch, obatch):
        """Repeat fewer times, depending on the size of obatch"""
        assert len(ibatch) == len(self.ishape), 'length of ibatch should match length of ishape'
        assert len(obatch) == len(self.oshape), 'length of obatch should match length of oshape'
        return type(self)(
            n_repeats=slice_len(obatch[self.dim], self.n_repeats),
            dim=self.dim,
            ishape=self.ishape,
            oshape=self.oshape,
        )

    def size(self, dim: str):
        if dim == self.oshape[self.dim]:
            return self.n_repeats
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

    def _split(self, ibatch, obatch):
        assert ibatch == obatch, 'Diagonal linop must be split identically'
        self.weight[ibatch]
        return type(self)(self.weight[ibatch], self.ishape, self.oshape)


class Dense(NamedLinop):
    def __init__(self, mat, ishape, oshape):
        super().__init__(ishape, oshape)
        self.weight = weight


