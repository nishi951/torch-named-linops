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

    def forward(self, x):
        return self.fn(x, self.n_repeats)

    def fn(self, x, /, n_repeats):
        expand_size = [-1] * len(self.oshape)
        expand_size[self.dim] = n_repeats
        x = x.unsqueeze(self.dim)
        # print(x)
        return x.expand(*expand_size)

    def adj_fn(self, x, /):
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

class Dense(NamedLinop):
    def __init__(self, mat, ishape, oshape):
        super().__init__(ishape, oshape)
        self.weight = weight


