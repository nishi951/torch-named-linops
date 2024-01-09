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

class Repeat(NamedLinop):
    """Unsqueezes and expands a tensor along dim
    """
    def __init__(self, n_repeats, dim, ishape, oshape):
        super().__init__(ishape, oshape)
        self.n_repeats = n_repeats
        self.dim = dim

    def forward(self, x):
        return self.fn(x)

    def fn(self, x, /):
        x = x.unsqueeze(self.dim)
        # print(x)
        return torch.repeat_interleave(x, self.n_repeats, dim=self.dim)

    def adj_fn(self, x, /):
        return torch.sum(x, dim=self.dim, keepdim=False)


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


