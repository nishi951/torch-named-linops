from einops import rearrange
import torch


class Split(LinearOperator):
    def __init__(self, dim, chunks):
        self.dim = dim
        self.chunks = chunks

    def forward(self, x: torch.Tensor):
        chunks = torch.chunk(x, dim=self.dim)
        return torch.stack(chunks, dim=self.dim+1)

    def adjoint(self, x: torch.Tensor):
        ...

    # Adjoint is Sum
    # Closer is VerticalStack

class Rearrange(LinearOperator):
    def __init__(self, ishape: str, oshape: str, **shape_kwargs):
        self.ishape = ishape
        self.oshape = oshape
        self.shape_kwargs = shape_kwargs

    def forward(self, x: torch.Tensor):
        return rearrange(x,
                         f'{self.ishape} -> {self.oshape}',
                         self.shape_kwargs)


class VerticalStack(LinearOperator):
    ...
    # Adjoint is HorizontalSplit
    # Opener is VerticalSplit


class HoriontalSplit(LinearOperator):
    ...
    # Adjoint is VerticalStack
    # Closer is Sum

class Sum(LinearOperator):
    ...
    # Adjoint is VerticalSplit
    # Opener is HorizontalSplit

### Named versions (!)

class Split(NamedLinopShapes)
