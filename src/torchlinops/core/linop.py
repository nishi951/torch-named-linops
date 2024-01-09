from copy import copy

import torch
import torch.nn as nn

__all__ = ['LinearOperator']

class LinearOperator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """matrix-vector multiply"""
        return self.H(y)

    def compose(self, L):
        """i.e. matrix-matrix multiply"""
        return Chain(self, L)

    @property
    def H(self):
        if self.adj is None:
            self.adj = copy(self) # shallow
            self.adj.forward = self.adjoint
            self.adj.adjoint = self.forward
        return self.adj

    @property
    def N(self):
        """matrix-vector multiply"""
        if self.normal is None:
            self.normal = self.H @ self
        return self.normal

    def __matmul__(self, V):
        return self.compose(V)

    def __rmatmul__(self, U):
        return U.compose(self)

    def __getitem__(self, idx):
        out_slc, in_slc = idx
        raise NotImplementedError(
            f'LinearOperator {self.__class__.__name__} has no slice implemented.'
        )

    def __repr__(self):
        return f'{self.__class__.__name__}[{tuple(self.output_shape)}x{tuple(self.input_shape)}]'


class Chain(LinearOperator):
    def __init__(self, *linops):
        self.linops = nn.ModuleList(linops)

    def forward(self, x: torch.Tensor):
        for linop in reversed(self.linops):
            x = linop(x)
        return x

    def adjoint(self, y: torch.Tensor):
        for linop in self.linops:
            y = linop.H(y)
        return y

    def normal(self, x: torch.Tensor):
        return self.adjoint(self.forward(x))
