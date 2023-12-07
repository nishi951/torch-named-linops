from copy import copy

import torch
import torch.nn as nn

__all__ = ['LinearOperator']

class LinearOperator(nn.Module):
    def __init__(self, input_shape, input_names, output_shape, output_names):
        super().__init__()
        self.input_shape = input_shape
        self.input_names = input_names
        self.output_shape = output_shape
        self.output_names = output_names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """matrix-vector multiply"""
        return NotImplemented

    def compose(self, L):
        """i.e. matrix-matrix multiply"""
        return NotImplemented

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

    def __getitem__(self, key):
        raise NotImplementedError(
            f'LinearOperator {self.__class__.__name__} cannot be sliced'
        )

    def __repr__(self):
        return f'{self.__class__.__name__}[{tuple(self.output_shape)}x{tuple(self.input_shape)}]'
