import torch
import torch.nn as nn

from ..core.linop import LinearOperator

__all__ = [
    'SenseMapsLinop'
]

class SenseMapsLinop(LinearOperator):
    def __init__(
            self,
            mps: torch.Tensor,
    ):
        """
        C: number of coils
        im_size: Image size (tuple)
        mps: [C *im_size]
        """
        input_names = ('im_size')
        output_names = ('C', 'im_size')
        super().__init__(
            input_shape=mps.shape[1:],
            input_names=input_names,
            output_shape=mps.shape,
            output_names=output_names
        )
        self.mps = nn.Parameter(mps, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mps * x

    def __getitem__(self, idx):
        return SenseMapsLinop(self.mps[idx])

class DiagonalLinop(LinearOperator):
    def __init__(self, A, input_names, output_names):
        self.A = A
        input_shape = A.shape
        output_shape = A.shape
        super().__init__(
            input_shape=input_shape,
            input_names=input_names,
            output_shape=output_shape,
            output_names=output_names,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.A * x

    def __getitem__(self, idx):
        return DiagonalLinop(self.A[idx], self.input_names, self.output_names)

class NUFFT(LinearOperator):
    def __init__(self, trj):
        ...
