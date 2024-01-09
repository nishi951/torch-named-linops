from dataclasses import dataclass
from itertools import chain
import copy

from typing import Optional, Tuple

@dataclass
class Dim:
    name: Optional[str] = None
    """Name of this dimension"""

@dataclass
class BatchedDim(Dim):
    batch_size: Optional[int] = None
    """Batch dimension or None if dimension is not batched"""


class NamedLinopShape:
    def __init__(self,
            inames: Optional[List[Dim, ...]] = None,
            onames: Optional[List[Dim, ...]] = None,
    ):
        self.inames = inames
        self.onames = onames

    def forward(self, s: List[Dim, ...]):
        assert s == self.inames
        return self.onames

    def adjoint(self, t: List[Dim, ...]):
        assert t == self.onames
        return self.inames

    @property
    def H(self):
        return NamedLinopShape(
            inames=self.onames,
            onames=self.inames,
        )

    @property
    def N(self):
        return NamedLinopShape(
            inames=self.inames,
            onames=self.inames,
        )

    @staticmethod
    def check_shapes(left, right):
        if left.inames != right.onames:
            raise ValueError(f'Mismatched shapes for product of {left} and {right}')
        return

    def compose(self, L, consolidate=False):
        self.check_shapes(self, L)
        if consolidate:
            return NamedLinopShape(
                inames=L.inames,
                onames=self.onames,
            )
        return NamedLinopShapeChain(self, L)

    def __matmul__(self, V):
        return self.compose(V)

    def __rmatmul__(self, U):
        return U.compose(self)


class NamedLinopShapeChain(NamedLinopShape):
    def __init__(self, *shapes):
        self.shapes = shapes
        self.inames = shapes[-1].inames
        self.onames = shapes[0].onames

    @property
    def adjoint(self):
        return NamedLinopShapeChain(
            shape.adjoint for shape in reversed(self.shapes)
        )

    @property
    def normal(self):
        return NamedLinopShapeChain(
            chain(self.adjoint.shapes, self.shapes)
        )


