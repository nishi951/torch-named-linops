from typing import Optional

from ..nameddim import NamedShape as NS, Shape
from .namedlinop import NamedLinop

__all__ = ["BreakpointLinop"]


class BreakpointLinop(NamedLinop):
    """Debugging identity operator that drops into ``pdb`` on forward/adjoint.

    Useful for inspecting intermediate tensor values inside a ``Chain``.
    """

    def __init__(self, ioshape: Optional[Shape] = None):
        super().__init__(NS(ioshape))

    @staticmethod
    def fn(linop, x, /):
        breakpoint()
        return x

    @staticmethod
    def adj_fn(linop, x, /):
        breakpoint()
        return x

    def split_forward(self, ibatch, obatch):
        return self
