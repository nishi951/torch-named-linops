# pragma: exclude file

from ..nameddim import NamedShape as NS
from ..nameddim import Shape
from .namedlinop import NamedLinop

__all__ = ["BreakpointLinop"]


class BreakpointLinop(NamedLinop):
    """Debugging identity operator that drops into ``pdb`` on forward/adjoint.

    Useful for inspecting intermediate tensor values inside a ``Chain``.
    """

    def __init__(self, ioshape: Shape | None = None):
        super().__init__(NS(ioshape))

    @staticmethod
    def fn(linop, x, /):
        breakpoint()  # noqa: T100
        return x

    @staticmethod
    def adj_fn(linop, x, /):
        breakpoint()  # noqa: T100
        return x

    @staticmethod
    def split(linop, tile):
        return linop
