from typing import Callable, Optional

from torch import Tensor

from ..nameddim import NamedShape, Shape
from .namedlinop import NamedLinop

__all__ = ["SimpleLinop"]


class SimpleLinop(NamedLinop):
    """Ad-hoc linear operator created from plain callables.

    Enables creating linear operators with minimal ceremony, without requiring
    users to subclass NamedLinop. Accepts plain (x) -> y callables for forward,
    adjoint, and optional normal operations.

    Attributes
    ----------
    _forward_fn : Callable
        User-provided forward operation (x) -> y
    _adjoint_fn : Callable
        User-provided adjoint operation (x) -> y
    _normal_fn : Callable, optional
        User-provided normal operation (x) -> y
    """

    def __init__(
        self,
        forward: Callable[[Tensor], Tensor],
        adjoint: Callable[[Tensor], Tensor],
        normal: Optional[Callable[[Tensor], Tensor]] = None,
        ishape: Shape = ("...",),
        oshape: Optional[Shape] = None,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        forward : Callable[[Tensor], Tensor]
            Plain callable (x) -> y implementing the forward operation
        adjoint : Callable[[Tensor], Tensor]
            Plain callable (x) -> y implementing the adjoint operation
        normal : Callable[[Tensor], Tensor], optional
            Plain callable (x) -> y implementing the normal operation.
            If None, uses base class default (adj_fn(fn(x)))
        ishape : Shape, optional
            Input named dimensions, defaults to ("...",)
        oshape : Shape, optional
            Output named dimensions, defaults to ishape
        name : str, optional
            Display name for the linop
        """
        self._forward_fn = forward
        self._adjoint_fn = adjoint
        self._normal_fn = normal

        super().__init__(NamedShape(ishape, oshape), name=name)

    @staticmethod
    def fn(linop, x, /, context=None):
        return linop._forward_fn(x)

    @staticmethod
    def adj_fn(linop, x, /, context=None):
        return linop._adjoint_fn(x)

    @staticmethod
    def normal_fn(linop, x, /, context=None):
        if linop._normal_fn is not None:
            return linop._normal_fn(x)
        return NamedLinop.normal_fn(linop, x, context=context)

    def adjoint(self):
        return SimpleLinop(
            forward=self._adjoint_fn,
            adjoint=self._forward_fn,
            normal=None,
            ishape=self.oshape,
            oshape=self.ishape,
            name=self.name + ".H" if self.name else None,
        )
