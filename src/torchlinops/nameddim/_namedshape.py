from collections.abc import Sequence
from typing import Optional

from ._matching import isequal
from ._nameddim import NamedDimension as ND
from ._nameddimcollection import NamedDimCollection

__all__ = ["NamedShape", "Shape"]

Shape = Sequence[ND | str]


class NamedShape(NamedDimCollection):
    """A linop shape with input and output dimensions
    Inherit from this to define custom behavior
    - e.g. splitting ishape and oshape into subparts that are linked
    """

    def __init__(
        self,
        ishape: Optional["Shape | NamedShape"],
        oshape: Optional[Shape] = None,
        **other_shapes,
    ):
        """Construct a NamedShape from input and output dimension names.

        Parameters
        ----------
        ishape : Shape or NamedShape or None
            Input dimension names.  If a ``NamedShape`` instance is passed,
            it is copied directly and *oshape* / *other_shapes* are ignored.
            If ``None``, defaults to the ellipsis shape ``('...',)``.
        oshape : Shape or None, optional
            Output dimension names.  If ``None`` while *ishape* is provided,
            the operator is treated as diagonal (``oshape = ishape``).  If
            both *ishape* and *oshape* are ``None``, both default to
            ``('...',)``.
        **other_shapes
            Additional named shape sequences stored alongside ishape and
            oshape (e.g. auxiliary dimensions for specialised operators).
        """
        # Pass-through
        if isinstance(ishape, type(self)):
            super().__init__(**ishape.shapes)
            return

        if oshape is None:
            if ishape is None:
                # Empty shape
                oshape = ("...",)
            else:
                # Diagonal
                oshape = ishape
        if ishape is None:
            ishape = ("...",)
        super().__init__(ishape=ishape, oshape=oshape, **other_shapes)

    @property
    def other_shapes(self):
        """Shapes that are not ishape or oshape."""
        other_shapes = self.shapes.copy()
        for name in ["ishape", "oshape"]:  # Special attributes
            other_shapes.pop(name)
        return other_shapes

    def adjoint(self):
        """Return a new NamedShape with ishape and oshape swapped.

        Override this method in subclasses that need custom adjoint
        behaviour (e.g. swapping auxiliary shapes as well).

        Returns
        -------
        NamedShape
            A new instance with ``ishape`` and ``oshape`` exchanged.
        """
        new = type(self)(self.oshape, self.ishape, **self.other_shapes)
        return new

    def normal(self):
        """Return the NamedShape for the normal operator (A^H A).

        The resulting shape has ``ishape`` equal to the original ``ishape``
        and ``oshape`` derived from ``ishape`` with indices incremented to
        avoid collisions, representing the domain-to-domain mapping of the
        normal equation.

        Returns
        -------
        NamedShape
            A new instance representing the normal operator shape.
        """
        new_oshape = tuple(d.next_unused(self.ishape) for d in self.ishape)
        new = type(self)(self.ishape, new_oshape, **self.other_shapes)
        return new

    @property
    def H(self) -> "NamedShape":
        """The adjoint NamedShape (ishape and oshape swapped)."""
        return self.adjoint()

    @property
    def N(self) -> "NamedShape":
        """The normal NamedShape for the operator A^H A."""
        return self.normal()

    def __repr__(self):
        return f"{self.ishape} -> {self.oshape}"

    def __add__(self, right) -> "NamedShape":
        try:
            _ishape = self.ishape + right.ishape
        except TypeError as e:
            raise TypeError(
                f"Problem combining shapes {self.ishape} + {right.ishape}"
            ) from e
        try:
            _oshape = self.oshape + right.oshape
        except TypeError as e:
            raise TypeError(
                f"Problem combining shapes {self.oshape} + {right.oshape}"
            ) from e
        new = type(self)(ishape=_ishape, oshape=_oshape)
        new.update(self.other_shapes)
        new.update(right.other_shapes)
        return new

    def __radd__(self, left):
        if left is None:
            return self
        return left.__add__(self)

    def __eq__(self, other):
        return isequal(self.ishape, other.ishape) and isequal(self.oshape, other.oshape)
