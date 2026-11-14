"""Temporary staging area for new functionality

Once functionality is established, we will refactor the entire module.
"""

from collections.abc import Sequence

from ._matching import coalesce
from ._nameddim import ELLIPSES, shape
from ._nameddim import NamedDimension as ND

Shape = tuple[ND, ...]
AnyShape = (ELLIPSES,)
EmptyShape = ()


def normalize(value: str | Sequence | None) -> Shape:
    """Canonicalize an input into a shape object.

    Coalesces adjacent ellipses together.

    Examples
    --------
    >>> normalize("AB")
    (A, B)
    >>> normalize(("A", "...", "...", "B"))
    (A, ..., B)
    >>> normalize(())
    ()
    >>> normalize(None)
    (...,)
    >>> normalize("(1)")
    ((1),)
    """
    if value is None:
        return AnyShape
    if isinstance(value, list | tuple):
        output = [ND.infer(s) for s in value]
    elif isinstance(value, str):
        output = [ND.infer(s) for s in shape(value)]
    # Coalesce ellipses
    output = coalesce(output, lambda x: x == ELLIPSES)
    return tuple(output)


def match(): ...


def unify(): ...


def assign(): ...
