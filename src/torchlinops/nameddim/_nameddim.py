import re
from copy import copy
from dataclasses import dataclass
from typing import Any

__all__ = ["ANY", "ELLIPSES", "AnyDim", "Dim", "NamedDimension", "shape"]

# Special dim names
ELLIPSES = "..."
ANY = "()"


### Various regular expressions
_ANY_DIM = r"\(([0-9]*)\)"
_REGULAR_DIM = r"([a-zA-Z_]*)([0-9]*)"

ANY_PATTERN = re.compile(_ANY_DIM, re.VERBOSE)
REGULAR_PATTERN = re.compile(_REGULAR_DIM, re.VERBOSE)

# Uppercase is used as delimiting info for Shape/Dim only
_REGULAR_DIM_UPPERCASE = "([A-Z][a-z_]*)([0-9]*)"
DIM_PATTERN = re.compile(f"{_REGULAR_DIM_UPPERCASE}|{_ANY_DIM}", re.VERBOSE)


def shape(s: str | None = None) -> tuple[str, ...]:
    """Convenience function for splitting a string into a tuple of dimension names.

    Parses a compact dimension string into individual dimension names using
    simple rules:

    - A new dimension begins at each uppercase letter.
    - A dimension name cannot start with a digit.
    - The special token ``()`` (ANY) is recognised and split out.
    - Ordinal ANY tokens ``(1)``, ``(2)``, ``(23)``, etc. are recognised.

    Parameters
    ----------
    s : str or None, optional
        The compact dimension string to parse. If ``None`` or empty, an
        empty tuple is returned.

    Returns
    -------
    tuple of str
        A tuple of individual dimension name strings.

    Examples
    --------
    >>> shape("ABCD")
    ('A', 'B', 'C', 'D')
    >>> shape("NxNyNz")
    ('Nx', 'Ny', 'Nz')
    >>> shape("A1B2Kx1Ky2")
    ('A1', 'B2', 'Kx1', 'Ky2')
    >>> shape("()A()B")
    ('()', 'A', '()', 'B')
    >>> shape("(1)A(23)")
    ('(1)', 'A', '(23)')
    >>> shape()
    """
    if s is None or len(s) == 0:
        return ()
    shape = []
    pos = 0
    while pos < len(s):
        m = DIM_PATTERN.match(s, pos)
        if m is None:
            raise ValueError(f"no token matches at index {pos}: {s[pos : pos + 10]!r}")
        shape.append(m.group())
        pos = m.end()
    return tuple(shape)


# Alias for backwards compatibility
Dim = shape


@dataclass(slots=True, frozen=True)
class NamedDimension:
    """Fundamental named dimension type used throughout the library.

    Each dimension has a ``name`` and an optional integer index ``i`` for
    creating indexed variants (e.g. ``A1``, ``A2``).  Two
    ``NamedDimension`` instances are considered equal when their string
    representations match; the index is folded into the representation
    rather than compared separately.

    Parameters
    ----------
    name : str
        The base name of the dimension (e.g. ``'A'``, ``'Nx'``).
    i : int, optional
        Integer index for indexed variants.  Defaults to ``0``, which is
        omitted from the string representation.

    Examples
    --------
    >>> NamedDimension("A")
    A
    >>> NamedDimension("A", 1)
    A1
    >>> NamedDimension("A") == "A"
    True
    """

    name: str
    i: int = 0

    @classmethod
    def infer(cls, dim: Any):
        """Create a NamedDimension by inferring the name and optional index.

        If *dim* is already a ``NamedDimension`` it is returned as-is.
        A two-character string whose second character is a digit is
        interpreted as ``name=dim[0], i=int(dim[1])``.  Sequences are
        inferred element-wise.

        Parameters
        ----------
        dim : Any
            A ``NamedDimension``, a string, or a list/tuple of those.

        Returns
        -------
        NamedDimension or sequence thereof
            The inferred dimension(s).
        """
        if isinstance(dim, cls):
            return dim
        if dim == ELLIPSES:
            return cls(ELLIPSES)
        if isinstance(dim, str):
            m = ANY_PATTERN.match(dim)
            if m is not None:
                val = m.groups()[0]
                val = int(val) if len(val) > 0 else 0
                return cls(ANY, val)
            m = REGULAR_PATTERN.match(dim)
            if m is not None:
                name, val = m.groups()
                val = int(val) if len(val) > 0 else 0
                return cls(name, val)
            raise ValueError(f"Could not infer named dimension from string {dim}")
        if isinstance(dim, (tuple, list)):
            return type(dim)(cls.infer(d) for d in dim)
        return cls(dim)

    def next_unused(self, avoid=None):
        """Get the next dim by index that does not occur in tup"""
        curr = copy(self)
        if avoid is None:
            return curr + 1
        if self.name == ELLIPSES:
            return curr
        while curr in avoid:
            curr = curr + 1
        return curr

    def __repr__(self):
        if self.name == ANY and self.i > 0:
            return f"({self.i})"
        return self.name + ("" if self.i == 0 else str(self.i))

    def __add__(self, k):
        if self.name == ELLIPSES:
            return self
        try:
            return type(self)(self.name, self.i + k)
        except TypeError as e:
            raise TypeError(f"Unsupported NamedDimension add: {self} + {k}", e)

    def __eq__(self, other):
        """Tests for simple string equality"""
        return repr(self) == other

    def __hash__(self):
        """Allow dictionary lookups to work with strings too."""
        return hash(repr(self))


def AnyDim(i: int | None = None) -> NamedDimension:
    """Create an ANY dimension with optional ordinal.

    Parameters
    ----------
    i : int, optional
        Ordinal index. If None, returns base ANY (i=0).

    Returns
    -------
    NamedDimension
        An ANY dimension, e.g. (), (1), (2), etc.

    Examples
    --------
    >>> AnyDim()
    ()
    >>> AnyDim(1)
    (1)
    >>> AnyDim(2)
    (2)
    """
    return NamedDimension(ANY, i if i is not None else 0)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
