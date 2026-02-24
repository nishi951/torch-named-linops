from copy import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

__all__ = ["NamedDimension", "ELLIPSES", "ANY", "Dim"]

# Special dim names
ELLIPSES = "..."
ANY = "()"


def Dim(s: Optional[str] = None) -> tuple[str]:
    """Convenience function for splitting a string into a tuple of dimension names.

    Parses a compact dimension string into individual dimension names using
    simple rules:

    - A new dimension begins at each uppercase letter.
    - A dimension name cannot start with a digit.
    - The special token ``()`` (ANY) is recognised and split out.

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
    >>> Dim("ABCD")
    ('A', 'B', 'C', 'D')
    >>> Dim("NxNyNz")
    ('Nx', 'Ny', 'Nz')
    >>> Dim("A1B2Kx1Ky2")
    ('A1', 'B2', 'Kx1', 'Ky2')
    >>> Dim("()A()B")
    ('()', 'A', '()', 'B')
    """
    if s is None or len(s) == 0:
        return tuple()
    parts = []
    current = s[0]
    for char in s[1:]:
        if char == "(":
            parts.append(current)
            current = char
        elif current == "()":
            parts.append(current)
            current = char
        elif char.isupper():
            # New dim
            if char.isdigit():
                raise ValueError(f"Dim cannot start with a digit in dim string {s}")
            parts.append(current)
            current = char
        else:
            current += char
    parts.append(current)
    return tuple(parts)


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
        if isinstance(dim, str) and len(dim) == 2:
            if dim[1].isdigit():
                return cls(dim[0], int(dim[1]))
        elif dim == ELLIPSES:
            return cls(ELLIPSES)
        elif isinstance(dim, Tuple) or isinstance(dim, List):
            return type(dim)(cls.infer(d) for d in dim)
        return cls(dim)

    def next_unused(self, tup):
        """Get the next dim by index that does not occur in tup"""
        curr = copy(self)
        if self.name == ELLIPSES or self.name == ANY:
            return curr
        while curr in tup:
            curr = curr + 1
        return curr

    def __repr__(self):
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
