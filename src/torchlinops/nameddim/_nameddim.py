from copy import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

__all__ = ["NamedDimension", "ELLIPSES", "ANY", "Dim"]

# Special dim names
ELLIPSES = "..."
ANY = "()"


def Dim(s: Optional[str] = None) -> tuple[str]:
    """Convenience function for splitting a string into a tuple based on simple parsing rules.

    Rules:
    - dim begins with an uppercase letter
    - dim cannot start with a number

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
    name: str
    i: int = 0

    @classmethod
    def infer(cls, dim: Any):
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
