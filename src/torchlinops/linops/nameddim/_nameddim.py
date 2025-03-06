from typing import Optional

from copy import copy
from dataclasses import dataclass
from typing import Any, Tuple, List

__all__ = ["ND", "NamedDimension", "ELLIPSES", "ANY", "Dim", "parse_dim_str"]

# Special dim names
ELLIPSES = "..."
ANY = "()"


# Convenience function
def parse_dim_str(s: Optional[str] = None) -> tuple[str]:
    """
    Rules:
    - dim begins with an uppercase letter
    - dim cannot start with a number

    Examples
    --------
    >>> parse_dim_str("ABCD")
    ('A', 'B', 'C', 'D')
    >>> parse_dim_str("NxNyNz")
    ('Nx', 'Ny', 'Nz')
    >>> parse_dim_str("A1B2Kx1Ky2")
    ('A1', 'B2', 'Kx1', 'Ky2')
    """
    if s is None or len(s) == 0:
        return tuple()
    parts = []
    current = s[0]
    for char in s[1:]:
        if char.isupper():
            if char.isdigit():
                raise ValueError(f"Dim cannot start with a digit in dim string {s}")
            parts.append(current)
            current = char
        else:
            current += char
    parts.append(current)
    return tuple(parts)


Dim = parse_dim_str


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


ND = NamedDimension
