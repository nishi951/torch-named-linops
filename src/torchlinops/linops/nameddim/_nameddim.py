from copy import copy
from dataclasses import dataclass
from typing import Any, Tuple, List

__all__ = [
    "ND",
    "NamedDimension",
    "ELLIPSES",
    "ANY",
]

# Special dim names
ELLIPSES = "..."
ANY = "()"


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
