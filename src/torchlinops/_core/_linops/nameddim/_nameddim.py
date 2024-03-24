from copy import copy
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ND",
    "NamedDimension",
]


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
        return cls(dim)

    @classmethod
    def from_tuple(cls, tup):
        return tuple(cls.infer(i) for i in tup)

    def next_unused(self, tup):
        """Get the next dim by index that does not occur in tup"""
        curr = copy(self)
        while curr in tup:
            curr = curr + 1
        return curr

    def __repr__(self):
        return self.name + ("" if self.i == 0 else str(self.i))

    def __add__(self, k):
        try:
            return type(self)(self.name, self.i + k)
        except TypeError as e:
            raise TypeError(f"Unsupported NamedDimension add: {self} + {k}", e)

    def __eq__(self, other):
        """Tests for simple string equality"""
        return repr(self) == other

ND = NamedDimension
