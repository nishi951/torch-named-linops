from copy import copy
from typing import Iterable, Sequence, Tuple, Union, Optional, OrderedDict
from collections import OrderedDict

from ._nameddim import ND
from ._nameddimcollection import NamedDimCollection
from ._shapes import get2dor3d

__all__ = [
    "NS",
    "NamedShape",
]

NDorStr = Union[ND, str]


def NS(ishape: NDorStr, oshape: Optional[NDorStr] = None):
    """
    Iif shape is empty, use tuple(), not None
    """
    if ishape is None:
        return NamedShape(ishape=tuple(), oshape=tuple())
    if oshape is None:
        if isinstance(ishape, NamedShape):
            return ishape
        return NamedShape(ishape=ishape, oshape=ishape)
    return NamedShape(ishape=ishape, oshape=oshape)

class NamedShape(NamedDimCollection):
    """A linop shape with input and output dimensions
    Inherit from this to define custom behavior
    - e.g. splitting ishape and oshape into subparts that are linked
    """

    def __init__(self, ishape: Iterable[NDorStr], oshape: Iterable[NDorStr]):
        super().__init__(_ishape=ishape, _oshape=oshape)

    @staticmethod
    def convert(a: Iterable[NDorStr]):
        return list(ND.infer(a))

    def adjoint(self):
        """Return the adjoint shape. Don't call this method directly, but definitely override it"""
        new = type(self)(self.oshape, self.ishape)
        for shape in self.shapes:
            if shape not in ['_ishape', '_oshape']:
                new.add(shape, self.lookup(shape))
        return new

    def normal(self):
        new_oshape = tuple(d.next_unused(self.ishape) for d in self.ishape)
        new = type(self)(self.ishape, new_oshape)
        for shape in self.shapes:
            if shape not in ['_ishape', '_oshape']:
                new.add(shape, self.lookup(shape))
        return new

    @property
    def ishape(self) -> Tuple[ND]:
        return self._ishape

    @ishape.setter
    def ishape(self, val: Iterable[NDorStr]):
        _ishape = self.convert(val)
        self._ishape = _ishape

    @property
    def oshape(self) -> Tuple[ND]:
        return self._oshape

    @oshape.setter
    def oshape(self, val: Iterable[NDorStr]):
        _oshape = self.convert(val)
        self._oshape = _oshape

    @property
    def H(self):
        return self.adjoint()

    @property
    def N(self):
        return self.normal()

    def __repr__(self):
        return f"{self.ishape} -> {self.oshape}"

    def __add__(self, right):
        _ishape = self.ishape + right.ishape
        _oshape = self.oshape + right.oshape
        new = type(self)(ishape=_ishape, oshape=_oshape)
        for shape in self.shapes:
            if shape not in ['_ishape', '_oshape']:
                new.add(shape, self.lookup(shape))
        for shape in right.shapes:
            if shape not in ['_ishape', '_oshape']:
                new.add(shape, right.lookup(shape))
        return new

    def __radd__(self, left):
        if left is None:
            return self
        return left.__add__(self)

    def __eq__(self, other):
        return self.ishape == other.ishape and self.oshape == other.oshape


