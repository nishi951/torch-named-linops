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
        self._adjoint = None
        self._normal = None
        self._unnormal = None
        self._updated = {k: False for k in self.shapes}

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

    @property
    def ishape(self) -> Tuple[ND]:
        return self._ishape

    @ishape.setter
    def ishape(self, val: Iterable[NDorStr]):
        if self._updated['_ishape']:
            return
        _ishape = self.convert(val)
        self._ishape = _ishape
        self._updated['_ishape'] = True
        if self._adjoint is not None:
            self._adjoint.oshape = _ishape
        self._updated['_ishape'] = False

    @property
    def oshape(self) -> Tuple[ND]:
        return self._oshape

    @oshape.setter
    def oshape(self, val: Iterable[NDorStr]):
        if self._updated['_oshape']:
            return
        _oshape = self.convert(val)
        self._oshape = _oshape
        self._updated['_oshape'] = True
        if self._adjoint is not None:
            self._adjoint.ishape = _oshape
        self._updated['_oshape'] = False

    @property
    def H(self):
        _adjoint = self.adjoint()
        _adjoint._adjoint = self
        self._adjoint = _adjoint
        return self._adjoint

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


