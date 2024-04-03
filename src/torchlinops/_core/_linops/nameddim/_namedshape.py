from copy import copy
from typing import Iterable, Sequence, Tuple, Union, Optional

from ._nameddim import ND
from ._shapes import get2dor3d

__all__ = [
    "NS",
    "NamedShape",
    "NamedDiagShape",
    "NamedComboShape",
]

NDorStr = Union[ND, str]


def NS(ishape: NDorStr, oshape: Optional[NDorStr] = None):
    if oshape is None:
        return NamedDiagShape(ishape)
    return NamedShape(ishape, oshape)


class NamedShape:
    """A linop shape with named dimensions
    Behaves like a linop
    Inherit from this to define custom behavior
    - e.g. splitting ishape and oshape into subparts that are linked
    """

    def __init__(self, ishape: Iterable[NDorStr], oshape: Iterable[NDorStr]):
        self._ishape = self.convert(ishape)
        self._oshape = self.convert(oshape)

        self._adjoint = None
        self._normal = None
        self._unnormal = None

        # Tracking updates to prevent infinite recursion
        self._ishape_updated = False
        self._oshape_updated = False

    @staticmethod
    def convert(a: Iterable[NDorStr]):
        return list(ND.infer(a))

    @property
    def ishape(self) -> Tuple[ND]:
        return tuple(self._ishape)

    @ishape.setter
    def ishape(self, val: Iterable[NDorStr]):
        if self._ishape_updated:
            return
        _ishape = self.convert(val)
        self._ishape = _ishape
        self._ishape_updated = True
        if self._adjoint is not None:
            self._adjoint.oshape = _ishape
        if self._unnormal is not None:
            self._unnormal.ishape = _ishape
        if self._normal is not None:
            self._normal.ishape = _ishape
            new_oshape = tuple(d.next_unused(_ishape) for d in _ishape)
            self._normal.oshape = new_oshape
        self._ishape_updated = False

    @property
    def oshape(self) -> Tuple[ND]:
        return tuple(self._oshape)

    @oshape.setter
    def oshape(self, val: Iterable[NDorStr]):
        if self._oshape_updated:
            return
        _oshape = self.convert(val)
        self._oshape = _oshape
        self._oshape_updated = True
        if self._adjoint is not None:
            self._adjoint.ishape = _oshape
        self._oshape_updated = False

    def adjoint(self):
        """Return the adjoint shape. Don't call this method directly, but definitely override it"""
        return type(self)(self.oshape, self.ishape)

    def normal(self):
        new_ishape = tuple(d.next_unused(self.ishape) for d in self.ishape)
        return type(self)(self.ishape, new_ishape)

    @property
    def H(self):
        _adjoint = self.adjoint()
        _adjoint._adjoint = self
        self._adjoint = _adjoint
        return self._adjoint

    @property
    def N(self):
        _normal = self.normal()
        self._normal = _normal
        _normal._unnormal = self
        return self._normal

    def __repr__(self):
        return f"{self.ishape} -> {self.oshape}"


class NamedDiagShape(NamedShape):
    """A namedshape where ishape == oshape"""

    def __init__(self, ioshape: Iterable[NDorStr]):
        self._ioshape = self.convert(ioshape)

    @staticmethod
    def convert(a: Iterable[NDorStr]):
        return list(ND.infer(a))

    @property
    def ishape(self) -> Tuple[ND]:
        return tuple(self._ioshape)

    @ishape.setter
    def ishape(self, val: Iterable[NDorStr]):
        _ioshape = self.convert(val)
        self._ioshape = _ioshape

    @property
    def oshape(self) -> Tuple[ND]:
        return tuple(self._ioshape)

    @oshape.setter
    def oshape(self, val: Iterable[NDorStr]):
        _ioshape = self.convert(val)
        self._ioshape = _oshape

    def adjoint(self):
        return self

    def normal(self):
        return self

    @property
    def H(self):
        return self

    @property
    def N(self):
        return self

    def __len__(self):
        return len(self._ioshape)


class NamedComboShape(NamedShape):
    """A shape that combines parts of a diag and a regular shape

    Might be unnecessary lol
    """

    def __init__(self, ishape, oshape):
        shared_shape = []
        idense = []
        odense = []

        shared = set(ishape) & set(oshape)
        idense = set(ishape) - shared
        odense = set(oshape) - shared

        self.diag = NamedDiagShape(shared_shape)
        self.num_diag = len(self.diag.ishape)
        self.dense = NamedShape(ishape, oshape)

        self.iperm = iperm
        self.operm = operm

        self._adjoint = None
        self._normal = None
        self._unnormal = None

        self._ishape_updated = False
        self._oshape_updated = False

    @property
    def ishape(self):
        return tuple((self.diag.ishape + self.dense.ishape)[self.iperm])

    @ishape.setter
    def ishape(self, val):
        if self._ishape_updated:
            return
        diag_ishape = val[self.iperm][: self.num_diag]
        dense_ishape = val[self.iperm][self.num_diag :]
        self.diag.ishape = diag_ishape
        self.dense.ishape = dense_ishape
        self._ishape_updated = True
        if self._adjoint is not None:
            self._adjoint.diag.ishape = diag_ishape
            self._adjoint.dense.oshape = dense_ishape
        if self._unnormal is not None:
            self._unnormal.diag.ishape = diag_ishape
            self._unnormal.dense.ishape = dense_ishape
        if self._normal is not None:
            self._normal.diag.ishape = diag_ishape
            self._normal.dense.ishape = dense_ishape
            self._normal.dense.oshape = dense_ishape
        self._ishape_updated = False

    @property
    def oshape(self):
        return tuple((self.diag.oshape + self.dense.oshape)[self.operm])

    @oshape.setter
    def oshape(self, val):
        if self._oshape_updated:
            return
        diag_oshape = val[: self.num_diag]
        dense_oshape = val[self.num_diag :]
        self.diag.oshape = diag_oshape
        self.dense.oshape = dense_oshape
        self._oshape_updated = True
        if self._adjoint is not None:
            self._adjoint.diag.oshape = diag_oshape
            self._adjoint.dense.ishape = dense_oshape
        if self._unnormal is not None:
            self._unnormal.diag.ishape = diag_oshape
            self._unnormal.dense.ishape = dense_oshape
            self._unnormal.dense.oshape = dense_oshape
        self._oshape_updated = False

    def adjoint(self):
        return type(self)(self.diag.ishape, self.dense.oshape, self.dense.ishape)

    def normal(self):
        return type(self)(self.diag.ishape, self.dense.ishape, self.dense.ishape)
