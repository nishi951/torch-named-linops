from copy import copy
from typing import (
    Iterable,
    Sequence,
    Tuple,
    Union,
    Optional,
    Mapping,
    List,
)
from collections import OrderedDict

from ._nameddim import NamedDimension, ND, ELLIPSES, ANY
from ._matching import iscompatible

__all__ = ["NamedDimCollection"]

NDorStr = Union[ND, str]


class NamedDimCollection:
    """A collection of named dimensions
    Updating some dimensions updates all of them
    Inherit from this to define custom behavior

    self.idx :
        Maps from a shape name to a data structure of indices into self._dims
    self._dims : List
        A List of all the dims in this shape
    """

    def __init__(self, **shapes):
        self.__dict__["idx"] = {}  # Avoids setattr weirdness
        self._dims = []
        self._adjoint = None
        self._normal = None
        self._unnormal = None

        for k, v in shapes.items():
            self.add(k, v)

    @property
    def shapes(self):
        return list(self.idx.keys())

    def __getattr__(self, key):
        if key.startswith("__"):
            raise AttributeError(f"Attempted to get missing private attribute: {key}")
        if key in self.__dict__["idx"]:
            return self.lookup(key)
        raise AttributeError(f"{key} not in index: {self.shapes}")

    def __setattr__(self, key, val):
        if key in self.idx:
            self.update(key, val)
        else:
            # New shape attributes must be created via `.add` first
            # This is to maintain the ability to add regular attributes
            super().__setattr__(key, val)

    def index(self, data: Iterable[NDorStr]):
        """Get index i of data stream (integer-valued)"""
        if isinstance(data, Mapping):
            return {self._dims.index(k): v for k, v in data.items()}
        elif isinstance(data, Iterable):
            return [self._dims.index(d) for d in data]
        else:
            # Singleton
            return self._dims.index(data)

    def lookup(self, shape_name):
        """Lookup a shape by its name"""
        data = self.idx[shape_name]
        if isinstance(data, Mapping):
            return {self._dims[k]: v for k, v in data.items()}
        elif isinstance(data, Tuple):
            return tuple(self._dims[i] for i in self.idx[shape_name])
        else:
            return self._dims[self.idx[shape_name]]

    def add(self, shape_name, data):
        """
        data : Tuple, List, or Mapping
            If Tuple or List, all values should be nameddim-able
            IF Mapping, all keys should be nameddim-able
        """
        if shape_name in self.idx:
            raise ValueError(f"{shape_name} already in index of shape: {self}")
        if isinstance(data, Tuple) or isinstance(data, List):
            indexed_shape = []
            for d in data:
                if d not in self._dims:
                    self._dims.append(ND.infer(d))
                indexed_shape.append(self._dims.index(d))
            indexed_shape = tuple(indexed_shape)
        elif isinstance(data, Mapping):
            indexed_shape = {}
            for d, v in data.items():
                if d not in self._dims:
                    self._dims.append(ND.infer(d))
                indexed_shape[self._dims.index(d)] = v
        else:
            # Single dim
            if data not in self._dims:
                self._dims.append(ND.infer(data))
            indexed_shape = self._dims.index(data)
        self.idx[shape_name] = indexed_shape

    def update(self, oldshape_name, newshape):
        """Update some shape with a new shape"""
        oldshape = self.lookup(oldshape_name)
        if isinstance(oldshape, NamedDimension):  # Singleton
            if not isinstance(newshape, NamedDimension):
                raise ValueError(
                    f"Trying to update singleton shape {oldshape_name} with non-singleton {newshape}"
                )
            self._dims[self.index(oldshape)] = ND.infer(newshape)
            return
        # List, Tuple, or Mapping
        iscompat, assignments = iscompatible(
            oldshape, newshape, return_assignments=True
        )
        assert (
            iscompat
        ), f"Updated shape {newshape} not compatible with current: {oldshape}"
        for i, olddim in enumerate(oldshape):
            if olddim != ELLIPSES and olddim != ANY:
                newdims_i = assignments[i]
                if len(newdims_i) != 1:
                    raise ValueError(
                        f"Non-ellipses dim {olddim} received invalid number of assignment dims {[newshape[j] for j in newdims_i]}."
                        + " If this happens, try specifying the shapes further."
                        + f"oldshape: {oldshape} newshape: {newshape}"
                    )
                j = newdims_i[0]
                newdim = newshape[j]
                if newdim != ELLIPSES and newdim != ANY:
                    k = self.index(olddim)
                    self._dims[k] = ND.infer(newdim)

    def __repr__(self):
        return f"{type(self).__name__}({self._dims})"
