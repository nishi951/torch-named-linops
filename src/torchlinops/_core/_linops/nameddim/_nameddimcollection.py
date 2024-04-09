from copy import copy
from typing import (
    Iterable,
    Sequence,
    Tuple,
    Union,
    Optional,
    OrderedDict,
    Mapping,
    List,
)
from collections import OrderedDict

from ._nameddim import ND

__all__ = ["NamedDimCollection"]

NDorStr = Union[ND, str]


class NamedDimCollection:
    """A collection of named dimensions
    Updating some dimensions updates all of them
    Inherit from this to define custom behavior
    """

    def __init__(self, **shapes):
        self.__dict__["idx"] = {}
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
        if isinstance(data, Mapping):
            return {self._dims.index(k): v for k, v in data.items()}
        elif isinstance(data, Iterable):
            return [self._dims.index(d) for d in data]
        else:
            # Singleton
            return self._dims.index(data)

    def lookup(self, shape_name):
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

    def update(self, shape_name, shape):
        assert (
            len(shape) == len(self.idx[shape_name])
        ), f"Updated shape differs from current (immutable) shape length: shape: {shape} current: {self.lookup(shape_name)}"
        for i, j in enumerate(self.idx[shape_name]):
            self._dims[j] = ND.infer(shape[i])

    def __repr__(self):
        return f"{type(self).__name__}({self._dims})"
