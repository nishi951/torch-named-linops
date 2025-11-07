from collections import OrderedDict
from copy import copy
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ._matching import iscompatible
from ._nameddim import ANY, ELLIPSES, ND, NamedDimension

__all__ = ["NamedDimCollection"]

NDorStr = Union[ND, str]


class NamedDimCollection:
    """A collection of NamedDimensions, grouped into named shapes.

    Note: Starting to look a lot like a normal dictionary...

    Attributes
    ----------
    idx :
        Maps from a shape name to a data structure of indices into self._dims
        e.g. {'ishape': (A, B), 'oshape': (C,)}
    self._dims : list
        A list of all the dims in this shape
        e.g. [A, B, C]

    Notes
    -----
    shape = a sequence or mapping of NamedDimensions, or a bare NamedDimension.
    """

    def __init__(self, **shapes):
        self.__dict__["idx"] = {}  # Avoids setattr weirdness
        self.__dict__["_dims"] = []

        for k, v in shapes.items():
            self._add(k, v)

    def __getitem__(self, key):
        return self._lookup(key)

    def __setitem__(self, key, newvalue):
        setattr(self, key, newvalue)

    def update(self, shape_mapping: dict):
        """Add all shapes in the mapping to the collection.
        Behaves similarly to dict.update.
        """
        for key, value in shape_mapping.items():
            self[key] = value

    @property
    def shapes(self) -> dict:
        """The shapes in this collection."""
        # return list(self.idx.keys())
        return {shape_name: self._lookup(shape_name) for shape_name in self.idx.keys()}

    def __getattr__(self, key):
        """Enables attribute-style access of shapes from their names.

        Examples
        --------
        >>> collection = NamedDimCollection(some_shape=("A", "B"))
        >>> collection.some_shape
        (A, B)
        """
        if key == "_dims":  # Normal access
            return self.__dict__["_dims"]
        if key.startswith("__"):
            raise AttributeError(f"Attempted to get missing private attribute: {key}")
        if key in self.__dict__["idx"]:
            return self._lookup(key)
        raise AttributeError(f"{key} not in index: {self.shapes}")

    def __setattr__(self, key, val):
        """Enables attribute-style mutation of shapes from their names.

        Mutating a dim in one shape mutates it in all shapes that include it.

        Shapes must be length-compatible.

        Examples
        --------
        >>> collection = NamedDimCollection(shape1=("A", "B"), shape2=("B", "C"))
        >>> collection.shape1
        (A, B)
        >>> collection.shape1 = ("D", "E")  # (A -> D), (B -> E) across all shapes
        >>> collection.shape1
        (D, E)
        >>> collection.shape2
        (E, C)

        """
        if key in self.idx:
            self._update_shape(key, val)
        else:
            self._add(key, val)

    def _index(self, data: Iterable[NDorStr]):
        """Get index i of data stream (integer-valued)"""
        if isinstance(data, Mapping):
            return {self._dims.index(k): v for k, v in data.items()}
        elif isinstance(data, Iterable):
            return [self._dims.index(d) for d in data]
        else:
            # Singleton
            return self._dims.index(data)

    def _lookup(self, shape_name):
        """_Lookup a shape by its name"""
        data = self.idx[shape_name]
        if isinstance(data, Mapping):
            return {self._dims[k]: v for k, v in data.items()}
        elif isinstance(data, Tuple):
            return tuple(self._dims[i] for i in self.idx[shape_name])
        else:
            return self._dims[self.idx[shape_name]]

    def _add(self, shape_name, data):
        """Create a new shape.

        Parameters
        ----------
        shape_name : str
            The name of the shape.
        data : Tuple, List, or Mapping
            If Tuple or List, all values should be nameddim-able
            If Mapping, all keys should be nameddim-able
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

    def _update_shape(self, oldshape_name, newshape):
        """Update some shape with a new shape"""
        oldshape = self._lookup(oldshape_name)
        if isinstance(oldshape, NamedDimension):  # Updating a Singleton
            if not isinstance(newshape, NamedDimension):
                raise ValueError(
                    f"Trying to update singleton shape {oldshape_name} with non-singleton {newshape}"
                )
            self._dims[self._index(oldshape)] = ND.infer(newshape)
            return
        # Updating a List, Tuple, or Mapping
        iscompat, assignments = iscompatible(
            oldshape, newshape, return_assignments=True
        )
        if not iscompat:
            raise ValueError(
                f"Updated shape {newshape} not compatible with current: {oldshape}"
            )
        for i, olddim in enumerate(oldshape):
            if olddim != ELLIPSES:  # and olddim != ANY:
                # Get the new dim that should replace olddim
                newdims_i_list = assignments[i]
                if len(newdims_i_list) != 1:
                    raise ValueError(
                        f"Non-ellipses dim {olddim} received invalid number of assignment dims {[newshape[j] for j in newdims_i_list]}."
                        + " If this happens, try specifying the shapes further."
                        + f"oldshape: {oldshape} newshape: {newshape}"
                    )
                j = newdims_i_list[0]
                newdim = newshape[j]

                # Only replace olddim with concrete dim names, not ELLIPSES or ANY
                if newdim != ELLIPSES and newdim != ANY:
                    k = self._index(olddim)
                    if olddim != ANY:
                        # Replace all instances of olddim with newdim
                        self._dims[k] = ND.infer(newdim)
                    else:
                        # Replace this specific instance of () with newdim
                        newdim = ND.infer(newdim)
                        if newdim not in self._dims:
                            self._dims.append(newdim)
                        n = self._index(newdim)

                        # Modify idx directly
                        data = self.idx[oldshape_name]
                        # Change k -> n in data
                        if isinstance(data, Mapping):
                            # Fancy way to replace a dictionary key
                            data[n] = data.pop(k)
                        elif isinstance(data, Tuple):
                            # Less fancy way to replace a tuple entry
                            data = list(data)
                            data[data.index(k)] = n
                            data = tuple(data)
                        else:
                            # Unfancy way to replace an int
                            data = n
                        self.idx[oldshape_name] = data

    def __repr__(self):
        return f"{type(self).__name__}({self._dims})"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
