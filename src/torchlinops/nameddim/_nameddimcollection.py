from collections.abc import Iterable, Mapping, Sequence
from warnings import warn

from ._matching import coalesce
from ._nameddim import NamedDimension as ND
from .utils import (
    ELLIPSES_MATCH,
    Assignment,
    is_any,
    is_concrete,
    is_ellipses,
    match,
    normalize,
)

__all__ = ["NamedDimCollection"]


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

    def __getitem__(self, key) -> Sequence | Mapping:
        return self._lookup(key)

    def __setitem__(self, key, newvalue):
        """
        Examples
        --------
        >>> s = NamedDimCollection()
        >>> s.someshape = ("A", "B")
        >>> s.someshape
        (A, B)
        """
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
        return {shape_name: self._lookup(shape_name) for shape_name in self.idx}

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

    def _index(self, data: Iterable[ND | str]):
        """Get index i of data stream (integer-valued)"""
        if isinstance(data, Mapping):
            return {self._dims.index(k): v for k, v in data.items()}
        elif isinstance(data, tuple | list):
            return [self._dims.index(d) for d in data]
        else:
            # Singleton
            return self._dims.index(data)

    def _lookup(self, shape_name):
        """_Lookup a shape by its name"""
        data = self.idx[shape_name]
        if isinstance(data, Mapping):
            return {self._dims[k]: v for k, v in data.items()}
        elif isinstance(data, tuple):
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
        if isinstance(data, (tuple, list)):
            data = coalesce(data)
            indexed_shape = []
            for d in data:
                d = ND.infer(d)
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

    def _apply_rename(self, assignment: Assignment, oldshape_name):
        # First, add the new dims/reassign the old dims
        for dim in assignment.source:
            if (new_dim := assignment.apply(dim)) != dim:
                if (is_concrete(dim) or is_any(dim)) and is_concrete(new_dim):
                    # Replace dim in-place
                    if dim in self._dims:  # only if it hasn't been replaced yet
                        k = self._index(dim)
                        self._dims[k] = ND.infer(new_dim)
                elif is_ellipses(dim):
                    k = self._index(dim)
                    # Add new dims potentially
                    new_idxs = []
                    for new in new_dim:
                        new = ND.infer(new)
                        if new not in self._dims:
                            # Don't affect the indices of existing dims
                            self._dims.append(new)
                        new_idxs.append(self._index(new))
                    data = self.idx[oldshape_name]
                    if isinstance(data, Mapping):
                        oldval = data.pop(k)
                        for n in new_idxs:
                            data[n] = oldval
                    elif isinstance(data, tuple):
                        newdata = []
                        for i in data:
                            if self._dims[i] == dim:
                                newdata.extend(new_idxs)
                            else:
                                newdata.append(i)
                        data = tuple(newdata)
                    else:
                        # Replace int with tuple
                        data = tuple(new_idxs)
                    self.idx[oldshape_name] = data
                else:
                    warn(
                        f"Trying to replace {dim} with {new_dim} is not supported - no change to the dim will be made."
                    )

    def _update_shape(self, oldshape_name, newshape: tuple[ND | str] | ND | str):
        """Update some shape with a new shape
        Shapes must be broadly compatible.

        Allowed Transformations:
        - Changing a Concrete or Any dim to a Concrete Dim
        - Changing an Ellipsis to any combination of dims

        Disallowed Transformations:
        - Changing a Concrete Dim to an Any Dim
        - Changing anything besides an Ellipsis to an Ellipsis

        TBD:
        - Adding Ellipsis to a shape e.g. (A,) -> (..., A)

        """

        oldshape = self._lookup(oldshape_name)
        if isinstance(oldshape, ND | str):  # Updating a Singleton
            oldshape = normalize((oldshape,))
        newshape = normalize(newshape)

        assignment = match(oldshape, newshape, replace=True)
        if assignment is None:  # Length compatibility
            raise ValueError(
                f"Updated shape {newshape} not compatible with current: {oldshape}"
            )
        self._apply_rename(assignment, oldshape_name)
        self._coalesce_ellipses()

    def _coalesce_ellipses(self):
        """Remove duplicate ellipses from tracking and coalesce ellipses in tuple-type shapes."""
        if any(is_ellipses(d) for d in self._dims):
            for k, v in self.idx.items():
                if isinstance(v, tuple):
                    v = tuple(
                        coalesce(
                            v, lambda x, y: ELLIPSES_MATCH(self._dims[x], self._dims[y])
                        )
                    )
                self.idx[k] = v

    def __repr__(self):
        return f"{type(self).__name__}({self._dims})"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
