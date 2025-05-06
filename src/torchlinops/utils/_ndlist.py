from typing import Optional

__all__ = ["NDList"]


class NDList:
    def __init__(self, shape, default_value=None, labels: Optional[tuple] = None):
        """
        Initializes an n-dimensional nested with the given shape.
        :param shape: Tuple representing the shape of the array.
        :param default_value: The default value to fill the array.
        """
        self.shape = shape
        self.data = self._create_nd_array(shape, default_value)
        if labels is None:
            self.labels = (None,) * len(self.shape)
        else:
            if len(labels) != len(self.shape):
                raise ValueError(
                    f"If specified, labels must have same length as shape but got {len(labels)} != {len(shape)}"
                )
            self.labels = labels

    def _create_nd_array(self, shape, default_value, depth=0):
        if depth == len(shape) - 1:
            return [default_value] * shape[depth]
        return [
            self._create_nd_array(shape, default_value, depth + 1)
            for _ in range(shape[depth])
        ]

    def _get_nested(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        elem = self.data
        for index in indices:
            if index == slice(None):
                return [
                    self._get_nested((i,) + indices[1:]) for i in range(self.shape[0])
                ]
            elem = elem[index]
        return elem

    def _set_nested(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if indices[0] == slice(None):
            for i in range(self.shape[0]):
                self._set_nested((i,) + indices[1:], value)
            return
        elem = self.data
        for index in indices[:-1]:
            elem = elem[index]
        elem[indices[-1]] = value

    def __getitem__(self, index):
        return self._get_nested(index)

    def __setitem__(self, index, value):
        self._set_nested(index, value)

    def __repr__(self):
        return repr(self.data)
