import numpy as np

from torchlinops.linops.split import (
    flatten_recursive,
    is_broadcastable,
    repeat_along_axes,
    tile_along_axes,
)


def test_flatten_recursive_nested():
    result = flatten_recursive([[1, 2], [3, 4]])
    assert result == [1, 2, 3, 4]


def test_flatten_recursive_1d():
    result = flatten_recursive([1, 2, 3])
    assert result == [1, 2, 3]


def test_flatten_recursive_max_depth():
    result = flatten_recursive([[1, 2], [3, [4, 5]]], max_depth=1)
    assert result == [1, 2, 3, [4, 5]]


def test_is_broadcastable():
    assert is_broadcastable((3, 1), (1, 4))
    assert is_broadcastable((1,), (5,))
    assert not is_broadcastable((3,), (4,))


def test_repeat_along_axes():
    arr = np.array([[1, 2]])
    result = repeat_along_axes(arr, [3])
    assert result.shape[0] == 3


def test_tile_along_axes():
    arr = np.array([[1, 2]])
    result = tile_along_axes(arr, [3])
    assert result.shape[0] == 3
