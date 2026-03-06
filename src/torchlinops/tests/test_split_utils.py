import numpy as np

from torchlinops.linops.split import (
    flatten_recursive,
    fuzzy_broadcast_to,
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


def test_fuzzy_broadcast_to_truncate_and_repeat():
    """When source is shorter than target, values should wrap (tile then trim)."""
    arr = np.array([1, 2, 3])
    result = fuzzy_broadcast_to(arr, (4,))
    assert result.shape == (4,)
    np.testing.assert_array_equal(result, [1, 2, 3, 1])


def test_fuzzy_broadcast_to_2d_expand():
    """2-D target: values broadcast across both axes with tiling."""
    arr2 = np.array([1, 2])
    result = fuzzy_broadcast_to(arr2, (3, 5))
    assert result.shape == (3, 5)
    # Each row should be [1, 2, 1, 2, 1]
    np.testing.assert_array_equal(result[0], [1, 2, 1, 2, 1])
    np.testing.assert_array_equal(result[1], result[0])


def test_fuzzy_broadcast_to_ndim_greater_than_arr():
    """When target has more dims than source, source is expanded via np.expand_dims."""
    arr = np.array([1, 2, 3])
    result = fuzzy_broadcast_to(arr, (2, 3))
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result[0], [1, 2, 3])
    np.testing.assert_array_equal(result[1], [1, 2, 3])
