import warnings

import numpy as np
import pytest

from torchlinops.linops.split import (
    BatchSpec,
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


def test_flatten_recursive_max_depth_zero():
    """flatten_recursive with max_depth=0 should not flatten any nested lists."""
    nested = [[1, 2], [3, [4, 5]]]
    result = flatten_recursive(nested, max_depth=0)
    assert result == [[1, 2], [3, [4, 5]]]


def test_flatten_recursive_max_depth_one():
    """flatten_recursive with max_depth=1 should flatten exactly one level."""
    nested = [[1, 2], [3, [4, 5]]]
    result = flatten_recursive(nested, max_depth=1)
    assert result == [1, 2, 3, [4, 5]]


def test_flatten_recursive_full():
    """flatten_recursive with no max_depth should fully flatten."""
    nested = [[1, [2, [3]]], [4]]
    result = flatten_recursive(nested)
    assert result == [1, 2, 3, 4]


def test_repeat_along_axes_more_repeats_than_ndim():
    """When len(repeats) > arr.ndim, singleton dims are added then repeated."""
    arr = np.array([[1, 2]])  # shape (1, 2)
    result = repeat_along_axes(arr, [3, 1, 2])  # 3 repeats requested on 3 axes
    assert result.shape[0] == 3


def test_batch_spec_non_dict_warns():
    """BatchSpec should emit a UserWarning when batch_sizes is not a dict."""
    with pytest.warns(UserWarning):
        BatchSpec(batch_sizes=[("N", 2)])
