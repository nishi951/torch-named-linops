"""Tests for utility modules: _batching, _defaults, _pad."""

import pytest

from torchlinops.utils._batching import (
    batch_iterator,
    batch_tqdm,
    ceildiv,
    dict_product,
)
from torchlinops.utils._defaults import default_to, default_to_dict
from torchlinops.utils._pad import end_pad_with_zeros
import torch


# --- ceildiv ---


def test_ceildiv_exact():
    assert ceildiv(9, 3) == 3


def test_ceildiv_ceiling():
    assert ceildiv(10, 3) == 4


def test_ceildiv_single():
    assert ceildiv(1, 1) == 1


# --- batch_iterator ---


def test_batch_iterator_basic():
    result = list(batch_iterator(10, 3))
    assert result == [(0, 3), (3, 6), (6, 9), (9, 10)]


def test_batch_iterator_exact_multiple():
    result = list(batch_iterator(6, 2))
    assert result == [(0, 2), (2, 4), (4, 6)]


def test_batch_iterator_zero_total_asserts():
    """batch_iterator must assert total > 0."""
    with pytest.raises(AssertionError):
        list(batch_iterator(0, 5))


# --- batch_tqdm ---


def test_batch_tqdm_yields_correct_ranges():
    result = list(batch_tqdm(7, 3, disable=True))
    assert result == [(0, 3), (3, 6), (6, 7)]


# --- dict_product ---


def test_dict_product_single_key():
    result = dict_product({"a": [1, 2, 3]})
    assert result == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_dict_product_two_keys():
    result = dict_product({"a": [1, 2], "b": ["x", "y"]})
    assert len(result) == 4
    assert {"a": 1, "b": "x"} in result
    assert {"a": 2, "b": "y"} in result


# --- default_to ---


def test_default_to_returns_first_non_none():
    assert default_to(1, None, 3) == 3  # reversed order: last non-None wins


def test_default_to_typecast():
    result = default_to((1, 2, 3), [4, 5, 6], typecast=True)
    # typecls is tuple (from vals[0]), and typecast=True casts the result
    assert isinstance(result, tuple)


def test_default_to_zero_args():
    assert default_to() is None


def test_default_to_single_arg():
    assert default_to(42) == 42


# --- default_to_dict ---


def test_default_to_dict_basic():
    result = default_to_dict({"a": 1}, {"b": 2})
    assert result == {"a": 1, "b": 2}


def test_default_to_dict_none_entry():
    result = default_to_dict(None, {"a": 1})
    assert result == {"a": 1}


def test_default_to_dict_non_dict_raises():
    with pytest.raises(ValueError, match="Non-dictionary"):
        default_to_dict("not a dict")


def test_default_to_dict_empty():
    assert default_to_dict() == {}


# --- end_pad_with_zeros ---


def test_end_pad_with_zeros_basic():
    x = torch.tensor([1.0, 2.0, 3.0])
    result = end_pad_with_zeros(x, dim=0, pad_length=2)
    assert result.shape == (5,)
    assert result[3] == 0.0
    assert result[4] == 0.0
    assert torch.allclose(result[:3], x)


def test_end_pad_with_zeros_zero_length():
    x = torch.tensor([1.0, 2.0])
    result = end_pad_with_zeros(x, dim=0, pad_length=0)
    assert torch.allclose(result, x)
