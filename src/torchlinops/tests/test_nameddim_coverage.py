import warnings

import pytest
import torch

from torchlinops.nameddim._nameddim import Dim, NamedDimension as ND
from torchlinops.nameddim._namedshape import NamedShape as NS
from torchlinops.nameddim._shapes import (
    N2K,
    K2N,
    fake_dims,
    get_nd_shape,
    is_spatial_dim,
)
from torchlinops.nameddim._matching import iscompatible, isequal, partition


# --- _shapes.py ---


def test_get_nd_shape_1d():
    result = get_nd_shape((10,))
    assert len(result) == 1


def test_get_nd_shape_2d():
    result = get_nd_shape((10, 20))
    assert len(result) == 2


def test_get_nd_shape_3d():
    result = get_nd_shape((10, 20, 30))
    assert len(result) == 3


def test_get_nd_shape_invalid():
    with pytest.raises(ValueError):
        get_nd_shape((1, 2, 3, 4))


def test_get_nd_shape_kspace():
    result = get_nd_shape((10, 20), kspace=True)
    assert len(result) == 2


def test_fake_dims():
    result = fake_dims("x", 3)
    assert len(result) == 3


def test_is_spatial_dim():
    assert is_spatial_dim(ND("Nx"))
    assert is_spatial_dim(ND("Ky"))
    assert not is_spatial_dim(ND("B"))


def test_N2K():
    dims = (ND("Nx"), ND("Ny"))
    result = N2K(dims)
    assert len(result) == 2


def test_K2N():
    dims = (ND("Kx"), ND("Ky"))
    result = K2N(dims)
    assert len(result) == 2


# --- _nameddim.py ---


def test_dim_none():
    result = Dim(None)
    assert result == tuple()


def test_dim_empty():
    result = Dim("")
    assert result == tuple()


def test_dim_with_any():
    result = Dim("()A")
    assert "()" in result
    assert "A" in result


def test_nd_add_ellipsis():
    nd = ND("...")
    result = nd + 1
    assert result == nd  # Ellipsis stays unchanged


def test_nd_add_invalid():
    nd = ND("A")
    with pytest.raises(TypeError):
        nd + [1, 2]


# --- _namedshape.py ---


def test_namedshape_repr():
    ns = NS(("A", "B"), ("C", "D"))
    r = repr(ns)
    assert "->" in r


def test_namedshape_radd_none():
    ns = NS(("A",), ("B",))
    result = None + ns
    assert result is ns


# --- _matching.py ---


def test_partition():
    seq = ("A", "B", "C", "D")
    first, mid, last = partition(seq, "B")
    assert first == ("A",)
    assert last == ("C", "D")


def test_partition_not_found():
    seq = ("A", "B")
    result = partition(seq, "Z")
    assert result[0] == seq


def test_isequal_incompatible():
    result = isequal(("A", "B"), ("C",), return_assignments=True)
    assert result[0] is False


def test_iscompatible_string_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        iscompatible("AB", ("C",))
        assert len(w) >= 1
