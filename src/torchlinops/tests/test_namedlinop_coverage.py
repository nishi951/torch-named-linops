import copy
import warnings

import pytest
import torch

import torchlinops.config as config
from torchlinops import Dense, Diagonal, Identity
from torchlinops.linops.namedlinop import ForwardedAttribute, NamedLinop
from torchlinops.nameddim import NamedShape as NS


def test_base_adj_fn():
    A = Identity(ishape=("N",), oshape=("N",))
    x = torch.randn(5)
    result = NamedLinop.adj_fn(A, x)
    assert torch.allclose(result, x)


def test_shape_setter():
    A = Identity(ishape=("N",), oshape=("N",))
    A.shape = NS(("A",), ("B",))
    assert A.ishape == ("A",)
    assert A.oshape == ("B",)


def test_repr():
    A = Identity(ishape=("N",), oshape=("N",))
    r = repr(A)
    assert "N" in r


def test_radd():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    B = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    result = B + A
    assert result is not None


def test_mul_scalar():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    result = A * 2.0
    assert result is not None
    x = torch.randn(2)
    y1 = result(x)
    y2 = 2.0 * A(x)
    assert torch.allclose(y1, y2)


def test_rmul_scalar():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    result = 2.0 * A
    assert result is not None


def test_mul_invalid():
    A = Identity(ishape=("N",), oshape=("N",))
    # __mul__ with invalid type returns NotImplemented, but Python raises TypeError
    with pytest.raises(TypeError):
        A * "invalid"


def test_matmul_invalid():
    A = Identity(ishape=("N",), oshape=("N",))
    with pytest.raises(TypeError):
        A @ 42


def test_rmatmul_invalid():
    A = Identity(ishape=("N",), oshape=("N",))
    with pytest.raises(ValueError):
        42.0 @ A


def test_deepcopy():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    B = copy.deepcopy(A)
    assert B is not A
    assert torch.allclose(A.weight, B.weight)


def test_to_cpu():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    B = A.to("cpu")
    assert B is not None


def test_to_memory_aware():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    B = A.to("cpu", memory_aware=True)
    assert B is not None


# --- ForwardedAttribute ---


def test_forwarded_attribute_no_obj():
    fa = ForwardedAttribute(_value=42)
    assert fa.value == 42
    assert not fa.is_forwarded
    fa.value = 99
    assert fa.value == 99


def test_forwarded_attribute_forwarded():
    class Holder:
        x = 10

    fa = ForwardedAttribute()
    fa.forward_to(Holder(), "x")
    assert fa.is_forwarded
    assert fa.value == 10
