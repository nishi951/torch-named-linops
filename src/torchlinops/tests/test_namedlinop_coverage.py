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


# --- int support for __mul__/__rmul__ ---


def test_mul_int():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    result = A * 2
    x = torch.randn(2)
    assert torch.allclose(result(x), 2 * A(x))


def test_rmul_int():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    result = 2 * A
    x = torch.randn(2)
    assert torch.allclose(result(x), 2 * A(x))


def test_mul_negative_int():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    result = A * (-3)
    x = torch.randn(2)
    assert torch.allclose(result(x), (-3) * A(x))


# --- __neg__ tests ---


def test_neg_basic():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    neg_A = -A
    assert neg_A is not None


def test_neg_correctness():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    x = torch.randn(2)
    assert torch.allclose((-A)(x), -A(x))


def test_neg_adjoint():
    A = Dense(torch.randn(3, 2, dtype=torch.complex64), ("M", "N"), ("N",), ("M",))
    y = torch.randn(3, dtype=torch.complex64)
    assert torch.allclose((-A).H(y), -(A.H(y)))


def test_neg_double():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    x = torch.randn(2)
    assert torch.allclose((-(-A))(x), A(x))


def test_neg_complex():
    A = Dense(torch.randn(3, 2, dtype=torch.complex64), ("M", "N"), ("N",), ("M",))
    x = torch.randn(2, dtype=torch.complex64)
    assert torch.allclose((-A)(x), -A(x))


def test_neg_identity():
    A = Identity(ishape=("N",), oshape=("N",))
    x = torch.randn(5)
    assert torch.allclose((-A)(x), -x)


# --- __sub__ tests ---


def test_sub_basic():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    B = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    result = A - B
    assert result is not None


def test_sub_correctness():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    B = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    x = torch.randn(2)
    assert torch.allclose((A - B)(x), A(x) - B(x))


def test_sub_adjoint():
    A = Dense(torch.randn(3, 2, dtype=torch.complex64), ("M", "N"), ("N",), ("M",))
    B = Dense(torch.randn(3, 2, dtype=torch.complex64), ("M", "N"), ("N",), ("M",))
    y = torch.randn(3, dtype=torch.complex64)
    assert torch.allclose((A - B).H(y), A.H(y) - B.H(y))


def test_sub_shape_mismatch():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    B = Dense(torch.randn(4, 5), ("P", "Q"), ("Q",), ("P",))
    with pytest.raises(AssertionError):
        _ = A - B


# --- __rsub__ tests ---


def test_rsub_basic():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    B = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    result = B.__rsub__(A)
    x = torch.randn(2)
    assert torch.allclose(result(x), A(x) - B(x))


def test_rsub_with_nonlinop():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    with pytest.raises(TypeError):
        _ = 5 - A


def test_rsub_tensor():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    tensor = torch.randn(3)
    with pytest.raises((TypeError, AttributeError)):
        _ = tensor - A


# --- additional edge cases ---


def test_sub_add_equivalence():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    B = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    x = torch.randn(2)
    sub_result = (A - B)(x)
    add_result = (A + (-B))(x)
    assert torch.allclose(sub_result, add_result)


def test_chain_with_neg():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    B = Dense(torch.randn(2, 4), ("N", "K"), ("K",), ("N",))
    x = torch.randn(4)
    result = ((-A) @ B)(x)
    expected = -(A @ B)(x)
    assert torch.allclose(result, expected)


def test_mul_tensor_weight():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    scalar_tensor = torch.tensor(2.5)
    x = torch.randn(2)
    result = A * scalar_tensor
    assert torch.allclose(result(x), 2.5 * A(x))


def test_neg_linop_repr():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    neg_A = -A
    assert repr(neg_A)


def test_sub_chain():
    A = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    B = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    C = Dense(torch.randn(3, 2), ("M", "N"), ("N",), ("M",))
    x = torch.randn(2)
    result = ((A - B) - C)(x)
    expected = A(x) - B(x) - C(x)
    assert torch.allclose(result, expected)
