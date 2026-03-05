import pytest
import torch

from torchlinops import Add, Dense, Identity
from torchlinops.utils import is_adjoint


@pytest.fixture
def dense_linops():
    A = Dense(torch.randn(4, 3, dtype=torch.complex64), ("N", "M"), ("N",), ("N", "M"))
    B = Dense(torch.randn(4, 3, dtype=torch.complex64), ("N", "M"), ("N",), ("N", "M"))
    C = Dense(torch.randn(4, 3, dtype=torch.complex64), ("N", "M"), ("N",), ("N", "M"))
    return A, B, C


class TestAddAdjoint:
    def test_adjoint_method(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C)
        adj = add.adjoint()
        assert isinstance(adj, Add)
        assert len(adj.linops) == 3

    def test_adjoint_property(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C)
        adj = add.H
        assert isinstance(adj, Add)


class TestAddNormal:
    def test_normal_method(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C)
        normal = add.normal()
        assert isinstance(normal, Add)

    def test_normal_property(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C)
        normal = add.N
        assert isinstance(normal, Add)


class TestAddMethods:
    def test_size_returns_first_non_none(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C)
        size = add.size("N")
        assert size == 4

    def test_size_returns_none_when_all_none(self, dense_linops):
        A = Identity(ishape=("N",), oshape=("N",))
        B = Identity(ishape=("N",), oshape=("N",))
        add = Add(A, B)
        size = add.size("N")
        assert size is None

    def test_dims(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C)
        dims = add.dims
        assert "N" in dims
        assert "M" in dims


class TestAddDunderMethods:
    def test_len(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C)
        assert len(add) == 3

    def test_getitem(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C)
        assert add[0] is A
        assert add[1] is B
        assert add[2] is C

    def test_flatten(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C)
        flat = add.flatten()
        assert flat == [add]

    def test_repr(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C)
        repr_str = repr(add)
        assert "Add" in repr_str or "+" in repr_str or "Dense" in repr_str


class TestAddAdjointness:
    def test_add_is_linear(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C)
        x = torch.randn(4, dtype=torch.complex64)
        y = torch.randn(4, 3, dtype=torch.complex64)
        assert is_adjoint(add, x, y)


class TestAddWithDifferentShapes:
    def test_add_must_have_same_ishape(self):
        A = Dense(torch.randn(4, 3), ("N", "M"), ("N",), ("N", "M"))
        B = Dense(torch.randn(4, 5), ("N", "K"), ("N",), ("N", "K"))
        with pytest.raises(AssertionError):
            Add(A, B)

    def test_add_must_have_same_oshape(self):
        A = Dense(torch.randn(4, 3), ("N", "M"), ("N",), ("N", "M"))
        B = Dense(torch.randn(5, 3), ("N", "M"), ("N",), ("N", "K"))
        with pytest.raises(AssertionError):
            Add(A, B)
