import pytest
import torch

from torchlinops import Add, Dense, Diagonal, Dim, Identity, config
from torchlinops.tests.test_base import BaseNamedLinopTests
from torchlinops.utils import is_adjoint


class TestAdd(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        A = Dense(
            torch.randn((5, 5), dtype=torch.complex64), ("M", "N"), ("N",), ("M",)
        )
        B = Dense(
            torch.randn((5, 5), dtype=torch.complex64), ("M", "N"), ("N",), ("M",)
        )
        C = Dense(
            torch.randn((5, 5), dtype=torch.complex64), ("M", "N"), ("N",), ("M",)
        )
        add = Add(A, B, C)
        add.threaded = False
        x = torch.randn(5, dtype=torch.complex64)
        y = torch.randn(5, dtype=torch.complex64)
        return add, x, y


class TestAddDunder:
    def test_len(self):
        A = Dense(torch.randn(4, 3), ("N", "M"), ("N",), ("N", "M"))
        B = Dense(torch.randn(4, 3), ("N", "M"), ("N",), ("N", "M"))
        add = Add(A, B)
        assert len(add) == 2
        assert repr(add)

    def test_flatten(self):
        A = Dense(torch.randn(4, 3), ("N", "M"), ("N",), ("N", "M"))
        B = Dense(torch.randn(4, 3), ("N", "M"), ("N",), ("N", "M"))
        add = Add(A, B)
        flat = add.flatten()
        assert flat == [add]

    def test_size(self):
        A = Dense(torch.randn(4, 3), ("N", "M"), ("N",), ("N", "M"))
        B = Dense(torch.randn(4, 3), ("N", "M"), ("N",), ("N", "M"))
        add = Add(A, B)
        assert add.size("N") == 4
        assert "N" in add.dims
        assert "M" in add.dims

    def test_size_none(self):
        A = Identity(ishape=("N",), oshape=("N",))
        B = Identity(ishape=("N",), oshape=("N",))
        add = Add(A, B)
        assert add.size("N") is None


class TestAddValidation:
    def test_must_have_same_ishape(self):
        A = Dense(torch.randn(4, 3), ("N", "M"), ("N",), ("N", "M"))
        B = Dense(torch.randn(4, 5), ("N", "K"), ("N",), ("N", "K"))
        with pytest.raises(AssertionError):
            Add(A, B)

    def test_must_have_same_oshape(self):
        A = Dense(torch.randn(4, 3), ("N", "M"), ("N",), ("N", "M"))
        B = Dense(torch.randn(5, 3), ("N", "M"), ("N",), ("N", "K"))
        with pytest.raises(AssertionError):
            Add(A, B)
