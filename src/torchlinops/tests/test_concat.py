import pytest
import torch

from torchlinops import Add, Concat, Dense, Dim, Stack
from torchlinops.tests.test_base import BaseNamedLinopTests
from torchlinops.utils import inner


class TestConcatHorizontal(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        N, P, Q = 1, 3, 4
        A = Dense(
            torch.randn(N, P, Q, dtype=torch.complex64),
            ("N", "P", "Q"),
            ("N", "Q"),
            ("N", "P"),
        )
        B = Dense(
            torch.randn(N, P, Q, dtype=torch.complex64),
            ("N", "P", "Q"),
            ("N", "Q"),
            ("N", "P"),
        )
        C = Concat(A, B, idim="Q")
        x = torch.randn(N, 2 * Q, dtype=torch.complex64)
        y = torch.randn(N, P, dtype=torch.complex64)
        return C, x, y


class TestConcatVertical(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        N, P, Q = 1, 3, 4
        A = Dense(
            torch.randn(N, P, Q, dtype=torch.complex64),
            ("N", "P", "Q"),
            ("N", "Q"),
            ("N", "P"),
        )
        B = Dense(
            torch.randn(N, P, Q, dtype=torch.complex64),
            ("N", "P", "Q"),
            ("N", "Q"),
            ("N", "P"),
        )
        C = Concat(A, B, odim="P")
        x = torch.randn(N, Q, dtype=torch.complex64)
        y = torch.randn(N, 2 * P, dtype=torch.complex64)
        return C, x, y


class TestConcatDiagonal(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        N, P, Q = 1, 3, 4
        A = Dense(
            torch.randn(N, P, Q, dtype=torch.complex64),
            ("N", "P", "Q"),
            ("N", "Q"),
            ("N", "P"),
        )
        B = Dense(
            torch.randn(N, P, Q, dtype=torch.complex64),
            ("N", "P", "Q"),
            ("N", "Q"),
            ("N", "P"),
        )
        C = Concat(A, B, idim="Q", odim="P")
        x = torch.randn(N, 2 * Q, dtype=torch.complex64)
        y = torch.randn(N, 2 * P, dtype=torch.complex64)
        return C, x, y


def test_concat_shape_inference():
    from torchlinops import ND

    shape = (ND("A"), ND("B"), ND("..."), ND("C"), ND("D"), ND("E"))
    d1 = Concat._infer_dim_idx("B", shape)
    assert d1 == 1
    d2 = Concat._infer_dim_idx("C", shape)
    assert d2 == -3


def test_concat_forward_horizontal():
    N, P, Q = 1, 3, 4
    A = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    B = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    C = Concat(A, B, idim="Q")
    assert C.size("Q") == 2 * Q
    x = torch.randn(C.size("N"), C.size("Q"))
    Cx = C(x)
    xs = x.tensor_split(C.islices, dim=C.idim_idx)[:-1]
    Cx_ref = A(xs[0]) + B(xs[1])
    assert Cx.allclose(Cx_ref)


def test_concat_split():
    N, P, Q = 1, 3, 4
    A = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    B = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    C = Concat(A, B, odim="P")
    C1 = C.split(C, {"P": slice(0, 3)})
    assert isinstance(C1, Dense)

    C2 = C.split(C, {"P": slice(2, 6)})
    assert isinstance(C2, Concat)


def test_concat_getitem():
    N, P, Q = 1, 3, 4
    A = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    B = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    C = Concat(A, B, odim="P")
    assert len(C) == 2
    assert repr(C)
