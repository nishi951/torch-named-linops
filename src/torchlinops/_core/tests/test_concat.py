import pytest

import torch

from torchlinops import Concat, Dense, Diagonal, ND
from einops import einsum

from torchlinops._core.tests.test_base import BaseNamedLinopTests


def test_concat_init():
    N = 1
    P, Q = 3, 4
    ishape = ("N", "Q")
    oshape = ("N", "P")

    # Diagonal stacking
    A = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ishape, oshape)
    B = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ishape, oshape)
    C = Concat(A, B, idim="N")
    C2 = Concat(A, B, idim="Q")
    C3 = Concat(A, B, idim="Q", odim="P")


def test_concat_shape_inference():
    shape = (ND("A"), ND("B"), ND("..."), ND("C"), ND("D"), ND("E"))

    d1 = Concat._infer_dim_idx("B", shape)
    assert d1 == 1
    d2 = Concat._infer_dim_idx("C", shape)
    assert d2 == -3


@pytest.fixture
def denselinops():
    N = 1
    P, Q = 3, 4
    ishape = ("N", "Q")
    oshape = ("N", "P")
    wA = torch.randn(N, P, Q)
    wB = torch.randn(N, P, Q)
    A = Dense(wA, ("N", "P", "Q"), ishape, oshape)
    B = Dense(wB, ("N", "P", "Q"), ishape, oshape)
    return A, B


def test_concat_horizontal(denselinops):
    A, B = denselinops
    # Horizontal stack
    C = Concat(A, B, idim="Q")
    assert C.size("Q") == A.size("Q") + B.size("Q")
    x = torch.randn(C.size("N"), C.size("Q"))
    Cx = C(x)
    xs = x.tensor_split(C.islices, dim=C.idim_idx)[:-1]
    Cx_ref = A(xs[0]) + B(xs[1])
    assert Cx.allclose(Cx_ref)


def test_concat_vertical(denselinops):
    A, B = denselinops
    # Vertical stack
    C = Concat(A, B, odim="P")
    assert C.size("P") == A.size("P") + B.size("P")
    x = torch.randn(C.size("N"), C.size("Q"))
    Cx = C(x)
    Cx_ref = torch.concatenate((A(x), B(x)), dim=-1)
    assert Cx.allclose(Cx_ref)
