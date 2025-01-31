import pytest

import torch

from torchlinops import Concat, Dense, Diagonal, ND
from einops import einsum

from torchlinops.tests.test_base import BaseNamedLinopTests


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


def test_concat_diagonal(denselinops):
    A, B = denselinops
    # Diagonal stack
    C = Concat(A, B, idim="Q", odim="P")
    assert C.size("Q") == A.size("Q") + B.size("Q")
    assert C.size("P") == A.size("P") + B.size("P")
    x = torch.randn(C.size("N"), C.size("Q"))
    Cx = C(x)
    xs = x.tensor_split(C.islices, dim=C.idim_idx)[:-1]
    Cx_ref = torch.concatenate((A(xs[0]), B(xs[1])), dim=-1)
    assert Cx.allclose(Cx_ref)


def test_concat_split(denselinops):
    A, B = denselinops
    # Vertical stack
    C = Concat(A, B, odim="P")

    # Single Linop
    ibatch = [slice(None), slice(None)]
    obatch = [slice(None), slice(0, 3)]
    C1 = C.split(C, ibatch, obatch)
    assert isinstance(C1, Dense)
    assert (C1.weight == A.weight).all()

    # Multiple linops (overlapping)
    ibatch = [slice(None), slice(None)]
    obatch = [slice(None), slice(2, 6)]
    C2 = C.split(C, ibatch, obatch)
    assert isinstance(C2, Concat)
    assert (C2[0].weight == A.weight[:, 2:, :]).all()
    assert (C2[1].weight == B.weight[:, :, :]).all()

    # Horizontal stack
    C = Concat(A, B, idim="Q")
    # Single Linop
    ibatch = [slice(None), slice(0, 4)]
    obatch = [slice(None), slice(None)]
    C3 = C.split(C, ibatch, obatch)
    assert isinstance(C3, Dense)
    assert (C3.weight == A.weight).all()

    # Multiple linops (overlapping)
    ibatch = [slice(None), slice(2, 6)]
    obatch = [slice(None), slice(None)]
    C4 = C.split(C, ibatch, obatch)
    assert isinstance(C4, Concat)
    assert (C4[0].weight == A.weight[:, :, 2:]).all()
    assert (C4[1].weight == B.weight[:, :, :2]).all()
