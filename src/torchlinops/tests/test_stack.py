import pytest

import torch

from torchlinops import Stack, Dense, Diagonal, ND, Dim
from einops import einsum

from torchlinops.tests.test_base import BaseNamedLinopTests


def test_stack_init():
    P, Q = 3, 4
    ishape = Dim("Q")
    oshape = Dim("P")

    A = Dense(torch.randn(P, Q), ("P", "Q"), ishape, oshape)
    B = Dense(torch.randn(P, Q), ("P", "Q"), ishape, oshape)
    C = Stack(A, B, idim_and_idx=("N", 0))
    C2 = Stack(A, B, odim_and_idx=("N", 1))
    C3 = Stack(A, B, idim_and_idx=("N", 0), odim_and_idx=("N", 1))


@pytest.fixture
def denselinops():
    P, Q = 3, 4
    ishape = Dim("Q")
    oshape = Dim("P")
    wA = torch.randn(P, Q)
    wB = torch.randn(P, Q)
    A = Dense(wA, ("P", "Q"), ishape, oshape)
    B = Dense(wB, ("P", "Q"), ishape, oshape)
    return A, B


def test_stack_horizontal(denselinops):
    A, B = denselinops
    idim = 0
    # Horizontal stack
    C = Stack(A, B, idim_and_idx=("N", idim))
    assert C.size("N") == len([A, B])
    x = torch.randn(C.size("N"), C.size("Q"))
    Cx = C(x)
    xs = x.tensor_split(len([A, B]), idim)
    xs = [xi.squeeze(idim) for xi in xs]
    Cx_ref = A(xs[0]) + B(xs[1])
    assert Cx.allclose(Cx_ref)


def test_stack_vertical(denselinops):
    A, B = denselinops
    odim = 1
    # Vertical stack
    C = Stack(A, B, odim_and_idx=("N", odim))
    assert C.size("N") == len([A, B])
    x = torch.randn(C.size("Q"))
    Cx = C(x)
    Cx_ref = torch.stack((A(x), B(x)), dim=odim)
    assert Cx.allclose(Cx_ref)


def test_stack_diagonal(denselinops):
    A, B = denselinops
    idim = 0
    odim = 1
    # Diagonal stack
    C = Stack(A, B, idim_and_idx=("N", idim), odim_and_idx=("M", odim))
    assert C.size("N") == len([A, B])
    assert C.size("M") == len([A, B])
    x = torch.randn(C.size("N"), C.size("Q"))
    Cx = C(x)
    xs = x.tensor_split(len([A, B]), idim)
    xs = [xi.squeeze(idim) for xi in xs]
    Cx_ref = torch.stack((A(xs[0]), B(xs[1])), dim=odim)
    assert Cx.allclose(Cx_ref)


def test_stack_split(denselinops):
    A, B = denselinops
    idim = 0
    odim = 1
    # Vertical stack
    C = Stack(A, B, odim_and_idx=("M", odim))

    # Slice along non-stack dim
    ibatch = [slice(None)]
    obatch = [slice(0, 1), slice(None)]
    C1 = C.split(C, ibatch, obatch)
    assert (C1[0].weight == A.weight[:1]).all()

    # Slice along stack dim
    ibatch = [slice(None)]
    obatch = [slice(None), slice(0, 2)]
    C2 = C.split(C, ibatch, obatch)
    assert isinstance(C2, Stack)
    assert (C2[0].weight == A.weight).all()
    assert (C2[1].weight == B.weight).all()

    # Horizontal stack
    C = Stack(A, B, idim_and_idx=("N", idim))
    # Slice along non-stack dim
    ibatch = [slice(None), slice(1, 2)]
    obatch = [slice(None)]
    C3 = C.split(C, ibatch, obatch)
    assert (C3[0].weight == A.weight[:, 1:2]).all()
    assert (C3[1].weight == B.weight[:, 1:2]).all()

    # Slice along stack dim
    ibatch = [slice(1, 2), slice(None)]
    obatch = [slice(None)]
    C4 = C.split(C, ibatch, obatch)
    assert isinstance(C4, Stack)
    assert (C4[0].weight == B.weight).all()
