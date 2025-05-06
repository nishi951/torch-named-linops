import pytest

import torch
from torchlinops import Dense, Batch


def test_batch_normal_adjoint():
    ishape = ("B", "N")
    oshape = ("B", "M")
    B = 10
    M, N = (3, 7)
    weight = torch.randn(B, M, N)
    weightshape = ("B", "M", "N")
    device = "cpu"
    A_nobatch = Dense(weight, weightshape, ishape, oshape)
    A = Batch(A_nobatch, device, device, weight.dtype, weight.dtype, **{"N": 1})
    x = torch.randn(B, N)
    y = torch.randn(B, M)

    # Test Forward
    Ax = A(x)
    Ax_ref = A_nobatch(x)
    assert Ax_ref.allclose(Ax)

    # Test Normal
    AN = A.N
    ANx = AN(x)
    ANx_ref = A_nobatch.N(x)
    assert len(AN._linops) == N * N
    assert ANx_ref.allclose(ANx)

    # Test Adjoint
    AH = A.H
    AHy = AH(y)
    AHy_ref = A_nobatch.H(y)
    assert len(AH._linops) == N
    assert AHy_ref.allclose(AHy)


def test_batch_chain_normal_adjoint(): ...
