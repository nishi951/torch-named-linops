import pytest

import torch
from torchlinops import Dense, split


def test_split():
    ishape = ("B", "N")
    oshape = ("B", "M")
    B = 10
    M, N = (3, 7)
    weight = torch.randn(B, M, N)
    weightshape = ("B", "M", "N")
    device = "cpu"
    A = Dense(weight, weightshape, ishape, oshape)
    linops, in_slc, out_slc = split(A, {"N": 2, "M": 1})

    # tile indices
    n, m = 1, 2
    # Input
    x_n = torch.randn(B, 2)
    y_m = linops[n, m](x_n)
    # True operator
    A_mn = Dense(weight[:, m : m + 1, 2 * n : 2 * (n + 1)], weightshape, ishape, oshape)
    y_m_ref = A_mn(x_n)
    assert torch.allclose(y_m, y_m_ref)


def test_batch_split(): ...
