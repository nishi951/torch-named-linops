import pytest

import torch
from torchlinops import Dense, split_linop, create_batched_linop, BatchSpec


def test_split_linop():
    ishape = ("B", "N")
    oshape = ("B", "M")
    B = 10
    M, N = (3, 7)
    weight = torch.randn(B, M, N)
    weightshape = ("B", "M", "N")
    device = "cpu"
    A = Dense(weight, weightshape, ishape, oshape)
    linops, in_slc, out_slc = split_linop(A, {"N": 2, "M": 1})

    # tile indices
    n, m = 1, 2
    # Input
    x_n = torch.randn(B, 2)
    y_m = linops[n, m](x_n)
    # True operator
    A_mn = Dense(weight[:, m : m + 1, 2 * n : 2 * (n + 1)], weightshape, ishape, oshape)
    y_m_ref = A_mn(x_n)
    assert torch.allclose(y_m, y_m_ref)


def test_create_batched_linop():
    ishape = ("B", "N")
    oshape = ("B", "M")
    B = 10
    M, N = (3, 7)
    weight = torch.randn(B, M, N)
    weightshape = ("B", "M", "N")
    device = "cpu"
    A = Dense(weight, weightshape, ishape, oshape)

    Abatch = create_batched_linop(A, BatchSpec(dict(N=2, M=1)))
    x = torch.randn(B, N)
    assert Abatch(x).allclose(A(x), rtol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_create_batched_linop_multi_device():
    ishape = ("B", "N")
    oshape = ("B", "M")
    B = 10
    M, N = (3, 7)
    weight = torch.randn(B, M, N)
    weightshape = ("B", "M", "N")
    device = "cpu"
    A = Dense(weight, weightshape, ishape, oshape)

    Abatch = create_batched_linop(
        A,
        [
            BatchSpec(
                dict(N=2),
                device_matrix=[torch.device("cpu"), torch.device("cuda:0")],
                base_device=torch.device("cpu"),
            ),
            BatchSpec(dict(M=1)),
        ],
    )
    for _ in range(10):
        # Fuzzing with multiple retries
        x = torch.randn(B, N)
        Abatch_x = Abatch(x)
        Ax = A(x)
        assert Abatch_x.allclose(Ax, rtol=1e-3)
