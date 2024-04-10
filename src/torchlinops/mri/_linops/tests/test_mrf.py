import pytest

import torch
from einops import repeat

from torchlinops import Diagonal, SumReduce, Batch
from torchlinops.mri import SENSE, NUFFT


@pytest.fixture
def subspace_linop():
    num_coils = 6
    N = 220
    im_size = (N, N, N)
    D = len(im_size)
    T, R, K = 500, 16, 1678
    A = 5
    mps = torch.randn(*(num_coils, *im_size), dtype=torch.complex64)
    trj = N * (torch.rand((T, R, K, D)) - 0.5)
    phi = torch.randn(A, T, dtype=torch.complex64)
    dcf = torch.ones((T, R, K))

    S = SENSE(mps, in_batch_shape=("A",))
    F = NUFFT(
        trj,
        im_size,
        in_batch_shape=("A", "C"),
        out_batch_shape=("T", "R", "K"),
        toeplitz=False,
        backend="fi",
    )
    P = Diagonal(
        repeat(phi, "A T -> A () T () ()"),  # Expand to match
        ioshape=("A", "C", "T", "R", "K"),
        broadcast_dims=["C", "R", "K"],
    )
    R = SumReduce(
        ishape=("A", "C", "T", "R", "K"),
        oshape=("C", "T", "R", "K"),
    )
    D = Diagonal(dcf, ioshape=("C", "T", "R", "K"))
    return (D ** (1 / 2)) @ R @ P @ F @ S


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_subspace_linop_big(subspace_linop):
    A = subspace_linop.to("cuda")
    isize = tuple(A.size(d) for d in A.ishape)
    osize = tuple(A.size(d) for d in A.oshape)
    x = torch.randn(isize, dtype=torch.complex64, device="cuda")
    b = torch.randn(osize, dtype=torch.complex64, device="cuda")

    AHAx = A.N(x)
    AHb = A.H(b)
    breakpoint()


@pytest.fixture
def real_subspace_linop():
    num_coils = 6
    N = 220
    im_size = (N, N, N)
    D = len(im_size)
    T, R, K = 500, 16, 1678
    A = 5
    mps = torch.randn(*(num_coils, *im_size), dtype=torch.complex64)
    trj = N * (torch.rand((T, R, K, D)) - 0.5)
    phi = torch.randn(A, T, dtype=torch.complex64)
    dcf = torch.ones((T, R, K))

    S = SENSE(mps, in_batch_shape=("A",))
    F = NUFFT(
        trj,
        im_size,
        in_batch_shape=("A", "C"),
        out_batch_shape=("T", "R", "K"),
        toeplitz=False,
        backend="fi",
    )
    P = Diagonal(
        repeat(phi, "A T -> A () T () ()"),  # Expand to match
        ioshape=("A", "C", "T", "R", "K"),
        broadcast_dims=["C", "R", "K"],
    )
    R = SumReduce(
        ishape=("A", "C", "T", "R", "K"),
        oshape=("C", "T", "R", "K"),
    )
    D = Diagonal(dcf, ioshape=("C", "T", "R", "K"))
    return (D ** (1 / 2)) @ R @ P @ F @ S


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_subspace_linop_big(subspace_linop):
    A = subspace_linop.to("cuda")
    isize = tuple(A.size(d) for d in A.ishape)
    osize = tuple(A.size(d) for d in A.oshape)
    x = torch.randn(isize, dtype=torch.complex64, device="cuda")
    b = torch.randn(osize, dtype=torch.complex64, device="cuda")

    AHAx = A.N(x)
    AHb = A.H(b)
    breakpoint()
