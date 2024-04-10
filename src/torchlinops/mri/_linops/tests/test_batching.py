import pytest

import torch
from einops import repeat

from torchlinops.mri.sim.spiral2d import (
    Spiral2dSimulator,
    Spiral2dSimulatorConfig,
)
from torchlinops.mri.sim.tgas_spi import (
    TGASSPISimulator,
    TGASSPISimulatorConfig,
)

from torchlinops import Diagonal, SumReduce, Batch
from torchlinops.mri import SENSE, NUFFT


@pytest.fixture
def subspace_linop():
    num_coils = 6
    N = 64
    im_size = (N, N, N)
    D = len(im_size)
    T, R, K = 50, 16, 21
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
        toeplitz=True,
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


def test_subspace_linop_batching(subspace_linop):
    A_batch = Batch(
        subspace_linop,
        "cpu",
        "cpu",
        torch.complex64,
        torch.complex64,
        C=1,
        A=1,
    )

    isize = tuple(A_batch.size(d) for d in A_batch.ishape)
    osize = tuple(A_batch.size(d) for d in A_batch.oshape)
    x = torch.randn(isize, dtype=torch.complex64)
    b = torch.randn(osize, dtype=torch.complex64)

    AHAx = A_batch.N(x)
    AHb = A_batch.H(b)

    AHAx_nobatch = subspace_linop.N(x)
    AHb_nobatch = subspace_linop.H(b)

    assert torch.isclose(AHAx, AHAx_nobatch, atol=1e-6, rtol=1e-5).all()
    assert torch.isclose(AHb, AHb_nobatch, atol=1e-6, rtol=1e-5).all()
