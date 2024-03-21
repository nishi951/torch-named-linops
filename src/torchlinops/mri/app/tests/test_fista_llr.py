import pytest

from torchlinops.mri.recon.prior.llr import LocallyLowRankConfig, LocallyLowRank
from torchlinops.mri.app.fista import FISTA


def test_fista_llr_mrf_invivo():

    llr_config = LocallyLowRankConfig(
        block_size=(10, 10, 10),
        block_stride=(10, 10, 10),
        threshold=1.5e-4,
        shift_increment='random'
    )

    # DCF
    D = DCF(...)
    S = SENSE(...)
    F = NUFFT(..., toeplitz=True)
    Phi = Diagonal(...)
    R = SumReduce(...)

    A = (D**(1/2)) @ R @ Phi @ F @ S
    b = (D**(1/2))(ksp)

    # Run recon
    fista = FISTA(
        A, b,
        prox=LocallyLowRank(input_size, config),
        num_iters=20,
        max_eig=None,
        max_eig_iters=30,
        precond=None,
    )
    recon = fista.run()

    breakpoint()

def test_fista_llr_mrf_timeseg_invivo():

    llr_config = LocallyLowRankConfig(
        block_size=(10, 10, 10),
        block_stride=(10, 10, 10),
        threshold=1.5e-4,
        shift_increment='random'
    )

    # DCF
    D = DCF(...)
    S = SENSE(...)
    F = NUFFT(..., toeplitz=True)
    F, D = timeseg(F, D, num_segments=10)
    Phi = Diagonal(...)
    R = SumReduce(...)

    A = (D**(1/2)) @ R @ Phi @ F @ S
    b = (D**(1/2))(ksp)

    # Run recon
    fista = FISTA(
        A, b,
        prox=LocallyLowRank(input_size, config),
        num_iters=20,
        max_eig=None,
        max_eig_iters=30,
        precond=None,
    )
    recon = fista.run()

    breakpoint()
