import pytest

import torch

from torchlinops import NamedLinop
from torchlinops.mri import (
    NUFFT
)
from torchlinops.mri._linops.nufft.backends import NUFFT_BACKENDS
from torchlinops.mri.sim.spiral2d import (
    Spiral2dSimulator,
    Spiral2dSimulatorConfig,
)

@pytest.fixture
def spiral2d_data():
    # TODO: Perhaps consolidate the spiral2d_data fixtures into a single fixture?
    config = Spiral2dSimulatorConfig(
        im_size=(64, 128),
        # im_size=(64, 64),
        num_coils=8,
        noise_std=0.0,
        spiral_2d_kwargs={
            "n_shots": 16,
            "alpha": 1.5,
            "f_sampling": 1.0,
        },
    )

    simulator = Spiral2dSimulator(config)
    data = simulator.data
    return data


def inner(x, y):
    """Complex inner product"""
    return torch.sum(x.conj() * y)


def is_adjoint(
    A: NamedLinop,
    x: torch.Tensor,
    y: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-8,
):
    """
    The adjoint test states that if A and AH are adjoints, then
    inner(y, Ax) = inner(AHy, x)
    """
    return torch.isclose(inner(y, A(x)), inner(A.H(y), x), atol=atol, rtol=rtol).all()

@pytest.fixture(params=NUFFT_BACKENDS)
def nufft(spiral2d_data, request):
    data = spiral2d_data
    F = NUFFT(
        data.trj,
        im_size=data.img.shape,
        out_batch_shape=(
            "R",
            "K",
        ),
        backend=request.param,
        toeplitz=False,
    )
    return F, request.param, data.img, data.ksp[0]

def test_nufft(nufft):
    F, backend, x, y = nufft
    # isize = F.im_size
    # osize = tuple(F.size(d) for d in F.oshape)
    if backend == 'fi':
        assert is_adjoint(F, x, y)
    elif backend == 'sigpy':
        assert is_adjoint(F, x, y, atol=1e-4, rtol=1e-7)
