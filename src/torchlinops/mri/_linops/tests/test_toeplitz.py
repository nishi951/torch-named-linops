import pytest

import torch

from torchlinops.mri import NUFFT, SENSE, DCF
from torchlinops.mri.sim.spiral2d import (
    Spiral2dSimulator,
    Spiral2dSimulatorConfig,
)
from torchlinops.mri.sim.tgas_spi import (
    TGASSPISimulator,
    TGASSPISimulatorConfig,
)


@pytest.fixture
def spiral2d_data():
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


@pytest.fixture
def tgas_spi_data():
    config = TGASSPISimulatorConfig(
        im_size=(64, 64, 128),
        num_coils=8,
        num_TRs=100,
        num_groups=18,
        groups_undersamp=1,
        noise_std=0.0,
        spiral_2d_kwargs={
            "alpha": 1.5,
            "f_sampling": 1.0,
        },
    )

    simulator = TGASSPISimulator(config)
    data = simulator.data
    return data


def test_toeplitz_2d(spiral2d_data):
    data = spiral2d_data
    D = DCF(data.trj, data.img.shape, ("C", "R", "K"), show_pbar=False)
    S = SENSE(data.mps)
    F_fi = NUFFT(
        data.trj,
        im_size=data.img.shape,
        in_batch_shape=("C",),
        out_batch_shape=(
            "R",
            "K",
        ),
        backend="fi",
        toeplitz=True,
    )
    F_sp = NUFFT(
        data.trj,
        im_size=data.img.shape,
        in_batch_shape=("C",),
        out_batch_shape=(
            "R",
            "K",
        ),
        backend="sigpy",
        toeplitz=True,
    )
    A_fi = (D ** (1 / 2)) @ F_fi @ S
    A_sp = (D ** (1 / 2)) @ F_sp @ S

    assert torch.isclose(
        A_fi.N(data.img), A_fi.H(A_fi(data.img)), atol=2e-1, rtol=1e-1
    ).all()
    assert torch.isclose(
        A_sp.N(data.img), A_sp.H(A_sp(data.img)), atol=2e-1, rtol=1e-1
    ).all()


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_toeplitz_3d(tgas_spi_data):
    data = tgas_spi_data
    device = torch.device("cuda:0")
    D = DCF(
        data.trj, data.img.shape, ("C", "R", "T", "K"), device_idx=0, show_pbar=False
    )
    S = SENSE(data.mps)
    F_fi = NUFFT(
        data.trj,
        im_size=data.img.shape,
        in_batch_shape=("C",),
        out_batch_shape=(
            "R",
            "T",
            "K",
        ),
        backend="fi",
        toeplitz=True,
    )
    F_sp = NUFFT(
        data.trj,
        im_size=data.img.shape,
        in_batch_shape=("C",),
        out_batch_shape=(
            "R",
            "T",
            "K",
        ),
        backend="sigpy",
        toeplitz=True,
    )
    A_fi = (D ** (1 / 2)) @ F_fi @ S
    A_sp = (D ** (1 / 2)) @ F_sp @ S
    A_fi.to(device)
    A_sp.to(device)

    assert torch.isclose(
        A_fi.N(data.img.to(device)),
        A_fi.H(A_fi(data.img.to(device))),
        atol=2e-1,
        rtol=1e-1,
    ).all()
    assert torch.isclose(
        A_sp.N(data.img.to(device)),
        A_sp.H(A_sp(data.img.to(device))),
        atol=2e-1,
        rtol=1e-1,
    ).all()
