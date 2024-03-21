import pytest

from torchlinops.mri import NUFFT, SENSE
from torchlinops.mri._linops.nufft.toeplitz import toeplitz
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
    S = SENSE(data.mps)
    F = NUFFT(
        data.trj,
        im_size=data.img.shape,
        in_batch_shape=S.out_batch_shape,
        out_batch_shape=("R",),
        backend="fi",
        toeplitz=True,
    )
    A = F @ S
    A.N
    breakpoint()
