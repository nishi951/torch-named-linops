import pytest

from torchlinops.mri.sim.spiral2d import (
    Spiral2dSimulator,
    Spiral2dSimulatorConfig,
)
from torchlinops.mri.sim.tgas_spi import (
    TGASSPISimulator,
    TGASSPISimulatorConfig,
)


def test_spiral2dsimulator():
    config = Spiral2dSimulatorConfig(
        im_size=(64, 64),
        num_coils=8,
        noise_std=0.1,
        spiral_2d_kwargs={
            "n_shots": 16,
            "alpha": 1.5,
            "f_sampling": 1.0,
        },
    )

    simulator = Spiral2dSimulator(config)
    data = simulator.data
    assert data.trj.shape[0] == config.spiral_2d_kwargs["n_shots"]
    assert data.trj.shape[2] == 2  # Dimension
    assert data.mps.shape[0] == config.num_coils
    assert tuple(data.mps.shape[1:]) == config.im_size


def test_tgasspisimulator():
    config = TGASSPISimulatorConfig(
        im_size=(64, 64, 64),
        num_coils=8,
        num_TRs=100,
        num_groups=18,
        groups_undersamp=1,
        noise_std=0.1,
        spiral_2d_kwargs={
            "alpha": 1.5,
            "f_sampling": 1.0,
        },
    )

    simulator = TGASSPISimulator(config)
    data = simulator.data
    assert data.trj.shape[0] == 3 * config.num_groups
    assert data.trj.shape[1] == config.num_TRs
    assert data.trj.shape[3] == 3  # Dimension
    assert data.mps.shape[0] == config.num_coils
    assert tuple(data.mps.shape[1:]) == config.im_size
