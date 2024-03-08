import pytest

from torchlinops.mri.sim import Spiral2dSimulator, Spiral2dSimulatorConfig


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
