import pytest
from math import floor, ceil

import torch

from torchlinops.mri.sim.spiral2d import (
    Spiral2dSimulator,
    Spiral2dSimulatorConfig,
)

from torchlinops.mri.data import (
    ReadoutInterpolator,
    ReadoutInterpolatorConfig,
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

def test_readoutinterpolator(spiral2d_data):
    data = spiral2d_data

    interp_config = ReadoutInterpolatorConfig(
        oversamp=1.,
        width=120,
        resamp_method='sinc_interp_kaiser',
    )

    ksp = data.ksp
    ksp_interp = ReadoutInterpolator(ksp, interp_config)
    c = torch.rand(3, 4, 1) * ksp.shape[-1]
    ksp_c = ksp_interp(c)
    assert ksp_c.shape == tuple(ksp.shape[:-1]) + tuple(c.shape[:-1])

    # Test one of the points
    t = c[0, 0, 0].item()
    l, u = floor(t), ceil(t)
    manual_interp = ksp[..., u] * (t - l)/(u - l) + ksp[..., l] * (u - t)/(u - l)
    assert torch.isclose(ksp_c[..., 0, 0], manual_interp).all()
