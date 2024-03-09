import pytest

import torch

from torchlinops.mri.sim.spiral2d import (
    Spiral2dSimulator,
    Spiral2dSimulatorConfig,
)
from torchlinops.mri.sim.tgas_spi import (
    TGASSPISimulator,
    TGASSPISimulatorConfig,
)

from torchlinops.mri._linops._fi_nufft import _nufft, _nufft_adjoint
from torchlinops.mri._linops.convert_trj import sp2fi, fi2sp

@pytest.fixture
def spiral2d_data():
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
    return data

@pytest.fixture
def tgas_spi_data():
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
    return data

def test_finufft2d(spiral2d_data):
    data = spiral2d_data
    ksp = _nufft(data.img, sp2fi(data.trj, data.img.shape))
    assert torch.isclose(ksp, data.ksp).all()

def test_finufft3d(tgas_spi_data):
    ...

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is required but not available")
def test_cufinufft2d():
    device = torch.device('cuda')


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is required but not available")
def test_cufinufft3d():
    ...
