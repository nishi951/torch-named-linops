import pytest

import torch

from torchlinops.mri.sim.mrf.tgas_spi_mrf import (
    TGASSPISubspaceMRFSimulator,
    TGASSPISubspaceMRFSimulatorConfig,
)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_tgas_spi_mrf_small():
    config = TGASSPISubspaceMRFSimulatorConfig(
        im_size=(180, 216, 180),
        num_coils=1.0,
        num_TRs=500,
        num_groups=16,
        groups_undersamp=1.0,
        num_bases=5,
        noise_std=0.0,
        voxel_batch_size=10000,
        tr_batch_size=1,
        coil_batch_size=4,
        nufft_backend="fi",
        spiral_2d_kwargs={
            "alpha": 1.5,
            "f_sampling": 0.4,
            "g_max": 40.0,
            "s_max": 100.0,
        },
        debug=True,
    )
    device = torch.device("cuda")
    sim = TGASSPISubspaceMRFSimulator(config, device)
    data = sim.simulate()
    assert True


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_tgas_spi_mrf_full():
    config = TGASSPISubspaceMRFSimulatorConfig(
        im_size=(180, 216, 180),
        num_coils=16,
        num_TRs=500,
        num_groups=16,
        groups_undersamp=1.0,
        num_bases=5,
        noise_std=0.0,
        voxel_batch_size=10000,
        tr_batch_size=1,
        coil_batch_size=4,
        nufft_backend="fi",
        spiral_2d_kwargs={
            "alpha": 1.5,
            "f_sampling": 0.4,
            "g_max": 40.0,
            "s_max": 100.0,
        },
        debug=False,
    )
    device = torch.device("cuda")
    sim = TGASSPISubspaceMRFSimulator(config, device)
    data = sim.simulate()
    assert True
