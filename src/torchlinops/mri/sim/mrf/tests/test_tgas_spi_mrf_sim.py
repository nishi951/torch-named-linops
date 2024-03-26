import pytest

import torch

from torchlinops._core._linops import Repeat
from torchlinops.mri._linops.nufft.backends import NUFFT_BACKENDS
from torchlinops.mri.sim.mrf.tgas_spi_mrf import (
    TGASSPISubspaceMRFSimulator,
    TGASSPISubspaceMRFSimulatorConfig,
)

TOLERANCES = {
    "fi": {"atol": 1e-5, "rtol": 1e-4},
    "sigpy": {"atol": 1e-2, "rtol": 1e-4},
}


def mask_by_img(x, reference_img, eps=1e-2):
    mask = torch.abs(reference_img) < eps
    out = x.clone()
    out[mask] = 0.0
    return out


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
    assert True  # Simulation completes


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.big
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
    assert True  # Simulation completes


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
@pytest.mark.parametrize("backend", NUFFT_BACKENDS)
def test_tgas_spi_mrf_toeplitz(backend):
    config = TGASSPISubspaceMRFSimulatorConfig(
        im_size=(180, 216, 180),
        num_coils=2,
        num_TRs=500,
        num_groups=16,
        groups_undersamp=1.0,
        num_bases=5,
        noise_std=0.0,
        voxel_batch_size=10000,
        tr_batch_size=1,
        coil_batch_size=1,
        nufft_backend=backend,
        nufft_extras={
            "width": 4,
            "oversamp": 2.0,
        },
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

    A = sim.A
    A.to(device)
    img = data.sub_img.to(device)
    img = img / torch.max(torch.abs(img))
    toep = A.N(img)
    notoep = A.H(A(img))

    assert torch.isclose(toep, notoep, **TOLERANCES[backend]).all()

@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
@pytest.mark.parametrize("backend", NUFFT_BACKENDS)
def test_tgas_spi_mrf_timeseg_toeplitz(backend):
    config = TGASSPISubspaceMRFSimulatorConfig(
        im_size=(180, 216, 180),
        num_coils=2,
        num_TRs=500,
        num_groups=16,
        groups_undersamp=1.0,
        num_bases=5,
        noise_std=0.0,
        voxel_batch_size=10000,
        tr_batch_size=1,
        coil_batch_size=1,
        nufft_backend=backend,
        nufft_extras={
            "width": 4,
            "oversamp": 2.0,
        },
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

    A = sim.A

    # Insert time segmentation
    num_segments = 4
    segment_dim = "B"
    A_tseg = A.linops[-2].timeseg(num_segments, segment_dim)
    mps_shape = A.linops[-1].oshape
    tseg_shape = A_tseg.ishape
    # Simulate B0 with repeat - no actual b0 applied
    B0 = Repeat({segment_dim: num_segments}, mps_shape, tseg_shape)
    A = A[:-2] @ A_tseg @ B0 @ A[-1]
    A.to(device)

    img = data.sub_img.to(device)
    img = img / torch.max(torch.abs(img))
    toep = A.N(img)
    notoep = A.H(A(img))

    assert torch.isclose(toep, notoep, **TOLERANCES[backend]).all()
