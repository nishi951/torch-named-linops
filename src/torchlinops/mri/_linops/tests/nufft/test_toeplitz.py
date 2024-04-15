import pytest

import torch

from torchlinops.mri import NUFFT, SENSE, DCF
from torchlinops.mri._linops.nufft.backends import NUFFT_BACKENDS
from torchlinops.mri.sim.spiral2d import (
    Spiral2dSimulator,
    Spiral2dSimulatorConfig,
)
from torchlinops.mri.sim.tgas_spi import (
    TGASSPISimulator,
    TGASSPISimulatorConfig,
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


# Different image sizes
@pytest.fixture(params=[(64, 64), (64, 128)])
def spiral2d_data(request):
    config = Spiral2dSimulatorConfig(
        im_size=request.param,
        # im_size=(64, 64),
        num_coils=8,
        noise_std=0.0,
        spiral_2d_kwargs={
            "n_shots": 1,
            "alpha": 1.0,
            "f_sampling": 0.2,
        },
    )

    simulator = Spiral2dSimulator(config)
    data = simulator.data
    return data


@pytest.mark.filterwarnings("ignore:No Inner linop")
@pytest.mark.parametrize("backend", NUFFT_BACKENDS)
def test_toeplitz_2d_nufft_only(spiral2d_data, backend):
    data = spiral2d_data
    F = NUFFT(
        data.trj,
        im_size=data.img.shape,
        out_batch_shape=(
            "R",
            "K",
        ),
        backend=backend,
        toeplitz=True,
    )
    toep = F.N(data.img)
    toep = mask_by_img(toep, data.img)
    notoep = F.H(F(data.img))
    notoep = mask_by_img(notoep, data.img)
    assert torch.isclose(toep, notoep, **TOLERANCES[backend]).all()


@pytest.mark.filterwarnings("ignore:No Inner linop")
@pytest.mark.parametrize("backend", NUFFT_BACKENDS)
def test_toeplitz_2d_with_coils(spiral2d_data, backend):
    data = spiral2d_data
    F = NUFFT(
        data.trj,
        im_size=data.img.shape,
        in_batch_shape=("C",),
        out_batch_shape=(
            "R",
            "K",
        ),
        backend=backend,
        toeplitz=True,
    )
    S = SENSE(data.mps)
    A = F @ S
    toep = mask_by_img(A.N(data.img), data.img)
    notoep = mask_by_img(A.H(A(data.img)), data.img)
    assert torch.isclose(toep, notoep, **TOLERANCES[backend]).all()


@pytest.mark.parametrize("backend", NUFFT_BACKENDS)
def test_toeplitz_2d_full(spiral2d_data, backend):
    data = spiral2d_data
    D = DCF(data.trj, data.img.shape, ("C", "R", "K"), show_pbar=False)
    S = SENSE(data.mps)
    F = NUFFT(
        data.trj,
        im_size=data.img.shape,
        in_batch_shape=("C",),
        out_batch_shape=(
            "R",
            "K",
        ),
        backend=backend,
        toeplitz=True,
    )
    A = (D ** (1 / 2)) @ F @ S
    toep = mask_by_img(A.N(data.img), data.img)
    notoep = mask_by_img(A.H(A(data.img)), data.img)
    assert torch.isclose(toep, notoep, **TOLERANCES[backend]).all()


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
            "alpha": 1.0,
            "f_sampling": 0.2,
        },
    )

    simulator = TGASSPISimulator(config, device=torch.device("cuda:0"))
    data = simulator.data
    return data


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
@pytest.mark.parametrize("backend", NUFFT_BACKENDS)
def test_toeplitz_3d(tgas_spi_data, backend):
    data = tgas_spi_data
    device = torch.device("cuda:0")
    D = DCF(data.trj.to(device), data.img.shape, ("C", "R", "T", "K"), show_pbar=False)
    S = SENSE(data.mps)
    F = NUFFT(
        data.trj,
        im_size=data.img.shape,
        in_batch_shape=("C",),
        out_batch_shape=(
            "R",
            "T",
            "K",
        ),
        backend=backend,
        toeplitz=True,
    )
    A = (D ** (1 / 2)) @ F @ S
    A.to(device)
    img = data.img.to(device)
    toep = mask_by_img(A.N(img), img).detach().cpu()
    notoep = mask_by_img(A.H(A(img)), img).detach().cpu()
    assert torch.isclose(toep, notoep, **TOLERANCES[backend]).all()
