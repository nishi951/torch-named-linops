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
    F.N # get it
    x = data.img
    # import matplotlib.pyplot as plt
    # curr = ""
    # for linop in reversed(F.N):
    #     plt.figure()
    #     plt.title(f'{curr}, abs')
    #     plt.imshow(torch.abs(x))
    #     plt.figure()
    #     plt.title(f'{curr}, angle')
    #     plt.imshow(torch.angle(x))
    #     x = linop(x)
    #     curr += f' {type(linop).__name__}'
    # plt.figure()
    # plt.title(f'{curr}, abs')
    # plt.imshow(torch.abs(x))
    # plt.figure()
    # plt.title(f'{curr}, angle')
    # plt.imshow(torch.angle(x))
    toep = F.N(data.img)
    # toep = x

    notoep = F.H(F(data.img))
    # plt.figure()
    # plt.title('notoep, abs')
    # plt.imshow(torch.abs(notoep))
    # plt.figure()
    # plt.title('notoep, angle')
    # plt.imshow(torch.angle(notoep))

    mask = torch.abs(data.img) < 1e-2
    toep_mask = torch.abs(toep).clone()
    toep_mask[mask] = 0.
    notoep_mask = torch.abs(notoep).clone()
    notoep_mask[mask] = 0.

    # plt.figure()
    # plt.title('mask')
    # plt.imshow(mask)

    # plt.figure()
    # plt.title('abs(toep) / abs(notoep)')
    # plt.imshow(toep_mask / notoep_mask)
    # plt.colorbar()

    # plt.show()
    # breakpoint()

    if backend == 'fi':
        assert torch.isclose(toep_mask, notoep_mask, atol=1e-5, rtol=1e-4).all()
    elif backend == 'sigpy':
        assert torch.isclose(toep_mask, notoep_mask, atol=1e-2, rtol=1e-4).all()

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
    assert torch.isclose(A.N(data.img), A.H(A(data.img))).all()

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
    assert torch.isclose(A.N(data.img), A.H(A(data.img))).all()

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
            "alpha": 1.,
            "f_sampling": 0.2,
        },
    )

    simulator = TGASSPISimulator(config)
    data = simulator.data
    return data

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

