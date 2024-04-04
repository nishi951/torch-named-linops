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

from torchlinops.mri._linops.nufft.backends.fi.functional import (
    nufft as fi_nufft,
    nufft_adjoint as fi_nufft_adjoint,
)
from torchlinops.mri._linops.nufft.backends.sp.functional import (
    nufft as sp_nufft,
    nufft_adjoint as sp_nufft_adjoint,
)
from torchlinops.mri._linops.nufft.backends.fi.convert_trj import sp2fi


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


def test_finufft2d(spiral2d_data):
    data = spiral2d_data
    fi_trj = sp2fi(data.trj.clone(), data.img.shape)
    # Test forward
    fi_ksp = fi_nufft(data.img * data.mps, fi_trj)
    assert torch.isclose(fi_ksp, data.ksp, atol=1e-2, rtol=1e-2).all()

    # Test adjoint
    sp_img_adjoint = sp_nufft_adjoint(data.ksp, data.trj, data.mps.shape)
    fi_img_adjoint = fi_nufft_adjoint(data.ksp, fi_trj, tuple(data.mps.shape))
    assert torch.isclose(fi_img_adjoint, sp_img_adjoint, atol=1e-1, rtol=1e-2).all()

    # Test normal
    fi_normal = fi_nufft_adjoint(fi_nufft(data.img, fi_trj), fi_trj, data.img.shape)
    sp_normal = sp_nufft_adjoint(sp_nufft(data.img, data.trj), data.trj, data.img.shape)
    assert torch.isclose(fi_normal, sp_normal, atol=1e-1, rtol=1e-2).all()


@pytest.mark.slow
def test_finufft3d(tgas_spi_data):
    data = tgas_spi_data
    fi_trj = sp2fi(data.trj.clone(), data.img.shape)
    # Test forward
    fi_ksp = fi_nufft(data.img * data.mps, fi_trj)
    assert torch.isclose(fi_ksp, data.ksp, atol=1e-1, rtol=1e-1).all()

    sp_img_adjoint = sp_nufft_adjoint(data.ksp, data.trj, data.mps.shape)
    fi_img_adjoint = fi_nufft_adjoint(data.ksp, fi_trj, tuple(data.mps.shape))
    assert torch.isclose(fi_img_adjoint, sp_img_adjoint, atol=1e-1, rtol=1e-1).all()

    # Test normal
    fi_normal = fi_nufft_adjoint(fi_nufft(data.img, fi_trj), fi_trj, data.img.shape)
    sp_normal = sp_nufft_adjoint(sp_nufft(data.img, data.trj), data.trj, data.img.shape)
    assert torch.isclose(fi_normal, sp_normal, atol=1e-1, rtol=1e-1).all()


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_cufinufft2d(spiral2d_data):
    device = torch.device("cuda")
    data = spiral2d_data
    fi_trj = sp2fi(data.trj.clone(), data.img.shape)
    # Test forward
    ksp = fi_nufft(data.img.to(device) * data.mps.to(device), fi_trj.to(device))
    ksp = ksp.cpu()
    assert torch.isclose(ksp, data.ksp, atol=1e-2, rtol=1e-2).all()

    # Test adjoint
    sp_img_adjoint = sp_nufft_adjoint(
        data.ksp.to(device), data.trj.to(device), data.mps.shape
    )
    fi_img_adjoint = fi_nufft_adjoint(
        data.ksp.to(device),
        fi_trj.to(device),
        tuple(data.mps.shape),
    )
    assert torch.isclose(
        fi_img_adjoint.cpu(), sp_img_adjoint.cpu(), atol=1e-1, rtol=1e-2
    ).all()

    # Test normal
    fi_normal = fi_nufft_adjoint(
        fi_nufft(data.img.to(device), fi_trj.to(device)),
        fi_trj.to(device),
        data.img.shape,
    )
    sp_normal = sp_nufft_adjoint(
        sp_nufft(data.img.to(device), data.trj.to(device)),
        data.trj.to(device),
        data.img.shape,
    )
    assert torch.isclose(fi_normal.cpu(), sp_normal.cpu(), atol=1e-1, rtol=1e-2).all()


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_cufinufft3d(tgas_spi_data):
    device = torch.device("cuda")
    data = tgas_spi_data
    fi_trj = sp2fi(data.trj.clone(), data.img.shape)
    # Test forward
    ksp = fi_nufft(data.img.to(device) * data.mps.to(device), fi_trj.to(device))
    ksp = ksp.cpu()
    assert torch.isclose(ksp, data.ksp, atol=1e-1, rtol=1e-1).all()

    # Test adjoint
    sp_img_adjoint = sp_nufft_adjoint(
        data.ksp.to(device), data.trj.to(device), data.mps.shape
    )
    fi_img_adjoint = fi_nufft_adjoint(
        data.ksp.to(device),
        fi_trj.to(device),
        tuple(data.mps.shape),
    )
    assert torch.isclose(
        fi_img_adjoint.cpu(), sp_img_adjoint.cpu(), atol=1e-1, rtol=1e-1
    ).all()

    # Test normal
    fi_normal = fi_nufft_adjoint(
        fi_nufft(data.img.to(device), fi_trj.to(device)),
        fi_trj.to(device),
        data.img.shape,
    )
    sp_normal = sp_nufft_adjoint(
        sp_nufft(data.img.to(device), data.trj.to(device)),
        data.trj.to(device),
        data.img.shape,
    )
    assert torch.isclose(fi_normal.cpu(), sp_normal.cpu(), atol=1e-1, rtol=1e-1).all()
