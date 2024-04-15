import pytest

import torch

from torchlinops.mri.sim.spiral2d import (
    Spiral2dSimulator,
    Spiral2dSimulatorConfig,
)
from torchlinops.mri import NUFFT
from torchlinops.mri._linops.nufft.backends.sp.functional import (
    nufft as sp_nufft,
    nufft_adjoint as sp_nufft_adjoint,
)
from torchlinops.mri._linops.nufft.backends.fi.functional import (
    nufft as fi_nufft,
    nufft_adjoint as fi_nufft_adjoint,
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
            "f_sampling": 0.2,
        },
    )

    simulator = Spiral2dSimulator(config)
    data = simulator.data
    return data


# Sigpy tests
def test_sigpy_linop_vs_functional_scaling(spiral2d_data):
    data = spiral2d_data
    F_sp = NUFFT(
        data.trj,
        data.mps.shape[1:],
        in_batch_shape=tuple(),
        out_batch_shape=("R", "K"),
        toeplitz=False,
        backend="sigpy",
    )
    AHAx_sp = F_sp.N(data.img)
    Ax_sp_fn = sp_nufft(data.img, data.trj)
    AHAx_sp_fn = sp_nufft_adjoint(Ax_sp_fn, data.trj, oshape=data.mps.shape[1:])

    assert (AHAx_sp_fn.abs().max() == AHAx_sp.abs().max()).all()


def test_sigpy_subspace_linop_vs_functional_scaling(spiral2d_data):
    data = spiral2d_data
    F_sp = NUFFT(
        data.trj,
        data.mps.shape[1:],
        in_batch_shape=("A",),
        out_batch_shape=("R", "K"),
        toeplitz=False,
        backend="sigpy",
    )
    subspace_img = data.img[None] * torch.randn(5, 1, 1)
    AHAx_sp = F_sp.N(subspace_img)
    Ax_sp_fn = sp_nufft(subspace_img, data.trj)
    AHAx_sp_fn = sp_nufft_adjoint(Ax_sp_fn, data.trj, oshape=subspace_img.shape)

    assert (AHAx_sp_fn.abs().max() == AHAx_sp.abs().max()).all()


# Fi tests
def test_fi_linop_vs_functional_scaling(spiral2d_data):
    data = spiral2d_data
    F_fi = NUFFT(
        data.trj,
        data.mps.shape[1:],
        in_batch_shape=tuple(),
        out_batch_shape=("R", "K"),
        toeplitz=False,
        backend="fi",
    )
    AHAx_fi = F_fi.N(data.img)
    Ax_fi_fn = fi_nufft(data.img, data.trj)
    AHAx_fi_fn = fi_nufft_adjoint(Ax_fi_fn, data.trj, oshape=data.mps.shape[1:])

    assert (AHAx_fi_fn.abs().max() == AHAx_fi.abs().max()).all()


def test_fi_linop_vs_functional_scaling(spiral2d_data):
    data = spiral2d_data
    F_fi = NUFFT(
        data.trj,
        data.mps.shape[1:],
        in_batch_shape=tuple(),
        out_batch_shape=("R", "K"),
        toeplitz=False,
        backend="fi",
    )
    subspace_img = data.img[None] * torch.randn(5, 1, 1)
    AHAx_fi = F_fi.N(subspace_img)
    Ax_fi_fn = fi_nufft(subspace_img, data.trj)
    AHAx_fi_fn = fi_nufft_adjoint(Ax_fi_fn, data.trj, oshape=subspace_img.shape)

    assert (AHAx_fi_fn.abs().max() == AHAx_fi.abs().max()).all()


def test_fi_planned_vs_functional_scaling(spiral2d_data):
    data = spiral2d_data
    F_fi = NUFFT(
        data.trj,
        data.mps.shape[1:],
        in_batch_shape=tuple(),
        out_batch_shape=("R", "K"),
        toeplitz=False,
        backend="fi",
        extras={
            "plan_ahead": "cpu",
            "N_shape": (1,),
        },
    )
    AHAx_fi = F_fi.N(data.img)
    Ax_fi_fn = fi_nufft(data.img, data.trj)
    AHAx_fi_fn = fi_nufft_adjoint(Ax_fi_fn, data.trj, oshape=data.mps.shape[1:])

    assert (AHAx_fi_fn.abs().max() == AHAx_fi.abs().max()).all()


def test_fi_subspace_planned_vs_functional_scaling(spiral2d_data):
    data = spiral2d_data
    A = 5
    F_fi = NUFFT(
        data.trj,
        data.mps.shape[1:],
        in_batch_shape=("A",),
        out_batch_shape=("R", "K"),
        toeplitz=False,
        backend="fi",
        extras={
            "plan_ahead": "cpu",
            "N_shape": (A,),
        },
    )
    subspace_img = data.img[None] * torch.randn(A, 1, 1)
    AHAx_fi = F_fi.N(subspace_img)
    Ax_fi_fn = fi_nufft(subspace_img, data.trj)
    AHAx_fi_fn = fi_nufft_adjoint(Ax_fi_fn, data.trj, oshape=subspace_img.shape)

    assert (AHAx_fi_fn.abs().max() == AHAx_fi.abs().max()).all()


def test_fi_vs_sigpy_scaling(spiral2d_data):
    data = spiral2d_data
    F_fi = NUFFT(
        data.trj,
        data.mps.shape[1:],
        in_batch_shape=tuple(),
        out_batch_shape=("R", "K"),
        toeplitz=False,
        backend="fi",
    )

    F_fi_plan = NUFFT(
        data.trj,
        data.mps.shape[1:],
        in_batch_shape=tuple(),
        out_batch_shape=("R", "K"),
        toeplitz=False,
        backend="fi",
        extras={
            "plan_ahead": "cpu",
            "N_shape": (1,),
        },
    )

    F_sp = NUFFT(
        data.trj,
        data.mps.shape[1:],
        in_batch_shape=tuple(),
        out_batch_shape=("R", "K"),
        toeplitz=False,
        backend="sigpy",
    )

    AHAx_fi = F_fi.N(data.img)
    AHAx_fi_plan = F_fi_plan.N(data.img)
    AHAx_sigpy = F_sp.N(data.img)

    assert torch.isclose(AHAx_fi, AHAx_fi_plan).all()


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_fi_vs_sigpy_scaling_gpu(spiral2d_data):
    data = spiral2d_data
    F_fi = NUFFT(
        data.trj,
        data.mps.shape[1:],
        in_batch_shape=tuple(),
        out_batch_shape=("R", "K"),
        toeplitz=False,
        backend="fi",
    )

    F_fi_plan = NUFFT(
        data.trj,
        data.mps.shape[1:],
        in_batch_shape=tuple(),
        out_batch_shape=("R", "K"),
        toeplitz=False,
        backend="sigpy",
    )

    F_sp = NUFFT(
        data.trj,
        data.mps.shape[1:],
        in_batch_shape=tuple(),
        out_batch_shape=("R", "K"),
        toeplitz=False,
        backend="sigpy",
    )
