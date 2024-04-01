import pytest

from typing import Tuple

import sigpy as sp
import sigpy.mri as mri
import torch
import torch.fft as fft
from torchlinops import Identity
from torchlinops.mri.recon._pcg import CGHparams, ConjugateGradient
from torchlinops.mri._linops.nufft.grid import GriddedNUFFT
from torchlinops.mri import DCF, NUFFT, SENSE
from torchlinops.mri._grid.third_party._igrog.grogify import grogify, training_params, gridding_params
from torchlinops.mri.data import trj_mask
from torchlinops.utils import ordinal, to_pytorch, from_pytorch, cfft, cifft

from torchlinops.mri.sim.spiral2d import (
    Spiral2dSimulator,
    Spiral2dSimulatorConfig,
)

@pytest.fixture
def spiral2d_data():
    config = Spiral2dSimulatorConfig(
        im_size=(128, 128),
        num_coils=16,
        noise_std=1e-5,
        spiral_2d_kwargs={
            "n_shots": 16,
            "alpha": 1.0,
            "f_sampling": 0.2,
            "g_max": 0.04,
            "s_max": 100.
        },
    )

    simulator = Spiral2dSimulator(config)
    data = simulator.data
    return data

def mask_by_img(x, reference_img, eps=1e-2):
    mask = torch.abs(reference_img) < eps
    out = x.clone()
    out[mask] = 0.0
    return out



@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_gridded_nufft_2d(spiral2d_data):
    data = spiral2d_data
    cal_frac = 0.5
    calib_width = 16
    kernel_width = 6

    cg_lambda = 1e-2
    cg_iters = 200
    epochs = 20
    im_size = torch.tensor(data.img.shape)
    #cal_im_size = torch.round(im_size * cal_frac).long()

    mask = trj_mask(data.trj/im_size, max_k = cal_frac)
    trj_trunc = data.trj[mask, :]
    ksp_trunc = data.ksp[:, mask]
    D_trunc = DCF(trj_trunc, data.img.shape, ("C", "K"), show_pbar=False)
    F_trunc = NUFFT(
        trj_trunc,
        im_size=data.img.shape,
        in_batch_shape=("C",),
        out_batch_shape=("K",),
        backend='fi',
        toeplitz=True,
    )
    A = (D_trunc ** (1/2)) @ F_trunc
    b = (D_trunc ** (1/2))(ksp_trunc)
    AHb = A.H(b)
    img_cal = ConjugateGradient(
        A.N + cg_lambda * Identity(A.ishape),
        CGHparams(num_iter=cg_iters)
    )(AHb, AHb)

    assert img_cal.shape == data.mps.shape

    # Grid data
    device = torch.device('cuda')
    gparams = gridding_params(kern_width=3)
    tparams = training_params(show_loss=False, epochs=epochs)
    ksp_grd, trj_grd = grogify(data.ksp.to(device), data.trj.to(device), img_cal.to(device), tparams, gparams)
    D = DCF(trj_grd, data.img.shape, ("C", "R", "K"), show_pbar=False, device_idx=ordinal(device))
    F = NUFFT(
        trj_grd,
        im_size,
        in_batch_shape=("C",),
        out_batch_shape=("R", "K"),
        toeplitz=True,
        toeplitz_oversamp=2.,
        backend='grid',
    )

    # Estimate sensitivity maps
    kgrid = cfft(img_cal, dim=tuple(range(-len(im_size), 0)), norm='ortho')
    mps = to_pytorch(
        mri.app.EspiritCalib(
            from_pytorch(kgrid),
            calib_width=calib_width,
            kernel_width=kernel_width,
            device=sp.Device(ordinal(device)),
        ).run()
    )
    S = SENSE(mps)

    # Construct gridded linop
    A = (D ** (1/2)) @ F @ S
    b = (D ** (1/2))(ksp_grd)

    # Do recon with gridding
    AHb = A.H(b)
    recon_grd = ConjugateGradient(
        A.N + cg_lambda * Identity(A.ishape),
        CGHparams(num_iter=cg_iters)
    )(AHb, AHb)


    # Do recon without gridding
    D = DCF(data.trj, data.img.shape, ("C", "R", "K"), show_pbar=False, device_idx=ordinal(device))
    F = NUFFT(
        data.trj,
        im_size,
        in_batch_shape=("C",),
        out_batch_shape=("R", "K"),
        toeplitz=True,
        toeplitz_oversamp=2.,
        backend='fi',
    )
    A = (D ** (1/2)) @ F @ S
    A.to(device)
    b = (D ** (1/2))(data.ksp.to(device))
    AHb = A.H(b)
    recon = ConjugateGradient(
        A.N + cg_lambda * Identity(A.ishape),
        CGHparams(num_iter=cg_iters),
    )(AHb, AHb)

    # Make sure both recons are similar
    # Visualize
    # import numpy as np
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('WebAgg')
    # plt.figure()
    # plt.title('Ground Truth')
    # plt.imshow(np.abs(data.img.detach().cpu().numpy()))

    # plt.figure()
    # plt.title('img_cal')
    # plt.imshow(np.abs(img_cal[0].detach().cpu().numpy()))

    # plt.figure()
    # plt.title('Mps[0]')
    # plt.imshow(np.abs(mps[0].detach().cpu().numpy()))

    # plt.figure()
    # plt.title('Recon (Gridded)')
    # plt.imshow(np.abs(recon_grd.detach().cpu().numpy()))

    # plt.figure()
    # plt.title('Recon (NUFFT)')
    # plt.imshow(np.abs(recon.detach().cpu().numpy()))

    # plt.show()
    # breakpoint

    assert torch.isclose(recon, recon_grd, atol=1e-1, rtol=1e-1).all()
