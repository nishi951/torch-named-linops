import pytest

from typing import Tuple

import torch
from torchlinops.mri.recon._pcg import CGHparams, ConjugateGradient
from torchlinops.mri._linops.nufft.grid import GriddedNUFFT
from torchlinops.mri import DCF, NUFFT
from torchlinops.mri._grid.third_party._igrog.grogify import grogify, training_params, gridding_params
from torchlinops.utils import ordinal, to_pytorch, from_pytorch

from torchlinops.mri.sim.spiral2d import (
    Spiral2dSimulator,
    Spiral2dSimulatorConfig,
)

@pytest.fixture
def spiral2d_data():
    config = Spiral2dSimulatorConfig(
        im_size=(64, 128),
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


def trj_mask(trj: torch.Tensor, max_k: float, ord=float('inf')) -> Tuple[torch.Tensor, torch.Tensor]:
    """Truncate a kspace trajectory to a specific radius

    Parameters
    ----------
    trj : torch.Tensor
        The ksp trajectory to truncate
        where N is the input matrix size in that dimension
    max_k : float
        The radius of the kspace region to truncate to.

    Returns
    -------
    torch.Tensor, size [K', D] | float
        The truncated kspace trajectory. Note the readout has collapsed.
    torch.Tensor, size [K...] | bool
        A boolean tensor mask

    Note: May need to pre-scale trj in each dimension to have desired behavior
    """
    return torch.linalg.norm(trj, ord=ord, dim=-1, keepdim=False) <= max_k

@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_gridded_nufft_2d(spiral2d_data):
    data = spiral2d_data
    cal_frac = 0.5
    calib_width = 16
    kernel_width = 6
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
    img_cal = ConjugateGradient(A.N, CGHparams(num_iter=10))(AHb, AHb)

    assert img_cal.shape == data.mps.shape

    # Grid data
    device = torch.device('cuda')
    gparams = gridding_params(kern_width=3)
    tparams = training_params(show_loss=False, epochs=20)
    ksp_grd, trj_grd = grogify(data.ksp.to(device), data.trj.to(device), img_cal.to(device), tparams, gparams)
    D = DCF(trj_grd, data.img.shape, ("C", "R", "K"), show_pbar=False, device_idx=ordinal(device))
    F = NUFFT(trj_grd, im_size)

    # Estimate sensitivity maps
    kgrid = torch.fft.fftn(img_cal, dim=tuple(range(-len(im_size), 0)), norm='ortho')
    mps = to_pytorch(
        mri.app.EspiritCalib(
            from_pytorch(kgrid),
            calib_width=calib_width,
            kernel_width=kernel_width,
            device=sp.Device(ordinal(device)),
        ).run()
    )
    S = SENSE(mps)
    breakpoint()

    # Construct gridded linop
    A = (D ** (1/2)) @ F @ S
    # Do recon with and without gridding
    # Make sure both recons are similar
