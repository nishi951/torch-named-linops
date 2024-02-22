import tyro
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import sigpy.mri as mri

from torchlinops.utils import numpy2torch, get_device
import torchlinops.mri.app as app
from mr_sim.trajectory.trj import spiral_2d

from torchlinops.mri.gridding.backends import igrog_grogify, grog_params


def main(device_idx: int = -1):
    device = get_device(device_idx)
    # Simulate data
    im_size = (64, 64)
    num_coils = 8
    img = sp.shepp_logan(im_size).astype(np.complex64)
    # trj = np.round(spiral_2d(im_size))
    trj = np.round(spiral_2d(im_size))
    dcf = mri.pipe_menon_dcf(
        trj, im_size, device=sp.Device(device_idx), show_pbar=False
    )
    mps = mri.birdcage_maps((num_coils, *im_size))
    data = app.Simulate(img, trj, mps, device_idx=0).run()

    # Extract calibration
    data.mps_recon, data.kgrid_recon = app.Calib(
        data.trj,
        data.ksp,
        data.mps.shape[1:],
        device=sp.get_device(data.ksp),
    ).run()

    # Run Naive Recon
    data.dcf = dcf
    data_torch = numpy2torch(data)
    cgsense_recon = app.CGSENSE(
        data_torch.ksp,
        data_torch.trj,
        data_torch.mps_recon,
        data_torch.dcf,
        gridded=False,
    ).run()

    # Run iGROG Recon
    gparams = grog_params(
        lamda_tikonov=1.0,
        num_inputs=5,
        kernel_width_readout=1.0,
        oversamp_grid=1.0,
        no_grid=False,
        verbose=False,
    )
    data.trj_grd, data.ksp_grd, _ = igrog_grogify(
        data.trj,
        data.ksp,
        data.kgrid_recon,
        im_size=im_size,
        gparams=gparams,
        device_idx=0,
    )
    data_torch = numpy2torch(data)
    cgsense_gridded_recon = app.CGSENSE(
        data_torch.ksp_grd,
        data_torch.trj_grd,
        data_torch.mps_recon,
        data_torch.dcf,
        gridded=True,
    ).run()

    matplotlib.use('WebAgg')
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(img)
    ax[1].imshow(cgsense_recon)
    ax[2].imshow(cgsense_gridded_recon)

    breakpoint()
    # Run FISTA+LLR Recon
    prior = LocallyLowRank()
    fistallr_recon = app.FISTA(
        data.ksp_grd,
        data.trj_grd,
        data.mps_recon,
        prior=prior,
        # FISTA Params
    ).run()

    ###

    # Grid trj and ksp
    data.ksp_grid, data.trj_grid = app.VanillaImplicitGROG(
        data.kgrid_recon,  # Calibration region
        data.ksp,  # Kspace data
        data.trj,  # Trajectory
    ).run()

    # Rerun the old recons
    cgsense_grid_recon = app.CGSENSE(
        data.ksp_grid,
        data.trj_grid,
        data.mps_recon,
    ).run()

    fistallr_grid_recon = app.FISTA(
        data.ksp_grid,
        data.trj_grid,
        data.mps_recon,
        prior=prior,
        # FISTA params
    ).run()

    breakpoint()


if __name__ == '__main__':
    tyro.cli(main)
