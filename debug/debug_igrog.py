from einops import rearrange
import tyro
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import sigpy.mri as mri

from torchlinops.utils import numpy2torch, get_device, print_shapes
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
    trj = spiral_2d(im_size) # Noncartesian
    trj = rearrange(trj, 'K R D -> R K D')
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

    # Run rounded recon
    cgsense_rounded_recon = app.CGSENSE(
        data_torch.ksp,
        torch.round(data_torch.trj),
        data_torch.mps_recon,
        data_torch.dcf,
        gridded=True,
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
        data.trj.detach().cpu().numpy(),
        data.ksp.detach().cpu().numpy(),
        data.kgrid_recon.detach().cpu().numpy(),
        im_size=im_size,
        gparams=gparams,
        device_idx=0,
    )
    data_torch = numpy2torch(data)

    cgsense_gridded_recon_app = app.CGSENSE(
        data_torch.ksp_grd,
        data_torch.trj_grd,
        data_torch.mps_recon,
        data_torch.dcf,
        gridded=True,
    )
    cgsense_gridded_recon = cgsense_gridded_recon_app.run()
    # Debug
    matplotlib.use('WebAgg')
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax.flat[0].imshow(np.abs(img))
    ax.flat[0].set_title('Ground truth')
    ax.flat[1].imshow(np.abs(cgsense_recon.detach().cpu().numpy()))
    ax.flat[1].set_title('CGSENSE Recon')
    ax.flat[2].imshow(np.abs(cgsense_rounded_recon.detach().cpu().numpy()))
    ax.flat[2].set_title('CGSENSE Rounded Recon')
    ax.flat[3].imshow(np.abs(cgsense_gridded_recon.detach().cpu().numpy()))
    ax.flat[3].set_title('CGSENSE igrog Recon')

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax.flat[0].imshow(np.angle(img), vmin=-np.pi, vmax=np.pi)
    ax.flat[0].set_title('Ground truth')
    ax.flat[1].imshow(np.angle(cgsense_recon.detach().cpu().numpy()),
                      vmin=-np.pi, vmax=np.pi)
    ax.flat[1].set_title('CGSENSE Recon')
    ax.flat[2].imshow(np.angle(cgsense_rounded_recon.detach().cpu().numpy()),
                      vmin=-np.pi, vmax=np.pi)
    ax.flat[2].set_title('CGSENSE Rounded Recon')
    ax.flat[3].imshow(np.angle(cgsense_gridded_recon.detach().cpu().numpy()),
                      vmin=-np.pi, vmax=np.pi)
    ax.flat[3].set_title('CGSENSE igrog Recon')
    plt.show()
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
