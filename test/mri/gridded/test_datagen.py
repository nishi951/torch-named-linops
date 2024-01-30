import marimo

__generated_with = "0.1.88"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    from typing import Tuple

    from einops import rearrange
    import numpy as np
    import torch
    import sigpy as sp
    import matplotlib
    import matplotlib.pyplot as plt

    from torchlinops.mri.gridded.backend.datagen import (
        ImplicitGROGDataset,
        ImplicitGROGDatasetConfig,
    )
    from mr_sim.trajectory.trj import spiral_2d, tgas_spi
    return (
        ImplicitGROGDataset,
        ImplicitGROGDatasetConfig,
        Tuple,
        matplotlib,
        mo,
        np,
        plt,
        rearrange,
        sp,
        spiral_2d,
        tgas_spi,
        torch,
    )


@app.cell
def __(mo):
    mo.md("# Testing Implicit GROG Data Generation")
    return


@app.cell
def __(np, sp, spiral_2d, tgas_spi):
    def gen_mri_dataset(im_size, num_coils):
        # Image
        img = sp.shepp_logan(im_size).astype(np.complex64)

        # Trajectory
        if len(im_size) == 2:
            trj = spiral_2d(im_size)
        elif len(im_size) == 3:
            trj = tgas_spi(im_size, ntr=500)
        else:
            raise ValueError(f'Unsupported image dimension: {len(im_size)} (size {im_size})')

        # Coils
        mps = sp.mri.birdcage_maps((num_coils, *im_size))
        return img, trj, mps


    def gen_mrf_dataset(im_size, num_coils):
        ...
    return gen_mrf_dataset, gen_mri_dataset


@app.cell
def __(gen_mri_dataset):
    im_size = (100, 100)
    num_coils = 8
    img, trj, mps = gen_mri_dataset(im_size, num_coils)
    return im_size, img, mps, num_coils, trj


@app.cell
def __(mo, mps, trj):
    coil_idx = mo.ui.slider(start=0, stop=mps.shape[0]-1, label='Coil Index')
    trj_idx = mo.ui.slider(start=0, stop=trj.shape[1]-1, label='Trajectory Index')
    mo.vstack([trj_idx, coil_idx])
    return coil_idx, trj_idx


@app.cell
def __(coil_idx, img, mps, np, plt, trj, trj_idx):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(np.abs(img))
    ax[0].set_title('Phantom')
    ax[1].plot(trj[:, trj_idx.value, 0], trj[:, trj_idx.value, 1])
    ax[1].axis('square')
    ax[1].set_xlim(-img.shape[0]//2, img.shape[0]//2)
    ax[1].set_ylim(-img.shape[1]//2, img.shape[1]//2)
    ax[1].set_title(f'Trj[{trj_idx.value}]')
    ax[2].imshow(np.abs(mps[coil_idx.value]))
    ax[2].set_title(f'Mps[{coil_idx.value}]')
    fig
    return ax, fig


@app.cell
def __(mo):
    # Get ksp data from forward NUFFT
    mo.md("""
    Now we simulate the forward operator using the NUFFT linop:
    """)
    return


@app.cell
def __(Tuple, np, rearrange, torch):
    from torchlinops.core.linops import Repeat
    from torchlinops.mri.linops import NUFFT, SENSE
    def get_linop(im_size: Tuple, trj: np.ndarray, mps: np.ndarray):
        trj_tkbn = torch.from_numpy(trj)
        trj_tkbn = trj_tkbn / torch.tensor(im_size).float() * (2*np.pi)
        trj_tkbn = rearrange(trj_tkbn, 'K R D -> R D K')

        F = NUFFT(trj_tkbn, im_size, batch_shape=('R',))
        C = SENSE(torch.from_numpy(mps).to(torch.complex64))
        R = Repeat(trj_tkbn.shape[0], dim=0, ishape=('C', 'Nx', 'Ny'), oshape=('R', 'C', 'Nx', 'Ny'))
        return F @ R @ C
    return NUFFT, Repeat, SENSE, get_linop


@app.cell
def __(get_linop, im_size, img, mps, torch, trj):
    linop = get_linop(im_size, trj, mps)
    ksp = linop(torch.from_numpy(img).to(torch.complex64))
    ksp.shape
    return ksp, linop


if __name__ == "__main__":
    app.run()
