import marimo

__generated_with = "0.2.4"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import functools
    from typing import Tuple, Optional

    import matplotlib.pyplot as plt
    from einops import rearrange
    import torch
    import torch.fft as fft
    import numpy as np
    import sigpy as sp
    import sigpy.mri as mri

    from torchlinops.mri.recon.pcg import CGHparams, ConjugateGradient
    from torchlinops.core.base import NamedLinop
    from torchlinops.core.linops import Diagonal, Identity
    from torchlinops.mri.linops import NUFFT, SENSE
    from torchlinops.mri.igrog.linops import GriddedNUFFT
    from torchlinops.app.calib import Calib

    from mr_sim.trajectory.trj import spiral_2d, tgas_spi
    return (
        CGHparams,
        Calib,
        ConjugateGradient,
        Diagonal,
        GriddedNUFFT,
        Identity,
        NUFFT,
        NamedLinop,
        Optional,
        SENSE,
        Tuple,
        fft,
        functools,
        mo,
        mri,
        np,
        plt,
        rearrange,
        sp,
        spiral_2d,
        tgas_spi,
        torch,
    )


@app.cell
def __(
    GriddedNUFFT,
    NUFFT,
    NamedLinop,
    SENSE,
    Tuple,
    np,
    rearrange,
    sp,
    spiral_2d,
    tgas_spi,
    torch,
):
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

    def get_sp_linop(im_size: Tuple, trj: np.ndarray, mps: np.ndarray):
        trj = rearrange(trj, 'K R D -> R K D')
        trj = torch.from_numpy(trj)
        # trj_tkbn = to_tkbn(trj, im_size)
        F = NUFFT(trj, im_size,
                  in_batch_shape=('C',),
                  out_batch_shape=('R',),
                 )
        S = SENSE(torch.from_numpy(mps).to(torch.complex64))
        # R = Repeat(trj_tkbn.shape[0], dim=0, ishape=('C', 'Nx', 'Ny'), oshape=('R', 'C', 'Nx', 'Ny'))
        return F @ S

    def get_gridded_linop(im_size: Tuple, trj: np.ndarray, mps: np.ndarray):
        trj = rearrange(trj, 'K R D -> R K D')
        trj = torch.from_numpy(trj)
        trj = torch.round(trj)
        F = GriddedNUFFT(
            trj, im_size,
            in_batch_shape=('C',),
            out_batch_shape=('R',),
        )
        S = SENSE(torch.from_numpy(mps).to(torch.complex64))
        return F @ S

    def simulate(A: NamedLinop, img: np.ndarray, sigma: float = 0.):
        ksp = linop(torch.from_numpy(img).to(torch.complex64))
        ksp = ksp + sigma * torch.randn_like(ksp)
        return ksp

    im_size = (100, 100)
    num_coils = 8
    img, trj, mps = gen_mri_dataset(im_size, num_coils)

    # Non-cartesian version
    linop = get_sp_linop(im_size, trj, mps)
    ksp = simulate(linop, img, sigma=0.01)

    # Gridded version
    grid_linop = get_gridded_linop(im_size, trj, mps)
    ksp_grid = simulate(grid_linop, img, sigma=0.01)
    print(ksp.shape)
    print(ksp_grid.shape)
    return (
        gen_mri_dataset,
        get_gridded_linop,
        get_sp_linop,
        grid_linop,
        im_size,
        img,
        ksp,
        ksp_grid,
        linop,
        mps,
        num_coils,
        simulate,
        trj,
    )


@app.cell
def __(mo, mps, trj):
    coil_idx = mo.ui.slider(start=0, stop=mps.shape[0]-1, label='Coil Index')
    trj_idx = mo.ui.slider(start=0, stop=trj.shape[1]-1, label='Trajectory Index')
    mo.vstack([trj_idx, coil_idx])
    return coil_idx, trj_idx


@app.cell
def __(coil_idx, grid_linop, img, mps, np, plt, trj, trj_idx):
    fig, ax = plt.subplots(nrows=2, ncols=3)
    ax[0,0].imshow(np.abs(img))
    ax[0,0].set_title('Phantom')
    ax[0,1].plot(trj[:, trj_idx.value, 0], trj[:, trj_idx.value, 1])
    ax[0,1].axis('square')
    ax[0,1].set_xlim(-img.shape[0]//2, img.shape[0]//2)
    ax[0,1].set_ylim(-img.shape[1]//2, img.shape[1]//2)
    ax[0,1].set_title(f'Trj[{trj_idx.value}]')
    ax[0,2].imshow(np.abs(mps[coil_idx.value]))
    ax[0,2].set_title(f'Mps[{coil_idx.value}]')

    trj_grd = grid_linop.linops[0].trj
    ax[1,1].plot(trj_grd[trj_idx.value, :, 0], trj_grd[trj_idx.value, :, 1])
    ax[1,1].axis('square')
    ax[1,1].set_xlim(-img.shape[0]//2, img.shape[0]//2)
    ax[1,1].set_ylim(-img.shape[1]//2, img.shape[1]//2)
    ax[1,1].set_title(f'Trj[{trj_idx.value}]')
    print(trj.shape)
    print(trj_grd.shape)
    fig
    return ax, fig, trj_grd


@app.cell
def __(coil_idx, ksp, ksp_grid, np, plt, trj_idx):
    # Plot simulated kspace
    readout = ksp[coil_idx.value, trj_idx.value].detach().cpu().numpy()
    readout_grid = ksp_grid[coil_idx.value, trj_idx.value].detach().cpu().numpy()

    fig2, ax2 = plt.subplots(nrows=2, ncols=1)
    ax2[0].plot(np.abs(readout))
    ax2[0].set_xlabel('Readout Sample')
    ax2[0].set_ylabel('Abs(readout)')
    ax2[0].set_title(f'Ksp[trj[{trj_idx.value}], coil[{coil_idx.value}]]')
    ax2[1].plot(np.abs(readout_grid))
    ax2[1].set_xlabel('Readout Sample')
    ax2[1].set_ylabel('Abs(readout)')
    ax2[1].set_title(f'Ksp[trj[{trj_idx.value}], coil[{coil_idx.value}]]')
    fig2
    return ax2, fig2, readout, readout_grid


@app.cell
def __(Calib, fft, im_size, ksp, mo, sp, trj):
    mo.md('## Reconstruction Phase')
    def sp_fft(x, dim=None):
        """Matches Sigpy's fft, but in torch"""
        x = fft.ifftshift(x, dim=dim)
        x = fft.fftn(x, dim=dim, norm='ortho')
        x = fft.fftshift(x, dim=dim)
        return x

    def sp_ifft(x, dim=None, norm=None):
        """Matches Sigpy's fft adjoint, but in torch"""
        x = fft.ifftshift(x, dim=dim)
        x = fft.ifftn(x, dim=dim, norm='ortho')
        x = fft.fftshift(x, dim=dim)
        return x
    mps_recon, kgrid = Calib(
        trj, ksp,
        im_size=im_size,
        calib_width=24,
        kernel_width=7,
        device=sp.Device(0),
    ).run()
    return kgrid, mps_recon, sp_fft, sp_ifft


@app.cell
def __(mo, mps_recon):
    mps_idx = mo.ui.slider(start=0, stop=mps_recon.shape[0]-1, label='mps recon')
    mps_idx
    return mps_idx,


@app.cell
def __(mo):
    mo.md('### Reconstructed maps')
    return


@app.cell
def __(mps_idx, mps_recon, np, plt):
    fig3, ax3 = plt.subplots(nrows=1, ncols=2)
    ax3[0].imshow(np.abs(mps_recon[mps_idx.value]))
    ax3[0].set_title('Magnitude')
    ax3[1].imshow(np.angle(mps_recon[mps_idx.value]), vmin=-np.pi, vmax=np.pi)
    ax3[1].set_title('Angle')
    fig3.suptitle(f'Mps[{mps_idx.value}]')
    fig3
    return ax3, fig3


@app.cell
def __(mo):
    mo.md("### CG-SENSE Recon")
    return


@app.cell
def __(mo):
    num_iter = mo.ui.slider(start=1, stop=20, label='CG Iters')
    lam = mo.ui.dropdown(options=['1e-2', '1e-1', '1e0', '1e1'], label='Lambda', value='1e-1')
    mo.vstack([num_iter, lam])
    return lam, num_iter


@app.cell
def __(
    CGHparams,
    ConjugateGradient,
    Diagonal,
    GriddedNUFFT,
    Identity,
    NUFFT,
    Optional,
    SENSE,
    TorchNUFFT,
    functools,
    ksp_grid_np,
    mps_recon,
    np,
    to_tkbn,
    torch,
    trj_grid_np,
):
    # def get_mps_kgrid(trj, ksp, im_size, calib_width, kernel_width, device_idx, **espirit_kwargs):

    def cgsense(
        ksp: np.ndarray,
        trj: np.ndarray,
        mps: np.ndarray,
        dcf: Optional[np.ndarray] = None,
        lam: float = 0.1,
        num_iter=10,
        device_idx: int = -1,
    ):
        device = torch.device(
            f'cuda:{device_idx}' if device_idx >= 0 else 'cpu'
        )
        im_size = mps.shape[1:]
        ksp = torch.from_numpy(ksp).to(device)
        omega = to_tkbn(trj, im_size).to(device).to(torch.float32)
        mps = torch.from_numpy(mps).to(device).to(torch.complex64)
        batch = tuple(f'B{i}' for i in range(len(ksp.shape[1:-1])))
        # Create simple linop
        F = TorchNUFFT(omega, im_size,
                  in_batch_shape=('C',),
                  out_batch_shape=batch).to(device)
        if dcf is not None:
            D = Diagonal(torch.sqrt(dcf),
                         ioshape=('C', *batch, 'K'),
                        ).to(device)
        else:
            D = Identity(ioshape=('C', *batch, 'K')).to(device)
        S = SENSE(mps).to(device)

        A = (D @ F @ S).to(device)
        AHb = A.H(D(ksp)).to(device)
        def A_reg(x):
            return A.N(x) + lam*x
        cg = ConjugateGradient(A_reg, hparams=CGHparams(num_iter=num_iter)
    ).to(device)
        recon = cg(AHb, AHb)
        return recon

    def sp_cgsense(
        ksp: np.ndarray,
        trj: np.ndarray,
        mps: np.ndarray,
        dcf: Optional[np.ndarray] = None,
        lam: float = 0.1,
        num_iter=10,
        device_idx: int = -1,
    ):
        device = torch.device(
            f'cuda:{device_idx}' if device_idx >= 0 else 'cpu'
        )
        im_size = mps.shape[1:]
        ksp = torch.from_numpy(ksp).to(device)
        # omega = to_tkbn(trj, im_size).to(device).to(torch.float32)
        mps = torch.from_numpy(mps).to(device).to(torch.complex64)
        batch = tuple(f'B{i}' for i in range(len(ksp.shape[1:-1])))
        # Create simple linop
        F = NUFFT(torch.from_numpy(trj), im_size,
                  in_batch_shape=('C',),
                  out_batch_shape=batch).to(device)
        if dcf is not None:
            D = Diagonal(torch.sqrt(dcf),
                         ioshape=('C', *batch, 'K'),
                        ).to(device)
        else:
            D = Identity(ioshape=('C', *batch, 'K')).to(device)
        S = SENSE(mps).to(device)

        A = (D @ F @ S).to(device)
        AHb = A.H(D(ksp)).to(device)
        def A_reg(x):
            return A.N(x) + lam*x
        cg = ConjugateGradient(A_reg, hparams=CGHparams(num_iter=num_iter)
    ).to(device)
        recon = cg(AHb, AHb)
        return recon

    def sp_cgsense_grid(
        ksp: np.ndarray,
        trj: np.ndarray,
        mps: np.ndarray,
        dcf: Optional[np.ndarray] = None,
        lam: float = 0.1,
        num_iter=10,
        device_idx: int = -1,
    ):
        device = torch.device(
            f'cuda:{device_idx}' if device_idx >= 0 else 'cpu'
        )
        im_size = mps.shape[1:]
        ksp = torch.from_numpy(ksp).to(device)
        # omega = to_tkbn(trj, im_size).to(device).to(torch.float32)
        trj = torch.from_numpy(trj).long()
        mps = torch.from_numpy(mps).to(device).to(torch.complex64)
        batch = tuple(f'B{i}' for i in range(len(ksp.shape[1:-1])))
        # Create simple linop
        F = GriddedNUFFT(trj, im_size,
                  in_batch_shape=('C',),
                  out_batch_shape=batch).to(device)
        if dcf is not None:
            D = Diagonal(torch.sqrt(dcf),
                         ioshape=('C', *batch, 'K'),
                        ).to(device)
        else:
            D = Identity(ioshape=('C', *batch, 'K')).to(device)
        S = SENSE(mps).to(device)

        A = (D @ F @ S).to(device)
        AHb = A.H(D(ksp)).to(device)
        def A_reg(x):
            return A.N(x) + lam*x
        cg = ConjugateGradient(A_reg, hparams=CGHparams(num_iter=num_iter)
    ).to(device)
        recon = cg(AHb, AHb)
        return recon

    # @functools.cache
    # def recon_interactive(num_iter: int, lam: float):
    #     return sp_cgsense(ksp_np, trj_np, mps_recon, lam=lam, num_iter=num_iter, device_idx=0).detach().cpu().numpy()

    @functools.cache
    def recon_interactive_grid(num_iter: int, lam: float):
        return sp_cgsense_grid(ksp_grid_np, trj_grid_np, mps_recon, lam=lam, num_iter=num_iter, device_idx=0).detach().cpu().numpy()
    return cgsense, recon_interactive_grid, sp_cgsense, sp_cgsense_grid


@app.cell
def __(lam, np, num_iter, plt, recon_interactive_grid):
    #recon = cgsense(ksp_np, trj_np, mps_recon, device_idx=0)
    recon = recon_interactive_grid(num_iter.value, float(lam.value))
    plt.imshow(np.abs(recon))
    plt.title('Recon (Sigpy backend)')
    return recon,


if __name__ == "__main__":
    app.run()
