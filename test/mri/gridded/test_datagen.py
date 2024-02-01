import marimo

__generated_with = "0.1.88"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    from typing import Tuple, Optional
    import functools

    from einops import rearrange
    import numpy as np
    import torch
    import torch.fft as fft
    import sigpy as sp
    import sigpy.mri as mri
    import matplotlib
    import matplotlib.pyplot as plt

    from torchlinops.mri.gridded.backend.datagen import (
        ImplicitGROGDataset,
        ImplicitGROGDatasetConfig,
    )
    from torchlinops.mri.recon.pcg import CGHparams, ConjugateGradient
    from torchlinops.core.linops import Diagonal, Rearrange, Identity, NamedLinop
    from torchlinops.mri.linops import NUFFT, SENSE

    from mr_sim.trajectory.trj import spiral_2d, tgas_spi
    return (
        CGHparams,
        ConjugateGradient,
        Diagonal,
        Identity,
        ImplicitGROGDataset,
        ImplicitGROGDatasetConfig,
        NUFFT,
        NamedLinop,
        Optional,
        Rearrange,
        SENSE,
        Tuple,
        fft,
        functools,
        matplotlib,
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
def __(mo):
    mo.md("# Testing Implicit GROG Data Generation")
    return


@app.cell
def __(mo):
    mo.md('## Simulation Phase')
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
    return gen_mri_dataset,


@app.cell
def __(NUFFT, SENSE, Tuple, np, rearrange, torch):
    def to_tkbn(trj: np.ndarray, im_size: Tuple):
        """Input trj should be sigpy-style
        i.e. in [-N/2, N/2] and shape [..., K, D]
        """
        trj_tkbn = torch.from_numpy(trj)
        trj_tkbn = trj_tkbn / torch.tensor(im_size).float() * (2*np.pi)
        trj_tkbn = rearrange(trj_tkbn, '... K D -> ... D K')
        return trj_tkbn

    def get_linop(im_size: Tuple, trj: np.ndarray, mps: np.ndarray):
        trj = rearrange(trj, 'K R D -> R K D')
        trj_tkbn = to_tkbn(trj, im_size)
        F = NUFFT(trj_tkbn, im_size,
                  in_batch_shape=('C',),
                  out_batch_shape=('R',),
                 )
        S = SENSE(torch.from_numpy(mps).to(torch.complex64))
        # R = Repeat(trj_tkbn.shape[0], dim=0, ishape=('C', 'Nx', 'Ny'), oshape=('R', 'C', 'Nx', 'Ny'))
        return F @ S
    return get_linop, to_tkbn


@app.cell
def __(NamedLinop, gen_mri_dataset, get_linop, np, torch):
    def simulate(A: NamedLinop, img: np.ndarray, sigma: float = 0.):
        ksp = linop(torch.from_numpy(img).to(torch.complex64))
        ksp = ksp + sigma * torch.randn_like(ksp)
        return ksp
    im_size = (100, 100)
    num_coils = 8
    img, trj, mps = gen_mri_dataset(im_size, num_coils)
    linop = get_linop(im_size, trj, mps)
    ksp = simulate(linop, img, sigma=0.01)
    return im_size, img, ksp, linop, mps, num_coils, simulate, trj


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
def __(coil_idx, ksp, np, plt, trj_idx):
    # Plot simulated kspace
    readout = ksp[trj_idx.value, coil_idx.value].detach().cpu().numpy()
    plt.plot(np.abs(readout))
    plt.xlabel('Readout Sample')
    plt.ylabel('Abs(readout)')
    plt.title(f'Ksp[trj[{trj_idx.value}], coil[{coil_idx.value}]]')
    return readout,


@app.cell
def __(mo):
    mo.md('## Reconstruction Phase')
    return


@app.cell
def __(mo):
    mo.md("""
    In this phase, we only make use of:

    - The kspace data `ksp` (numpy)
    - The trajectory `trj` (numpy)

    We must estimate:

    - The density compensation function `dcf`
    - The sensitivity maps `mps`

    Once we do that, we can reconstruct the image.
    """)
    return


@app.cell
def __(
    CGHparams,
    ConjugateGradient,
    Diagonal,
    Identity,
    NUFFT,
    Optional,
    Tuple,
    fft,
    np,
    to_tkbn,
    torch,
):
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

    def truncate_trj_ksp(trj, ksp, max_k, dcf: Optional[np.ndarray] = None):
        mask = np.all(np.abs(trj) <= max_k, axis=-1)
        trj_truncated = trj[mask, :] # [B... K D] -> [K' D]
        ksp_truncated = ksp[:, mask] # [C B... K] -> [C K']
        if dcf is not None:
            dcf_truncated = dcf[mask] # [B... K] -> [K']
            return trj_truncated, ksp_truncated, dcf_truncated
        return trj_truncated, ksp_truncated


    def inufft(
        omega: torch.Tensor,
        ksp: torch.Tensor,
        im_size: Tuple,
        dcf: Optional[torch.Tensor] = None,
        num_iter=10,
        device='cpu',
    ):
        """Inverse NUFFT aka least squares via PCG
        omega: [B... D K] tkbn-style tensor
        ksp: [C B... K] tensor where B... is the same batch as omega's B...
        dcf: if provided, [B... K]
        """
        hparams = CGHparams(num_iter=num_iter)
        device = ksp.device
        C = ksp.shape[0]
        batch = tuple(f'B{i}' for i in range(len(ksp.shape[1:-1])))
        # Create simple linop
        F = NUFFT(omega, im_size,
                  in_batch_shape=('C',),
                  out_batch_shape=batch).to(device)
        if dcf is not None:
            D = Diagonal(torch.sqrt(dcf),
                         ioshape=('C', *batch, 'K'),
                        ).to(device)
        else:
            D = Identity(ioshape=('C', *batch, 'K')).to(device)
        A = D @ F
        AHb = A.H(D(ksp))
        cg = ConjugateGradient(A.N, hparams).to(device)
        return cg(AHb, AHb)

    def synth_cal(
        trj,
        ksp,
        acs_size: int,
        dcf: Optional[np.ndarray] = None,
        device: torch.device = 'cpu',
    ):
        D = trj.shape[-1]
        cal_size = (acs_size,) * D
        if dcf is not None:
            trj, ksp, dcf = truncate_trj_ksp(trj, ksp, max_k=acs_size/2, dcf=dcf)
        else:
            trj, ksp = truncate_trj_ksp(trj, ksp, max_k=acs_size/2)
        omega = to_tkbn(trj, cal_size)
        ksp = torch.from_numpy(ksp).to(torch.complex64)
        if dcf is not None:
            dcf = torch.as_tensor(dcf)
        img_cal = inufft(omega, ksp, cal_size, dcf=dcf, num_iter=10, device=device)
        kgrid = sp_fft(img_cal, dim=(-2, -1))
        return kgrid.detach().cpu().numpy()
    return inufft, sp_fft, sp_ifft, synth_cal, truncate_trj_ksp


@app.cell
def __(im_size, ksp, mri, rearrange, sp, synth_cal, torch, trj):
    def get_mps_kgrid(trj, ksp, im_size, calib_width, kernel_width, device_idx, **espirit_kwargs):
        """
        """
        if len(espirit_kwargs) == 0:
            # Defaults
            espirit_kwargs = {
                'crop': 0.8,
                'thresh': 0.05,
            }
        device = torch.device(f'cuda:{device_idx}' if device_idx >= 0 else 'cpu')
        dcf = mri.pipe_menon_dcf(trj, im_size, device=sp.Device(device_idx))
        xp = sp.get_device(dcf).xp
        dcf /= xp.linalg.norm(dcf)
        kgrid = synth_cal(trj, ksp, acs_size, dcf, device)
        kgrid_pad = sp.resize(kgrid, (kgrid.shape[0], *im_size))
        mps = mri.app.EspiritCalib(
            kgrid_pad,
            calib_width=calib_width,
            kernel_width=kernel_width,
            device=sp.Device(device_idx),
            **espirit_kwargs,
        ).run()
        return sp.to_device(mps, sp.cpu_device), kgrid

    acs_size = 24
    kernel_width = 7
    trj_np = rearrange(trj, 'K R D -> R K D')
    ksp_np = ksp.detach().cpu().numpy()
    mps_recon, kgrid = get_mps_kgrid(
        trj_np,
        ksp_np,
        im_size,
        calib_width=acs_size,
        kernel_width=kernel_width,
        device_idx=0
    )
    return (
        acs_size,
        get_mps_kgrid,
        kernel_width,
        kgrid,
        ksp_np,
        mps_recon,
        trj_np,
    )


@app.cell
def __(mps_recon):
    mps_recon.shape
    return


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
    plt.imshow(np.abs(mps_recon[mps_idx.value]))
    plt.title(f'Mps[{mps_idx.value}]')
    return


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
    Identity,
    NUFFT,
    Optional,
    SENSE,
    functools,
    ksp_np,
    mps_recon,
    np,
    to_tkbn,
    torch,
    trj_np,
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
        F = NUFFT(omega, im_size,
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

    @functools.cache
    def recon_interactive(num_iter: int, lam: float):
        return cgsense(ksp_np, trj_np, mps_recon, lam=lam, num_iter=num_iter, device_idx=0).detach().cpu().numpy()
    return cgsense, recon_interactive


@app.cell
def __(lam, np, num_iter, plt, recon_interactive):
    #recon = cgsense(ksp_np, trj_np, mps_recon, device_idx=0)
    recon = recon_interactive(num_iter.value, float(lam.value))
    plt.imshow(np.abs(recon))
    plt.title('Recon')
    return recon,


@app.cell
def __(mo):
    mo.md("## Sigpy baseline")
    return


@app.cell
def __(mri, sp):
    import sigpy.linop as sp_linop
    def run_sigpy_sense_recon(
        ksp,
        trj,
        mps,
        dcf,
        lam=0.1,
        device_idx=-1
    ):
        """
        trj: [B... K D]
        mps: [C *im_size]
        dcf: [B... K]
        """

        def mvd(x):
            return sp.to_device(x, sp.Device(device_idx))
        def mvc(x):
            return sp.to_device(x, sp.cpu_device)
        xp = sp.Device(device_idx).xp
        im_size = mps.shape[1:]
        if dcf is None:
            dcf = mri.pipe_menon_dcf(trj, im_size, device=sp.Device(device_idx))
        B = trj.shape[:-2]
        A = mri.linop.Sense(
            mvd(mps),
            mvd(trj),
            mvd(dcf),
        )
        recon = mri.app.SenseRecon(
            mvd(ksp),
            mvd(mps),
            lamda=lam,
            weights=mvd(dcf),
            coord=mvd(trj),
            device=sp.Device(device_idx)
        ).run()
        # F = sp_linop.NUFFT(mps.shape, mvd(trj))
        # S = sp_linop.Multiply(im_size, mvd(mps))
        # D = sp_linop.Multiply(dcf.shape, mvd(xp.sqrt(dcf)))
        return recon, A, dcf
    return run_sigpy_sense_recon, sp_linop


@app.cell
def __(ksp, mps, rearrange, run_sigpy_sense_recon, trj):
    print(ksp.shape)
    print(trj.shape)
    print(mps.shape)
    ksp_sp = rearrange(ksp.detach().cpu().numpy(),
                      'C R K -> C K R')
    recon_sp, A, dcf = run_sigpy_sense_recon(ksp_sp, trj, mps, None, lam=0.1, device_idx=0)
    return A, dcf, ksp_sp, recon_sp


@app.cell
def __(np, plt, recon_sp):
    plt.imshow(np.abs(recon_sp.get()))
    return


@app.cell
def __(mo):
    mo.md("## Sigpy + Pytorch functionality")
    return


@app.cell
def __(CGHparams, ConjugateGradient, ksp_sp, sp, torch):
    from sigpy.app import LinearLeastSquares
    def sigpy_pytorch_solve(A, dcf, ksp, device_idx):
        def mvd(x):
            return sp.to_device(x, sp.Device(device_idx))
        def mvc(x):
            return sp.to_device(x, sp.cpu_device)
        xp = sp.Device(device_idx).xp
        AHA = sp.to_pytorch_function(A.N, input_iscomplex=True, output_iscomplex=True)
        AHb = A.H(xp.sqrt(mvd(dcf)) * mvd(ksp_sp))
        AHb = sp.to_pytorch(AHb)
        cg = ConjugateGradient(AHA.apply, hparams=CGHparams(num_iter=100))
        recon = cg(AHb, AHb)
        return torch.view_as_complex(recon)
    return LinearLeastSquares, sigpy_pytorch_solve


@app.cell
def __(A, dcf, ksp_sp, sigpy_pytorch_solve):
    recon_sp2 = sigpy_pytorch_solve(A, dcf, ksp_sp, device_idx=0)
    return recon_sp2,


@app.cell
def __(np, plt, recon_sp2):
    plt.imshow(np.abs(recon_sp2.detach().cpu().numpy()))
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
