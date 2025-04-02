import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from jaxtyping import Float
    from torch import Tensor

    from einops import rearrange
    from sigpy import mri
    import torch
    import matplotlib.pyplot as plt
    from torchlinops import Dim
    from torchlinops.utils import cfft2, cifft2
    from torchlinops.linops.nufft import toeplitz_psf, NUFFT
    import numpy as np
    from sigpy.fourier import (
        toeplitz_psf as sp_toeplitz_psf,
        fft as sp_fft,
        ifft as sp_ifft,
    )

    def spiral_2d(
        im_size: tuple,
        n_shots: int = 16,
        alpha: float = 1.5,
        f_sampling: float = 0.4,
        g_max: float = 40.0,
        s_max: float = 100.0,
    ) -> Float[Tensor, "R K D"]:
        """
        Generates an 2-dimensional variable density spiral

        Parameters:
        ----------
        im_size: Tuple
            2D Image resolution tuple
        n_shots : int
            number of phase encodes to cover k-space
        alpha : float
            controls variable density. 1.0 means no variable density, center denisty increases with alpha
        g_max : float
            Maximum gradient amplitude in T/m
        s_max : float
            Maximum gradient slew rate in T/m/s

        Returns:
        ----------
        Float[Tensor, "R K D"]
            k-space trajector with shape (n_shots, n_readout_points,  d), d = len(im_size)
            sigpy scaling - each dim in range [-N//2, N//2]
        """

        # Gen spiral
        trj = mri.spiral(
            fov=1,
            N=max(im_size),
            f_sampling=f_sampling,  # TODO function of self.n_read
            R=1,
            ninterleaves=n_shots,
            alpha=alpha,
            gm=g_max,  # Tesla / m
            sm=s_max,  # Tesla / m / s
        )
        assert trj.shape[0] % n_shots == 0
        trj = rearrange(trj, "(R K) D -> R K D", R=n_shots)
        # Normalize to sigpy scaling
        im_size_2 = np.array(im_size) / 2.0
        trj = trj * im_size_2 / trj.max((0, 1), keepdims=False)
        trj = torch.from_numpy(trj).to(torch.float32)
        return trj

    return (
        Dim,
        Float,
        NUFFT,
        Tensor,
        cfft2,
        cifft2,
        mo,
        mri,
        np,
        plt,
        rearrange,
        sp_fft,
        sp_ifft,
        sp_toeplitz_psf,
        spiral_2d,
        toeplitz_psf,
        torch,
    )


@app.cell
def _(Dim, NUFFT, cifft2, spiral_2d, toeplitz_psf, torch):
    import sigpy as sp

    Nx, Ny = 64, 64
    im_size = (Nx, Ny)

    def make_locs(im_size, mode="cartesian", R=1):
        Nx, Ny = im_size
        if mode == "cartesian":
            kx = torch.linspace(-Nx // 2, Nx // 2 - 1, Nx)
            ky = torch.linspace(-Ny // 2, Ny // 2 - 1, Ny)
            locs = torch.meshgrid(kx, ky, indexing="ij")
            locs = torch.stack(locs, dim=-1)
        elif mode == "random":
            kx = torch.rand(im_size) * (Nx - 1) - Nx // 2
            ky = torch.rand(im_size) * (Ny - 1) - Ny // 2
            locs = torch.stack([kx, ky], dim=-1)
        elif mode == "spiral":
            locs = spiral_2d(im_size)
            locs = locs[::R]
        return locs

    def make_nufft(locs, im_size, oversamp):
        return NUFFT(
            locs, im_size, output_shape=Dim("KxKy"), oversamp=oversamp, width=4.0
        )

    locs = make_locs(im_size, mode="random", R=4)
    # plt.scatter(locs[..., 0], locs[..., 1])
    print(locs)
    nufft = make_nufft(locs.clone(), im_size, oversamp=1.25)
    kern, _, _, nufft_os = toeplitz_psf(nufft, None)
    otf = kern.weight[0, 0]
    psf = cifft2(otf)
    return (
        Nx,
        Ny,
        im_size,
        kern,
        locs,
        make_locs,
        make_nufft,
        nufft,
        nufft_os,
        otf,
        psf,
        sp,
    )


@app.cell
def _(otf, plt):
    with plt.rc_context({"figure.dpi": 256}):
        plt.imshow(otf.abs())
    # print(psf.abs()[64:66, 64:66])
    plt.colorbar()
    plt.gcf()
    return


@app.cell
def _(im_size, locs, np, plt, sp_ifft, sp_toeplitz_psf):
    otf_sp = sp_toeplitz_psf(locs.numpy(), im_size)
    psf_sp = sp_ifft(otf_sp)
    with plt.rc_context({"figure.dpi": 256}):
        plt.imshow(np.abs(otf_sp))
        plt.colorbar()
    plt.gcf()
    return otf_sp, psf_sp


@app.cell
def _(im_size, np, psf, psf_sp):
    np.abs(psf_sp[im_size[0] // 2, im_size[1] // 2]) / np.abs(
        psf.numpy()[im_size[0] // 2, im_size[1] // 2]
    )
    return


@app.cell
def _(np, otf):
    np.abs(otf)
    return


@app.cell
def _(np, otf_sp):
    np.abs(otf_sp)
    return


@app.cell
def _(nufft_os):
    nufft_os.interp.locs
    return


@app.cell
def _(nufft_os):
    nufft_os.interp.locs.max()
    return


@app.cell
def _(NUFFT):
    width = 4
    oversamp = 1.25
    beta = NUFFT.beta(width, oversamp)
    return beta, oversamp, width


@app.cell
def _(NUFFT, beta, oversamp, width):
    img_size = (64, 64)
    oversamp_size = tuple(int(oversamp * s) for s in img_size)
    apod = NUFFT.apodize_weights(img_size, oversamp_size, oversamp, width, beta)
    return apod, img_size, oversamp_size


@app.cell
def _(apod, plt):
    plt.imshow(apod)
    plt.colorbar()
    plt.gcf()
    return


@app.cell
def _(nufft):
    nufft.linops[0].weight[20, 20] * 1000
    return


@app.cell
def _(im_size, nufft, torch):
    delta = torch.zeros(im_size)
    center_idx = tuple(s // 2 for s in im_size)
    delta[center_idx] = 1.0
    (1.0 / nufft(delta)).abs()[center_idx]
    return center_idx, delta


@app.cell
def _(delta, nufft):
    1.0 / nufft(delta)
    return


@app.cell
def _(im_size, np):
    from math import prod

    np.sqrt(prod(im_size))
    return (prod,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
