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

    from torchlinops.mri.linops import TorchNUFFT
    from torchlinops.mri._sp_nufft import nufft, nufft_adjoint

    from mr_sim.trajectory.trj import spiral_2d, tgas_spi

    def gen_mri_dataset(im_size, num_coils):
        # Image
        img = sp.shepp_logan(im_size).astype(np.complex64)

        # Trajectory
        if len(im_size) == 2:
            trj = spiral_2d(
                im_size, n_shots=3, f_sampling=1.0, alpha=1.0, g_max=1.0, s_max=3.0
            )
        elif len(im_size) == 3:
            trj = tgas_spi(im_size, ntr=500)
        else:
            raise ValueError(
                f"Unsupported image dimension: {len(im_size)} (size {im_size})"
            )

        # Coils
        mps = sp.mri.birdcage_maps((num_coils, *im_size))
        return img, trj, mps

    return (
        Optional,
        TorchNUFFT,
        Tuple,
        fft,
        functools,
        gen_mri_dataset,
        matplotlib,
        mo,
        mri,
        np,
        nufft,
        nufft_adjoint,
        plt,
        rearrange,
        sp,
        spiral_2d,
        tgas_spi,
        torch,
    )


@app.cell
def __(gen_mri_dataset):
    im_size = (4, 4)
    num_coils = 8
    img, trj, _ = gen_mri_dataset(im_size, num_coils)
    return im_size, img, num_coils, trj


@app.cell
def __(mo):
    mo.md("# NUFFT Gradient Comparison")
    return


@app.cell
def __(TorchNUFFT, Tuple, im_size, img, np, nufft, rearrange, torch, trj):
    def to_tkbn(trj: np.ndarray, im_size: Tuple):
        """Input trj should be sigpy-style
        i.e. in [-N/2, N/2] and shape [..., K, D]
        """
        trj_tkbn = torch.from_numpy(trj)
        trj_tkbn = trj_tkbn / torch.tensor(im_size).float() * (2 * np.pi)
        trj_tkbn = rearrange(trj_tkbn, "... K D -> ... D K")
        return trj_tkbn

    def test_torch_linop(im_size: Tuple, img: np.ndarray, trj: np.ndarray):
        img = torch.from_numpy(img).requires_grad_(True)
        trj = rearrange(trj, "K R D -> R K D")
        trj_tkbn = to_tkbn(trj, im_size)
        F = TorchNUFFT(
            trj_tkbn,
            im_size,
            in_batch_shape=("C",),
            out_batch_shape=("R",),
        )
        Fimg = F(img)
        loss = torch.mean(torch.abs(Fimg) ** 2)
        loss.backward()
        return img.grad, loss

    def test_sigpy_linop(im_size: Tuple, img: np.ndarray, trj: np.ndarray):
        img = torch.from_numpy(img).requires_grad_(True)

        trj = rearrange(trj, "K R D -> R K D")
        trj = torch.from_numpy(trj)

        Fimg = nufft(img, trj, oversamp=1.25, width=4) * 2
        loss = torch.mean(torch.abs(Fimg) ** 2)
        loss.backward()
        return img.grad, loss

    torchgrad, torchloss = test_torch_linop(im_size, img, trj)
    sigpygrad, sigpyloss = test_sigpy_linop(im_size, img, trj)
    return (
        sigpygrad,
        sigpyloss,
        test_sigpy_linop,
        test_torch_linop,
        to_tkbn,
        torchgrad,
        torchloss,
    )


@app.cell
def __(np, plt, torchgrad, torchloss):
    plt.imshow(
        np.abs(torchgrad.detach().cpu().numpy()) / torchloss.detach().cpu().numpy()
    )
    plt.colorbar()
    plt.title("Gradient (tkbn)")
    return


@app.cell
def __(np, plt, sigpygrad, sigpyloss):
    plt.imshow(
        np.abs(sigpygrad.detach().cpu().numpy()) / sigpyloss.detach().cpu().numpy()
    )
    plt.colorbar()
    plt.title("Gradient (sigpy)")
    return


@app.cell
def __(mo):
    mo.md("# NUFFT Adjoint Comparison")
    return


@app.cell
def __(
    TorchNUFFT,
    Tuple,
    im_size,
    img,
    np,
    nufft,
    nufft_adjoint,
    rearrange,
    to_tkbn,
    torch,
    trj,
):
    def test_torch_adjoint_linop(im_size: Tuple, img: np.ndarray, trj: np.ndarray):
        img = torch.from_numpy(img)
        trj = rearrange(trj, "K R D -> R K D")
        trj_tkbn = to_tkbn(trj, im_size).float()
        F = TorchNUFFT(
            trj_tkbn,
            im_size,
            in_batch_shape=("C",),
            out_batch_shape=("R",),
        )
        Fimg = F(img).detach()
        Fimg = Fimg.requires_grad_(True)
        print("tkbn", Fimg.abs().max())
        img2 = F.H(Fimg)
        loss = torch.mean(torch.abs(img2) ** 2)
        loss.backward()
        return Fimg.grad, loss

    def test_sigpy_adjoint_linop(im_size: Tuple, img: np.ndarray, trj: np.ndarray):
        img = torch.from_numpy(img)

        trj = rearrange(trj, "K R D -> R K D")
        trj = torch.from_numpy(trj)

        Fimg = nufft(img, trj, oversamp=2.0, width=4).detach().requires_grad_(True)
        print("sigpy", Fimg.abs().max())
        img2 = nufft_adjoint(Fimg, trj, oshape=im_size, oversamp=1.25, width=4)
        loss = torch.mean(torch.abs(img2) ** 2)
        loss.backward()
        return Fimg.grad, loss

    torchadjgrad, torchadjloss = test_torch_adjoint_linop(im_size, img, trj)
    sigpyadjgrad, sigpyadjloss = test_sigpy_adjoint_linop(im_size, img, trj)
    return (
        sigpyadjgrad,
        sigpyadjloss,
        test_sigpy_adjoint_linop,
        test_torch_adjoint_linop,
        torchadjgrad,
        torchadjloss,
    )


@app.cell
def __(np, plt, torchadjgrad, torchadjloss):
    plt.plot(
        np.abs(torchadjgrad[0].detach().cpu().numpy())
        / torchadjloss.detach().cpu().numpy()
    )
    print(torchadjloss)
    plt.title("Gradient (tkbn)")
    return


@app.cell
def __(np, plt, sigpyadjgrad, sigpyadjloss):
    plt.plot(
        np.abs(sigpyadjgrad[0].detach().cpu().numpy())
        / sigpyadjloss.detach().cpu().numpy()
    )
    print(sigpyadjloss)
    plt.title("Gradient (sigpy)")
    return


@app.cell
def __(mo):
    mo.md(
        """
    # Gradcheck (adjoint only)

    This part doesn't really work for me. ¯\\\__(ツ)\__/¯

    Gradcheck is really slow for larger spirals too - beware!

    """
    )
    return


@app.cell
def __(
    TorchNUFFT,
    Tuple,
    np,
    nufft,
    nufft_adjoint,
    rearrange,
    to_tkbn,
    torch,
):
    from torch.autograd.gradcheck import gradcheck
    from functools import partial

    def gradcheck_torch_adjoint_linop(im_size: Tuple, img: np.ndarray, trj: np.ndarray):
        img = torch.from_numpy(img)
        trj = rearrange(trj, "K R D -> R K D")
        trj_tkbn = to_tkbn(trj, im_size).float().to("cuda")
        F = TorchNUFFT(
            trj_tkbn,
            im_size,
            in_batch_shape=("C",),
            out_batch_shape=("R",),
        ).to("cuda")
        img = img.to("cuda")
        Fimg = F(img).detach().requires_grad_(True)

        # breakpoint()
        def loss_fn(Fimg):
            img2 = F.H(Fimg)
            return torch.sum(torch.abs(img2) ** 2)

        result = gradcheck(loss_fn, Fimg)
        print(result)

    def gradcheck_sigpy_adjoint_linop(im_size: Tuple, img: np.ndarray, trj: np.ndarray):
        img = torch.from_numpy(img)

        trj = rearrange(trj, "K R D -> R K D")
        trj = torch.from_numpy(trj)

        Fimg = nufft(img, trj, oversamp=2.0, width=4).detach().requires_grad_(True)
        inputs = (Fimg.to("cuda"), trj.to("cuda"), im_size, 1.25, 4)
        nufft_adjoint_partial = partial(
            nufft_adjoint, coord=trj, oshape=im_size, oversamp=1.25, width=4
        )

        def loss_fn(Fimg):
            img2 = nufft_adjoint_partial(Fimg)
            return torch.sum(torch.abs(img2) ** 2)

        result = gradcheck(loss_fn, Fimg)
        print(result)

    return (
        gradcheck,
        gradcheck_sigpy_adjoint_linop,
        gradcheck_torch_adjoint_linop,
        partial,
    )


@app.cell
def __(gradcheck_torch_adjoint_linop, im_size, img, trj):
    gradcheck_torch_adjoint_linop(im_size, img, trj)
    return


@app.cell
def __(gradcheck_sigpy_adjoint_linop, im_size, img, trj):
    gradcheck_sigpy_adjoint_linop(im_size, img, trj)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
