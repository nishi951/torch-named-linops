import marimo

__generated_with = "0.1.67"
app = marimo.App()


@app.cell
def __():
    from copy import copy, deepcopy
    from inspect import signature

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import sigpy as sp
    import sigpy.mri as mri
    from torchkbnufft import KbNufft, KbNufftAdjoint
    from einops import rearrange

    from torchlinops.core.base import Chain
    from torchlinops.core.linops import Diagonal, Repeat
    from torchlinops.core.tiling import Batch
    from torchlinops.mri.linops import NUFFT, SENSE
    return (
        Batch,
        Chain,
        Diagonal,
        KbNufft,
        KbNufftAdjoint,
        NUFFT,
        Repeat,
        SENSE,
        copy,
        deepcopy,
        mo,
        mri,
        nn,
        np,
        rearrange,
        signature,
        sp,
        torch,
    )


@app.cell
def __(Diagonal, torch):
    d = Diagonal(torch.randn(3, 3), ('a', 'b'), ('a','b'))
    d2 = d.split(ibatch=(slice(0, 2), slice(None)), obatch=(slice(0, 2), slice(None)))
    d2.weight.shape
    print(d.weight)
    print(d2.weight)
    print(d2.H.weight)


    return d, d2


@app.cell
def __(NUFFT, Repeat, SENSE, mri, np, rearrange, torch):
    # B, Nx, Ny, T, K, D, C all defined
    B = 5
    Nx = 64

    Ny = 64
    C = 12
    T = 10
    K = 100
    D = 2
    num_interleaves = 16
    trj = mri.spiral(
        fov=1,
        N=Nx,
        f_sampling=0.2,
        R=1,
        ninterleaves=num_interleaves,
        alpha=1.5,
        gm=40e-3,
        sm=100,
    )
    trj = rearrange(trj, '(r k) d -> r k d', r=num_interleaves)
    # print(trj.shape)

    x = torch.randn((Nx, Ny), dtype=torch.complex64)
    x_dims = ('B', 'Nx', 'Ny')
    # Convert sigpy trj to tkbn trj
    trj = torch.from_numpy(trj)
    trj = rearrange(trj, '... k d -> ... d k')
    trj = trj * 2 * np.pi

    mps = torch.randn((C, Nx, Ny), dtype=torch.complex64)
    F = NUFFT(trj, im_size=(Nx, Ny),
              img_batch_shape=('R', 'C'),
              trj_batch_shape=('R',),
             )
    S = SENSE(mps)
    R = Repeat(n_repeats=num_interleaves, dim=0,
               ishape=('C', 'Nx', 'Ny'), oshape=('R', 'C', 'Nx', 'Ny'))
    # BC = Broadcast(('B', 'Nx', 'Ny'), ('B', '1', 'Nx', 'Ny'))
    A = F @ R @ S

    print(A)
    # print(A.batchable_dims())
    # TODO
    # A = Batch(A, B=3, C=1, T=2)

    # Optional:
    # A = torch.compile(A)

    # Run
    y = A(x)
    print(y.shape)

    # Also should run!
    print(A.H)
    x2 = A.H(y)
    print(x2.shape)

    # Also also should run!!
    print(A.N)
    x3 = A.N(x)
    print(x3.shape)

    # You get the idea
    Sx = S.fn(x, mps)
    RSx = R.fn(Sx, num_interleaves)
    print(RSx.shape)
    FRSx = F.fn(RSx, trj)
    FRSx.shape
    print(FRSx.isclose(y).all())

    Ax = A.fn(x, [trj, num_interleaves, mps])
    print(FRSx.isclose(Ax).all())


    return (
        A,
        Ax,
        B,
        C,
        D,
        F,
        FRSx,
        K,
        Nx,
        Ny,
        R,
        RSx,
        S,
        Sx,
        T,
        mps,
        num_interleaves,
        trj,
        x,
        x2,
        x3,
        x_dims,
        y,
    )


@app.cell
def __(A):
    for dim in A.dims:
        print(dim, ':', A.size(dim))


    # Phi = Dense(phi)
    # D = Diagonal(dcf)
    # T = ImplicitGROGToepNUFFT(trj, inner=(Phi.H @ D @ Phi)
    # A = S.H @ T @ Sj

    return dim,


@app.cell
def __(A, Batch, mps, num_interleaves, torch, trj, x, y):
    # Batching
    Abatch = Batch(A, C=1, R=3)
    ybatch = Abatch(x)
    print(ybatch.isclose(y.to(torch.complex64)).all())
    # print(trj.shape)
    ybatch_fn = Abatch.fn(x, [trj, num_interleaves, mps])
    # print(ybatch_fn.shape)
    print(ybatch_fn.isclose(y.to(torch.complex64)).all())
    return Abatch, ybatch, ybatch_fn


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
