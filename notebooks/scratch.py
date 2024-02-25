import marimo

__generated_with = "0.2.4"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    from einops import rearrange
    return mo, np, rearrange


@app.cell
def __(np, rearrange):
    cal_size = (32, 64, 64)
    coords = tuple((np.arange(im_size) - im_size // 2) for im_size in cal_size)
    xyz = np.meshgrid(*coords)
    xyz = np.stack(xyz, axis=-1)
    xyz = rearrange(xyz, '... d -> (...) d')
    i = np.random.randint(xyz.shape[0], size=3)
    print(xyz[i])
    return cal_size, coords, i, xyz


@app.cell
def __():
    import torch
    from math import prod
    # from torchlinops.mri.igrog.indexing import ravel
    def ravel(x, shape, dim):
        out = 0
        for s, i in zip(shape[1:], range(x.shape[dim]-1)):
            out = s * (out + torch.select(x, dim, i))
            # out = s * (out + torch.remainder(torch.select(x, dim, i), s))
        out += torch.select(x, dim, -1)
        # out += torch.remainder(torch.select(x, dim, -1), shape[-1])
        return torch.remainder(out, prod(shape))
    return prod, ravel, torch


@app.cell
def __(ravel, torch):
    a = torch.stack(torch.meshgrid(torch.arange(5), torch.arange(10), indexing='ij'), dim=-1)
    print(a.shape)
    a[..., 0] -= 2
    # a[..., 1] -= 5
    # print(a)
    print(a[..., 0])
    print(torch.remainder(a[..., 0], 5))
    print(torch.remainder(a[..., 1], 10))

    b = torch.randn((5, 10))
    idx = ravel(a, (5, 10), dim=-1)
    print(idx)
    return a, b, idx


@app.cell
def __():
    return


@app.cell
def __(b):
    print(b)
    print(b.flatten())
    return


if __name__ == "__main__":
    app.run()
