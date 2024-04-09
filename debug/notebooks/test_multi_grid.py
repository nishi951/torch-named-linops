import marimo

__generated_with = "0.2.4"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    from typing import Tuple
    from math import prod
    import copy

    from einops import rearrange
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    import sigpy as sp

    from torchlinops.mri.gridding.indexing import multi_grid

    # def ravel(x: torch.Tensor, shape: Tuple, dim: int):
    #     """
    #     x: torch.LongTensor, arbitrary shape,
    #     shape: Shape of the array that x indexes into
    #     dim: dimension of x that is the "indexing" dimension

    #     Returns:
    #     torch.LongTensor of same shape as x but with indexing dimension removed
    #     """
    #     out = 0
    #     for s, i in zip(shape, range(x.shape[dim]-1)):
    #         out = s * (out + torch.remainder(torch.select(x, dim, i), s))
    #     out += torch.remainder(torch.select(x, dim, -1), shape[-1])
    #     # return torch.remainder(out, prod(shape))
    #     return out

    # def multi_grid(x: torch.Tensor, idx: torch.Tensor, final_size: Tuple, raveled: bool = False):
    #     """Grid values in x to im_size with indices given in idx
    #     x: [N... I...]
    #     idx: [I... ndims] or [I...] if raveled=True
    #     raveled: Whether the idx still needs to be raveled or not

    #     Returns:
    #     Tensor with shape [N... final_size]

    #     Notes:
    #     Adjoint of multi_index
    #     Might need nonnegative indices
    #     """
    #     if not raveled:
    #         assert len(final_size) == idx.shape[-1], f'final_size should be of dimension {idx.shape[-1]}'
    #         idx = ravel(idx, final_size, dim=-1)
    #     ndims = len(idx.shape)
    #     assert x.shape[-ndims:] == idx.shape, f'x and idx should correspond in last {ndims} dimensions'
    #     x_flat = torch.flatten(x, start_dim=-ndims, end_dim=-1) # [N... (I...)]
    #     idx_flat = torch.flatten(idx)

    #     batch_dims = x_flat.shape[:-1]
    #     y = torch.zeros((*batch_dims, *final_size), dtype=x_flat.dtype, device=x_flat.device)
    #     y = y.reshape((*batch_dims, -1))
    #     y = y.index_add_(-1, idx_flat, x_flat)
    #     y = y.reshape(*batch_dims, *final_size)
    #     return y
    return Tuple, copy, mo, multi_grid, np, plt, prod, rearrange, sp, torch


@app.cell
def __(np, rearrange):
    # img = sp.shepp_logan((64, 64))
    img = np.ones((64, 64))
    # mask = np.random.randint(0, 64, size=(32, 32 ,2))
    all_inds = rearrange(
        np.stack(np.meshgrid(range(64), range(64), indexing="ij")), "d x y -> (x y) d"
    )
    inds_idx = np.random.choice(64**2, size=32**2, replace=False)
    inds = all_inds[inds_idx]
    mask = inds.reshape(32, 32, 2) - 32
    return all_inds, img, inds, inds_idx, mask


@app.cell
def __(img, np, plt):
    plt.imshow(np.abs(img), vmin=0, vmax=1)
    return


@app.cell
def __(copy, img, mask, np, plt):
    masked = copy.deepcopy(img)
    masked[mask[..., 0], mask[..., 1]] = 0.0
    plt.imshow(np.abs(masked))
    return (masked,)


@app.cell
def __(mask, multi_grid, np, plt, torch):
    gridded_vals = np.ones(mask.shape[:-1])
    # filled = copy.deepcopy(masked)
    filled = (
        multi_grid(
            torch.from_numpy(gridded_vals),
            torch.from_numpy(mask),
            final_size=(64, 64),
        )
        .detach()
        .cpu()
        .numpy()
    )
    plt.imshow(np.abs(filled))
    return filled, gridded_vals


@app.cell
def __(filled, masked, np, plt):
    plt.imshow(np.abs(filled) + np.abs(masked), vmin=0, vmax=1)

    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
