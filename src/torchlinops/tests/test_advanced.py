import pytest

import torch

from torchlinops import Diagonal, NUFFT, Concat
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs


def test_sense_mri():
    # Define shapes
    grid_size = (128, 128, 128)
    Nx, Ny, Nz = grid_size
    R, K = (6, 28)
    C = 11

    ishape = (Nx, Ny, Nz)
    oshape = (C, R, K)
    x = torch.randn(*ishape)
    S_weight = torch.randn(C, Nx, Ny, Nz)
    S = Diagonal.from_weight(
        S_weight,
        weight_shape=("C", "Nx", "Ny", "Nz"),
        ioshape=("C", "Nx", "Ny", "Nz"),
    )
    locs = get_valid_locs(
        (R, K),
        grid_size,
        ndim=len(grid_size),
        width=4.0,
        device="cpu",
        centered=True,
    )
    F = NUFFT(locs, grid_size, output_shape=("R", "K"))

    A = F @ S
    b = A(x)


def test_subspace_mri(): ...


def test_subspace_timeseg_mri(): ...
