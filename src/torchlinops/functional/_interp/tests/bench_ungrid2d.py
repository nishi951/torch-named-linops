from math import prod, ceil

import torch
import sigpy as sp

from torchlinops.utils.benchmark import benchmark_and_summarize
from torchlinops.utils._device import device_ordinal
from torchlinops.utils import to_pytorch, from_pytorch
from torchlinops.functional import ungrid

from torchlinops.functional._interp.tests._valid_pts import get_valid_locs


def main():
    device = torch.device("cuda")
    dtype = torch.complex64
    N = (2,)
    grid_size = (120, 120)
    npts = (600,)
    width = 2.0
    kernel = "kaiser_bessel"

    ishape = (*N, *grid_size)

    ndim = len(grid_size)
    grid_size = torch.tensor(grid_size, device=device)

    def gen_random_triton():
        vals = torch.arange(prod(ishape)).reshape(ishape).to(dtype).to(device)
        locs = get_valid_locs(npts, grid_size, ndim, width, device)
        return

    def ungrid_random_triton():
        vals = torch.arange(prod(ishape)).reshape(ishape).to(dtype).to(device)
        locs = get_valid_locs(npts, grid_size, ndim, width, device)
        out = ungrid(vals, locs, width, kernel)
        return out

    benchmark_and_summarize(
        gen_random_triton,
        num_iters=100,
        name="torch gen random",
        ignore_first=10,
    )

    res_triton, out_triton = benchmark_and_summarize(
        ungrid_random_triton,
        num_iters=100,
        name="triton",
        ignore_first=10,
    )

    dev = sp.Device(device_ordinal(device))

    def gen_random_sigpy():
        xp = dev.xp
        with dev:
            if dtype == torch.complex64:
                vals = xp.random.randn(*ishape, dtype=float) + 1j * xp.random.randn(
                    *ishape, dtype=float
                )
            else:
                vals = xp.random.randn(N, Nx, dtype=cp_dtype)
            locs = from_pytorch(get_valid_locs(npts, grid_size, ndim, width, device))

        return

    def ungrid_random_sigpy():
        xp = dev.xp
        with dev:
            if dtype == torch.complex64:
                vals = xp.random.randn(*ishape, dtype=float) + 1j * xp.random.randn(
                    *ishape, dtype=float
                )
            else:
                vals = xp.random.randn(N, Nx, dtype=cp_dtype)
            locs = from_pytorch(get_valid_locs(npts, grid_size, ndim, width, device))
            out = sp.interpolate(vals, locs, kernel=kernel, width=width)
        return out

    benchmark_and_summarize(
        gen_random_sigpy,
        num_iters=100,
        name="sigpy gen random",
        backend="cupy",
        ignore_first=10,
    )

    res_sigpy, out_sigpy = benchmark_and_summarize(
        ungrid_random_sigpy,
        num_iters=100,
        name="sigpy",
        backend="cupy",
        ignore_first=10,
    )


if __name__ == "__main__":
    main()
