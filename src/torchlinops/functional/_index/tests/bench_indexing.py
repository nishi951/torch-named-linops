import torch

from torchlinops.utils.benchmark import benchmark_and_summarize
import torchlinops.functional as F


def main():
    device = torch.device("cuda:0")
    B = 15
    Nx, Ny, Nz = (220, 220, 220)
    grid_size = (Nx, Ny, Nz)
    # ishape = (B, *grid_size)
    oshape = (48, 500, 1600)

    def gen_data():
        vals = torch.randn(*oshape, device=device)
        idx = [torch.randint(0, N, oshape, device=device) for N in grid_size]
        return vals, idx

    # Define random test functions
    def index_index_put():
        vals, idx = gen_data()
        # idx = torch.stack(idx, dim=-1)
        # return multi_grid(vals, idx, final_size=grid_size)
        return F.index_adjoint(vals, idx, grid_size=grid_size)

    def index_interp0():
        vals, idx = gen_data()
        locs = torch.stack(idx, dim=-1).float()
        return F.grid(vals, locs, grid_size=grid_size, width=1.0, kernel="spline")

    def index_multi_grid():
        vals, idx = gen_data()
        idx = torch.stack(idx, dim=-1)
        return multi_grid(vals, idx, final_size=grid_size)

    benchmark_and_summarize(gen_data, name="gen_data", num_iters=100)
    benchmark_and_summarize(
        index_interp0, name="grid with kernel=spline and width=0", num_iters=100
    )
    benchmark_and_summarize(index_index_put, name="index_put_", num_iters=100)
    benchmark_and_summarize(index_multi_grid, name="multi_grid", num_iters=100)

    # Test for correctness
    breakpoint()
    vals, idx = gen_data()
    y1 = F.index_adjoint(vals, idx, grid_size=grid_size)
    locs = torch.stack(idx, dim=-1).float()
    y2 = F.grid(vals, locs, grid_size=grid_size, width=1.0, kernel="spline")
    idx = torch.stack(idx, dim=-1)
    y3 = multi_grid(vals, idx, final_size=grid_size)

    assert y1.allclose(y2)
    assert y2.allclose(y3)


def multi_grid(
    x: torch.Tensor, idx: torch.Tensor, final_size: tuple, raveled: bool = False
):
    """Grid values in x to im_size with indices given in idx
    x: [N... I...]
    idx: [I... ndims] or [I...] if raveled=True
    raveled: Whether the idx still needs to be raveled or not

    Returns:
    Tensor with shape [N... final_size]

    Notes:
    Adjoint of multi_index
    Might need nonnegative indices
    """
    if not raveled:
        assert (
            len(final_size) == idx.shape[-1]
        ), f"final_size should be of dimension {idx.shape[-1]}"
        idx = ravel(idx, final_size, dim=-1)
    ndims = len(idx.shape)
    assert (
        x.shape[-ndims:] == idx.shape
    ), f"x and idx should correspond in last {ndims} dimensions"
    x_flat = torch.flatten(x, start_dim=-ndims, end_dim=-1)  # [N... (I...)]
    idx_flat = torch.flatten(idx)

    batch_dims = x_flat.shape[:-1]
    y = torch.zeros(
        (*batch_dims, *final_size), dtype=x_flat.dtype, device=x_flat.device
    )
    y = y.reshape((*batch_dims, -1))
    y = y.index_add_(-1, idx_flat, x_flat)
    y = y.reshape(*batch_dims, *final_size)
    return y


def ravel(x: torch.Tensor, shape: tuple, dim: int):
    """
    x: torch.LongTensor, arbitrary shape,
    shape: Shape of the array that x indexes into
    dim: dimension of x that is the "indexing" dimension

    Returns:
    torch.LongTensor of same shape as x but with indexing dimension removed
    """
    out = 0
    shape_shifted = tuple(shape[1:]) + (1,)
    for s, s_next, i in zip(shape, shape_shifted, range(x.shape[dim])):
        out += torch.select(x, dim, i) % s
        out *= s_next
    return out


if __name__ == "__main__":
    main()
