from math import ceil
import torch


def get_valid_locs(locs_batch_size, grid_size, ndim, width, device):
    """Avoid circular padding weirdness by sampling valid locations only"""
    out = []
    for d in range(ndim):
        lower = ceil(width / 2)
        upper = grid_size[d] - 1 - lower
        locs = torch.rand(*locs_batch_size, device=device)
        locs = locs * (upper - lower) + lower
        out.append(locs)
    return torch.stack(out, dim=-1).contiguous()
