from math import ceil
import torch


def get_valid_locs(
    locs_batch_size,
    grid_size,
    ndim,
    width,
    device,
    centered: bool = False,
    valid: bool = True,
):
    """Avoid circular padding weirdness by sampling valid locations only
    Allow non-valid locations by setting valid=False
    """
    out = []
    for d in range(ndim):
        if valid:
            lower = ceil(width / 2)
            upper = grid_size[d] - 1 - lower
        else:
            lower = 0
            upper = grid_size[d] - 1
        locs = torch.rand(*locs_batch_size, device=device)
        locs = locs * (upper - lower) + lower
        out.append(locs)
    out = torch.stack(out, dim=-1).contiguous()
    if centered:
        sz = torch.tensor(grid_size)
        out -= sz / 2
    return out
