import torch
import pytest
from torchlinops.linops.nufft.utils import fftshift_locs, ifftshift_locs


def test_fftshift_locs_inverse():
    size = (5, 8, 11)
    locs = torch.stack(tuple(torch.randint(s, size) for s in size), dim=-1)
    out1 = ifftshift_locs(fftshift_locs(locs, size), size)
    out2 = fftshift_locs(ifftshift_locs(locs, size), size)
    assert (locs == out1).all()
    assert (locs == out2).all()
