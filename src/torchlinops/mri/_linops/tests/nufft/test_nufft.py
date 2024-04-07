import pytest

import torch

from torchlinops.mri import NUFFT
from torchlinops.utils import same_storage


@pytest.fixture
def nufft():
    T, R, K = 3, 4, 5
    D = 2
    trj = 64 * (torch.rand((T, R, K, D)) - 0.5)
    F = NUFFT(
        trj,
        im_size=(64, 64),
        in_batch_shape=tuple(),
        out_batch_shape=("T", "R", "K"),
        toeplitz=False,
        backend="fi",
    )
    return F


def test_split(nufft):
    F = nufft

    ibatch = [slice(None), slice(None)]
    obatch = [slice(0, 1), slice(None), slice(None)]
    F_split = F.split(ibatch, obatch)

    assert tuple(F_split.trj.shape) == (1, 4, 5, 2)
    assert tuple(F.trj.shape) == (3, 4, 5, 2)
    assert (F.trj[0] == F_split.trj[0]).all()
    assert same_storage(F.trj, F_split.trj)
