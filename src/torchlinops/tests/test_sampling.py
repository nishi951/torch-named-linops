import pytest

import torch

import torchlinops.functional as F
from torchlinops import Sampling
from torchlinops.tests.test_base import BaseNamedLinopTests


class TestSampling(BaseNamedLinopTests):
    equality_check = "approx"

    isclose_kwargs = dict(rtol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        N = 64
        ndim = 2
        R, K = 13, 17
        B = 3  # Batch
        idx = torch.randint(0, N - 1, (R, K, ndim))
        x = torch.randn(B, N, N)
        y = torch.randn(B, R, K)
        linop = Sampling.from_stacked_idx(idx, (N, N), ("R", "K"))
        return linop, x, y


def test_sampling_slc():
    N = 64
    ndim = 2
    R, K = 13, 17
    B = 3  # Batch
    idx = torch.randint(0, N - 1, (R, K, ndim))
    x = torch.randn(B, N, N)
    # y = torch.randn(B, R, K)

    linop = Sampling.from_stacked_idx(idx, (N, N), ("R", "K"))

    ibatch = [slice(None)] * 3
    obatch = [slice(None), slice(2, 5), slice(4, 10)]
    linop_split = linop.split(linop, ibatch, obatch)
    Ax = linop(x)
    Ax_split = linop_split(x)
    assert (Ax[:, 2:5, 4:10] == Ax_split).all()
