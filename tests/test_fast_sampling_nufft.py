"""Tests for FastSamplingNUFFT (issue #207: shift-free sampling mode)."""

import pytest
import torch

from torchlinops import FastSamplingNUFFT
from torchlinops.utils import cfftn


@pytest.mark.parametrize(
    "padded",
    [
        (8, 10, 12),  # all even axes -> parity sign
        (7, 5, 9),  # all odd axes  -> full complex phase
        (6, 7, 8),  # mixed parity
    ],
)
def test_fold_centering_matches_sandwich(padded):
    """fold_centering must reproduce the exact centered-FFT sandwich gather."""
    torch.manual_seed(0)
    x = torch.randn(*padded, dtype=torch.complex64)
    centered = cfftn(x, dim=(-3, -2, -1), norm="ortho")  # the roll sandwich
    plain = torch.fft.fftn(x, dim=(-3, -2, -1), norm="ortho")

    locs_prepared = torch.stack(
        [torch.randint(0, n, (40,), dtype=torch.int64) for n in padded], dim=-1
    )

    idx, phase = FastSamplingNUFFT.fold_centering(locs_prepared, padded)
    assert idx.dtype == torch.int64
    assert phase.shape == locs_prepared.shape[:-1]
    assert phase.is_complex()

    gathered = plain[idx[..., 0], idx[..., 1], idx[..., 2]]
    got = gathered * phase
    ref = centered[locs_prepared[..., 0], locs_prepared[..., 1], locs_prepared[..., 2]]
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
