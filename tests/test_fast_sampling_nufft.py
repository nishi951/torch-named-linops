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


from torchlinops import SamplingNUFFT  # noqa: E402


def make_spec(
    batch=(2, 1), grid_size=(16, 16, 24), locs_batch_size=(3, 5), oversamp=1.25
):
    """Integer-lattice centered locs (same helper as tests/test_sampling_nufft.py).

    Requires oversamp * grid_size to be an exact integer so prep_locs rounds
    back onto the lattice without loss.
    """
    padded_size = tuple(int(oversamp * g) for g in grid_size)
    idx = torch.stack(
        [torch.randint(0, p, locs_batch_size) for p in padded_size], dim=-1
    )
    locs = (idx - torch.tensor(padded_size) // 2).float() / oversamp
    return {
        "batch": batch,
        "grid_size": grid_size,
        "padded_size": padded_size,
        "locs": locs.contiguous(),
        "oversamp": oversamp,
    }


def test_forward_matches_sampling_nufft():
    spec = make_spec()
    common = dict(output_shape=("R", "K"), oversamp=spec["oversamp"])
    slow = SamplingNUFFT(spec["locs"].clone(), spec["grid_size"], **common)
    fast = FastSamplingNUFFT(spec["locs"].clone(), spec["grid_size"], **common)
    x = torch.rand(*spec["batch"], *spec["grid_size"], dtype=torch.complex64) + 0.5j
    y_fast = fast(x)
    y_slow = slow(x)
    assert y_fast.shape == y_slow.shape
    torch.testing.assert_close(y_fast, y_slow, rtol=1e-3, atol=1e-4)


def test_chain_contains_no_centered_fft():
    """The regression this class exists to fix: no shift sandwich in the chain."""
    spec = make_spec()
    fast = FastSamplingNUFFT(
        spec["locs"].clone(),
        spec["grid_size"],
        output_shape=("R", "K"),
        oversamp=spec["oversamp"],
    )
    assert fast.fft.centered is False
    assert not any(getattr(linop, "centered", False) for linop in fast.linops)
