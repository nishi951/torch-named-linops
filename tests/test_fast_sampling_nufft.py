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


# The parity that drives the fold is that of PADDED_SIZE (the FFT grid the
# sandwich shifts), not grid_size: grid_size parity only affects the shared Pad
# placement and is invisible to the index/phase identity. Both are swept here.
# oversamp * grid must be an exact integer (see make_spec docstring).
@pytest.mark.parametrize(
    "grid_size, oversamp",
    [
        ((16, 16, 24), 1.25),  # padded (20,20,30): all even -> parity sign
        ((12, 16, 24), 1.25),  # padded (15,20,30): one odd axis -> complex phase
        ((20, 12, 12), 1.25),  # padded (25,15,15): all odd axes
        ((15, 15, 15), 1.2),  # odd grid_size, even padded: guards shared-Pad path
    ],
    ids=["padded-even", "padded-mixed", "padded-odd", "grid-odd"],
)
def test_forward_and_adjoint_match_grid_parity(grid_size, oversamp):
    spec = make_spec(
        batch=(1,), grid_size=grid_size, locs_batch_size=(3, 5), oversamp=oversamp
    )
    common = dict(output_shape=("R", "K"), oversamp=spec["oversamp"])
    slow = SamplingNUFFT(spec["locs"].clone(), spec["grid_size"], **common)
    fast = FastSamplingNUFFT(spec["locs"].clone(), spec["grid_size"], **common)
    torch.manual_seed(0)
    x = torch.rand(*spec["batch"], *spec["grid_size"], dtype=torch.complex64) + 0.5j
    y = torch.rand(*spec["batch"], *spec["locs"].shape[:-1], dtype=torch.complex64)

    torch.testing.assert_close(fast(x), slow(x), rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(fast.H(y), slow.H(y), rtol=1e-3, atol=1e-4)


def test_phase_is_unit_modulus():
    _, phase = FastSamplingNUFFT.fold_centering(
        torch.randint(0, 8, (100, 3), dtype=torch.int64), (8, 10, 12)
    )
    # gather idx must stay in-bounds for Sampling's range validation
    torch.testing.assert_close(phase.abs(), torch.ones_like(phase.real))


from torchlinops.testing import BaseNamedLinopTests  # noqa: E402


class TestFastSamplingNUFFT(BaseNamedLinopTests):
    """Shared linop-behavior suite: adjoint consistency, normal, split, backprop."""

    equality_check = "approx"
    isclose_kwargs: dict = {"rtol": 1e-3}

    instances = ["even_padded_3d", "odd_padded_3d"]

    @pytest.fixture(scope="class", params=instances)
    @classmethod
    def linop_input_output(cls, request):
        spec = request.getfixturevalue(request.param)
        grid_size = spec["grid_size"]
        locs_batch = spec["locs"].shape[:-1]
        linop = FastSamplingNUFFT(
            spec["locs"].clone(),
            grid_size,
            output_shape=("R", "K"),
            oversamp=spec["oversamp"],
        )
        ishape = (*spec["batch"], *grid_size)
        oshape = (*spec["batch"], *locs_batch)
        x = 0.5 * torch.rand(ishape, dtype=torch.complex64) + 0.5
        y = 0.5 * torch.rand(oshape, dtype=torch.complex64) + 0.5
        return linop, x, y

    @pytest.fixture(scope="class")
    @classmethod
    def even_padded_3d(self):
        return make_spec()  # grid (16,16,24) -> padded (20,20,30)

    @pytest.fixture(scope="class")
    @classmethod
    def odd_padded_3d(self):
        # The odd axis lives in the PADDED grid: grid (12,16,24) -> padded (15,20,30)
        return make_spec(grid_size=(12, 16, 24))

    def test_size(self, linop_input_output):
        A, _, _ = linop_input_output
        assert A.size("R") == 3
        assert A.size("K") == 5

    def test_split_preserves_class_and_results(self, linop_input_output):
        A, x, _ = linop_input_output
        op = type(A).split(A, {})
        assert type(op) is type(A)
        assert torch.isclose(A(x), op(x), rtol=1e-5).all()


from torchlinops.linops.nufft.toeplitz import toeplitz_psf  # noqa: E402


def test_toeplitz_psf_raises_for_fast_sampling_nufft():
    spec = make_spec(batch=(1,), locs_batch_size=(4, 6))
    op = FastSamplingNUFFT(
        spec["locs"],
        spec["grid_size"],
        output_shape=("R", "K"),
        oversamp=spec["oversamp"],
    )
    with pytest.raises(NotImplementedError, match="SamplingNUFFT"):
        toeplitz_psf(op)
