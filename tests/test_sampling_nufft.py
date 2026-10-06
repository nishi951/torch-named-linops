from math import prod

import pytest
import torch

from torchlinops import NUFFT, SamplingNUFFT
from torchlinops.linops.nufft.toeplitz import toeplitz_psf
from torchlinops.testing import BaseNamedLinopTests


def make_spec(batch=(2, 1), grid_size=(16, 16, 24), locs_batch_size=(3, 5),
              oversamp=1.25):
    """Integer-lattice centered locs: locs = (idx - padded_size//2) / oversamp.

    These locs round exactly onto the oversampled lattice, so
    SamplingNUFFT.prep_locs recovers the integer indices without wrap loss.
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


class TestSamplingNUFFT(BaseNamedLinopTests):
    """Shared linop-behavior suite for the sampling NUFFT backend."""

    equality_check = "approx"
    isclose_kwargs: dict = {"rtol": 1e-3}

    instances = ["small3d"]

    @pytest.fixture(scope="class", params=instances)
    def linop_input_output(self, request):
        spec = request.getfixturevalue(request.param)
        grid_size = spec["grid_size"]
        locs_batch = spec["locs"].shape[:-1]
        linop = SamplingNUFFT(
            spec["locs"].clone(),
            grid_size,
            output_shape=("R", "K"),
            oversamp=spec["oversamp"],
        )
        ishape = (*spec["batch"], *grid_size)
        oshape = (*spec["batch"], *locs_batch)
        x = 0.5 * torch.rand(ishape, dtype=torch.complex64) + 0.5
        y = 0.5 * torch.rand(oshape, dtype=torch.complex64) + 0.5
        linop._locs_orig = spec["locs"].clone()
        return linop, x, y

    @pytest.fixture(scope="class")
    def small3d(self):
        return make_spec()

    def test_size(self, linop_input_output):
        A, _, _ = linop_input_output
        assert A.size("R") == 3
        assert A.size("K") == 5

    def test_split_preserves_class_and_results(self, linop_input_output):
        A, x, _ = linop_input_output
        op = type(A).split(A, {})  # split is a staticmethod; NUFFTBase.split
        assert type(op) is type(A)
        assert torch.isclose(A(x), op(x), rtol=1e-5).all()


def test_sampling_prep_locs_round_and_wrap():
    grid, padded = (16, 16, 24), (20, 20, 30)
    locs = torch.tensor([[0.0, 0.0, 0.0], [-8.0, 8.0, 12.0]])
    wrapped = SamplingNUFFT.prep_locs(locs, grid, padded, pad_mode="circular")
    assert wrapped.dtype == torch.int64
    assert ((wrapped >= 0) & (wrapped < torch.tensor(padded))).all()
    expected = torch.remainder(
        torch.round(locs * torch.tensor(padded) / torch.tensor(grid)
                    + torch.tensor(padded) // 2),
        torch.tensor(padded),
    ).long()
    assert (wrapped == expected).all()


def test_sampling_prep_locs_zero_clamp():
    padded = (12, 12, 12)
    locs = torch.tensor([[-100.0, 0.0, 100.0], [5.9, 5.9, 5.9]])
    clamped = SamplingNUFFT.prep_locs(
        locs, (10, 10, 10), padded, pad_mode="zero"
    )
    assert clamped.dtype == torch.int64
    assert clamped[0].tolist() == [0, 6, 11]
    assert ((clamped >= 0) & (clamped <= 11)).all()


def test_sampling_prep_locs_bad_pad_mode():
    with pytest.raises(ValueError, match="padding mode"):
        SamplingNUFFT.prep_locs(
            torch.zeros(4, 3), (4, 4, 4), (5, 5, 5), pad_mode="octagonal"
        )


def test_nufft_class_split_rejects_mode():
    grid = (8, 8, 8)
    locs = torch.zeros(4, 3)
    with pytest.raises(ValueError, match="SamplingNUFFT"):
        NUFFT(locs, grid, output_shape=("K",), mode="sampling")
    with pytest.raises(ValueError, match="deprecated"):
        NUFFT(locs, grid, output_shape=("K",), mode="interpolate")


def test_nufft_class_split_identity():
    grid = (8, 8, 8)
    locs = torch.zeros(4, 3)
    assert type(NUFFT(locs, grid, output_shape=("K",))) is NUFFT
    assert type(SamplingNUFFT(locs, grid, output_shape=("K",))) is SamplingNUFFT


def test_toeplitz_psf_raises_for_sampling_nufft():
    spec = make_spec(batch=(1,), locs_batch_size=(4, 6))
    op = SamplingNUFFT(
        spec["locs"], spec["grid_size"], output_shape=("K",),
        oversamp=spec["oversamp"],
    )
    with pytest.raises(NotImplementedError, match="SamplingNUFFT"):
        toeplitz_psf(op)
