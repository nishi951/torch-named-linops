"""Tests for cache_nufft_parameters opt-in behavior and cache hygiene (issue #221)."""

import functools
import gc
import subprocess
import sys
import weakref

import pytest
import torch

import torchlinops.config as config
from torchlinops import FastSamplingNUFFT, NUFFT, SamplingNUFFT
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs
from torchlinops.linops.nufft import NUFFTBase, clear_nufft_parameter_caches

_GRID = (16, 16, 24)
_PADDED = tuple(int(1.25 * g) for g in _GRID)
_OPTS = dict(oversamp=1.25, width=4.0)


def _test_locs():
    return get_valid_locs((3, 5), _GRID, 3, 4.0, "cpu", centered=True)


def _lattice_locs(locs_batch=(3, 5)):
    """Integer-lattice centered locs so SamplingNUFFT.prep_locs rounds exactly."""
    idx = torch.stack([torch.randint(0, p, locs_batch) for p in _PADDED], dim=-1)
    return ((idx - torch.tensor(_PADDED) // 2).float() / 1.25).contiguous()


@pytest.fixture
def clean_caches():
    """Clear all NUFFT-family caches and pin the flag off, restoring after."""
    clear_nufft_parameter_caches()
    saved = config.cache_nufft_parameters
    config.cache_nufft_parameters = False
    yield
    config.cache_nufft_parameters = saved
    clear_nufft_parameter_caches()


@pytest.fixture
def flag_restored():
    saved = config.cache_nufft_parameters
    yield
    config.cache_nufft_parameters = saved


def test_default_flag_off_in_fresh_interpreter():
    """Subprocess so suite-order pollution (earlier tests set the flag) can't hide the module default."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torchlinops.config as c; print(c.cache_nufft_parameters)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_no_caching_by_default(clean_caches):
    prep_info = NUFFT.prep_locs.cache_info
    apod_info = NUFFT.apodize_weights.cache_info
    locs = _test_locs()
    p_before, a_before = prep_info().currsize, apod_info().currsize
    NUFFT(locs, _GRID, output_shape=("K",), **_OPTS)
    NUFFT(locs.clone(), _GRID, output_shape=("K",), **_OPTS)
    assert prep_info().currsize == p_before
    assert apod_info().currsize == a_before


def test_optin_populates_caches_and_exit_clears(clean_caches):
    prep_info = NUFFT.prep_locs.cache_info
    apod_info = NUFFT.apodize_weights.cache_info
    with config.using(cache_nufft_parameters=True):
        assert config.cache_nufft_parameters is True
        locs = _test_locs()
        p_before, a_before = prep_info().currsize, apod_info().currsize
        NUFFT(locs, _GRID, output_shape=("K",), **_OPTS)
        assert prep_info().currsize - p_before == 1
        NUFFT(locs, _GRID, output_shape=("K",), **_OPTS)
        # Same locs object -> dedup hit, no new entry
        assert prep_info().currsize - p_before == 1
        assert apod_info().currsize - a_before == 1
    assert config.cache_nufft_parameters is False
    assert prep_info().currsize == 0
    assert apod_info().currsize == 0


def test_optin_scope_releases_retained_input_tensors(clean_caches):
    with config.using(cache_nufft_parameters=True):
        locs = _test_locs()
        ref = weakref.ref(locs)
        NUFFT(locs, _GRID, output_shape=("K",), **_OPTS)
        del locs
        # While the cache is live, the key tuple retains the input tensor
        assert ref() is not None
    gc.collect()
    assert ref() is None


def test_optin_clears_on_exception(clean_caches):
    with pytest.raises(RuntimeError), config.using(cache_nufft_parameters=True):
        NUFFT(_test_locs(), _GRID, output_shape=("K",), **_OPTS)
        raise RuntimeError("boom")
    assert config.cache_nufft_parameters is False
    assert NUFFT.prep_locs.cache_info().currsize == 0
    assert NUFFT.apodize_weights.cache_info().currsize == 0


def test_turning_flag_off_in_nested_context_clears_at_outer_exit(clean_caches):
    with config.using(cache_nufft_parameters=True):
        NUFFT.prep_locs(_test_locs(), _GRID, _PADDED)
        NUFFT.apodize_weights(_GRID, _PADDED, 4.0, NUFFT.beta(4.0, 1.25))
        p_before = NUFFT.prep_locs.cache_info().currsize
        a_before = NUFFT.apodize_weights.cache_info().currsize
        with config.using(cache_nufft_parameters=False):
            pass
        # Returning to an opt-in (True) state keeps the cache entries
        assert NUFFT.prep_locs.cache_info().currsize == p_before
        assert NUFFT.apodize_weights.cache_info().currsize == a_before
    # Exiting back to the default-off world clears the caches
    assert config.cache_nufft_parameters is False
    assert NUFFT.prep_locs.cache_info().currsize == 0
    assert NUFFT.apodize_weights.cache_info().currsize == 0


def test_clear_walks_subclasses_and_new_cached_staticmethods(clean_caches):
    class Probe(NUFFTBase):
        @staticmethod
        @functools.lru_cache(maxsize=64)
        def probe(x):
            return x

    Probe.probe(_GRID)
    NUFFT.prep_locs(_test_locs(), _GRID, _PADDED)
    assert Probe.probe.cache_info().currsize == 1
    assert NUFFT.prep_locs.cache_info().currsize == 1
    clear_nufft_parameter_caches()
    clear_nufft_parameter_caches()
    assert Probe.probe.cache_info().currsize == 0
    assert NUFFT.prep_locs.cache_info().currsize == 0


def test_clear_nufft_parameter_caches_clears_all_families(clean_caches):
    NUFFT.prep_locs(_test_locs(), _GRID, _PADDED)
    NUFFT.apodize_weights(_GRID, _PADDED, 4.0, NUFFT.beta(4.0, 1.25))
    SamplingNUFFT.prep_locs(_lattice_locs(), _GRID, _PADDED)
    prepared = SamplingNUFFT.prep_locs.__wrapped__(_lattice_locs(), _GRID, _PADDED)
    FastSamplingNUFFT.fold_centering(prepared, _PADDED)
    assert NUFFT.prep_locs.cache_info().currsize == 1
    assert NUFFT.apodize_weights.cache_info().currsize == 1
    assert SamplingNUFFT.prep_locs.cache_info().currsize == 1
    assert FastSamplingNUFFT.fold_centering.cache_info().currsize == 1

    clear_nufft_parameter_caches()

    assert NUFFT.prep_locs.cache_info().currsize == 0
    assert NUFFT.apodize_weights.cache_info().currsize == 0
    assert SamplingNUFFT.prep_locs.cache_info().currsize == 0
    assert FastSamplingNUFFT.fold_centering.cache_info().currsize == 0
