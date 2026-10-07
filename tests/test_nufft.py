from math import prod

import numpy as np
import pytest
import sigpy as sp
import torch

from torchlinops import NUFFT, Interpolate, Pad
from torchlinops.functional import nufft, nufft_adjoint
from torchlinops.functional._interp.tests._valid_pts import get_valid_locs
from torchlinops.testing import BaseNamedLinopTests


class TestNUFFT(BaseNamedLinopTests):
    equality_check = "approx"

    oversamp = [1.0, 1.25]

    instances = ["small3d"]

    # Unstable numerical behavior
    isclose_kwargs: dict = {"rtol": 1e-3}

    @pytest.fixture(scope="class", params=instances)
    def linop_input_output(self, request):
        spec = request.param
        spec = request.getfixturevalue(spec)
        width = spec["width"]
        oversamp = spec["oversamp"]
        grid_size = spec["grid_size"]
        locs_batch_size = spec["locs_batch_size"]
        ndim = len(grid_size)
        npts = prod(locs_batch_size)
        batch_size = spec["N"]
        ishape = (*batch_size, *grid_size)
        oshape = (*batch_size, *locs_batch_size)
        locs = get_valid_locs(
            locs_batch_size, grid_size, ndim, width, "cpu", centered=True
        )

        linop = NUFFT(
            locs.clone(),
            grid_size,
            output_shape=("R", "K"),
            width=width,
            oversamp=oversamp,
        )
        # Limit randomness
        x = 0.5 * torch.rand(ishape, dtype=torch.complex64, device="cpu") + 1
        y = 0.5 * torch.rand(oshape, dtype=torch.complex64, device="cpu") + 1
        y /= torch.linalg.vector_norm(locs, dim=-1)

        # Save original locs
        linop._locs_orig = locs

        return linop, x, y

    @pytest.fixture(scope="class")
    def small3d(self, request):
        N = (2, 1)
        # grid_size = (16, 16, 24)
        grid_size = (32, 32, 32)
        locs_batch_size = (3, 5)
        width = 4.0
        oversamp = 1.25

        spec = {
            "N": N,
            "grid_size": grid_size,
            "locs_batch_size": locs_batch_size,
            "width": width,
            "oversamp": oversamp,
        }
        return spec

    def test_size(self, linop_input_output):
        A, x, y = linop_input_output
        assert A.size("R") == 3
        assert A.size("K") == 5

    def test_normal_fn(self, linop_input_output):
        A, x, y = linop_input_output
        ANx = A.N(x)
        normal_fn_result = A.normal_fn(A, x.clone())
        assert torch.isclose(ANx, normal_fn_result, rtol=1e-2).all()

    def test_split(self, linop_input_output):
        pytest.skip("NUFFT split not fully supported")

    def test_nufft_sigpy(self, linop_input_output):
        A, x, y = linop_input_output
        coord = A._locs_orig.numpy()  # Not usually a param, only here for testing
        # sz = np.array(A.grid_size)
        # coord = np.where(coord <= (sz / 2), coord, coord - sz)
        width = A.options["width"]
        oversamp = A.options["oversamp"]

        Ax = A(x).numpy()
        Ax_sp = sp.nufft(x.numpy(), coord, oversamp=oversamp, width=width)
        assert np.allclose(Ax, Ax_sp, **self.isclose_kwargs)

        AHy = A.H(y).numpy()
        AHy_sp = sp.nufft_adjoint(
            y.numpy(), coord, x.shape, oversamp=oversamp, width=width
        )
        assert np.allclose(AHy, AHy_sp, **self.isclose_kwargs)

    def test_nufft_sigpy_functional(self, linop_input_output):
        A, x, y = linop_input_output
        coord = A._locs_orig.numpy()  # Not usually a param, only here for testing
        # sz = np.array(A.grid_size)
        # coord = np.where(coord <= (sz / 2), coord, coord - sz)
        width = A.options["width"]
        oversamp = A.options["oversamp"]

        Ax = nufft(x, A._locs_orig, oversamp, width).numpy()
        Ax_sp = sp.nufft(x.numpy(), coord, oversamp=oversamp, width=width)
        assert np.allclose(Ax, Ax_sp, **self.isclose_kwargs)

        ndim = A._locs_orig.shape[-1]
        grid_size = x.shape[-ndim:]
        AHy = nufft_adjoint(y, A._locs_orig, grid_size, oversamp, width).numpy()
        AHy_sp = sp.nufft_adjoint(
            y.numpy(), coord, x.shape, oversamp=oversamp, width=width
        )
        assert np.allclose(AHy, AHy_sp, **self.isclose_kwargs)


@pytest.fixture
def nufft_params():
    width = 6.0
    oversamp = 2.0
    grid_size = (120, 119, 146)
    padded_size = tuple(int(i * oversamp) for i in grid_size)
    locs = get_valid_locs((10,), grid_size, len(grid_size), width, "cpu", centered=True)
    return {
        "width": width,
        "oversamp": oversamp,
        "grid_size": grid_size,
        "padded_size": padded_size,
        "locs": locs,
    }


def test_apodize(nufft_params):
    width = nufft_params["width"]
    oversamp = nufft_params["oversamp"]
    grid_size = nufft_params["grid_size"]
    padded_size = nufft_params["padded_size"]

    beta = NUFFT.beta(width, oversamp)
    apod = NUFFT.apodize_weights(grid_size, padded_size, width, beta)

    x = np.ones(grid_size)
    apod_sp = sp.fourier._apodize(x, len(grid_size), oversamp, width, beta)
    assert np.allclose(apod, apod_sp)


def test_nufft_os_pad(nufft_params):
    grid_size = nufft_params["grid_size"]
    padded_size = nufft_params["padded_size"]
    pad = Pad(padded_size, grid_size)

    x = torch.randn(*grid_size)
    padx = pad(x).numpy()

    padx_sp = sp.util.resize(x.numpy(), padded_size)

    assert np.allclose(padx, padx_sp)


def test_scale_locs(nufft_params):
    grid_size = nufft_params["grid_size"]
    padded_size = nufft_params["padded_size"]
    oversamp = nufft_params["oversamp"]

    # Torch version
    locs = nufft_params["locs"]
    locs_scaled = NUFFT.prep_locs(locs.clone(), grid_size, padded_size)

    coord = locs.clone().numpy()
    sz = np.array(grid_size)
    coord_scaled = sp.fourier._scale_coord(coord, grid_size, oversamp)
    assert np.allclose(locs_scaled, coord_scaled)


def test_nufft_interp(nufft_params):
    locs = nufft_params["locs"]
    grid_size = nufft_params["grid_size"]
    padded_size = nufft_params["padded_size"]
    width = nufft_params["width"]
    oversamp = nufft_params["oversamp"]
    beta = NUFFT.beta(width, oversamp)

    locs_prepared = NUFFT.prep_locs(locs.clone(), grid_size, padded_size)
    interp = Interpolate(
        locs_prepared,
        padded_size,
        batch_shape=None,
        locs_batch_shape=None,
        grid_shape=None,
        width=width,
        kernel="kaiser_bessel",
        kernel_params=dict(beta=beta),
    )

    x = torch.randn(*padded_size, dtype=torch.complex64)
    interpx = interp(x)

    interpx_sp = sp.interp.interpolate(
        x.numpy(),
        locs_prepared.numpy(),
        kernel="kaiser_bessel",
        width=width,
        param=beta,
    )

    assert np.allclose(interpx, interpx_sp, rtol=1e-3)


def test_prep_locs_zero_mode(nufft_params):
    """prep_locs with pad_mode='zero' should clamp locs to [0, padded_size-1]."""
    grid_size = nufft_params["grid_size"]
    padded_size = nufft_params["padded_size"]
    locs = nufft_params["locs"].clone()
    result = NUFFT.prep_locs(locs, grid_size, padded_size, pad_mode="zero")
    for i, ps in enumerate(padded_size):
        assert result[..., -(len(padded_size) - i)].min() >= 0
        assert result[..., -(len(padded_size) - i)].max() <= ps - 1


def test_prep_locs_invalid_mode_raises(nufft_params):
    """prep_locs with an unknown pad_mode should raise ValueError."""
    grid_size = nufft_params["grid_size"]
    padded_size = nufft_params["padded_size"]
    locs = nufft_params["locs"].clone()
    with pytest.raises(ValueError, match="Unrecognized padding mode"):
        NUFFT.prep_locs(locs, grid_size, padded_size, pad_mode="invalid_mode")


# def test_nufft_unknown_mode_raises(nufft_params):
#     """NUFFT with an unrecognised mode should raise ValueError at construction."""
#     locs = nufft_params["locs"].clone()
#     grid_size = nufft_params["grid_size"]
#     with pytest.raises(ValueError, match="Unrecognized NUFFT mode"):
#         NUFFT(locs, grid_size, output_shape=("K",), mode="bad_mode")


# def test_nufft_nan_apodize_weights_raises(nufft_params):
#     """NUFFT should raise ValueError when apodize_weights contains NaN."""
#     locs = nufft_params["locs"].clone()
#     grid_size = nufft_params["grid_size"]
#     bad_weights = torch.full(grid_size, float("nan"))
#     with pytest.raises(ValueError, match="Nan/Inf"):
#         NUFFT(locs, grid_size, output_shape=("K",), apodize_weights=bad_weights)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_nufft_device(nufft_params):
    locs = nufft_params["locs"]
    grid_size = nufft_params["grid_size"]
    width = nufft_params["width"]
    oversamp = nufft_params["oversamp"]
    linop = NUFFT(
        locs.clone(),
        grid_size,
        output_shape=("K",),
        width=width,
        oversamp=oversamp,
    )
    assert linop.device.type == "cpu"
    linop.to(torch.device("cuda"))
    assert linop.device.type == "cuda"


# ---------------- class-split & caching machinery ----------------


def _cache_test_locs():
    from torchlinops.functional._interp.tests._valid_pts import get_valid_locs

    return get_valid_locs((3, 5), (16, 16, 24), 3, 4.0, "cpu", centered=True)


_CACHE_GRID = (16, 16, 24)
_CACHE_OPTS = {"oversamp": 1.25, "width": 4.0}


def test_nufft_mode_kwarg_rejected():
    with pytest.raises(ValueError, match="SamplingNUFFT"):
        NUFFT(_cache_test_locs(), _CACHE_GRID, output_shape=("K",), mode="sampling")
    with pytest.raises(ValueError, match="deprecated"):
        NUFFT(_cache_test_locs(), _CACHE_GRID, output_shape=("K",), mode="interpolate")


def test_nufft_class_identity():
    linop = NUFFT(_cache_test_locs(), _CACHE_GRID, output_shape=("K",), **_CACHE_OPTS)
    assert type(linop) is NUFFT


def _enable_and_clear_caches():
    import torchlinops.config as config

    config.cache_nufft_parameters = True
    NUFFT.prep_locs.cache_clear()
    NUFFT.apodize_weights.cache_clear()


def test_locs_cache_shared_by_object_identity():
    _enable_and_clear_caches()
    info = NUFFT.prep_locs.cache_info
    locs = _cache_test_locs()  # fresh object every call
    before = info().currsize
    n1 = NUFFT(locs, _CACHE_GRID, output_shape=("K",), **_CACHE_OPTS)
    assert info().currsize - before == 1  # fresh object: miss
    n2 = NUFFT(locs, _CACHE_GRID, output_shape=("K",), **_CACHE_OPTS)
    assert info().currsize - before == 1  # same object: hit, no new entry
    assert n1.interp.locs.data_ptr() == n2.interp.locs.data_ptr()
    NUFFT(_cache_test_locs(), _CACHE_GRID, output_shape=("K",), **_CACHE_OPTS)
    assert info().currsize - before == 2  # another fresh object: miss


def test_apod_cache_keyed_by_geometry():
    _enable_and_clear_caches()
    g = _CACHE_GRID
    p125 = tuple(int(1.25 * x) for x in g)
    p15 = tuple(int(1.5 * x) for x in g)
    w = NUFFT.apodize_weights
    b125 = NUFFT.beta(4.0, 1.25)
    b15 = NUFFT.beta(4.0, 1.5)
    w0 = w(g, p125, 4.0, b125)
    assert w.cache_info().misses == 1
    assert w(g, p125, 4.0, b125) is w0  # same geometry: hit, shared tensor
    assert w.cache_info().hits == 1
    assert w(g, p15, 4.0, b15) is not w0  # different oversamp: new key
    assert w.cache_info().currsize == 2


def test_skip_prep_locs_bypasses_cache():
    _enable_and_clear_caches()
    padded = tuple(int(1.25 * g) for g in _CACHE_GRID)
    prepared = NUFFT.prep_locs.__wrapped__(_cache_test_locs(), _CACHE_GRID, padded)
    before = NUFFT.prep_locs.cache_info().currsize
    NUFFT(
        prepared, _CACHE_GRID, output_shape=("K",), skip_prep_locs=True, **_CACHE_OPTS
    )
    assert NUFFT.prep_locs.cache_info().currsize == before


def test_results_independent_of_cache_state():
    _enable_and_clear_caches()
    locs = _cache_test_locs()
    op = NUFFT(locs.clone(), _CACHE_GRID, output_shape=("K",), **_CACHE_OPTS)
    x = torch.randn(2, *_CACHE_GRID, dtype=torch.complex64)
    y1 = op(x).clone()
    NUFFT.prep_locs.cache_clear()
    NUFFT.apodize_weights.cache_clear()
    op2 = NUFFT(locs, _CACHE_GRID, output_shape=("K",), **_CACHE_OPTS)
    assert torch.allclose(y1, op2(x), rtol=1e-5)


def test_caches_bypassed_when_flag_off():
    import torchlinops.config as config

    _enable_and_clear_caches()
    config.cache_nufft_parameters = False
    try:
        locs = _cache_test_locs()
        for _ in range(3):
            NUFFT(locs, _CACHE_GRID, output_shape=("K",), **_CACHE_OPTS)
        assert NUFFT.prep_locs.cache_info().currsize == 0
        assert NUFFT.apodize_weights.cache_info().currsize == 0
    finally:
        config.cache_nufft_parameters = True


def test_options_override_defaults():
    linop = NUFFT(
        _cache_test_locs(), _CACHE_GRID, output_shape=("K",), oversamp=1.5, width=3.0
    )
    assert linop.options["oversamp"] == 1.5
    assert linop.options["width"] == 3.0
    assert linop.options["toeplitz"] is False  # untouched default survives
