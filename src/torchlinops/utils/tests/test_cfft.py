"""Tests for centered FFT utility functions."""

import torch
import pytest

from torchlinops.utils._fft import cfft, cifft, cfft2, cifft2, cfftn, cifftn


def test_cfftn_round_trip_1d():
    """cifftn(cfftn(x)) should recover x."""
    x = torch.randn(32, dtype=torch.complex64)
    assert torch.allclose(cifftn(cfftn(x)), x, atol=1e-5)


def test_cfftn_round_trip_2d():
    """2-D round-trip."""
    x = torch.randn(16, 20, dtype=torch.complex64)
    assert torch.allclose(cifftn(cfftn(x)), x, atol=1e-5)


def test_cfftn_preserves_shape():
    x = torch.randn(8, 12, dtype=torch.complex64)
    assert cfftn(x).shape == x.shape


def test_cfftn_with_explicit_dim():
    x = torch.randn(4, 8, 16, dtype=torch.complex64)
    y = cfftn(x, dim=(-1,))
    assert y.shape == x.shape
    assert torch.allclose(cifftn(y, dim=(-1,)), x, atol=1e-5)


def test_cfftn_differs_from_plain_fftn():
    """Centered FFT should differ from uncentered FFT on a non-symmetric signal."""
    import torch.fft as fft

    x = torch.randn(16, dtype=torch.complex64)
    centered = cfftn(x)
    plain = fft.fftn(x)
    assert not torch.allclose(centered, plain)


def test_cfft_is_cfftn_last_dim():
    """cfft should match cfftn(x, dim=(-1,))."""
    x = torch.randn(4, 8, dtype=torch.complex64)
    assert torch.allclose(cfft(x), cfftn(x, dim=(-1,)), atol=1e-6)


def test_cifft_round_trip():
    x = torch.randn(16, dtype=torch.complex64)
    assert torch.allclose(cifft(cfft(x)), x, atol=1e-5)


def test_cfft2_is_cfftn_last_two_dims():
    """cfft2 should match cfftn(x, dim=(-2,-1))."""
    x = torch.randn(4, 8, 10, dtype=torch.complex64)
    assert torch.allclose(cfft2(x), cfftn(x, dim=(-2, -1)), atol=1e-6)


def test_cifft2_round_trip():
    x = torch.randn(8, 10, dtype=torch.complex64)
    assert torch.allclose(cifft2(cfft2(x)), x, atol=1e-5)


def test_cfftn_modulate():
    x = torch.randn(8, 5, 1)
    y_shift = cfftn(x, method="shift")
    y_modulate = cfftn(x, method="modulate")
    assert torch.allclose(y_shift, y_modulate, atol=1e-6)


def test_cifftn_modulate():
    x = torch.randn(8, 5, 1)
    y_shift = cifftn(x, method="shift")
    y_modulate = cifftn(x, method="modulate")
    assert torch.allclose(y_shift, y_modulate, atol=1e-6)


# The (8, 5, 1) fixture above misses two classes of failure that need explicit
# coverage: even axes with N % 4 == 2 (the fold constant is ( -1 ) ** ( N // 2 ),
# i.e. -1 for e.g. N=6/150), and negative / subset dims (as used by FFT.linop).


@pytest.mark.parametrize("inv", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (8, 5, 1),  # odd axis, degenerate N=1 axis
        (16, 6, 7),  # even axis with N % 4 == 2 (constant = -1)
        (300, 100, 75),  # production-like mixed parity
        (5,),  # single odd axis
        (6,),  # single N % 4 == 2 axis
    ],
)
def test_modulate_matches_shift_on_last_dims(shape, inv):
    g = torch.Generator().manual_seed(42)
    x = (torch.randn(*shape, generator=g) + 1j * torch.randn(*shape, generator=g)).to(
        torch.complex64
    )
    fn = cifftn if inv else cfftn
    y_shift = fn(x, dim=(-1,), method="shift")
    y_mod = fn(x, dim=(-1,), method="modulate")
    assert y_mod.dtype == y_shift.dtype == torch.complex64
    assert y_mod.shape == y_shift.shape
    # c64 cuFFT rounding is data-position dependent, so NOT bitwise equal
    # (see dce-cones #9 diagnosis); 1e-5 is ~30x the complex64 eps.
    assert torch.allclose(y_shift, y_mod, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("inv", [False, True])
def test_modulate_dim_subset_and_negative_dims(inv):
    x = torch.randn(4, 7, 6, 5).to(torch.complex64)
    fn = cifftn if inv else cfftn
    for dim in [(-4, -2), (0, 2), (-3, -1)]:
        y_shift = fn(x, dim=dim, method="shift")
        y_mod = fn(x, dim=dim, method="modulate")
        assert torch.allclose(y_shift, y_mod, rtol=1e-5, atol=1e-5), dim


@pytest.mark.parametrize("inv", [False, True])
@pytest.mark.parametrize("method", ["nonsense", "", "shiftish"])
def test_bad_method_raises(inv, method):
    with pytest.raises(ValueError, match="method"):
        (cifftn if inv else cfftn)(torch.randn(4, 4), method=method)


@pytest.mark.parametrize("inv", [False, True])
def test_modulate_preserves_dtype(inv):
    fn = cifftn if inv else cfftn
    for src, want in [
        (torch.float32, torch.complex64),
        (torch.complex64, torch.complex64),
        (torch.complex128, torch.complex128),
    ]:
        x = torch.randn(8, 6, 5).to(src)
        y_shift = fn(x, method="shift")
        y_mod = fn(x, method="modulate")
        assert y_mod.dtype == want
        assert y_shift.dtype == want
        assert torch.allclose(y_shift, y_mod, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("inv", [False, True])
def test_modulate_round_trip_matches_shift_round_trip(inv):
    # cifftn(cfftn(.)) must be identity for BOTH methods, incl. odd sizes
    # where the round-trip silently cancels a wrong fold constant and so
    # cannot be the only check.
    x = torch.randn(7, 6, 5).to(torch.complex64)
    forward, inverse = (cfftn, cifftn) if not inv else (cifftn, cfftn)
    for method in ("shift", "modulate"):
        y = inverse(forward(x, method=method), method=method)
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-4), method
