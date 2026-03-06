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
