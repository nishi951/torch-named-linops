"""Linop-level tests for FFT centered method.

The utils-level tests (src/torchlinops/utils/tests/test_cfft.py) check cfftn/
cifftn directly; these check the FFT NamedLinop wiring: forward and adjoint
agreement across parities and dtype.
"""

import pytest
import torch

from torchlinops import FFT

SHAPES = [
    (8, 8, 8),  # all even, N % 4 == 0
    (6, 6, 6),  # all even, N % 4 == 2 (fold constant = -1 per axis)
    (5, 7, 9),  # all odd
    (400, 400, 150),  # production-like mixed parity
]


@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_manual_sandwich(shape):
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(*shape, generator=g) + 1j * torch.randn(*shape, generator=g)).to(
        torch.complex64
    )
    manual = torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x), dim=(0, 1, 2), norm="ortho"),
        dim=(0, 1, 2),
    )
    F = FFT(ndim=len(shape), centered=True)
    y = F(x)
    assert y.dtype == torch.complex64
    assert torch.allclose(y, manual, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
def test_adjoint_matches_manual_sandwich(shape):
    g = torch.Generator().manual_seed(1)
    y = (torch.randn(*shape, generator=g) + 1j * torch.randn(*shape, generator=g)).to(
        torch.complex64
    )
    manual = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(y), dim=(0, 1, 2), norm="ortho"),
        dim=(0, 1, 2),
    )
    F = FFT(ndim=len(shape), centered=True)
    assert torch.allclose(F.H(y), manual, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
def test_round_trip_identity(shape):
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(*shape, generator=g) + 1j * torch.randn(*shape, generator=g)).to(
        torch.complex64
    )
    F = FFT(ndim=len(shape), centered=True)
    assert torch.allclose(F.H(F(x)), x, rtol=1e-4, atol=1e-4)
