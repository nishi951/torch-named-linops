"""Linop-level tests for FFT centered_method ("shift" vs "modulate").

The utils-level tests (src/torchlinops/utils/tests/test_cfft.py) check cfftn/
cifftn directly; these check the FFT NamedLinop wiring: default method, forward
and adjoint agreement across parities, dtype, and error surfacing. The
modulate path is the default (FFT.__init__), so every centered FFT in the
library — including NUFFT's interp pipeline (nufft.py) — flows through it.
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


def _linop(method, shape):
    return FFT(ndim=len(shape), centered=True, centered_method=method)


@pytest.mark.parametrize("method", ["shift", "modulate"])
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_manual_sandwich(method, shape):
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(*shape, generator=g) + 1j * torch.randn(*shape, generator=g)).to(
        torch.complex64
    )
    manual = torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x), dim=(0, 1, 2), norm="ortho"),
        dim=(0, 1, 2),
    )
    y = _linop(method, shape)(x)
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
    assert torch.allclose(_linop("modulate", shape).H(y), manual, rtol=1e-5, atol=1e-5)
    assert torch.allclose(_linop("shift", shape).H(y), manual, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
def test_round_trip_identity(shape):
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(*shape, generator=g) + 1j * torch.randn(*shape, generator=g)).to(
        torch.complex64
    )
    for method in ("shift", "modulate"):
        F = _linop(method, shape)
        assert torch.allclose(F.H(F(x)), x, rtol=1e-4, atol=1e-4), method


def test_unknown_method_raises_through_linop():
    F = FFT(ndim=1, centered=True, centered_method="banana")
    x = torch.randn(5).to(torch.complex64)
    with pytest.raises(ValueError, match="method"):
        F(x)
