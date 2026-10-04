import pytest
import torch
from torchlinops.functional import roll, fftshift, ifftshift


def test_roll_fused():
    x = torch.randn(3, 5, 8)

    assert (torch.roll(x, (1, 1, 1), (0, 1, 2)) == roll(x, (1, 1, 1), (0, 1, 2))).all()
    assert (
        torch.roll(x, (-1, -1, -9), (0, 1, 2)) == roll(x, (-1, -1, -9), (0, 1, 2))
    ).all()


def test_fftshift_fused():
    x = torch.randn(3, 5, 8)
    assert (torch.fft.fftshift(x, (0, 1, 2)) == fftshift(x, (0, 1, 2))).all()
    assert (torch.fft.ifftshift(x, (0, 1, 2)) == ifftshift(x, (0, 1, 2))).all()
