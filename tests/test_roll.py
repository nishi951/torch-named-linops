import pytest
import torch
from torchlinops.utils._roll import roll, fftshift, ifftshift


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


@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0, 3)])
def test_roll_empty_dim_matches_torch(shape):
    # torch.roll returns an empty tensor of the same shape when a rolled
    # dimension has size 0; roll_fused must not divide by zero.
    x = torch.randn(*shape)
    dims = tuple(range(len(shape)))
    out = roll(x, (1,) * len(shape), dims)
    assert out.shape == x.shape
    assert out.numel() == 0
