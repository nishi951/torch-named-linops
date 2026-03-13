import pytest
import torch

from torchlinops.functional import center_crop, center_pad
from torchlinops.utils import inner


@pytest.mark.parametrize("dtype", ["real", "complex"])
def test_center_pad_1d(dtype):
    if dtype == "complex":
        x = torch.randn(10, dtype=torch.complex64)
    else:
        x = torch.randn(10)
    im_size = (10,)
    pad_im_size = (20,)

    y = center_pad(x, im_size, pad_im_size)

    assert y.shape == (20,)


@pytest.mark.parametrize("dtype", ["real", "complex"])
def test_center_pad_2d(dtype):
    if dtype == "complex":
        x = torch.randn(10, 15, dtype=torch.complex64)
    else:
        x = torch.randn(10, 15)
    im_size = (10, 15)
    pad_im_size = (20, 30)

    y = center_pad(x, im_size, pad_im_size)

    assert y.shape == (20, 30)


@pytest.mark.parametrize("dtype", ["real", "complex"])
def test_center_pad_3d(dtype):
    if dtype == "complex":
        x = torch.randn(8, 10, 12, dtype=torch.complex64)
    else:
        x = torch.randn(8, 10, 12)
    im_size = (8, 10, 12)
    pad_im_size = (16, 20, 24)

    y = center_pad(x, im_size, pad_im_size)

    assert y.shape == (16, 20, 24)


@pytest.mark.parametrize("dtype", ["real", "complex"])
def test_center_pad_nd(dtype):
    if dtype == "complex":
        x = torch.randn(5, 10, 15, 20, dtype=torch.complex64)
    else:
        x = torch.randn(5, 10, 15, 20)
    im_size = (5, 10, 15, 20)
    pad_im_size = (10, 20, 30, 40)

    y = center_pad(x, im_size, pad_im_size)

    assert y.shape == (10, 20, 30, 40)


def test_center_pad_odd_input_even_output():
    x = torch.randn(11, 11)
    im_size = (11, 11)
    pad_im_size = (20, 20)

    y = center_pad(x, im_size, pad_im_size)

    assert y.shape == (20, 20)


def test_center_pad_even_input_odd_output():
    x = torch.randn(10, 10)
    im_size = (10, 10)
    pad_im_size = (21, 21)

    y = center_pad(x, im_size, pad_im_size)

    assert y.shape == (21, 21)


def test_center_pad_odd_input_odd_output():
    x = torch.randn(11, 11)
    im_size = (11, 11)
    pad_im_size = (21, 21)

    y = center_pad(x, im_size, pad_im_size)

    assert y.shape == (21, 21)


def test_center_pad_even_input_even_output():
    x = torch.randn(10, 10)
    im_size = (10, 10)
    pad_im_size = (20, 20)

    y = center_pad(x, im_size, pad_im_size)

    assert y.shape == (20, 20)


def test_center_pad_preserves_center():
    x = torch.arange(9).reshape(3, 3).float()
    im_size = (3, 3)
    pad_im_size = (6, 6)

    y = center_pad(x, im_size, pad_im_size)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0, 4.0, 5.0, 0.0],
            [0.0, 0.0, 6.0, 7.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(y, expected)


@pytest.mark.parametrize("dtype", ["real", "complex"])
def test_center_crop_1d(dtype):
    if dtype == "complex":
        x = torch.randn(20, dtype=torch.complex64)
    else:
        x = torch.randn(20)
    im_size = (20,)
    crop_im_size = (10,)

    y = center_crop(x, im_size, crop_im_size)

    assert y.shape == (10,)


@pytest.mark.parametrize("dtype", ["real", "complex"])
def test_center_crop_2d(dtype):
    if dtype == "complex":
        x = torch.randn(20, 30, dtype=torch.complex64)
    else:
        x = torch.randn(20, 30)
    im_size = (20, 30)
    crop_im_size = (10, 15)

    y = center_crop(x, im_size, crop_im_size)

    assert y.shape == (10, 15)


@pytest.mark.parametrize("dtype", ["real", "complex"])
def test_center_crop_3d(dtype):
    if dtype == "complex":
        x = torch.randn(16, 20, 24, dtype=torch.complex64)
    else:
        x = torch.randn(16, 20, 24)
    im_size = (16, 20, 24)
    crop_im_size = (8, 10, 12)

    y = center_crop(x, im_size, crop_im_size)

    assert y.shape == (8, 10, 12)


@pytest.mark.parametrize("dtype", ["real", "complex"])
def test_center_crop_nd(dtype):
    if dtype == "complex":
        x = torch.randn(10, 20, 30, 40, dtype=torch.complex64)
    else:
        x = torch.randn(10, 20, 30, 40)
    im_size = (10, 20, 30, 40)
    crop_im_size = (5, 10, 15, 20)

    y = center_crop(x, im_size, crop_im_size)

    assert y.shape == (5, 10, 15, 20)


def test_center_crop_odd_input_even_output():
    x = torch.randn(21, 21)
    im_size = (21, 21)
    crop_im_size = (10, 10)

    y = center_crop(x, im_size, crop_im_size)

    assert y.shape == (10, 10)


def test_center_crop_even_input_odd_output():
    x = torch.randn(20, 20)
    im_size = (20, 20)
    crop_im_size = (11, 11)

    y = center_crop(x, im_size, crop_im_size)

    assert y.shape == (11, 11)


def test_center_crop_odd_input_odd_output():
    x = torch.randn(21, 21)
    im_size = (21, 21)
    crop_im_size = (11, 11)

    y = center_crop(x, im_size, crop_im_size)

    assert y.shape == (11, 11)


def test_center_crop_even_input_even_output():
    x = torch.randn(20, 20)
    im_size = (20, 20)
    crop_im_size = (10, 10)

    y = center_crop(x, im_size, crop_im_size)

    assert y.shape == (10, 10)


def test_center_crop_extracts_center():
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0, 4.0, 5.0, 0.0],
            [0.0, 0.0, 6.0, 7.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    im_size = (6, 6)
    crop_im_size = (3, 3)

    y = center_crop(x, im_size, crop_im_size)

    expected = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ]
    )
    assert torch.allclose(y, expected)


@pytest.mark.parametrize("dtype", ["real", "complex"])
def test_center_pad_adjoint(dtype):
    if dtype == "complex":
        x = torch.randn(10, 15, dtype=torch.complex64)
        y = torch.randn(20, 30, dtype=torch.complex64)
    else:
        x = torch.randn(10, 15)
        y = torch.randn(20, 30)

    im_size = (10, 15)
    pad_im_size = (20, 30)

    pad_x = center_pad(x, im_size, pad_im_size)
    crop_y = center_crop(y, pad_im_size, im_size)

    assert torch.allclose(inner(pad_x, y), inner(x, crop_y), rtol=1e-3)
