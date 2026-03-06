import torch

from torchlinops.utils._pad import end_pad_with_zeros


def test_end_pad_with_zeros():
    t = torch.tensor([1, 2, 3])
    result = end_pad_with_zeros(t, dim=0, pad_length=2)
    assert result.shape == (5,)
    assert result.tolist() == [1, 2, 3, 0, 0]


def test_end_pad_with_zeros_2d():
    t = torch.tensor([[1, 2], [3, 4]])
    result = end_pad_with_zeros(t, dim=0, pad_length=2)
    assert result.shape == (4, 2)
    assert result.tolist() == [[1, 2], [3, 4], [0, 0], [0, 0]]


def test_end_pad_with_zeros_dim1():
    t = torch.tensor([[1, 2], [3, 4]])
    result = end_pad_with_zeros(t, dim=1, pad_length=3)
    assert result.shape == (2, 5)
    assert result.tolist() == [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0]]


def test_end_pad_with_zeros_complex():
    t = torch.tensor([1 + 2j, 3 + 4j])
    result = end_pad_with_zeros(t, dim=0, pad_length=2)
    assert result.shape == (4,)
    assert result.dtype == torch.complex64


def test_end_pad_with_zeros_device():
    t = torch.tensor([1, 2, 3])
    result = end_pad_with_zeros(t, dim=0, pad_length=2)
    assert result.device == t.device
