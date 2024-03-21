import torch

__all__ = ["end_pad_with_zeros"]


def end_pad_with_zeros(t, dim, pad_length):
    """Extend a tensor with a specified number of zeros along a dimension"""
    pad_shape = list(t.shape)
    pad_shape[dim] = pad_length
    pad = torch.zeros(*tuple(pad_shape), dtype=t.dtype, device=t.device)
    t_pad = torch.concatenate((t, pad), dim=dim)
    return t_pad
