import torch

__all__ = ["get_device", "device_ordinal", "same_storage"]


def get_device(device_idx: int = -1):
    return torch.device(f"cuda:{device_idx}" if device_idx >= 0 else "cpu")


def device_ordinal(device: torch.device):
    return torch.zeros(1, device=device).get_device()


def same_storage(x, y):
    """Determine if tensors share the same storage or not"""
    x_ptrs = set(e.data_ptr() for e in x.view(-1))
    y_ptrs = set(e.data_ptr() for e in y.view(-1))
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)
