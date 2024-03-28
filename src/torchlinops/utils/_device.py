import torch

__all__ = ["get_device", "ordinal"]


def get_device(device_idx: int = -1):
    return torch.device(f"cuda:{device_idx}" if device_idx >= 0 else "cpu")


def ordinal(device: torch.device):
    return torch.zeros(1, device=device).get_device()
