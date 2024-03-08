import torch

__all__ = ['get_device']

def get_device(device_idx: int = -1):
    return torch.device(f'cuda:{device_idx}' if device_idx >= 0 else 'cpu')
