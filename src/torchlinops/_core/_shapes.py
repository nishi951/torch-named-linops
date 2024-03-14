from typing import Tuple

__all__ = [
    "fake_dims",
    "get2dor3d",
]


def get2dor3d(im_size, kspace=False):
    if len(im_size) == 2:
        im_dim = ("Kx", "Ky") if kspace else ("Nx", "Ny")
    elif len(im_size) == 3:
        im_dim = ("Kx", "Ky", "Kz") if kspace else ("Nx", "Ny", "Nz")
    else:
        raise ValueError(f"Image size {im_size} - should have length 2 or 3")
    return im_dim


def fake_dims(letter: str, n: int) -> Tuple:
    """Helper function for generating fake dimension names"""
    return tuple(f"{letter}_{i}" for i in range(n))
