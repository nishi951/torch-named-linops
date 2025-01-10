from typing import Tuple, Any, Sequence

from ._nameddim import NamedDimension as ND

__all__ = ["fake_dims", "get_nd_shape", "N2K", "K2N"]


def get_nd_shape(im_size, kspace=False):
    if len(im_size) == 1:
        im_dim = ("Kx",) if kspace else ("Nx",)
    elif len(im_size) == 2:
        im_dim = ("Kx", "Ky") if kspace else ("Nx", "Ny")
    elif len(im_size) == 3:
        im_dim = ("Kx", "Ky", "Kz") if kspace else ("Nx", "Ny", "Nz")
    else:
        raise ValueError(f"Image size {im_size} - should have length 2 or 3")
    return im_dim


def fake_dims(letter: str, n: int) -> Tuple:
    """Helper function for generating fake dimension names"""
    return tuple(f"{letter}_{i}" for i in range(n))


def is_spatial_dim(d: ND):
    return "x" in d.name or "y" in d.name or "z" in d.name


def N2K(tup: Tuple[ND]):
    out = []
    for d in tup:
        if is_spatial_dim(d):
            # Flip 'N' to 'K'
            out.append(ND(d.name.replace("N", "K"), d.i))
        else:
            out.append(d)
    return tuple(out)


def K2N(tup: Tuple[ND]):
    out = []
    for d in tup:
        if is_spatial_dim(d):
            # Flip 'N' to 'K'
            out.append(ND(d.name.replace("K", "N"), d.i))
        else:
            out.append(d)
    return tuple(out)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
