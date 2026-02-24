from typing import Tuple

from ._nameddim import NamedDimension as ND

__all__ = ["fake_dims", "get_nd_shape", "N2K", "K2N"]


def get_nd_shape(im_size, kspace=False):
    """Return spatial dimension names for a given image size.

    Maps a 1-D, 2-D, or 3-D image size tuple to the corresponding named
    dimension tuple (e.g. ``('Nx', 'Ny')`` or ``('Kx', 'Ky')``).

    Parameters
    ----------
    im_size : tuple
        Image size tuple whose length (1, 2, or 3) determines the spatial
        dimensionality.
    kspace : bool, optional
        If ``True``, return k-space dimension names (``Kx``, ``Ky``, …)
        instead of image-space names (``Nx``, ``Ny``, …).  Defaults to
        ``False``.

    Returns
    -------
    tuple of str
        Named dimension strings for each spatial axis.

    Raises
    ------
    ValueError
        If ``im_size`` does not have length 1, 2, or 3.
    """
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
    """Check whether a dimension represents a spatial axis.

    A dimension is considered spatial if its name contains ``'x'``,
    ``'y'``, or ``'z'``.

    Parameters
    ----------
    d : NamedDimension
        The dimension to test.

    Returns
    -------
    bool
        ``True`` if *d* is a spatial dimension.
    """
    return "x" in d.name or "y" in d.name or "z" in d.name


def N2K(tup: Tuple[ND]):
    """Convert image-space dimension names to k-space dimension names.

    For each spatial dimension in *tup*, replaces the ``'N'`` character in
    the name with ``'K'`` (e.g. ``Nx`` → ``Kx``).  Non-spatial dimensions
    are passed through unchanged.

    Parameters
    ----------
    tup : tuple of NamedDimension
        Dimension names to convert.

    Returns
    -------
    tuple of NamedDimension
        The converted dimension names.
    """
    out = []
    for d in tup:
        if is_spatial_dim(d):
            # Flip 'N' to 'K'
            out.append(ND(d.name.replace("N", "K"), d.i))
        else:
            out.append(d)
    return tuple(out)


def K2N(tup: Tuple[ND]):
    """Convert k-space dimension names to image-space dimension names.

    For each spatial dimension in *tup*, replaces the ``'K'`` character in
    the name with ``'N'`` (e.g. ``Kx`` → ``Nx``).  Non-spatial dimensions
    are passed through unchanged.

    Parameters
    ----------
    tup : tuple of NamedDimension
        Dimension names to convert.

    Returns
    -------
    tuple of NamedDimension
        The converted dimension names.
    """
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
