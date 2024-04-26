from typing import Literal

from .backends import SigpyNUFFT, TorchNUFFT, FiNUFFT
from .backends.fi.convert_trj import sp2fi
from .grid import GriddedNUFFT
from .timeseg import timeseg
from .toeplitz import toeplitz

__all__ = [
    "NUFFT",
    "timeseg",
    "toeplitz",
]


def NUFFT(trj, im_size, backend: Literal["sigpy", "torch", "fi", "grid"] = "fi", *args, **kwargs):
    if backend == "sigpy":
        return SigpyNUFFT(trj, im_size, *args, **kwargs)
    elif backend == "torch":
        return TorchNUFFT(trj, im_size, *args, **kwargs)
    elif backend == "fi":
        trj.copy_(sp2fi(trj.clone(), im_size))
        return FiNUFFT(trj, im_size, *args, **kwargs)
    elif backend == "grid":
        return GriddedNUFFT(trj, im_size, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized NUFFT backend: {backend}")
