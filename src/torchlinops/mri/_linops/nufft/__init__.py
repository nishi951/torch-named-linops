from typing import Literal

from .backends import SigpyNUFFT, TorchNUFFT, FiNUFFT
from .grid import GriddedNUFFT
from .timeseg import timeseg
from .toeplitz import toeplitz

__all__ = [
    "NUFFT",
    "timeseg",
    "toeplitz",
]


def NUFFT(*args, backend: Literal["sigpy", "torch", "fi", "grid"] = "fi", **kwargs):
    if backend == "sigpy":
        return SigpyNUFFT(*args, **kwargs)
    elif backend == "torch":
        return TorchNUFFT(*args, **kwargs)
    elif backend == "fi":
        return FiNUFFT(*args, **kwargs)
    elif backend == "grid":
        return GriddedNUFFT(*args, **kwargs)
    else:
        raise ValueError(f"Unrecognized NUFFT backend: {backend}")
