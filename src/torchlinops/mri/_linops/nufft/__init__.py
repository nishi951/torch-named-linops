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


def NUFFT(*args, backend: Literal["sigpy", "torch", "fi", "grid"] = "fi", **kwargs):
    if backend == "sigpy":
        return SigpyNUFFT(*args, **kwargs)
    elif backend == "torch":
        return TorchNUFFT(*args, **kwargs)
    elif backend == "fi":
        trj = kwargs.get("trj")
        im_size = kwargs.get("im_size")
        if trj is None:
            trj = args[0]
        if im_size is None:
            im_size = args[1]
        trj.copy_(sp2fi(trj.clone(), im_size))
        return FiNUFFT(*args, **kwargs)
    elif backend == "grid":
        return GriddedNUFFT(*args, **kwargs)
    else:
        raise ValueError(f"Unrecognized NUFFT backend: {backend}")
