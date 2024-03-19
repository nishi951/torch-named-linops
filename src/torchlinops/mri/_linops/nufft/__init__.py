from typing import Literal

from .backends import SigpyNUFFT, TorchNUFFT, FiNUFFT

__all__ = [
    "NUFFT",
]


def NUFFT(*args, backend: Literal["sigpy", "torch", "fi"] = "fi", **kwargs):
    if backend == "sigpy":
        return SigpyNUFFT(*args, **kwargs)
    elif backend == "torch":
        return TorchNUFFT(*args, **kwargs)
    elif backend == "fi":
        return FiNUFFT(*args, **kwargs)
    else:
        raise ValueError(f"Unrecognized NUFFT backend: {backend}")
