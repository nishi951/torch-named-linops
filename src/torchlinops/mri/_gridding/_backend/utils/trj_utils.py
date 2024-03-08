from functools import partial

import numpy as np
from scipy.interpolate import CubicSpline


def oversample(trj, axis=0, factor=1.0, **interp_kwargs):
    """Oversample the trajectory"""
    if interp_kwargs is None:  # Defaults
        interp_kwargs = {
            "kind": "cubic",
            "fill_value": "extrapolate",
        }
    nro = trj.shape[axis]
    ro_orig = np.arange(nro)
    ro_new = np.arange(0, nro, 1 / factor)
    trj = CubicSpline(ro_orig, trj, axis=axis, **interp_kwargs)(ro_new)
    return trj


def readout_interp_1d(ksp_1d):
    """1D Linear interpolation with 0th-order extrapolation
    Returns a function that interpolates at new points fp

    ksp_1d: [K] array of readout points

    Notes:
    - This really makes the most sense when the readout is already very oversampled
    - In this regime, linear interpolation corresponds to sawtooth interpolation with
    a very narrow kernel, i.e. a very wide (and hopefully flat) filter
    """
    return partial(np.interp, xp=np.arange(len(ksp_1d)), x=ksp_1d)
