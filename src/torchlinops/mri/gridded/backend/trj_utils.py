import numpy as np
from scipy.interpolate import CubicSpline

def oversample(trj, axis=0, factor=1., **interp_kwargs):
    """Oversample the trajectory
    """
    if interp_kwargs is None: # Defaults
        interp_kwargs = {
            'kind': 'cubic',
            'fill_value': 'extrapolate',
        }
    nro = trj.shape[axis]
    ro_orig = np.arange(nro)
    ro_new = np.arange(0, nro, 1 / factor)
    trj = CubicSpline(ro_orig, trj, axis=axis, **interp_kwargs)(ro_new)
    return trj
