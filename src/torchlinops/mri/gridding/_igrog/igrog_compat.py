from typing import Tuple, Optional

from einops import rearrange
import numpy as np

from igrog.readout_grog import grog_params
from igrog.grog_lib import igrog

__all__ = [
    'grog_params',
    'grogify',
]

def grogify(
    trj: np.ndarray,
    ksp: np.ndarray,
    ksp_cal: np.ndarray,
    im_size: Tuple,
    gparams: grog_params,
    device_idx: int = -1,
):
    """
    Parameters
    ----------
    trj: [..., nro, ndim] np.ndarray
    ksp: [ncoil, nro, ...] np.ndarray
    ksp_cal: [ncoil, *im_size] np.ndarray
    """

    trj = rearrange(trj, '... K D -> K ... D')
    ksp = rearrange(ksp, 'C ... K -> C K ...')

    igrg = igrog(im_size=im_size, gparams=gparams, device_idx=device_idx)
    trj_grd, ksp_grd = igrg.grogify(trj, ksp, ksp_cal)

    trj_grd = rearrange(trj_grd, 'K ... D -> ... K D')
    ksp_grd = rearrange(ksp_grd, 'C K ... -> C ... K')

    return trj_grd, ksp_grd, igrog

# TODO: Field object for field correction
