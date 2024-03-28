from typing import Tuple

from einops import rearrange
import torch

from igrog.gridding import gridding_params
from igrog.training import training_params
from igrog.grogify import gridding_implicit_grogify

__all__ = [
    "gridding_params",
    "training_params",
    "grogify",
]


def grogify(
    ksp: torch.Tensor,
    trj: torch.Tensor,
    img_cal: torch.Tensor,
    tparams: training_params,
    gparams: gridding_params,
):
    """
    Parameters
    ----------
    trj: [..., nro, ndim] torch.Tensor
    ksp: [ncoil, nro, ...] torch.Tensor
    img_cal: [ncoil, *im_size] torch.Tensor
        Sensitivity maps or image-domain calibration region
    """

    ksp = rearrange(ksp, "C ... K -> C K ...")
    trj = rearrange(trj, "... K D -> K ... D")

    ksp_grd, trj_grd = gridding_implicit_grogify(
        ksp, trj, img_cal, train_params=tparams, grid_params=gparams
    )

    ksp_grd = rearrange(ksp_grd, "C K ... -> C ... K")
    trj_grd = rearrange(trj_grd, "K ... D -> ... K D")
    return ksp_grd, trj_grd


# TODO: Field object for field correction
