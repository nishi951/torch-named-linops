from typing import Tuple, Optional

from einops import rearrange
import torch

from igrog.gridding import gridding_params
from igrog.training import training_params
from igrog.grogify import gridding_implicit_grogify, imperfection_implicit_grogify

from mr_recon.imperfections.imperfection import imperfection
from mr_recon.imperfections.main_field import main_field_imperfection

__all__ = [
    "gridding_params",
    "training_params",
    "b02imperf",
    "grogify",
]


def b02imperf(
    b0_map: torch.Tensor,
    trj_size: Tuple,
    ro_dim: int,
    dt: float,
    L: int,
    method: Optional[str] = "ts",
    interp_type: Optional[str] = "zero",
    verbose: Optional[bool] = False,
):
    imperf = main_field_imperfection(
        b0_map,
        trj_size=trj_size,
        ro_dim=ro_dim,
        dt=dt,
        L=L,
        method=method,
        interp_type=interp_type,
        verbose=verbose,
    )
    return imperf


def grogify(
    ksp: torch.Tensor,
    trj: torch.Tensor,
    img_cal: torch.Tensor,
    tparams: training_params,
    gparams: gridding_params,
    imperfection: Optional[imperfection] = None,
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

    if imperfection is not None:
        ksp_grd, trj_grd = imperfection_implicit_grogify(
            ksp, trj, img_cal, imperfection, train_params=tparams, grid_params=gparams
        )
    else:
        ksp_grd, trj_grd = gridding_implicit_grogify(
            ksp, trj, img_cal, train_params=tparams, grid_params=gparams
        )

    ksp_grd = rearrange(ksp_grd, "C K ... -> C ... K")
    trj_grd = rearrange(trj_grd, "K ... D -> ... K D")
    return ksp_grd, trj_grd
