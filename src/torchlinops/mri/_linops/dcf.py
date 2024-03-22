from typing import Tuple

import torch
import sigpy as sp
import sigpy.mri as mri

from torchlinops._core._linops import Diagonal

__all__ = ["DCF"]


def DCF(
    trj: torch.Tensor,
    im_size: Tuple,
    ioshape: Tuple,
    max_iter: int = 30,
    show_pbar: bool = True,
):
    dcf = mri.pipe_menon_dcf(
        sp.from_pytorch(trj),
        img_shape=im_size,
        max_iter=max_iter,
        show_pbar=show_pbar,
    )
    dcf = sp.to_pytorch(dcf, requires_grad=False)
    dcf /= torch.max(dcf)
    return Diagonal(dcf, ioshape)
