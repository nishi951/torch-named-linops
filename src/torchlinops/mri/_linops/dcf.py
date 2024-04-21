from typing import Tuple, Optional

import torch
import sigpy as sp
import sigpy.mri as mri

from torchlinops._core._linops import Diagonal
from torchlinops.utils import ordinal

__all__ = ["DCF"]


def DCF(
    trj: torch.Tensor,
    im_size: Tuple,
    ioshape: Optional[Tuple] = None,
    max_iter: int = 30,
    show_pbar: bool = True,
    weight_only: bool = False,
):
    """Create a Diagonal linop representing the application of
    the full DCF to ksp data

    Note that many iterative recons will actually require D ** (1/2),
    where D is the linop produced by this function.

    Can pass weight_only to return the raw weight instead of the full linop
    - ioshape is not required if this is the case
    - False by default for backward compatibility

    """
    ioshape = ioshape if ioshape is not None else tuple()
    dcf = mri.pipe_menon_dcf(
        sp.from_pytorch(trj),
        img_shape=im_size,
        max_iter=max_iter,
        device=sp.Device(ordinal(trj.device)),
        show_pbar=show_pbar,
    )
    dcf = sp.to_pytorch(dcf, requires_grad=False)
    dcf /= torch.linalg.norm(dcf)
    if weight_only:
        return dcf
    broadcast_dims = ioshape[: -len(dcf.shape)]
    return Diagonal(dcf, ioshape, broadcast_dims)
