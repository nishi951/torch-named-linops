from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchlinops._core._linops import NamedLinop
from torchlinops._core._shapes import get2dor3d
from . import functional as F


class SigpyNUFFT(NamedLinop):
    """NUFFT with Sigpy backend"""

    def __init__(
        self,
        trj: torch.Tensor,
        im_size: Tuple,
        in_batch_shape: Optional[Tuple] = None,
        out_batch_shape: Optional[Tuple] = None,
        readout_dim: str = "K",
        nufft_kwargs=None,
    ):
        """
        img (input) [A... [C] Nx Ny [Nz]]
        trj: [B... D K] in -pi to pi (tkbn-style)
        img_batch_shape: Extra dimensions in front of the image, not including spatial dims (e.g. subspace/trs)
        trj_batch_shape: Extra dimensions after the trajectory, not including coils (e.g. interleaves)
        """
        self.in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        self.out_batch_shape = (
            out_batch_shape if out_batch_shape is not None else tuple()
        )
        ishape = self.in_batch_shape + get2dor3d(im_size)
        oshape = self.in_batch_shape + self.out_batch_shape + (readout_dim,)
        super().__init__(ishape, oshape)
        self.trj = nn.Parameter(trj, requires_grad=False)
        self.im_size = im_size
        self.readout_dim = readout_dim

        # Precompute
        self.D = len(im_size)

        # Sigpy-specific
        self.nufft_kwargs = nufft_kwargs if nufft_kwargs is not None else {}

    def forward(self, x: torch.Tensor):
        return self.fn(x, self.trj)

    def fn(self, x, /, trj):
        """
        x: [A...  Nx Ny [Nz]] # A... may include coils
        trj: [B... K D] (sigpy-style)
        output: [A... B... K]
        """
        y = F.nufft(x, trj, **self.nufft_kwargs)
        return y

    def adj_fn(self, y, /, trj):
        """

        y: [A... B... K]
        trj: [B... K D], Sigpy-style
        output: [A... Nx Ny [Nz]]
        """

        B = trj.shape[:-2]
        A = tuple(y.shape[: -(len(B) + 1)])
        oshape = A + self.im_size
        x = F.nufft_adjoint(y, trj, oshape, **self.nufft_kwargs)
        return x

    def normal_fn(self, x, /, trj):
        return self.adj_fn(self.fn(x, trj), trj)

    def split_forward(self, ibatch, obatch):
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.trj),
            im_size=self.im_size,
            in_batch_shape=self.in_batch_shape,
            out_batch_shape=self.out_batch_shape,
            coil_dim=self.coil_dim,
            readout_dim=self.readout_dim,
            norm=self.norm,
            kbnufft_kwargs=self.kbnufft_kwargs,
        )

    def split_forward_fn(self, ibatch, obatch, trj):
        # if self.coil_dim is None:
        #     # obatch is [... K]
        #     trj_slc = obatch[:-1] + [slice(None)] + obatch[-1:]
        # else:
        #     # obatch is [... C K]
        #     trj_slc = obatch[:-2] + [slice(None)] + obatch[-1:]

        # Get slice corresponding to trj
        # B_slc = obatch[len(self.in_batch_shape) :]
        # Add a free dim for the D dimension
        trj_slc = obatch[:-1] + [slice(None)] + obatch[-1:]
        return trj[trj_slc]

    def size(self, dim: str):
        return self.size_fn(dim, self.trj)

    def size_fn(self, dim: str, trj):
        if dim == self.readout_dim:
            return trj.shape[-2]
        # elif dim == self.oshape[0]:
        #     return trj.shape[0]
        return None
