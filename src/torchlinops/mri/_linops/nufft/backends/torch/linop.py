"""TODO: Fix this and make it workable"""

from math import prod
from typing import Optional, Tuple

from einops import rearrange

import torch
from torchkbnufft import KbNufft, KbNufftAdjoint

from torchlinops._core._linops import NamedLinop
from torchlinops._core._shapes import get2dor3d

__all__ = ["TorchNUFFT"]


class TorchNUFFT(NamedLinop):
    """Deprecated - not recommended"""

    def __init__(
        self,
        trj: torch.Tensor,
        im_size: Tuple,
        in_batch_shape: Optional[Tuple] = None,
        out_batch_shape: Optional[Tuple] = None,
        readout_dim: str = "K",
        norm="ortho",
        kbnufft_kwargs=None,
    ):
        """
        img (input) [A... [C] Nx Ny [Nz]]
        trj: [B... D K] in -pi to pi (tkbn-style)
        img_batch_shape: Extra dimensions in front of the image, not including spatial dims (e.g. subspace/trs)
        trj_batch_shape: Extra dimensions after the trajectory, not including coils (e.g. interleaves)
        """
        # self.batch_shape = img_batch_shape if img_batch_shape is not None else tuple()
        self.in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        self.out_batch_shape = (
            out_batch_shape if out_batch_shape is not None else tuple()
        )
        # if coil_dim is not None:
        #     ishape = self.in_batch_shape + (coil_dim,) + get2dor3d(im_size) # [A... C Nx Ny [Nz]]
        #     oshape = self.out_batch_shape + (coil_dim, readout_dim) # [R... C K]
        # else:
        #     ishape = self.in_batch_shape + get2dor3d(im_size)
        #     oshape = self.out_batch_shape + (readout_dim,)
        ishape = self.in_batch_shape + get2dor3d(im_size)
        oshape = self.in_batch_shape + self.out_batch_shape + (readout_dim,)
        super().__init__(ishape, oshape)
        self.trj = trj
        self.im_size = im_size
        # self.coil_dim = coil_dim
        self.readout_dim = readout_dim

        # expected_out_dim = (len(trj.shape)-2) + (1 if self.coil_dim else 0) + 1 # (K,)
        # assert len(self.oshape) == expected_out_dim, f'Output shape {self.oshape} does not match expected output dimension {expected_out_dim}'

        # Precompute
        self.D = len(im_size)

        # KbNufft-specific
        self.norm = norm
        self.kbnufft_kwargs = kbnufft_kwargs if kbnufft_kwargs is not None else {}
        self.nufft = KbNufft(im_size, **self.kbnufft_kwargs)
        self.nufft_adj = KbNufftAdjoint(im_size, **self.kbnufft_kwargs)
        self.oversamp_factor = 2.0  # Related to grid_size

    def forward(self, x: torch.Tensor):
        return self.fn(x, self.trj)

    def fn(self, x, /, trj):
        """
        x: [A...  Nx Ny [Nz]] # A... may include coils
        trj: [B... D K]
        output: [A... B... K]
        Note:
        - tkbn doesn't support batching over multiple non-trajectory dims, so we have to do this manually

        """
        spatial_dim = x.shape[-self.D :]  # doesn't May include coils
        A = x.shape[: -self.D]
        x = torch.reshape(x, (-1, *spatial_dim))  # [(A...) Nx Ny [Nz]]

        B = trj.shape[:-2]
        if len(B) > 0:
            trj = rearrange(trj, "... D K -> (...) D K")  # [(B...) D K]
        K = trj.shape[-1]

        y = torch.zeros(
            (prod(A), prod(B), K), dtype=x.dtype, layout=x.layout, device=x.device
        )
        for a, x_a in enumerate(x):
            # Add fake coil dim
            x_a = x_a[None, None, ...].expand(
                (prod(B), 1) + (-1,) * self.D
            )  # [(B...) 1 Nx Ny [Nz]]
            y_a = self.nufft(x_a, trj, norm=self.norm)
            y[a] = y_a[..., 0, :]  # Remove coil dim

        y = y.reshape((*A, *B, K)) * self.oversamp_factor
        return y

    def adj_fn(self, y, /, trj):
        """
        y: [A... B... K]
        trj: [B... D K]
        output: [A... Nx Ny [Nz]]
        """
        B = trj.shape[:-2]
        if len(B) > 0:
            trj = rearrange(trj, "... D K -> (...) D K")  # [(B...) D K]
        K = trj.shape[-1]

        A = y.shape[: -(len(B) + 1)]
        y = y.reshape((prod(A), prod(B), K))

        x = torch.zeros(
            (prod(A), *self.nufft_adj.im_size),
            dtype=y.dtype,
            layout=y.layout,
            device=y.device,
        )
        for a, y_a in enumerate(y):
            # Fake coil dim
            y_a = y_a[:, None, ...]  # [(B...) 1 K]
            x_a = self.nufft_adj(y_a, trj, norm=self.norm)
            x[a] = torch.sum(x_a, dim=(0, 1))  # Sum over batch and fake coil
        x = x.reshape((*A, *self.nufft_adj.im_size)) * self.oversamp_factor
        return x

    def normal_fn(self, x, /, trj):
        return self.adj_fn(self.fn(x, trj), trj)

    def split_forward(self, ibatch, obatch):
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.trj),
            im_size=self.im_size,
            in_batch_shape=self.in_batch_shape,
            out_batch_shape=self.out_batch_shape,
            readout_dim=self.readout_dim,
            norm=self.norm,
            kbnufft_kwargs=self.kbnufft_kwargs,
        )

    def split_forward_fn(self, ibatch, obatch, trj):
        # Get slice corresponding to trj
        B_slc = obatch[len(self.in_batch_shape) :]
        # Add a free dim for the D dimension
        trj_slc = B_slc + [slice(None)]
        return trj[trj_slc]

    def size(self, dim: str):
        return self.size_fn(dim, self.trj)

    def size_fn(self, dim: str, trj):
        if dim == self.readout_dim:
            return trj.shape[-1]
        elif dim == self.oshape[0]:
            return trj.shape[0]
        return None
