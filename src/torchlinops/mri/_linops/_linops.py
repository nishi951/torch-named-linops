from math import prod  # Numpy prod returns floats
from typing import Optional, Tuple, Literal

from einops import rearrange
import torch
import torch.nn as nn
from torchkbnufft import KbNufft, KbNufftAdjoint

from ..._core._linops import NamedLinop
from ..._core._shapes import get2dor3d
from . import _sp_nufft as spnufft
from . import _fi_nufft as finufft

from .convert_trj import sp2fi

__all__ = [
    "NUFFT",
    "SENSE",
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


class FiNUFFT(NamedLinop):
    def __init__(
        self,
        trj: torch.Tensor,
        im_size: Tuple,
        in_batch_shape: Optional[Tuple] = None,
        out_batch_shape: Optional[Tuple] = None,
        shared_batch_shape: Optional[Tuple] = None,
    ):
        """
        img (input) [S... N... Nx Ny [Nz]]
        trj: [S... K..., D] in sigpy style [-N/2, N/2]
        in_batch_shape : Tuple
            The shape of [N...] in img
        out_batch_shape : Tuple
            The shape of [K...] in trj.
        shared_batch_shape : Tuple
            The shape of [S...] in trj

        """
        self.in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        self.out_batch_shape = (
            out_batch_shape if out_batch_shape is not None else tuple()
        )
        self.shared_batch_shape = shared_batch_shape if shared_batch_shape is not None else tuple()
        self.shared_dims = len(self.shared_batch_shape)
        ishape = self.shared_batch_shape + self.in_batch_shape + get2dor3d(im_size)
        oshape = self.shared_batch_shape + self.in_batch_shape + self.out_batch_shape
        super().__init__(ishape, oshape)
        self.trj = nn.Parameter(trj, requires_grad=False)
        self.im_size = im_size

        # Precompute
        self.D = len(im_size)

    def forward(self, x: torch.Tensor):
        return self.fn(x, self.trj)

    def fn(self, x, /, trj):
        """
        x: [[S...] N...  Nx Ny [Nz]] # A... may include coils
        trj: [[S...] K... D] (sigpy-style)
        output: [[S...] N... K...]
        """
        if self.shared_dims == 0:
            return finufft.nufft(x, sp2fi(trj, self.im_size))
        assert x.shape[:self.shared_dims] == trj.shape[:self.shared_dims], f'First {self.shared_dims} dims of x, trj  must match but got x: {x.shape}, trj: {trj.shape}'
        S = x.shape[:self.shared_dims]
        x = torch.flatten(x, start_dim=0, end_dim=self.shared_dims-1)
        trj = torch.flatten(trj, start_dim=0, end_dim=self.shared_dims-1)
        N = x.shape[self.shared_dims:-self.D]
        K = trj.shape[self.shared_dims:-1]
        output_shape = (*S, *N, *K)
        y = torch.zeros((prod(S), *N, *K), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            finufft.nufft(x[i], sp2fi(trj[i], self.im_size), out=y[i])
        y = torch.reshape(y, output_shape)
        return y

    def adj_fn(self, y, /, trj):
        """
        y: [[S...] N... K...] # N... may include coils
        trj: [[S...] K... D] (sigpy-style)
        output: [[S...] N...  Nx Ny [Nz]]
        """
        if self.shared_dims == 0:
            return finufft.nufft_adjoint(y, sp2fi(trj, self.im_size), oshape)
        assert x.shape[:self.shared_dims] == trj.shape[:self.shared_dims], f'First {self.shared_dims} dims of x, trj  must match but got x: {x.shape}, trj: {trj.shape}'
        S = y.shape[:self.shared_dims]
        y = torch.flatten(y, start_dim=0, end_dim=self.shared_dims)
        trj = torch.flatten(trj, start_dim=0, end_dim=self.shared_dims)
        N = x.shape[self.shared_dims:-self.D]
        oshape = (*N, *self.im_size)
        output_shape = (*S, *N, *self.im_size)
        x = torch.zeros((prod(S), *N, *self.im_size), dtype=y.dtype, device=y.device)
        for i in x.shape[0]:
            finufft.nufft_adjoint(y, sp2fi(trj, self.im_size), oshape, out=x[i])
        x = torch.reshape(x, output_shape)
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
        )

    def split_forward_fn(self, ibatch, obatch, trj):
        # Get slice corresponding to trj
        B_slc = obatch[len(self.in_batch_shape) :]
        # Add a free dim for the D dimension
        trj_slc = obatch[:-1] + [slice(None)] + obatch[-1:]
        return trj[trj_slc]

    def size(self, dim: str):
        return self.size_fn(dim, self.trj)

    def size_fn(self, dim: str, trj):
        if dim == self.readout_dim:
            return trj.shape[-2]
        return None


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
        y = spnufft.nufft(x, trj, **self.nufft_kwargs)
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
        x = spnufft.nufft_adjoint(y, trj, oshape, **self.nufft_kwargs)
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
        B_slc = obatch[len(self.in_batch_shape) :]
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


class SENSE(NamedLinop):
    def __init__(
        self,
        mps: torch.Tensor,
        coil_str: str = "C",
        in_batch_shape: Optional[Tuple] = None,
    ):
        self.im_size = mps.shape[1:]
        self.D = len(self.im_size)
        self.coildim = -(self.D + 1)
        self.in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        self.out_batch_shape = self.in_batch_shape + (coil_str,)
        ishape = self.in_batch_shape + get2dor3d(self.im_size)
        oshape = self.out_batch_shape + get2dor3d(self.im_size)
        super().__init__(ishape, oshape)
        self.coil_str = coil_str
        self.mps = nn.Parameter(mps, requires_grad=False)

    def forward(self, x):
        return self.fn(x, self.mps)

    def fn(self, x, /, mps):
        return x.unsqueeze(self.coildim) * mps

    def adj_fn(self, x, /, mps):
        return torch.sum(x * torch.conj(mps), dim=self.coildim)

    def split_forward(self, ibatch, obatch):
        """Split over coil dim only"""
        for islc, oslc in zip(ibatch[-self.D :], obatch[-self.D :]):
            if islc != oslc:
                raise IndexError(
                    "SENSE currently only supports matched image input/output slicing."
                )
        return type(self)(self.split_forward_fn(ibatch, obatch, self.mps))

    def split_forward_fn(self, ibatch, obatch, /, weight):
        return self.mps[obatch[self.coildim :]]

    def size(self, dim: str):
        return self.size_fn(dim, self.mps)

    def size_fn(self, dim: str, mps):
        if dim in self.oshape[self.coildim :]:
            return mps.shape[self.oshape.index(dim)]
        return None
