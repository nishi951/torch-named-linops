from typing import Optional, Literal
from collections.abc import Callable
from jaxtyping import Float, Shaped
from torch import Tensor

from copy import copy
from math import prod
from itertools import product
from warnings import warn

import torch
import torch.nn as nn

from torchlinops.utils import default_to, cfftn

from .nameddim import NDorStr, ELLIPSES, NS, ND, get_nd_shape, Shape
from .namedlinop import NamedLinop
from .chain import Chain
from .dense import Dense
from .diagonal import Diagonal
from .scalar import Scalar
from .pad_last import PadLast
from .fft import FFT
from .interp import Interpolate
from .identity import Identity
from .sampling import Sampling


__all__ = ["NUFFT"]


class NUFFT(Chain):
    def __init__(
        self,
        locs: Float[Tensor, "... D"],
        grid_size: tuple[int, ...],
        output_shape: Shape,
        input_shape: Optional[Shape] = None,
        input_kshape: Optional[Shape] = None,
        batch_shape: Optional[Shape] = None,
        oversamp: float = 1.25,
        width: float = 4.0,
        mode: Literal["interpolate", "sampling"] = "interpolate",
        do_prep_locs: bool = True,
        apodize_weights: Optional[Float[Tensor, "..."]] = None,
        **options,
    ):
        """
        mode : "interpolate" or "sampling"
        do_prep_locs : bool, default True
            Whether to scale, shift, and clamp the locs to be amenable to interpolation
            By default (=True), assumes the locs lie in [-N/2, N/2]
                Scales, shifts and clamps them them to [0, oversamp*N - 1]
            If False, does not do this, which can have some benefits for memory reasons
        apodize_weights : Optional[Tensor]
            Provide apodization weights
            Only relevant for "intepolate" mode
            Can have memory benefits

        """
        device = locs.device
        self.mode = mode
        self.options = options
        # Infer shapes
        self.input_shape = ND.infer(default_to(get_nd_shape(grid_size), input_shape))
        self.input_kshape = ND.infer(
            default_to(get_nd_shape(grid_size, kspace=True), input_kshape)
        )
        self.output_shape = ND.infer(output_shape)
        self.batch_shape = ND.infer(default_to(("...",), batch_shape))
        batched_input_shape = NS(batch_shape) + NS(self.input_shape)

        # Initialize variables
        ndim = len(grid_size)
        padded_size = [int(i * oversamp) for i in grid_size]

        # Create Padding
        pad = PadLast(
            padded_size,
            grid_size,
            in_shape=self.input_shape,
            batch_shape=self.batch_shape,
        )

        # Create FFT
        fft = FFT(
            ndim=locs.shape[-1],
            centered=True,
            norm="ortho",
            batch_shape=self.batch_shape,
            grid_shapes=(pad.out_im_shape, self.input_kshape),
        )

        # Create Interpolator
        grid_shape = fft._shape.output_grid_shape
        if do_prep_locs:
            locs_prepared = self.prep_locs(
                locs, grid_size, padded_size, nufft_mode=mode
            )
        else:
            locs_prepared = locs
        if self.mode == "interpolate":
            beta = self.beta(width, oversamp)
            # Create Apodization
            if apodize_weights is None:
                weight = self.apodize_weights(
                    grid_size, padded_size, oversamp, width, beta
                ).to(device)  # Helps with batching later
            else:
                weight = apodize_weights
            if weight.isnan().any() or weight.isinf().any():
                raise ValueError(
                    f"Nan/Inf values detected in apodization weight (width={width}, oversamp={oversamp})."
                )
            apodize = Diagonal(weight, batched_input_shape.ishape)
            apodize.name = "Apodize"

            # Create Interpolator
            interp = Interpolate(
                locs_prepared,
                padded_size,
                batch_shape=self.batch_shape,
                locs_batch_shape=self.output_shape,
                grid_shape=grid_shape,
                width=width,
                kernel="kaiser_bessel",
                kernel_params=dict(beta=beta),
            )
            # Create scaling
            scale_factor = width**ndim * (prod(grid_size) / prod(padded_size)) ** 0.5
            scale = Scalar(weight=1.0 / scale_factor, ioshape=interp.oshape)
            scale.to(device)  # Helps with batching later
            linops = [apodize, pad, fft, interp, scale]
        elif self.mode == "sampling":
            if locs_prepared.is_complex() or locs_prepared.is_floating_point():
                raise ValueError(
                    f"Sampling linop requries integer-type locs but got {locs_prepared.dtype}"
                )
            # Clamp to within range
            interp = Sampling.from_stacked_idx(
                locs_prepared,
                dim=-1,
                # Arguments for Sampling
                input_size=padded_size,
                output_shape=self.output_shape,
                input_shape=grid_shape,
                batch_shape=self.batch_shape,
            )
            # No apodization or scaling needed
            linops = [pad, fft, interp]
        else:
            raise ValueError(f"Unrecognized NUFFT mode: {mode}")

        super().__init__(*linops, name="NUFFT")
        # Useful parameters to save
        self.locs = locs
        self.grid_size = grid_size
        self.oversamp = oversamp
        self.width = width

        # Handles to get modules directly
        self.pad = pad
        self.fft = fft
        self.interp = interp

    def adjoint(self):
        adj = super(Chain, self).adjoint()
        linops = [linop.H for linop in adj.linops]
        linops.reverse()
        adj.linops = nn.ModuleList(linops)
        return adj

    def normal(self, inner=None):
        if self.options.get("toeplitz", False):
            dtype = self.options.get("toeplitz_dtype")
            kernel_os, pad_os, fft_os = toeplitz_psf(self, inner, dtype=dtype)
            return pad_os.normal(fft_os.normal(kernel_os))
        else:
            normal = super().normal(inner)
        return normal

    @staticmethod
    def prep_locs(
        locs: Shaped[Tensor, "... D"],
        grid_size: tuple,
        padded_size: tuple,
        pad_mode: Literal["zero", "circular"] = "circular",
        nufft_mode: Literal["interpolate", "sampling"] = "interpolate",
    ):
        """
        Assumes centered locs
        """
        # out = locs.clone()
        out = locs
        for i in range(-len(grid_size), 0):
            out[..., i] *= padded_size[i] / grid_size[i]
            out[..., i] += padded_size[i] // 2
            if pad_mode == "zero":
                out[..., i] = torch.clamp(out[..., i], 0, padded_size[i] - 1)
            elif pad_mode == "circular":
                if nufft_mode == "interpolate":
                    out[..., i] = torch.remainder(
                        out[..., i], torch.tensor(padded_size[i])
                    )
                elif nufft_mode == "sampling":
                    out[..., i] = torch.clamp(out[..., i], 0, padded_size[i] - 1)
                    out[..., i] = torch.round(out[..., i])
            else:
                raise ValueError(f"Unrecognized padding mode during prep: {pad_mode}")
        return out.to(locs.dtype)

    @staticmethod
    def beta(width, oversamp):
        """
        https://sigpy.readthedocs.io/en/latest/_modules/sigpy/fourier.html#nufft

        References
        ----------
        Beatty PJ, Nishimura DG, Pauly JM. Rapid gridding reconstruction with a minimal oversampling ratio.
        IEEE Trans Med Imaging. 2005 Jun;24(6):799-808. doi: 10.1109/TMI.2005.848376. PMID: 15959939.
        """
        return torch.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5

    @staticmethod
    def apodize_weights(grid_size, padded_size, oversamp, width: float, beta: float):
        grid_size = torch.tensor(grid_size)
        padded_size = torch.tensor(padded_size)
        grid = torch.meshgrid(*(torch.arange(s) for s in grid_size), indexing="ij")
        grid = torch.stack(grid, dim=-1)

        # Sigpy compatibility
        apod = (
            beta**2 - (torch.pi * width * (grid - grid_size // 2) / padded_size) ** 2
        ) ** 0.5
        apod /= torch.sinh(apod)

        # Beatty paper
        # apod = (torch.pi * width * (grid - grid_size // 2) / padded_size) ** 2 - beta**2
        # print(apod)
        apod = torch.prod(apod, dim=-1)
        return apod

    # Special derived properties

    #     self.grid_size = tuple(grid_size)
    #     self.oversamp = oversamp
    #     self.width = width
    #     self.locs = locs
    #     self.options = default_to(
    #         {"toeplitz": False, "toeplitz_oversamp": 2.0}, options
    #     )

    # def normal(self, inner=None):
    #     if inner is not None:
    #         if self.options.get("toeplitz"):
    #             ...
    #         else:
    #             ...
    #     return NotImplemented

    def split_forward(self, ibatches, obatches):
        if len(ibatches) > 1 or len(obatches) > 1:
            raise ValueError(
                f"Got improper number of input and output batches for flattened chain linop {self}: ibatches: {ibatches}, obatches: {obatches}"
            )
        ibatch, obatch = ibatches[0], obatches[0]
        # Create ibatches and obatches from ibatch, obatch
        ibatch_lookup = {d: slc for d, slc in zip(self.ishape, ibatch)}
        obatch_lookup = {d: slc for d, slc in zip(self.oshape, obatch)}
        split_linops = []
        for linop in self.linops:
            sub_ibatch = [ibatch_lookup.get(dim, slice(None)) for dim in linop.ishape]
            sub_obatch = [obatch_lookup.get(dim, slice(None)) for dim in linop.oshape]
            split_linops.append(linop.split_forward(sub_ibatch, sub_obatch))
        out = copy(self)
        out.linops = nn.ModuleList(split_linops)
        return out

    def flatten(self):
        """Don't combine constituent linops into a chain with other linops"""
        return [self]

    @property
    def device(self):
        """Tracks device of interpolating/sampling linop
        Useful for toeplitz
        """

        if self.mode == "interpolate":
            return self.interp.locs.device
            # assert isinstance(
            #     self.linops[-2], Interpolate
            # ), f"Expected Interpolate linop but got {type(self.linops[-2])}"
            # return self.linops[-2].locs.device
        elif self.mode == "sampling":
            return self.interp.idx[0].device
            # assert isinstance(
            #     self.linops[-1], Sampling
            # ), f"Expected Sampling linop but got {type(self.linops[-1])}"
            # return self.linops[-1].idx[0].device
        raise ValueError(f"Unrecognized NUFFT mode: {self.mode}")


def toeplitz_psf(nufft, inner, dtype: Optional[torch.dtype] = None):
    if isinstance(nufft.interp, Sampling):
        raise NotImplementedError(
            f"Toeplitz embedding not yet implemented for Sampling-type NUFFT"
        )

    # Initialize variables
    dtype = default_to(torch.complex64, dtype)
    device = nufft.device
    # TODO: maybe accommodate other oversampling factors (more complicated)
    oversamp = 2.0
    grid_size = nufft.pad.im_size
    new_grid_size = scale_int(grid_size, oversamp)
    ndim = len(grid_size)
    width = nufft.pad.pad_im_size
    new_width = scale_int(width, oversamp)
    new_locs = rescale_locs(
        nufft.interp.locs.clone(),
        c0=tuple(w // 2 for w in width),
        w0=width,
        c1=tuple(w // 2 for w in new_width),
        w1=new_width,
    )
    nufft_os = NUFFT(
        # nufft.interp.locs.clone() * (oversamp / nufft.oversamp),
        new_locs,
        grid_size=new_grid_size,
        output_shape=nufft.output_shape,
        input_shape=nufft.input_shape,
        input_kshape=nufft.input_kshape,
        batch_shape=nufft.batch_shape,
        oversamp=nufft.oversamp,  # Oversample on top of toeplitz oversampling
        width=nufft.width,
        mode=nufft.mode,
        do_prep_locs=False,
    )
    pad_os = nufft_os.pad
    fft_os = nufft_os.fft

    # pad_os = PadLast(
    #     pad_im_size=pad_im_size,
    #     im_size=nufft.grid_size,
    #     in_shape=nufft.input_shape,
    #     batch_shape=nufft.batch_shape,
    # )
    # fft_os = FFT(
    #     ndim=len(pad_im_size),
    #     centered=True,
    #     norm="ortho",
    #     batch_shape=nufft.batch_shape,
    #     grid_shapes=(pad_os.out_im_shape, nufft.input_kshape),
    # )

    # Initialize inner if not provided
    if inner is None:
        inner = Identity(ishape=nufft.oshape)

    if len(inner.ishape) != len(inner.oshape):
        raise ValueError(
            f"Inner linop must have identical input and output shape lengths but got ishape={inner.ishape} and oshape={inner.oshape}"
        )

    # Get all useful shapes and sizes
    kernel_shape, ishape, oshape, kernel_size, input_size, batch_sizes = psf_sizing(
        nufft, inner, oversamp
    )

    # Create empty kernel
    kernel = torch.zeros(*kernel_size, dtype=dtype, device=device)

    # Allocate test input
    # Delta version
    # center_idx = tuple(s // 2 for s in new_grid_size)
    # delta = torch.zeros(*delta_size, dtype=dtype, device=device)

    # Ones version (saves 1 forward NUFFT)
    allones = torch.zeros(*input_size, dtype=dtype, device=device)
    scale_factor = oversamp**ndim / (prod(new_grid_size) ** 0.5)

    # Compute kernel by iterating through all possible input-output pairs
    dim = tuple(range(-len(new_grid_size), 0))
    for batch_idx in all_indices(batch_sizes):
        # TODO start from ones instead of from delta?
        # idx = (*batch_idx, *center_idx)
        # delta[idx] = 1.0  # set
        # psf = nufft_os.H(inner(nufft_os(delta)))
        allones[batch_idx] = 1.0
        psf = nufft_os.H(inner(allones))
        # kernel[batch_idx] = fft_os(pad_os(psf))
        # kernel[batch_idx] = fft_os(psf)
        # kernel[batch_idx] = cfftn(psf, dim=dim, norm=None) * 2**ndim
        kernel[batch_idx] = cfftn(psf, dim=dim, norm=None) * scale_factor
        # delta[idx] = 0.0  # reset
        allones[batch_idx] = 1.0  # reset
    kernel_os = Dense(
        weight=kernel,
        weightshape=kernel_shape,
        ishape=ishape,
        oshape=oshape,
    )
    return kernel_os, pad_os, fft_os, nufft_os


def scale_int(t: tuple[int, ...], scale_factor: float):
    return tuple(int(scale_factor * s) for s in t)


def psf_sizing(nufft, inner: NamedLinop, toeplitz_oversamp: float = 2.0):
    n_output_dims = len(nufft.output_shape)

    # Get all relevant shapes
    batch_ishape, batch_oshape = (
        inner.ishape[:-n_output_dims],
        inner.oshape[:-n_output_dims],
    )
    io_kshape = nufft.fft._shape.output_grid_shape
    ishape = batch_ishape + io_kshape
    oshape = batch_oshape + io_kshape
    kernel_shape = batch_ishape + batch_oshape + io_kshape

    # Get batch sizes
    batch_sizes = tuple(inner.size(d) for d in batch_ishape)
    batch_sizes = tuple(a if a is not None else 1 for a in batch_sizes)

    # Get kernel size
    im_size = nufft.pad.im_size
    kernel_ksize = scale_int(im_size, toeplitz_oversamp)
    kernel_size = batch_sizes + batch_sizes + kernel_ksize

    # Get test input size
    # input_size = batch_sizes + kernel_ksize
    output_size = tuple(nufft.size(d) for d in nufft.output_shape)
    input_size = batch_sizes + output_size

    return kernel_shape, ishape, oshape, kernel_size, input_size, batch_sizes


def all_indices(size: tuple[int]):
    ranges = tuple(range(s) for s in size)
    return product(*ranges)


def rescale_locs(locs, c0: tuple, w0: tuple, c1: tuple, w1: tuple, dim: int = -1):
    """Perform a scale-and-shift operation on a single dimension of a locs tensor.
    Parameters
    ----------
    locs : Tensor
        The locs to rescale, shape [... D ...]
    c0, w0 : tuple
        The center and width parameter for the current locs
    c1, w1: tuple
        The desired center and width parameters.
    dim : int
        The dimension of locs to unstack
    """
    ndim = locs.shape[dim]
    out = []
    for d in range(ndim):
        loc = torch.select(locs, dim, d)
        # Affine transform
        loc = (loc - c0[d]) * w1[d] / w0[d] + c1[d]
        out.append(loc)
    return torch.stack(out, dim=dim)


# def _changed_shape(ishape, oshape):
#     """Helper function for getting the kernel shape and slices
#     B = nufft shared + in batch shape
#     K = nufft out batch shape
#     Kx1, Ky1, Kz1 = output padded fourier-domain shapes

#     kernel_fourier_dim: [Kx1, Ky1, [Kz1]
#     B1 = nufft shared + in batch shape, different dims only. May be different length than B
#     kernel shape: [B1..., B..., Kx1, Ky1, [Kz1]]
#     ishape: [B..., K...]
#     in_batch_shape: [B...]
#     oshape: [B1..., K...]
#     out_batch_shape: [B1...]
#     """
#     changed_shape = []
#     changed = []
#     for idim, odim in zip(ishape, oshape):
#         if idim != odim:
#             changed_shape.append(idim)
#             changed.append(True)
#         else:
#             changed.append(False)
#     changed_shape = tuple(changed_shape)
#     return changed_shape, changed
