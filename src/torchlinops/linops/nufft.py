from typing import Optional, Literal
from jaxtyping import Float, Shaped
from torch import Tensor

from copy import copy
from math import prod
from warnings import warn

import torch
import torch.nn as nn

from torchlinops.utils import default_to

from .nameddim import NDorStr, ELLIPSES, NS, ND, get_nd_shape, Shape
from .namedlinop import NamedLinop
from .chain import Chain
from .diagonal import Diagonal
from .scalar import Scalar
from .pad_last import PadLast
from .fft import FFT
from .interp import Interpolate
from .sampling import Sampling


__all__ = ["NUFFT"]

# TODO create functional form based on this linop


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
        # Infer shapes
        input_shape = ND.infer(default_to(get_nd_shape(grid_size), input_shape))
        input_kshape = ND.infer(
            default_to(get_nd_shape(grid_size, kspace=True), input_kshape)
        )
        output_shape = ND.infer(output_shape)
        batch_shape = ND.infer(default_to(("...",), batch_shape))
        batched_input_shape = NS(batch_shape) + NS(input_shape)

        # Initialize variables
        ndim = len(grid_size)
        padded_size = [int(i * oversamp) for i in grid_size]

        # Create Padding
        pad = PadLast(
            padded_size,
            grid_size,
            in_shape=input_shape,
            batch_shape=batch_shape,
        )

        # Create FFT
        fft = FFT(
            ndim=locs.shape[-1],
            centered=True,
            norm="ortho",
            batch_shape=batch_shape,
            grid_shapes=(pad.out_im_shape, input_kshape),
        )

        # Create Interpolator
        grid_shape = fft._shape.output_grid_shape
        if do_prep_locs:
            locs_prepared = self.prep_locs(
                locs, grid_size, padded_size, nufft_mode=mode
            )
        else:
            locs_prepared = locs
        if mode == "interpolate":
            beta = self.beta(width, oversamp)
            # Create Apodization
            if apodize_weights is None:
                weight = self.apodize_weights(
                    grid_size, padded_size, oversamp, width, beta
                ).to(device)  # Helps with batching later
            else:
                weight = apodize_weights
            apodize = Diagonal(weight, batched_input_shape.ishape)
            apodize.name = "Apodize"

            # Create Interpolator
            interp = Interpolate(
                locs_prepared,
                padded_size,
                batch_shape=batch_shape,
                locs_batch_shape=output_shape,
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
        elif mode == "sampling":
            # Clamp to within range
            interp = Sampling.from_stacked_idx(
                locs_prepared.long(),
                dim=-1,
                # Arguments for Sampling
                input_size=padded_size,
                output_shape=output_shape,
                input_shape=grid_shape,
                batch_shape=batch_shape,
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

    def adjoint(self):
        adj = super(Chain, self).adjoint()
        linops = [linop.H for linop in adj.linops]
        linops.reverse()
        adj.linops = nn.ModuleList(linops)
        return adj

    # TODO: Replace with toeplitz version
    def normal(self, inner=None):
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
        out = locs.clone()
        for i in range(-len(grid_size), 0):
            out[..., i] *= padded_size[i] / grid_size[i]
            out[..., i] += padded_size[i] / 2
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
        apod = (
            beta**2 - (torch.pi * width * (grid - grid_size // 2) / padded_size) ** 2
        ) ** 0.5
        apod /= torch.sinh(apod)
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
