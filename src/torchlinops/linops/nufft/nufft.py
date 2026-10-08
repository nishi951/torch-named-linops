from functools import lru_cache
from math import prod
from typing import Literal

import torch
from jaxtyping import Shaped
from torch import Tensor

from torchlinops import config

from ...nameddim import (
    NamedShape as NS,
)
from ..diagonal import Diagonal
from ..fft import FFT
from ..interp import Interpolate
from ..pad_last import Pad
from ..scalar import Scalar
from ._base import NUFFTBase

__all__ = ["NUFFT"]


class NUFFT(NUFFTBase):
    """Non-uniform Fast Fourier Transform (type II) as a named linear operator.

    Implemented as a ``Chain`` of zero-padding, FFT, and interpolation. Supports
    forward (image-to-kspace) and adjoint (kspace-to-image) operations.

    Attributes
    ----------
    **options : dict
        oversamp : float
            Oversampling factor for fourier domain grid
        width : float
            Width of kernel to use for interpolation
        skip_prep_locs : bool
            Flag for skipping scaling/shifting/etc of locs
        toeplitz : bool
            If True, normal() performs toeplitz embedding calculation
        toeplitz_dtype : torch.dtype
            Data type for the toeplitz embedding. Probably should be torch.complex64
    """

    default_options = {
        "oversamp": 1.25,
        "width": 4.0,
        "toeplitz": False,
        "toeplitz_dtype": torch.complex64,
    }

    def __init__(self, *args, **kwargs):
        if "mode" in kwargs:
            raise ValueError(
                f"`mode` parameter was deprecated. Must now explicitly choose torchlinops.SamplingNUFFT or torchlinops.NUFFT."
            )
        super().__init__(*args, **kwargs)

    def build(self):
        grid_size = self.grid_size
        ndim = len(self.grid_size)
        padded_size = tuple(int(i * self.options["oversamp"]) for i in grid_size)
        if self.options.get("skip_prep_locs"):
            locs_prepared = self.locs
        else:
            if config.cache_nufft_parameters:
                locs_prepared = self.prep_locs(self.locs, self.grid_size, padded_size)
            else:
                locs_prepared = self.prep_locs.__wrapped__(
                    self.locs, self.grid_size, padded_size
                )
        pad = Pad(
            padded_size,
            self.grid_size,
            in_shape=self.input_shape,
            batch_shape=self.batch_shape,
        )

        # Create FFT
        fft = FFT(
            ndim=ndim,
            centered=True,
            batch_shape=self.batch_shape,
            grid_shapes=(pad.out_im_shape, self.input_kshape),
        )

        # Create Interpolator
        grid_shape = fft._shape.output_grid_shape

        # Create Apodization
        width, oversamp = self.options["width"], self.options["oversamp"]
        beta = self.beta(width, oversamp)
        if config.cache_nufft_parameters:
            weight = self.apodize_weights(grid_size, padded_size, width, beta)
        else:
            weight = self.apodize_weights.__wrapped__(
                grid_size, padded_size, width, beta
            )
        if weight.isnan().any() or weight.isinf().any():
            raise ValueError(
                f"Nan/Inf values detected in apodization weight (width={width}, oversamp={oversamp})."
            )
        batched_input_shape = NS(self.batch_shape) + NS(self.input_shape)
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
        linops = [apodize, pad, fft, interp, scale]
        return linops

    def post_init_hook(self):
        self.pad = self.linops[1]
        self.fft = self.linops[2]
        self.interp = self.linops[3]

    @staticmethod
    @lru_cache(maxsize=64)
    def prep_locs(
        locs: Shaped[Tensor, "... D"],
        grid_size: tuple,
        padded_size: tuple,
        pad_mode: Literal["zero", "circular"] = "circular",
    ):
        """
        Parameters
        ----------
        locs : Shaped[Tensor, "... D"]
            Input tensor representing locations in the grid. The last dimension corresponds to spatial dimensions.
            Range is [-N//2, N//2]
        grid_size : tuple
            The original size of the grid before padding.
        padded_size : tuple
            The size of the grid after padding.
        pad_mode : Literal["zero", "circular"], optional
            The type of padding applied. Can be "zero" for zero-padding or "circular" for circular padding.
            Default is "circular".
        Returns
        -------
        Shaped[Tensor, "... D"]
            Adjusted locations tensor in [0, padded_size - 1], float dtype.

        Raises
        ------
        ValueError
            If an unrecognized `pad_mode` is provided.

        Examples
        --------
        >>> _ = torch.manual_seed(0);
        >>> locs = torch.rand(1000, 3) * 64 - 32 # [-32, 32]
        >>> locs.min()
        tensor(-31.9949)
        >>> locs.max()
        tensor(31.9896)
        >>> grid_size = (64, 64, 64)
        >>> padded_size = (80, 80, 80) # oversamp = 1.25
        >>> locs_scaled_shifted = NUFFT.prep_locs(locs, grid_size, padded_size)
        >>> locs_scaled_shifted.min()
        tensor(0.0064)
        >>> locs_scaled_shifted.max()
        tensor(79.9871)

        Notes
        -----
        - Assumes that the input `locs` are centered.
        - Adjusts the locations by scaling and shifting them according to the grid and padded sizes.
        - Applies clamping or remainder operations based on the padding mode.
        """
        # Clone to prevent in-place scaling from modifying the original
        out = locs.clone()
        for i in range(-len(grid_size), 0):
            out[..., i] *= padded_size[i] / grid_size[i]
            out[..., i] += padded_size[i] // 2
            if pad_mode == "zero":
                out[..., i] = torch.clamp(out[..., i], 0, padded_size[i] - 1)
            elif pad_mode == "circular":
                out[..., i] = torch.remainder(out[..., i], torch.tensor(padded_size[i]))
            else:
                raise ValueError(f"Unrecognized padding mode during prep: {pad_mode}")
        return out

    @property
    def device(self):
        """Tracks device of interpolating/sampling linop
        Useful for toeplitz
        """
        return self.interp.locs.device

    @staticmethod
    @lru_cache(maxsize=64)
    def apodize_weights(grid_size, padded_size, width: float, beta: float):
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
