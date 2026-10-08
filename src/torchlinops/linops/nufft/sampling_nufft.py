from functools import lru_cache
from typing import ClassVar, Literal

import torch
from jaxtyping import Shaped
from torch import Tensor

from torchlinops import config

from ..fft import FFT
from ..pad_last import Pad
from ..sampling import Sampling
from ._base import NUFFTBase

__all__ = ["SamplingNUFFT"]


class SamplingNUFFT(NUFFTBase):
    """Non-uniform Fast Fourier Transform (type II) as a named linear operator.

    Attributes
    ----------
    **options : dict
        oversamp : float
            Oversampling factor for fourier domain grid

    """

    default_options: ClassVar[dict] = {"oversamp": 1.25}

    def build(self):
        ndim = len(self.grid_size)
        padded_size = tuple(int(i * self.options["oversamp"]) for i in self.grid_size)
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
        return linops

    def post_init_hook(self):
        self.pad = self.linops[0]
        self.fft = self.linops[1]
        self.interp = self.linops[2]

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
            Wrapped, rounded integer locations in [0, padded_size - 1].

        Raises
        ------
        ValueError
            If an unrecognized `pad_mode` is provided.

        Examples
        --------
        >>> _ = torch.manual_seed(0);
        >>> locs = torch.rand(1000, 3) * 64 - 32 # [-32, 32]
        >>> locs = torch.round(locs * 1.25) / 1.25
        >>> grid_size = (64, 64, 64)
        >>> padded_size = (80, 80, 80) # oversamp = 1.25
        >>> locs_scaled_shifted = SamplingNUFFT.prep_locs(locs, grid_size, padded_size)
        >>> locs_scaled_shifted.min()
        tensor(0)
        >>> locs_scaled_shifted.max()
        tensor(79)

        Notes
        -----
        - Assumes that the input `locs` are centered.
        - Adjusts the locations by scaling and shifting them according to the grid and padded sizes.
        - Rounds and wraps the resulting indices based on the padding mode.
        """
        # Clone to prevent in-place scaling from modifying the original
        out = locs.clone()
        for i in range(-len(grid_size), 0):
            out[..., i] *= padded_size[i] / grid_size[i]
            out[..., i] += padded_size[i] // 2
            if pad_mode == "zero":
                out[..., i] = torch.clamp(out[..., i], 0, padded_size[i] - 1)
            elif pad_mode == "circular":
                # Wrap rounded index to other side of kspace
                out[..., i] = torch.round(out[..., i])
                out[..., i] = torch.remainder(out[..., i], padded_size[i])
            else:
                raise ValueError(f"Unrecognized padding mode during prep: {pad_mode}")
        out = out.to(torch.int64)
        return out

    @property
    def device(self):
        """Tracks device of the sampling linop."""
        return self.interp.idx[0].device
