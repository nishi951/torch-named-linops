from typing import Optional
from jaxtyping import Float
from torch import Tensor

from torchlinops._core._linops.nameddim import NDorStr, ELLIPSES, NS, ND, get_nd_shape
from .chain import Chain
from .diagonal import Diagonal
from .pad_last import PadLast
from .fft import FFT
from .interp import Interpolate
from torchlinops.utils import default_to


Shape = tuple[NDorStr]


class NUFFT(Chain):
    def __init__(
        self,
        locs: Float[Tensor, "... D"],
        grid_size: tuple[int, ...],
        output_shape: Shape,
        input_shape: Optional[Shape] = None,
        batch_shape: Optional[Shape] = None,
        oversamp: float = 1.25,
        width: float = 4.0,
        **options,
    ):
        # Infer shapes
        self.grid_size = grid_size
        self.oversamp = oversamp
        self.width = width
        self.locs = locs

        input_shape = ND.infer(default_to(get_nd_shape(grid_size), input_shape))
        output_shape = ND.infer(output_shape)
        batch_shape = ND.infer(default_to(("...",), batch_shape))
        batched_input_shape = NS(batch_shape) + NS(input_shape)

        # Create Apodization
        beta = 1.0  # TODO steal this from sigpy
        weight = ...
        apodize = Diagonal(weight, batched_input_shape)
        apodize.name = "Apodize"

        # Create Padding
        padded_size = [int(i * oversamp) for i in grid_size]
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
            batch_shape=batch_shape,
            grid_shapes=pad.oshape,
        )

        # Create Interpolator
        interp = Interpolate(
            locs,
            grid_size,
            batch_shape=batch_shape,
            locs_batch_shape=output_shape,
            grid_shape=pad.oshape,
            width=width,
            kernel="kaiser_bessel",
            norm="1",
            beta=beta,
        )

        linops = [apodize, pad, fft, interp]
        super().__init__(*linops, name="NUFFT")

        self.options = default_to(
            {"toeplitz": False, "toeplitz_oversamp": 2.0}, options
        )

    # TODO: implement this
    # def normal(self, inner=None):
    #     if inner is not None:
    #         if self.options.get("toeplitz"):
    #             ...
    #         else:
    #             ...
    #     return NotImplemented
