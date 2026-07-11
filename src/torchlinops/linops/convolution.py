from copy import copy
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..functional._interp._circ_pad import circular_pad
from ..nameddim import NamedDimension as ND
from ..nameddim import NamedShape as NS
from ..nameddim import Shape, get_nd_shape
from .namedlinop import NamedLinop

__all__ = ["Convolution"]


class Convolution(NamedLinop):
    """N-dimensional convolution as a named linear operator.

    Supports 1D, 2D, and 3D convolutions with named batch, channel, and
    spatial dimensions. Uses PyTorch's native F.conv1d/2d/3d and F.conv_transpose1d/2d/3d.

    Note that unlike pytorch's Conv (which is really autocorrelation), the provided weight is flipped across each axis.

    Automatically broadcasts over channels.

    TODO:
    - Strided convolution
    - Dilation

    Assumes "center" of input weight is at [d1//2, d2//2, ...]

    Attributes
    ----------
    ndim : int
        Number of spatial dimensions (1, 2, or 3).
    weight : Tensor
        Convolution kernel of shape (out_channels, in_channels, *kernel_size).
    padding : str or int
        Padding mode or size.
    """

    def __init__(
        self,
        weight: Tensor,
        batch_shape: Optional[Shape] = None,
        in_grid_shape: Optional[Shape] = None,
        out_grid_shape: Optional[Shape] = None,
        padding_mode: Literal["zeros", "circular"] = "zeros",
        **options,
    ):
        """
        Parameters
        ----------
        weight : Tensor
            Convolution kernel of shape (*kernel_size).
        batch_shape : Shape, optional
            Named batch dimensions. Defaults to ("...",).
        in_grid_shape : Shape, optional
            Named input grid dimensions including channels.
            First dim is input channels, remaining are spatial.
        out_grid_shape : Shape, optional
            Named output grid dimensions including channels.
            First dim is output channels, remaining are spatial.
        padding_mode : "zeros" or "circular"
            Padding mode
        **options : dict
            Additional options (normal_mode, fft_dtype, etc.)
        """
        ndim = weight.dim()
        if ndim not in (1, 2, 3):
            raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")
        # if weight.dim() != ndim:
        #     raise ValueError(
        #         f"Expected weight with {ndim} dims but got shape {weight.shape}"
        #     )

        self.ndim = ndim
        self.options = options

        # Pad to odd size for simplicity
        # Later, might want to revisit for efficiency gains
        self.padding_mode = padding_mode
        weight = pad_to_odd(weight, ndim)
        _pad = [(s // 2, s // 2) for s in weight.shape[-ndim:]]
        self._crop = tuple(
            slice(first, -last if last > 0 else None) for first, last in _pad
        )
        _pad.reverse()
        self._pad = sum(_pad, start=tuple())

        # Set up shapes
        if batch_shape is None:
            batch_shape = ("...",)

        if in_grid_shape is None:
            in_grid_shape = ("Cin",) + get_nd_shape(ndim)

        if out_grid_shape is None:
            out_grid_shape = ("Cout",) + tuple(
                ND.infer(s).next_unused() for s in get_nd_shape(ndim)
            )

        # Build ishape and oshape
        ishape = batch_shape + in_grid_shape
        oshape = batch_shape + out_grid_shape

        super().__init__(NS(ishape, oshape))
        self._kernel = weight  # The original input, possibly padded
        self._shape.batch_shape = tuple(batch_shape)
        self._shape.in_grid_shape = tuple(in_grid_shape)
        self._shape.out_grid_shape = tuple(out_grid_shape)

        # Proper conv: flip but don't conjugate each dim
        weight = torch.flip(weight, dims=tuple(range(-ndim, 0)))
        weight = weight[None, None]  # [1, 1, *kernel_size]
        self.weight = nn.Parameter(weight, requires_grad=False)

    @staticmethod
    def fn(conv, x, /):
        """Forward convolution."""
        x, batch_size = compress_batch_and_channel(x, conv.ndim)
        # Broadcast weight
        conv_fn = (F.conv1d, F.conv2d, F.conv3d)[conv.ndim - 1]
        if conv.padding_mode == "circular":
            x = circular_pad(x, conv._pad)
            x = conv_fn(x, conv.weight, padding=0)
        elif conv.padding_mode == "zeros":
            x = conv_fn(x, conv.weight, padding="same")
        else:
            raise ValueError(f"Unrecognized padding_mode: {conv.padding_mode}")
        x = expand_batch_and_channel(x, batch_size)
        return x

    @staticmethod
    def adj_fn(conv, x, /):
        """Adjoint (transpose) convolution."""
        x, batch_size = compress_batch_and_channel(x, conv.ndim)

        if conv.padding_mode == "circular":
            # Adjoint of circular convolution is circular correlation
            # which is circular convolution with flipped and conjugated kernel
            # Flip the spatial dimensions of the kernel
            # Also transpose the input and output channels
            weight_flipped = torch.flip(conv.weight, dims=tuple(range(-conv.ndim, 0)))
            weight_flipped = weight_flipped.conj()
            conv_fn = (F.conv1d, F.conv2d, F.conv3d)[conv.ndim - 1]
            x = circular_pad(x, conv._pad)
            x = conv_fn(x, weight_flipped, padding=0)
        elif conv.padding_mode == "zeros":
            weight_conj = conv.weight.conj()
            conv_t_fn = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[
                conv.ndim - 1
            ]
            x = conv_t_fn(x, weight_conj, padding=0)
            slc = (slice(None), slice(None)) + conv._crop
            x = x[slc]
        else:
            raise ValueError(f"Unrecognized padding_mode: {conv.padding_mode}")
        x = expand_batch_and_channel(x, batch_size)
        return x

    def normal(self, inner=None):
        """Compute the normal operator A^H A.

        Parameters
        ----------
        inner : NamedLinop, optional
            Inner operator to sandwich between adjoint and forward.

        Returns
        -------
        NamedLinop
            The normal operator.

        Notes
        -----
        Three modes controlled by options["normal_mode"]:
        - None: Default fallback, compose adj_fn(fn(x))
        - "conv": Composed convolution with autocorrelated kernel (circular padding only)
        - "fft": FFT-based Toeplitz embedding (circular padding only)
        """
        if inner is not None:
            return super().normal(inner)

        mode = self.options.get("normal_mode")

        if mode is None:
            return super().normal()

        if mode == "fft":
            if self.padding_mode != "circular":
                raise ValueError("FFT normal mode only supports circular padding")
            return self._normal_fft()

        if mode == "conv":
            if self.padding_mode != "circular":
                raise ValueError("Conv normal mode only supports circular padding")
            return self._normal_conv()

        raise ValueError(f"Unknown normal_mode: {mode}")

    def _normal_conv(self):
        """Normal operator via composed convolution with autocorrelated kernel.

        This only works correctly for circular padding. For zero padding,
        the normal operator is not a simple convolution due to boundary effects.
        """

        kernel = self._kernel[None, None]
        kernel_conj = kernel.conj()
        conv_fn = (F.conv1d, F.conv2d, F.conv3d)[self.ndim - 1]
        in_grid_shape = self._shape.in_grid_shape
        double_pad = tuple(2 * p for p in self._pad)
        kernel_circ_pad = F.pad(kernel, pad=double_pad)
        new_weight = conv_fn(kernel_circ_pad, kernel_conj, padding=0)
        return type(self)(
            new_weight[0, 0],
            self._shape.batch_shape,
            in_grid_shape,
            ND.infer(tuple(s.next_unused(in_grid_shape) for s in in_grid_shape)),
            padding_mode=self.padding_mode,
            **self.options,
        )

    def _normal_fft(self):
        """Normal operator via FFT-based circular convolution (circular padding only)."""
        # For circular convolution, the normal operator is:
        # A^H A x = IFFT(|FFT(k)|^2 * FFT(x))
        # We implement this as a custom forward function that uses FFT directly.
        # The PSD is computed dynamically based on the input spatial size.

        # kernel = self._kernel
        # out_c, in_c = weight.shape[0], weight.shape[1]
        # kernel_size = weight.shape[2:]

        # Create a custom linop that computes PSD dynamically
        class FFTNormalLinop(NamedLinop):
            def __init__(self, weight, kernel_size, batch_shape, in_grid_shape, ndim):
                ishape = batch_shape + in_grid_shape
                oshape = batch_shape + in_grid_shape
                super().__init__(NS(ishape, oshape))
                self._weight = weight
                self._kernel_size = kernel_size
                self._batch_shape = batch_shape
                self._in_grid_shape = in_grid_shape
                self.ndim = ndim
                self._psd_cache = {}  # Cache PSD for different spatial sizes

            def _get_psd(self, spatial_size):
                """Compute PSD for given spatial size, with caching."""
                if spatial_size in self._psd_cache:
                    return self._psd_cache[spatial_size]

                # Pad kernel to input size, placing it at the origin
                pad_sizes = []
                for k, n in zip(self._kernel_size, spatial_size):
                    pad_sizes.extend([0, n - k])

                weight_padded = F.pad(self._weight, tuple(reversed(pad_sizes)))

                # FFT along spatial dimensions
                spatial_dims_fft = tuple(range(-self.ndim, 0))
                weight_fft = torch.fft.fftn(weight_padded, dim=spatial_dims_fft)

                # Compute |FFT(k)|^2, sum over output channels
                spatial_indices = "abc"[: self.ndim]
                einsum_str = (
                    f"oi{spatial_indices},oj{spatial_indices}->ij{spatial_indices}"
                )
                psd = torch.einsum(einsum_str, weight_fft.conj(), weight_fft)

                self._psd_cache[spatial_size] = psd
                return psd

            @staticmethod
            def fn(linop, x):
                # Get spatial size from input
                # ishape has format: (*batch, c_in, *spatial)
                # We need to figure out how many batch dims there are
                # Total dims in ishape = len(batch_shape) + 1 + ndim
                # But "..." can expand to 0 or more dims
                # So actual_batch_dims = x.ndim - 1 - linop.ndim
                actual_batch_dims = x.ndim - 1 - linop.ndim
                spatial_size = x.shape[
                    actual_batch_dims + 1 :
                ]  # Skip batch and channel dims

                psd = linop._get_psd(spatial_size)

                spatial_dims = tuple(range(-linop.ndim, 0))
                x_fft = torch.fft.fftn(x, dim=spatial_dims)

                # Multiply by PSD: result[i] = sum_j psd[i,j] * x_fft[j]
                spatial_indices = "abc"[: linop.ndim]
                einsum_str = (
                    f"ij{spatial_indices},j{spatial_indices}->i{spatial_indices}"
                )
                result_fft = torch.einsum(einsum_str, psd, x_fft)

                return torch.fft.ifftn(result_fft, dim=spatial_dims).real

            @staticmethod
            def adj_fn(linop, x):
                # PSD is Hermitian, so adjoint is the same
                return FFTNormalLinop.fn(linop, x)

            @staticmethod
            def split(linop, tile):
                return copy(linop)

            def size(self, dim):
                # Can't determine size without knowing input
                return None

        # return FFTNormalLinop(
        #     weight, kernel_size, self._batch_shape, self._in_grid_shape, self.ndim
        # )


def pad_to_odd(weight: Tensor, ndim: int):
    """
    Parameters
    ----------
    weight : Tensor
        Shape [out_channel, in_channel, *grid_shape]
    ndim : int
        Number of dimensions.
    """
    pad_first = []
    pad_last = []
    for s in weight.shape[-ndim:]:
        if s % 2:  # weight_shape is odd
            pad_first.append(0)
            pad_last.append(0)
        else:
            pad_first.append(0)
            pad_last.append(1)
    pad_first.reverse()
    pad_last.reverse()

    _pad = [(first, last) for first, last in zip(pad_first, pad_last)]
    _pad = sum(_pad, start=tuple())

    return F.pad(weight, pad=_pad, value=0)


def compress_batch_and_channel(x, ndim):
    """
    x has shape [B... C *dims]

    B and C are optional

    Returns
    -------
    Tensor
        Shape [(B... C) 1 *dims]

    """
    batch_size = x.shape[:-ndim]
    nbatch = len(batch_size)
    if nbatch > 0:
        x_flat = torch.flatten(x, start_dim=0, end_dim=nbatch - 1)
    else:
        x_flat = x[None]
    return x_flat[:, None], batch_size


def expand_batch_and_channel(x, batch_size):
    return x.reshape(batch_size + x.shape[2:])
