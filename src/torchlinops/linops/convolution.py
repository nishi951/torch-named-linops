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
from .pad_last import pad_to_size

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
    kernel : Tensor
        Convolution kernel of shape (out_channels, in_channels, *kernel_size).
    padding : str or int
        Padding mode or size.
    """

    def __init__(
        self,
        kernel: Tensor,
        batch_shape: Optional[Shape] = None,
        in_grid_shape: Optional[Shape] = None,
        out_grid_shape: Optional[Shape] = None,
        padding_mode: Literal["zeros", "circular"] = "zeros",
        **options,
    ):
        """
        Parameters
        ----------
        kernel : Tensor
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
        ndim = kernel.dim()
        if ndim not in (1, 2, 3):
            raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")

        self.ndim = ndim
        self.options = options

        # Pad to odd size for simplicity
        # Later, might want to revisit for efficiency gains
        self.padding_mode = padding_mode
        kernel = pad_to_odd(kernel, ndim)
        _pad = [(s // 2, s // 2) for s in kernel.shape[-ndim:]]
        self._crop = tuple(
            slice(first, -last if last > 0 else None) for first, last in _pad
        )
        _pad.reverse()
        self._pad = sum(_pad, start=tuple())

        # Set up shapes
        if batch_shape is None:
            batch_shape = ("...",)

        if in_grid_shape is None:
            in_grid_shape = get_nd_shape(ndim)

        if out_grid_shape is None:
            out_grid_shape = tuple(
                ND.infer(s).next_unused() for s in get_nd_shape(ndim)
            )

        # Build ishape and oshape
        ishape = batch_shape + in_grid_shape
        oshape = batch_shape + out_grid_shape

        super().__init__(NS(ishape, oshape))
        self.kernel = kernel  # The original input, possibly padded
        self._shape.batch_shape = tuple(batch_shape)
        self._shape.in_grid_shape = tuple(in_grid_shape)
        self._shape.out_grid_shape = tuple(out_grid_shape)

        # Proper conv: flip but don't conjugate each dim
        # kernel -> "weight" to indicate that we are moving from
        # math language to pytorch language
        weight = torch.flip(kernel, dims=tuple(range(-ndim, 0)))
        weight = weight[None, None]  # [1, 1, *kernel_size]
        self._weight = nn.Parameter(weight, requires_grad=False)

    @staticmethod
    def fn(conv, x, /):
        """Forward convolution."""
        x, batch_size = compress_batch_and_channel(x, conv.ndim)
        # Broadcast weight
        conv_fn = (F.conv1d, F.conv2d, F.conv3d)[conv.ndim - 1]
        if conv.padding_mode == "circular":
            x = circular_pad(x, conv._pad)
            x = conv_fn(x, conv._weight, padding=0)
        elif conv.padding_mode == "zeros":
            x = conv_fn(x, conv._weight, padding="same")
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
            weight_flipped = torch.flip(conv._weight, dims=tuple(range(-conv.ndim, 0)))
            weight_flipped = weight_flipped.conj()
            conv_fn = (F.conv1d, F.conv2d, F.conv3d)[conv.ndim - 1]
            x = circular_pad(x, conv._pad)
            x = conv_fn(x, weight_flipped, padding=0)
        elif conv.padding_mode == "zeros":
            weight_conj = conv._weight.conj()
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
        new_kernel = cross_correlation(self.kernel, self.kernel)
        batch_shape = self._shape.batch_shape
        in_grid_shape = self._shape.in_grid_shape

        # TODO change this?
        return type(self)(
            new_kernel,
            batch_shape,
            in_grid_shape,
            ND.infer(tuple(s.next_unused(in_grid_shape) for s in in_grid_shape)),
            padding_mode=self.padding_mode,
            **self.options,
        )

    def _normal_fft(self):
        """Normal operator via FFT-based circular convolution (circular padding only)."""

        batch_shape = self._shape.batch_shape
        in_grid_shape = self._shape.in_grid_shape
        out_grid_shape = self._shape.out_grid_shape
        return FFTConvolution(
            self.kernel,
            batch_shape,
            in_grid_shape,
            out_grid_shape,
        ).N


class FFTConvolution(Convolution):
    """Compute the convolution via the FFT."""

    def __init__(
        self,
        kernel,
        batch_shape,
        in_grid_shape,
        out_grid_shape,
    ):
        super().__init__(
            kernel,
            batch_shape,
            in_grid_shape,
            out_grid_shape,
            padding_mode="circular",
        )

        self._kernel_is_complex = torch.is_complex(self.kernel)
        self._fourier_kernel_cache = {}

    def _get_fourier_kernel(self, grid_size):
        """Compute fourier transform of kernel for a given input grid size."""
        if grid_size in self._fourier_kernel_cache:
            return self._fourier_kernel_cache[grid_size]

        if any(k > s for k, s in zip(self.kernel.shape, grid_size)):
            raise ValueError(
                f"Current FFT implementation requires all grid_sizes to be no smaller than kernel dims but got {grid_size} < {self.kernel.shape}"
            )

        # Pad kernel to input size, placing it at the origin
        pad_sizes = pad_to_size(self.kernel.shape, grid_size)
        kernel_padded = F.pad(self.kernel, pad_sizes)
        kernel_padded = torch.fft.ifftshift(
            kernel_padded, dim=tuple(range(-self.ndim, 0))
        )

        # FFT along spatial dimensions
        Fkernel = torch.fft.fftn(kernel_padded, dim=tuple(range(-self.ndim, 0)))

        self._fourier_kernel_cache[grid_size] = Fkernel
        return Fkernel

    @staticmethod
    def apply_fourier_kernel(linop, x, Fkernel):
        input_is_complex = torch.is_complex(x)
        x, batch_size = compress_batch_and_channel(x, linop.ndim)
        dims = tuple(range(-linop.ndim, 0))

        # Convolution theorem
        Fx = torch.fft.fftn(x, dim=dims)
        Fy = Fx * Fkernel
        y = torch.fft.ifftn(Fy, dim=dims)

        y = expand_batch_and_channel(y, batch_size)
        if not linop._kernel_is_complex and not input_is_complex:
            y = y.real
        return y

    @staticmethod
    def fn(linop, x):
        grid_size = x.shape[-linop.ndim :]
        Fkernel = linop._get_fourier_kernel(grid_size)
        return linop.apply_fourier_kernel(linop, x, Fkernel)

    @staticmethod
    def adj_fn(linop, x):
        grid_size = x.shape[-linop.ndim :]
        Fkernel = linop._get_fourier_kernel(grid_size).conj()
        return linop.apply_fourier_kernel(linop, x, Fkernel)

    @staticmethod
    def normal_fn(linop, x):
        grid_size = x.shape[-linop.ndim :]
        Fkernel = linop._get_fourier_kernel(grid_size).abs() ** 2
        return linop.apply_fourier_kernel(linop, x, Fkernel)

    def size(self, dim):
        # Can't determine size without knowing input
        return None


def cross_correlation(f: Tensor, g: Tensor) -> Tensor:
    """Compute the (non-circular) cross correlation f \\star g.

    Cross correlation (for complex values) is defined as

    (f \\star g)(t) = \int_{-\inf}^\inf \conj(f(t - \tau)) g(t) dt

    Note that it is not commutative.

    Parameters
    ----------
    f, g : Tensor
        The input tensors of shape [*dims]

    Returns
    -------
    Tensor
        The cross correlation f \\star g
    """
    if f.dim() != g.dim():
        raise ValueError(
            f"f and g must have the same dimension but got f.dim() {f.dim()} and g.dim() {g.dim()}"
        )
    if f.dim() > 3:
        raise ValueError(f"cross correlation only supported for input dimensions <= 3.")
    ndim = f.dim()
    f = f[None, None]
    g = g[None, None]
    conv_fn = (F.conv1d, F.conv2d, F.conv3d)[ndim - 1]  # really a correlation function
    full_pad = sum(((d - 1, d - 1) for d in reversed(g.shape)), start=tuple())
    g_padded = F.pad(g, pad=full_pad)
    out = conv_fn(g_padded, f.conj(), padding=0)  # no further padding required
    return out[0, 0]


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

    Parameters
    ----------
    x : Tensor
        Shape [B... C *dims]
        B... and C are optional
    ndim : int
        Number of dimensions to preserve at the end of x.

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
    """Return a tensor with converted batch dims to its original shape."""
    return x.reshape(batch_size + x.shape[2:])
