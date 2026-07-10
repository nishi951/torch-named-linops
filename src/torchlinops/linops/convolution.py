from copy import copy
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..nameddim import NamedDimension as ND, NamedShape as NS, Shape
from .namedlinop import NamedLinop

__all__ = ["Convolution"]


class Convolution(NamedLinop):
    """N-dimensional convolution as a named linear operator.

    Supports 1D, 2D, and 3D convolutions with named batch, channel, and
    spatial dimensions. Uses PyTorch's native F.conv1d/2d/3d and F.conv_transpose1d/2d/3d.

    Attributes
    ----------
    ndim : int
        Number of spatial dimensions (1, 2, or 3).
    weight : Tensor
        Convolution kernel of shape (out_channels, in_channels, *kernel_size).
    stride : tuple
        Convolution stride.
    padding : str or int
        Padding mode or size.
    """

    def __init__(
        self,
        weight: Tensor,
        ndim: int,
        batch_shape: Optional[Shape] = None,
        in_grid_shape: Optional[Shape] = None,
        out_grid_shape: Optional[Shape] = None,
        stride: Union[int, tuple] = 1,
        padding: Union[str, int, tuple] = "zeros",
        **options,
    ):
        """
        Parameters
        ----------
        weight : Tensor
            Convolution kernel of shape (out_channels, in_channels, *kernel_size).
        ndim : int
            Number of spatial dimensions (1, 2, or 3).
        batch_shape : Shape, optional
            Named batch dimensions. Defaults to ("...",).
        in_grid_shape : Shape, optional
            Named input grid dimensions including channels.
            First dim is input channels, remaining are spatial.
        out_grid_shape : Shape, optional
            Named output grid dimensions including channels.
            First dim is output channels, remaining are spatial.
        stride : int or tuple, default 1
            Convolution stride.
        padding : str, int, or tuple, default "zeros"
            Padding mode ("zeros", "circular") or explicit padding size.
        **options : dict
            Additional options (normal_mode, fft_dtype, etc.)
        """
        if ndim not in (1, 2, 3):
            raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")

        self.ndim = ndim
        self.options = options

        # Parse stride
        if isinstance(stride, int):
            self.stride = (stride,) * ndim
        else:
            self.stride = tuple(stride)

        # Parse padding
        self.padding = padding

        # Set up shapes
        if batch_shape is None:
            batch_shape = ("...",)
        self._batch_shape = tuple(batch_shape)

        if in_grid_shape is None:
            # Infer from weight
            in_channels = weight.shape[1]
            spatial_in = tuple(weight.shape[2 + i] for i in range(ndim))
            in_grid_shape = ("c_in",) + tuple(f"x{i}" for i in range(ndim))
        self._in_grid_shape = tuple(in_grid_shape)

        if out_grid_shape is None:
            out_channels = weight.shape[0]
            out_grid_shape = ("c_out",) + tuple(
                f"{self._in_grid_shape[i + 1]}_out" for i in range(ndim)
            )
        self._out_grid_shape = tuple(out_grid_shape)

        # Build ishape and oshape
        ishape = self._batch_shape + self._in_grid_shape
        oshape = self._batch_shape + self._out_grid_shape

        super().__init__(NS(ishape, oshape))

        self.weight = nn.Parameter(weight, requires_grad=False)

    @staticmethod
    def fn(linop, x):
        """Forward convolution."""
        conv_fn = (F.conv1d, F.conv2d, F.conv3d)[linop.ndim - 1]
        if linop.padding == "circular":
            # Manual circular padding
            pad_sizes = tuple(k // 2 for k in linop.weight.shape[2:])
            x = F.pad(x, tuple(p for ps in reversed(pad_sizes) for p in (ps, ps)), mode="circular")
            return conv_fn(x, linop.weight, stride=linop.stride, padding=0)
        elif linop.padding == "zeros":
            padding = tuple(k // 2 for k in linop.weight.shape[2:])
            return conv_fn(x, linop.weight, stride=linop.stride, padding=padding)
        else:
            # Explicit padding size
            return conv_fn(x, linop.weight, stride=linop.stride, padding=linop.padding)

    @staticmethod
    def adj_fn(linop, x):
        """Adjoint (transpose) convolution."""
        if linop.padding == "circular":
            # Adjoint of circular convolution is circular correlation
            # which is circular convolution with flipped and conjugated kernel
            # Flip the spatial dimensions of the kernel
            weight_flipped = linop.weight
            for i in range(linop.ndim):
                weight_flipped = torch.flip(weight_flipped, dims=[2 + i])
            weight_flipped = weight_flipped.conj()
            
            # Now do circular convolution with the flipped kernel
            # But we need to swap in_channels and out_channels
            # weight shape: (out_c, in_c, *kernel_size)
            # We need: (in_c, out_c, *kernel_size)
            weight_transposed = weight_flipped.transpose(0, 1)
            
            conv_fn = (F.conv1d, F.conv2d, F.conv3d)[linop.ndim - 1]
            pad_sizes = tuple(k // 2 for k in weight_transposed.shape[2:])
            x_padded = F.pad(x, tuple(p for ps in reversed(pad_sizes) for p in (ps, ps)), mode="circular")
            return conv_fn(x_padded, weight_transposed, stride=linop.stride, padding=0)
        elif linop.padding == "zeros":
            conv_t_fn = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[linop.ndim - 1]
            padding = tuple(k // 2 for k in linop.weight.shape[2:])
            return conv_t_fn(x, linop.weight, stride=linop.stride, padding=padding)
        else:
            conv_t_fn = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[linop.ndim - 1]
            return conv_t_fn(x, linop.weight, stride=linop.stride, padding=linop.padding)

    @staticmethod
    def split(linop, tile):
        """Split along batch dimensions."""
        new = copy(linop)
        return new

    def size(self, dim):
        """Return the size of a named dimension."""
        if dim in self.ishape:
            idx = self.ishape.index(dim)
            batch_offset = len(self._batch_shape)
            if idx == batch_offset:
                # Channel dimension
                return self.weight.shape[1]
            elif idx > batch_offset:
                # Spatial dimension
                spatial_idx = idx - batch_offset - 1
                return self.weight.shape[2 + spatial_idx]
        if dim in self.oshape:
            idx = self.oshape.index(dim)
            batch_offset = len(self._batch_shape)
            if idx == batch_offset:
                # Channel dimension
                return self.weight.shape[0]
        return None

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
        - "conv": Composed convolution with autocorrelated kernel (default)
        - "fft": FFT-based Toeplitz embedding (circular padding only)
        """
        if inner is not None:
            return super().normal(inner)

        mode = self.options.get("normal_mode", "conv")

        if mode is None:
            return super().normal()

        if mode == "fft":
            if self.padding != "circular":
                raise ValueError("FFT normal mode only supports circular padding")
            return self._normal_fft()

        if mode == "conv":
            return self._normal_conv()

        raise ValueError(f"Unknown normal_mode: {mode}")

    def _normal_conv(self):
        """Normal operator via composed convolution with autocorrelated kernel.
        
        This only works correctly for circular padding. For zero padding,
        the normal operator is not a simple convolution due to boundary effects.
        """
        if self.padding != "circular":
            # For zero padding, fall back to default composition
            return super().normal()
        
        # For circular padding, the normal operator is a convolution with
        # the autocorrelated kernel.
        weight = self.weight
        out_c, in_c = weight.shape[0], weight.shape[1]
        kernel_size = weight.shape[2:]

        # Compute autocorrelation using FFT
        # For circular autocorrelation, we can use the kernel as-is
        spatial_dims = tuple(range(-self.ndim, 0))
        k_fft = torch.fft.fftn(weight, dim=spatial_dims)

        # Cross-spectral density: sum over output channels
        # k_fft has shape (out_c, in_c, *freq_size)
        # Result: (in_c, in_c, *freq_size)
        spatial_indices = "abc"[:self.ndim]
        einsum_str = f"oi{spatial_indices},oj{spatial_indices}->ij{spatial_indices}"
        psd = torch.einsum(einsum_str, k_fft.conj(), k_fft)

        # IFFT to get autocorrelation kernel
        k_eff = torch.fft.ifftn(psd, dim=spatial_dims).real

        # Create new convolution with effective kernel
        normal_conv = Convolution(
            k_eff,
            ndim=self.ndim,
            batch_shape=self._batch_shape,
            in_grid_shape=self._in_grid_shape,
            out_grid_shape=self._in_grid_shape,
            stride=1,
            padding="circular",
            normal_mode=None,
        )
        return normal_conv

    def _normal_fft(self):
        """Normal operator via FFT-based circular convolution (circular padding only)."""
        # For circular convolution, the normal operator is:
        # A^H A x = IFFT(|FFT(k)|^2 * FFT(x))
        # We implement this as a custom forward function that uses FFT directly.
        # The PSD is computed dynamically based on the input spatial size.

        weight = self.weight
        out_c, in_c = weight.shape[0], weight.shape[1]
        kernel_size = weight.shape[2:]

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
                spatial_indices = "abc"[:self.ndim]
                einsum_str = f"oi{spatial_indices},oj{spatial_indices}->ij{spatial_indices}"
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
                spatial_size = x.shape[actual_batch_dims + 1:]  # Skip batch and channel dims

                psd = linop._get_psd(spatial_size)

                spatial_dims = tuple(range(-linop.ndim, 0))
                x_fft = torch.fft.fftn(x, dim=spatial_dims)

                # Multiply by PSD: result[i] = sum_j psd[i,j] * x_fft[j]
                spatial_indices = "abc"[:linop.ndim]
                einsum_str = f"ij{spatial_indices},j{spatial_indices}->i{spatial_indices}"
                result_fft = torch.einsum(einsum_str, psd, x_fft)

                return torch.fft.ifftn(result_fft, dim=spatial_dims).real

            @staticmethod
            def adj_fn(linop, x):
                # PSD is Hermitian, so adjoint is the same
                return FFTNormalLinop.fn(linop, x)

            @staticmethod
            def split(linop, tile):
                from copy import copy
                return copy(linop)

            def size(self, dim):
                # Can't determine size without knowing input
                return None

        return FFTNormalLinop(weight, kernel_size, self._batch_shape, self._in_grid_shape, self.ndim)
