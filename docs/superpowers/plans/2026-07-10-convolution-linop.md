# Convolution Linop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Convolution linop for 1D/2D/3D convolutions with named dimensions and multiple normal operator modes.

**Architecture:** The Convolution linop wraps PyTorch's `F.conv1d/2d/3d` and `F.conv_transpose1d/2d/3d`, supporting named batch, channel, and spatial dimensions. It provides three normal operator modes: default composition, composed convolution with autocorrelated kernel, and FFT-based Toeplitz embedding (circular padding only).

**Tech Stack:** PyTorch, torch-named-linops infrastructure

## Global Constraints

- Support 1D, 2D, 3D convolutions only
- Padding modes: `"zeros"` (default), `"circular"` only
- Normal modes: `None` (fallback), `"conv"` (default), `"fft"` (circular only)
- Stride: arbitrary int or tuple
- Named dimensions: `batch_shape`, `in_grid_shape`, `out_grid_shape`
- Weight tensor shape: `(out_channels, in_channels, *kernel_size)`

---

### Task 1: Basic Convolution Linop (Forward and Adjoint)

**Files:**
- Create: `src/torchlinops/linops/convolution.py`
- Create: `tests/test_convolution.py`
- Modify: `src/torchlinops/linops/__init__.py`

**Interfaces:**
- Consumes: `NamedLinop`, `NamedShape`, `Shape`, `Tensor`
- Produces: `Convolution` class with `fn`, `adj_fn`, `split`, `size` methods

- [ ] **Step 1: Write failing test for basic 2D convolution**

```python
# tests/test_convolution.py
import pytest
import torch

from torchlinops import Dim
from torchlinops.linops.convolution import Convolution
from torchlinops.testing import BaseNamedLinopTests


class TestConvolution2D(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(4, 3, 3, 3)  # (out_c, in_c, kx, ky)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
        )
        x = torch.randn(3, 8, 8)  # (c_in, x, y)
        y = torch.randn(4, 8, 8)  # (c_out, x, y)
        return conv, x, y
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convolution.py::TestConvolution2D -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'torchlinops.linops.convolution'"

- [ ] **Step 3: Create Convolution linop skeleton**

```python
# src/torchlinops/linops/convolution.py
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
        self.weight = nn.Parameter(weight, requires_grad=False)
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
        self._batch_shape = tuple(ND.infer(d) for d in batch_shape)

        if in_grid_shape is None:
            # Infer from weight
            in_channels = weight.shape[1]
            spatial_in = tuple(weight.shape[2 + i] for i in range(ndim))
            in_grid_shape = (Dim(f"c_in"),) + tuple(Dim(f"x{i}") for i in range(ndim))
        self._in_grid_shape = tuple(ND.infer(d) for d in in_grid_shape)

        if out_grid_shape is None:
            out_channels = weight.shape[0]
            out_grid_shape = (Dim(f"c_out"),) + tuple(
                ND.infer(self._in_grid_shape[i + 1]).next_unused([self._in_grid_shape])
                for i in range(ndim)
            )
        self._out_grid_shape = tuple(ND.infer(d) for d in out_grid_shape)

        # Build ishape and oshape
        ishape = self._batch_shape + self._in_grid_shape
        oshape = self._batch_shape + self._out_grid_shape

        super().__init__(NS(ishape, oshape))

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
        conv_t_fn = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[linop.ndim - 1]
        if linop.padding == "circular":
            # Adjoint of circular conv: transpose conv then fold
            return conv_t_fn(x, linop.weight, stride=linop.stride, padding=0)
        elif linop.padding == "zeros":
            padding = tuple(k // 2 for k in linop.weight.shape[2:])
            return conv_t_fn(x, linop.weight, stride=linop.stride, padding=padding)
        else:
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
            return self.weight.shape[1] if idx == len(self._batch_shape) else None
        if dim in self.oshape:
            idx = self.oshape.index(dim)
            return self.weight.shape[0] if idx == len(self._batch_shape) else None
        return None
```

- [ ] **Step 4: Export Convolution in __init__.py**

```python
# src/torchlinops/linops/__init__.py
# Add this line:
from .convolution import *
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_convolution.py::TestConvolution2D -v`
Expected: PASS (all BaseNamedLinopTests should pass)

- [ ] **Step 6: Commit**

```bash
git add src/torchlinops/linops/convolution.py src/torchlinops/linops/__init__.py tests/test_convolution.py
git commit -m "feat: add basic Convolution linop with forward and adjoint"
```

---

### Task 2: Forward and Adjoint Correctness Tests

**Files:**
- Modify: `tests/test_convolution.py`

**Interfaces:**
- Consumes: `Convolution` class
- Produces: Tests verifying forward/adjoint against PyTorch's native operations

- [ ] **Step 1: Write correctness tests for forward operation**

```python
# tests/test_convolution.py - add to existing file

class TestConvolutionCorrectness:
    """Verify Convolution matches PyTorch's native operations."""

    def test_forward_matches_pytorch(self):
        """Forward should match F.conv2d."""
        import torch.nn.functional as F

        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="zeros",
        )
        x = torch.randn(3, 8, 8)

        # Our implementation
        result = conv(x)

        # PyTorch native
        padding = tuple(k // 2 for k in weight.shape[2:])
        expected = F.conv2d(x.unsqueeze(0), weight, padding=padding).squeeze(0)

        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_adjoint_matches_pytorch(self):
        """Adjoint should match F.conv_transpose2d."""
        import torch.nn.functional as F

        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="zeros",
        )
        y = torch.randn(4, 8, 8)

        # Our implementation
        result = conv.H(y)

        # PyTorch native
        padding = tuple(k // 2 for k in weight.shape[2:])
        expected = F.conv_transpose2d(y.unsqueeze(0), weight, padding=padding).squeeze(0)

        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_forward_1d_matches_pytorch(self):
        """1D forward should match F.conv1d."""
        import torch.nn.functional as F

        weight = torch.randn(4, 3, 5)
        conv = Convolution(
            weight,
            ndim=1,
            in_grid_shape=(Dim("c_in"), Dim("x")),
            out_grid_shape=(Dim("c_out"), Dim("x")),
        )
        x = torch.randn(3, 16)

        result = conv(x)
        padding = tuple(k // 2 for k in weight.shape[2:])
        expected = F.conv1d(x.unsqueeze(0), weight, padding=padding).squeeze(0)

        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_forward_3d_matches_pytorch(self):
        """3D forward should match F.conv3d."""
        import torch.nn.functional as F

        weight = torch.randn(4, 3, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=3,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y"), Dim("z")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y"), Dim("z")),
        )
        x = torch.randn(3, 8, 8, 8)

        result = conv(x)
        padding = tuple(k // 2 for k in weight.shape[2:])
        expected = F.conv3d(x.unsqueeze(0), weight, padding=padding).squeeze(0)

        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_forward_with_stride(self):
        """Forward with stride should match F.conv2d."""
        import torch.nn.functional as F

        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x2"), Dim("y2")),
            stride=2,
        )
        x = torch.randn(3, 8, 8)

        result = conv(x)
        padding = tuple(k // 2 for k in weight.shape[2:])
        expected = F.conv2d(x.unsqueeze(0), weight, stride=2, padding=padding).squeeze(0)

        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run correctness tests**

Run: `pytest tests/test_convolution.py::TestConvolutionCorrectness -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_convolution.py
git commit -m "test: add correctness tests against PyTorch native operations"
```

---

### Task 3: Normal Operator Modes

**Files:**
- Modify: `src/torchlinops/linops/convolution.py`
- Modify: `tests/test_convolution.py`

**Interfaces:**
- Consumes: `Convolution` class
- Produces: `normal()` method with three modes: None, "conv", "fft"

- [ ] **Step 1: Write failing tests for normal modes**

```python
# tests/test_convolution.py - add to existing file

class TestConvolutionNormalModes:
    """Test different normal operator modes."""

    def test_normal_conv_mode(self):
        """Composed convolution mode should match default."""
        weight = torch.randn(4, 3, 3, 3)
        conv_default = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="zeros",
        )
        conv_conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="zeros",
            normal_mode="conv",
        )
        x = torch.randn(3, 8, 8)
        assert torch.allclose(conv_default.N(x), conv_conv.N(x), rtol=1e-4)

    def test_normal_fft_mode_circular(self):
        """FFT mode should work for circular padding."""
        weight = torch.randn(4, 3, 3, 3)
        conv_fft = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="circular",
            normal_mode="fft",
        )
        x = torch.randn(3, 8, 8)
        result = conv_fft.N(x)
        assert result.shape == x.shape

    def test_normal_fft_mode_error_non_circular(self):
        """FFT mode should raise error for non-circular padding."""
        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="zeros",
            normal_mode="fft",
        )
        x = torch.randn(3, 8, 8)
        with pytest.raises(ValueError, match="FFT normal mode only supports circular padding"):
            conv.N(x)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_convolution.py::TestConvolutionNormalModes -v`
Expected: FAIL (normal_mode not implemented yet)

- [ ] **Step 3: Implement normal operator modes**

```python
# src/torchlinops/linops/convolution.py - add normal method to Convolution class

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
        """Normal operator via composed convolution with autocorrelated kernel."""
        # For a convolution A with kernel k, the normal operator A^H A is
        # a convolution with effective kernel:
        #   k_eff[i0, i1, shift] = sum_o sum_s k[o, i0, s] * k[o, i1, s + shift]
        # This is a cross-correlation over spatial shifts.

        weight = self.weight
        out_c, in_c = weight.shape[0], weight.shape[1]
        kernel_size = weight.shape[2:]

        # Compute autocorrelation using conv_transpose
        # Reshape weight to (out_c * in_c, 1, *kernel_size)
        k_flat = weight.reshape(out_c * in_c, 1, *kernel_size)

        # Use conv_transpose to compute autocorrelation for each (o, i) pair
        # Groups = out_c * in_c means each channel is processed independently
        # Result shape: (out_c * in_c, 1, *2*kernel_size - 1)
        conv_t_fn = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[self.ndim - 1]
        k_auto = conv_t_fn(k_flat, k_flat, groups=out_c * in_c)

        # Reshape to (out_c, in_c, in_c, *2*kernel_size - 1)
        kernel_eff_size = tuple(2 * k - 1 for k in kernel_size)
        k_auto = k_auto.reshape(out_c, in_c, in_c, *kernel_eff_size)

        # Sum over output channels to get (in_c, in_c, *kernel_eff_size)
        k_eff = k_auto.sum(dim=0)

        # Create new convolution with effective kernel
        normal_conv = Convolution(
            k_eff,
            ndim=self.ndim,
            batch_shape=self._batch_shape,
            in_grid_shape=self._in_grid_shape,
            out_grid_shape=self._in_grid_shape,  # Output has same shape as input
            stride=1,
            padding=self.padding,
            normal_mode=None,  # Avoid recursion
        )
        return normal_conv

    def _normal_fft(self):
        """Normal operator via FFT-based Toeplitz embedding (circular padding only)."""
        # For circular convolution, the normal operator is:
        # A^H A x = IFFT(|FFT(k)|^2 * FFT(x))
        # where k is the kernel padded to input size

        weight = self.weight
        in_c = weight.shape[1]
        kernel_size = weight.shape[2:]

        # Get input spatial size from ishape
        spatial_dims = self._in_grid_shape[1:]  # Skip channel dim
        spatial_size = tuple(self.size(dim) for dim in spatial_dims)

        # For each input channel, compute |FFT(k)|^2
        # Weight shape: (out_c, in_c, *kernel_size)
        # We need to compute the effective kernel for each (in_c, in_c) pair

        # Pad kernel to input size
        pad_sizes = []
        for i, (k, n) in enumerate(zip(kernel_size, spatial_size)):
            pad_before = (n - k) // 2
            pad_after = n - k - pad_before
            pad_sizes.extend([pad_before, pad_after])

        # Pad each output channel's kernel
        # Shape: (out_c, in_c, *spatial_size)
        weight_padded = F.pad(weight, tuple(reversed(pad_sizes)))

        # FFT along spatial dimensions
        spatial_dims_fft = tuple(range(-self.ndim, 0))
        weight_fft = torch.fft.fftn(weight_padded, dim=spatial_dims_fft)

        # Compute |FFT(k)|^2 for each (in_c_out, in_c_in) pair
        # Sum over out_c dimension
        # Result shape: (in_c, in_c, *spatial_size)
        psd = (weight_fft.conj() * weight_fft).sum(dim=0)

        # Create a diagonal-like operator that applies IFFT(|FFT(k)|^2 * FFT(x))
        # This is a convolution with the IFFT of the PSD
        effective_kernel = torch.fft.ifftn(psd, dim=spatial_dims_fft).real

        # Create convolution with effective kernel
        normal_conv = Convolution(
            effective_kernel,
            ndim=self.ndim,
            batch_shape=self._batch_shape,
            in_grid_shape=self._in_grid_shape,
            out_grid_shape=self._in_grid_shape,
            stride=1,
            padding="circular",
            normal_mode=None,
        )
        return normal_conv
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_convolution.py::TestConvolutionNormalModes -v`
Expected: PASS

- [ ] **Step 5: Run all convolution tests**

Run: `pytest tests/test_convolution.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/torchlinops/linops/convolution.py tests/test_convolution.py
git commit -m "feat: add normal operator modes (None, conv, fft)"
```

---

### Task 4: Additional Test Coverage

**Files:**
- Modify: `tests/test_convolution.py`

**Interfaces:**
- Consumes: `Convolution` class
- Produces: Tests for 1D, 3D, batch dimensions, stride

- [ ] **Step 1: Add tests for 1D and 3D convolutions**

```python
# tests/test_convolution.py - add to existing file

class TestConvolution1D(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(4, 3, 5)  # (out_c, in_c, kx)
        conv = Convolution(
            weight,
            ndim=1,
            in_grid_shape=(Dim("c_in"), Dim("x")),
            out_grid_shape=(Dim("c_out"), Dim("x")),
        )
        x = torch.randn(3, 16)  # (c_in, x)
        y = torch.randn(4, 16)  # (c_out, x)
        return conv, x, y


class TestConvolution3D(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(4, 3, 3, 3, 3)  # (out_c, in_c, kx, ky, kz)
        conv = Convolution(
            weight,
            ndim=3,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y"), Dim("z")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y"), Dim("z")),
        )
        x = torch.randn(3, 8, 8, 8)  # (c_in, x, y, z)
        y = torch.randn(4, 8, 8, 8)  # (c_out, x, y, z)
        return conv, x, y
```

- [ ] **Step 2: Add tests for batch dimensions**

```python
# tests/test_convolution.py - add to existing file

class TestConvolutionBatched(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            batch_shape=(Dim("batch"),),
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
        )
        x = torch.randn(2, 3, 8, 8)  # (batch, c_in, x, y)
        y = torch.randn(2, 4, 8, 8)  # (batch, c_out, x, y)
        return conv, x, y
```

- [ ] **Step 3: Add tests for stride**

```python
# tests/test_convolution.py - add to existing file

class TestConvolutionStride(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x2"), Dim("y2")),
            stride=2,
        )
        x = torch.randn(3, 8, 8)  # (c_in, x, y)
        y = torch.randn(4, 4, 4)  # (c_out, x/2, y/2)
        return conv, x, y
```

- [ ] **Step 4: Add tests for circular padding**

```python
# tests/test_convolution.py - add to existing file

class TestConvolutionCircular(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="circular",
        )
        x = torch.randn(3, 8, 8)
        y = torch.randn(4, 8, 8)
        return conv, x, y
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/test_convolution.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_convolution.py
git commit -m "test: add comprehensive convolution tests"
```

---

### Task 5: Normal Mode Accuracy Tests

**Files:**
- Modify: `tests/test_convolution.py`

**Interfaces:**
- Consumes: `Convolution` class with normal modes
- Produces: Tests verifying normal modes match default composition

- [ ] **Step 1: Add normal mode accuracy tests**

```python
# tests/test_convolution.py - add to existing file

class TestNormalModeAccuracy:
    """Verify normal operator modes match default composition."""

    def test_normal_conv_matches_default_zeros(self):
        """Composed convolution normal should match default for zeros padding."""
        weight = torch.randn(4, 3, 3, 3)
        conv_default = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="zeros",
            normal_mode=None,
        )
        conv_conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="zeros",
            normal_mode="conv",
        )
        x = torch.randn(3, 8, 8)
        assert torch.allclose(conv_default.N(x), conv_conv.N(x), rtol=1e-4, atol=1e-4)

    def test_normal_conv_matches_default_circular(self):
        """Composed convolution normal should match default for circular padding."""
        weight = torch.randn(4, 3, 3, 3)
        conv_default = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="circular",
            normal_mode=None,
        )
        conv_conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="circular",
            normal_mode="conv",
        )
        x = torch.randn(3, 8, 8)
        assert torch.allclose(conv_default.N(x), conv_conv.N(x), rtol=1e-4, atol=1e-4)

    def test_normal_fft_matches_default_circular(self):
        """FFT normal should match default for circular padding."""
        weight = torch.randn(4, 3, 3, 3)
        conv_default = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="circular",
            normal_mode=None,
        )
        conv_fft = Convolution(
            weight,
            ndim=2,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="circular",
            normal_mode="fft",
        )
        x = torch.randn(3, 8, 8)
        assert torch.allclose(conv_default.N(x), conv_fft.N(x), rtol=1e-4, atol=1e-4)

    def test_normal_1d_accuracy(self):
        """Normal modes should work for 1D convolution."""
        weight = torch.randn(4, 3, 5)
        conv_default = Convolution(
            weight,
            ndim=1,
            in_grid_shape=(Dim("c_in"), Dim("x")),
            out_grid_shape=(Dim("c_out"), Dim("x")),
            padding="zeros",
            normal_mode=None,
        )
        conv_conv = Convolution(
            weight,
            ndim=1,
            in_grid_shape=(Dim("c_in"), Dim("x")),
            out_grid_shape=(Dim("c_out"), Dim("x")),
            padding="zeros",
            normal_mode="conv",
        )
        x = torch.randn(3, 16)
        assert torch.allclose(conv_default.N(x), conv_conv.N(x), rtol=1e-4, atol=1e-4)

    def test_normal_3d_accuracy(self):
        """Normal modes should work for 3D convolution."""
        weight = torch.randn(2, 2, 3, 3, 3)
        conv_default = Convolution(
            weight,
            ndim=3,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y"), Dim("z")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y"), Dim("z")),
            padding="zeros",
            normal_mode=None,
        )
        conv_conv = Convolution(
            weight,
            ndim=3,
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y"), Dim("z")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y"), Dim("z")),
            padding="zeros",
            normal_mode="conv",
        )
        x = torch.randn(2, 8, 8, 8)
        assert torch.allclose(conv_default.N(x), conv_conv.N(x), rtol=1e-4, atol=1e-4)

    def test_normal_with_batch_dims(self):
        """Normal modes should work with batch dimensions."""
        weight = torch.randn(4, 3, 3, 3)
        conv_default = Convolution(
            weight,
            ndim=2,
            batch_shape=(Dim("batch"),),
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="zeros",
            normal_mode=None,
        )
        conv_conv = Convolution(
            weight,
            ndim=2,
            batch_shape=(Dim("batch"),),
            in_grid_shape=(Dim("c_in"), Dim("x"), Dim("y")),
            out_grid_shape=(Dim("c_out"), Dim("x"), Dim("y")),
            padding="zeros",
            normal_mode="conv",
        )
        x = torch.randn(2, 3, 8, 8)
        assert torch.allclose(conv_default.N(x), conv_conv.N(x), rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run normal mode accuracy tests**

Run: `pytest tests/test_convolution.py::TestNormalModeAccuracy -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_convolution.py
git commit -m "test: add normal mode accuracy tests"
```

---

### Task 6: Final Verification and Cleanup

**Files:**
- Modify: `tests/test_convolution.py` (if needed)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/test_convolution.py -v`
Expected: All tests PASS

- [ ] **Step 2: Check for lint/type errors**

Run: `python -m py_compile src/torchlinops/linops/convolution.py`
Expected: No errors

- [ ] **Step 3: Verify imports work**

Run: `python -c "from torchlinops import Convolution; print('Import successful')"`
Expected: "Import successful"

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: final cleanup for convolution linop"
```
