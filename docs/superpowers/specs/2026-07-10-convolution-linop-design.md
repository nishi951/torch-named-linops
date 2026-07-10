# Convolution Linop Design Spec

**Date:** 2026-07-10  
**Issue:** #171  
**Status:** Draft

## Overview

Implement a `Convolution` linop for 1D/2D/3D convolutions using PyTorch's native `F.conv_nd` and `F.conv_transpose_nd`. The linop supports named dimensions for batch, channels, and spatial grids, with multiple normal operator computation modes.

## Constructor Signature

```python
class Convolution(NamedLinop):
    def __init__(
        self,
        weight: Tensor,
        ndim: int,
        batch_shape: Optional[Shape] = None,
        in_grid_shape: Optional[Shape] = None,
        out_grid_shape: Optional[Shape] = None,
        stride: Union[int, tuple[int, ...]] = 1,
        padding: Union[str, int, tuple[int, ...]] = "zeros",
        **options,
    ):
```

### Parameters

- **weight**: Convolution kernel tensor of shape `(out_channels, in_channels, *kernel_size)`
- **ndim**: Number of spatial dimensions (1, 2, or 3)
- **batch_shape**: Named batch dimensions, defaults to `("...",)`
- **in_grid_shape**: Named input grid dimensions including channels, e.g., `(Dim("c_in"), Dim("x"), Dim("y"))` for 2D
- **out_grid_shape**: Named output grid dimensions including channels, e.g., `(Dim("c_out"), Dim("x'"), Dim("y'"))` for 2D
- **stride**: Convolution stride (int or tuple)
- **padding**: Padding mode - `"zeros"` or `"circular"`, or explicit padding size (int or tuple)
- **\*\*options**:
  - `normal_mode`: `None | "fft" | "conv"` - how to compute normal operator (default: `"conv"`)
  - `fft_dtype`: Data type for FFT-based normal computation

### Dimension Mapping

```
Input:  (*batch_shape, *in_grid_shape)
Output: (*batch_shape, *out_grid_shape)
```

For 2D convolution:
- Input: `(batch, c_in, x, y)`
- Output: `(batch, c_out, x', y')` where `x', y'` depend on stride and kernel size

The `weight` tensor has shape `(out_channels, in_channels, *kernel_size)` where:
- `out_channels` = size of first dimension in `out_grid_shape`
- `in_channels` = size of first dimension in `in_grid_shape`
- `kernel_size` = sizes of remaining dimensions in `in_grid_shape` (spatial dims)

## Implementation Details

### Forward Operation

```python
@staticmethod
def fn(linop, x):
    return F.conv_nd(x, linop.weight, stride=linop.stride, padding=linop.padding_int)
```

For circular padding, manually pad before convolution:
```python
if linop.padding == "circular":
    x = circular_pad(x, linop.padding_int)
    return F.conv_nd(x, linop.weight, stride=linop.stride, padding=0)
```

### Adjoint Operation

```python
@staticmethod
def adj_fn(linop, x):
    return F.conv_transpose_nd(x, linop.weight, stride=linop.stride, padding=linop.padding_int)
```

For circular padding, manually handle the adjoint of circular padding (fold-back operation).

### Normal Operator

Three modes controlled by `options["normal_mode"]`:

1. **`None` (default fallback)**: Just compose `adj_fn(fn(x))`
2. **`"conv"` (default)**: Composed convolution with autocorrelated kernel
   - Compute effective kernel: `k_eff = conv_transpose(k, k)` (autocorrelation)
   - Apply as: `Convolution(k_eff, ...)`
   - Works exactly for both `"zeros"` and `"circular"` padding
3. **`"fft"`**: FFT-based Toeplitz embedding
   - Compute via: `IFFT(FFT(x) * |FFT(kernel)|^2)`
   - **Only valid for circular padding** - raise error otherwise
   - Can be more efficient for large kernels

```python
def normal(self, inner=None):
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
```

## Supported Features

- **Dimensions**: 1D, 2D, 3D
- **Padding modes**: `"zeros"` (default), `"circular"`
- **Stride**: Arbitrary integer or tuple
- **Normal modes**: `None`, `"conv"` (default), `"fft"` (circular only)

## Testing

### Standard Linop Tests

Use `BaseNamedLinopTests` to verify:
- Forward operation correctness
- Adjoint property: `<Ax, y> = <x, A^H y>`
- Normal operator: `A^H A x`
- Composition with other linops
- Multi-GPU splitting (if applicable)

### Normal Mode Accuracy Tests

1. **Composed convolution (`"conv"`)**: Verify matches default `adj_fn(fn(x))` within tolerance
2. **FFT-based (`"fft"`)**: Verify matches default for circular padding within tolerance
3. **Error handling**: Verify `"fft"` mode raises error for non-circular padding

### Test Cases

```python
class TestConvolution(BaseNamedLinopTests):
    # Test 1D, 2D, 3D convolutions
    # Test with and without batch dimensions
    # Test different strides
    # Test both padding modes
    
    def test_normal_conv_mode(self):
        # Verify normal_mode="conv" matches default
        
    def test_normal_fft_mode(self):
        # Verify normal_mode="fft" matches default for circular padding
        
    def test_normal_fft_error_non_circular(self):
        # Verify error when using FFT mode with non-circular padding
```

## Future Extensions

- Additional padding modes: `"reflect"`, `"replicate"` (requires implementing adjoint-of-padding)
- Dilation support
- Groups support
- Spatially-varying kernels (kernel as linop)
