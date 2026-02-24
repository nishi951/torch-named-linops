# Creating Custom Linops

This guide explains the conventions and requirements for creating your own `NamedLinop` subclass.

For a complete walkthrough with runnable code, see the [Custom Linops Tutorial](../tutorials/custom_linop.md).

## Minimal Example

```python
import torch
from torch import Tensor
from torchlinops import NamedLinop, Dim
from torchlinops.nameddim import NamedShape as NS


class Scale(NamedLinop):
    """Element-wise scaling by a scalar value."""

    def __init__(self, scale: float, ishape, oshape):
        super().__init__(NS(ishape, oshape))
        self.scale = scale

    @staticmethod
    def fn(x: Tensor, /, scale: float) -> Tensor:
        return scale * x

    @staticmethod
    def adj_fn(x: Tensor, /, scale: float) -> Tensor:
        # For real scalars, adjoint is the same as forward
        return scale * x

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x, self.scale)

    def adjoint(self):
        A_H = type(self)(self.scale, ishape=self._shape.oshape, oshape=self._shape.ishape)
        return A_H


# Usage
A = Scale(3.0, ishape=Dim("N"), oshape=Dim("N"))
x = torch.randn(5)
y = A(x)       # 3 * x
z = A.H(y)     # 3 * y (adjoint)
```

## Requirements

### 1. Call `super().__init__()` with a `NamedShape`

The constructor must initialize the linop's shape by passing a `NamedShape` to the parent:

```python
def __init__(self, ..., ishape, oshape):
    super().__init__(NS(ishape, oshape))
```

`NS` (aliased from `NamedShape`) pairs the input dimension names (`ishape`) with the output dimension names (`oshape`). For a diagonal operator where input and output have the same dimensions, you can use the shortcut:

```python
super().__init__(NS(ioshape=Dim("N")))  # ishape == oshape == ("N",)
```

### 2. Use `@staticmethod` for `fn()`, `adj_fn()`, and `normal_fn()`

These methods should be static so that the adjoint and normal operators can swap functions without being bound to a specific instance:

```python
@staticmethod
def fn(x: Tensor, /, weight: Tensor) -> Tensor:
    """Forward operation. First argument must be the input tensor."""
    return weight @ x

@staticmethod
def adj_fn(x: Tensor, /, weight: Tensor) -> Tensor:
    """Adjoint operation."""
    return weight.conj().T @ x

@staticmethod
def normal_fn(x: Tensor, /, weight: Tensor) -> Tensor:
    """Optional: optimized normal operation (A^H A)."""
    return weight.conj().T @ (weight @ x)
```

The first positional argument is always the input tensor `x`. Additional arguments are operator-specific data (weights, parameters, etc.) passed from `forward()`.

### 3. Use `type(self)` in `split_forward()`

If you override `split_forward()` for multi-GPU splitting, construct new instances using `type(self)(...)` instead of `copy.deepcopy(self)`. This avoids unnecessary tensor copies and ensures proper parameter isolation:

```python
def split_forward(self, ibatch, obatch):
    # Slice your data for this batch
    sub_weight = self.weight[obatch, ibatch]
    # Create a new instance (not a copy)
    return type(self)(sub_weight, ishape=..., oshape=...)
```

See [Copying Linops](../explanations/copying_linops.md) and [Multi-GPU Splitting](../explanations/multi_gpu.md) for more details on why this matters.

## Testing Your Linop

Use `is_adjoint` to verify that your forward and adjoint are mathematically consistent:

```python
from torchlinops.utils import is_adjoint

A = MyLinop(...)
x = torch.randn(...)  # Input-shaped random vector
y = torch.randn(...)  # Output-shaped random vector

# Checks that <Ax, y> == <x, A^H y>
assert is_adjoint(A, x, y), "Adjoint test failed!"
```

This verifies the identity $\langle Ax, y \rangle = \langle x, A^H y \rangle$ using random vectors, which should hold to numerical precision for a correct adjoint.
