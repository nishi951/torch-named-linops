# SimpleLinop: Ad-hoc Linop Creation API

**Issue:** #173  
**Date:** 2026-07-16  
**Status:** Design Complete

## Overview

Add a `SimpleLinop` class that enables creating ad-hoc linear operators with minimal ceremony, without requiring users to subclass `NamedLinop`.

## API

```python
SimpleLinop(
    forward: Callable[[Tensor], Tensor],
    adjoint: Callable[[Tensor], Tensor],
    normal: Optional[Callable[[Tensor], Tensor]] = None,
    ishape: Shape = ("...",),
    oshape: Optional[Shape] = None,  # defaults to ishape
    name: Optional[str] = None,
)
```

### Parameters

- **forward** (required): Plain callable `(x) -> y` implementing the forward operation
- **adjoint** (required): Plain callable `(x) -> y` implementing the adjoint operation
- **normal** (optional): Plain callable `(x) -> y` implementing the normal operation. If `None`, the base class default (`adj_fn(fn(x))`) is used automatically
- **ishape** (optional): Input named dimensions, defaults to `("...",)`
- **oshape** (optional): Output named dimensions, defaults to `ishape`
- **name** (optional): Display name for the linop

## Implementation

### Class Structure

`SimpleLinop` subclasses `NamedLinop` directly and stores user callables as instance attributes:

```python
class SimpleLinop(NamedLinop):
    def __init__(
        self,
        forward: Callable[[Tensor], Tensor],
        adjoint: Callable[[Tensor], Tensor],
        normal: Optional[Callable[[Tensor], Tensor]] = None,
        ishape: Shape = ("...",),
        oshape: Optional[Shape] = None,
        name: Optional[str] = None,
    ):
        # Store callables
        self._forward = forward
        self._adjoint = adjoint
        self._normal = normal
        
        # Initialize base class
        super().__init__(NamedShape(ishape, oshape), name=name)
```

### Static Method Wrappers

The user's plain callables are wrapped into the base class's staticmethod signature:

```python
@staticmethod
def fn(linop, x, /, context=None):
    return linop._forward(x)

@staticmethod
def adj_fn(linop, x, /, context=None):
    return linop._adjoint(x)

@staticmethod
def normal_fn(linop, x, /, context=None):
    if linop._normal is not None:
        return linop._normal(x)
    # Fall back to base class default
    return super().normal_fn(linop, x, context=context)
```

### Adjoint Behavior

Override `adjoint()` to return a new `SimpleLinop` with swapped callables:

```python
def adjoint(self):
    return SimpleLinop(
        forward=self._adjoint,
        adjoint=self._forward,
        normal=None,  # let base class compute it from swapped callables
        ishape=self.oshape,
        oshape=self.ishape,
        name=self.name + ".H" if self.name else None,
    )
```

**Important:** The user's custom `normal` callable is NOT passed to the adjoint's constructor. If the user provided `normal(x) = adjoint(forward(x))`, then for `A.H`, the normal should be `forward(adjoint(x))`, which is different. By passing `normal=None`, we let the base class compute the correct normal from the swapped callables.

### Normal Behavior

- If user provided a custom `normal` callable, `normal_fn` uses it directly
- Otherwise, inherits the base class default (`adj_fn(fn(x))`)
- The normal operator (`.N`) is self-adjoint, so its adjoint is itself

## Features and Limitations

### Supported

- Forward operation via `A(x)`
- Adjoint via `A.H` or `A.H(x)`
- Normal via `A.N` or `A.N(x)`
- Composition with other linops via `@`
- Addition/subtraction via `+`/`-`
- Scalar multiplication via `*`
- Custom named shapes
- Optional display name

### Not Supported (by design)

- **Splitting**: Inherits base class default (`copy(self)`), no custom splitting logic
- **Size queries**: Inherits base class default (returns `None`)
- **Pickling with lambdas**: Works only if user's callables are picklable (top-level functions). Lambdas won't pickle — this is expected and acceptable for ad-hoc use

## Usage Example

```python
import torch
from torchlinops import SimpleLinop

# Create a simple scaling operator
A = SimpleLinop(
    forward=lambda x: 2 * x,
    adjoint=lambda y: 2 * y,
    normal=lambda x: 4 * x,
    ishape=("Nx", "Ny"),
    name="ScaleBy2"
)

x = torch.randn(10, 20)
y = A(x)        # Forward: 2 * x
z = A.H(y)      # Adjoint: 2 * y
w = A.N(x)      # Normal: 4 * x

# Compose with other linops
from torchlinops import FFT
B = FFT(Dim("Nx")) @ A
```

## Testing

Tests should verify:

1. **Basic operations**: forward, adjoint, normal produce correct results
2. **Adjoint creation**: `.H` creates a new `SimpleLinop` with swapped callables
3. **Normal creation**: `.N` creates correct normal operator
4. **Custom normal**: User-provided normal callable is used correctly
5. **Composition**: Works with `@`, `+`, `-`, `*` operators
6. **Custom shapes**: `ishape` and `oshape` are set correctly
7. **Name handling**: Display name works correctly, including for adjoint (`.H` suffix)

## Files to Modify

- `src/torchlinops/linops/simple.py` (new file): Implement `SimpleLinop`
- `src/torchlinops/linops/__init__.py`: Export `SimpleLinop`
- `tests/test_simple_linop.py` (new file): Test suite

## Design Decisions

### Why subclass `NamedLinop`?

- Inherits all base class features (`.H`, `.N`, operators, etc.) for free
- Consistent with existing linops like `Identity`, `Diagonal`
- Simple and maintainable

### Why plain `(x) -> y` callables?

- Lower ceremony for ad-hoc use
- More intuitive for users who don't need to know about the `(linop, x)` pattern
- The wrapper overhead is negligible

### Why not pass `normal` to adjoint?

- If user provides `normal(x) = adjoint(forward(x))`, then `A.H.normal(x)` should be `forward(adjoint(x))`, which is different
- By passing `normal=None`, we let the base class compute the correct normal from the swapped callables
- This ensures mathematical correctness

### Why no splitting support?

- Ad-hoc linops are meant for simple use cases
- Users who need splitting should subclass `NamedLinop` directly
- Keeps the API simple and focused
