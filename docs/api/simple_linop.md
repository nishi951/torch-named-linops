# SimpleLinop: Ad-hoc Linear Operators

`SimpleLinop` enables creating linear operators with minimal ceremony, without requiring you to subclass `NamedLinop`.

## Basic Usage

```python
from torchlinops import SimpleLinop
import torch

# Create a simple scaling operator
A = SimpleLinop(
    forward=lambda x: 2 * x,
    adjoint=lambda y: 2 * y,
)

x = torch.randn(10, 20)
y = A(x)        # Forward: 2 * x
z = A.H(y)      # Adjoint: 2 * y
```

## Parameters

- **forward** (required): Callable `(x) -> y` implementing the forward operation
- **adjoint** (required): Callable `(x) -> y` implementing the adjoint operation
- **normal** (optional): Callable `(x) -> y` implementing the normal operation. If not provided, defaults to `adjoint(forward(x))`
- **ishape** (optional): Input named dimensions, defaults to `("...",)`
- **oshape** (optional): Output named dimensions, defaults to `ishape`
- **name** (optional): Display name for the linop

## Custom Normal Operation

If you have an efficient closed-form normal operation, provide it to avoid computing `adjoint(forward(x))`:

```python
A = SimpleLinop(
    forward=lambda x: 2 * x,
    adjoint=lambda y: 2 * y,
    normal=lambda x: 4 * x,  # More efficient than adjoint(forward(x))
)
```

## Named Dimensions

Specify input and output dimensions for composition with other linops:

```python
from torchlinops import SimpleLinop, FFT, Dim

# Create operator with named dimensions
A = SimpleLinop(
    forward=lambda x: 2 * x,
    adjoint=lambda y: 2 * y,
    ishape=("Nx", "Ny"),
    oshape=("Nx", "Ny"),
)

# Compose with FFT
B = FFT(Dim("Nx")) @ A
```

## Important Caveats

### No Adjoint Checking

`SimpleLinop` does **not** verify that your forward and adjoint operations are actually adjoints of each other. You can pass arbitrary functions and no errors will be raised:

```python
# This will NOT raise an error, even though the adjoint is wrong
A = SimpleLinop(
    forward=lambda x: 2 * x,
    adjoint=lambda y: 3 * y,  # Wrong! Should be 2 * y
)
```

**You are responsible for ensuring mathematical correctness.** If you need verification, use the testing utilities in `torchlinops.testing`.

### No Splitting Support

`SimpleLinop` never gets split during tiling operations. If you use a `SimpleLinop` in a context where splitting is required (e.g., multi-GPU execution), it will be copied as-is to each tile, which may lead to incorrect results or performance issues.

If you need splitting support, subclass `NamedLinop` directly and implement the `split()` method.

### Multiprocessing Limitations

Lambda functions cannot be pickled, which means `SimpleLinop` instances created with lambdas cannot be:
- Sent to other processes via `torch.multiprocessing`
- Saved/loaded with `torch.save()`/`torch.load()`
- Used with certain distributed computing frameworks

If you need multiprocessing support, define your forward/adjoint/normal functions at module level:

```python
# Module-level functions (picklable)
def my_forward(x):
    return 2 * x

def my_adjoint(y):
    return 2 * y

A = SimpleLinop(forward=my_forward, adjoint=my_adjoint)
```

## See Also

- [NamedLinop API](../reference/linops/namedlinop.md) - Base class for all linear operators
- [Creating Custom Linops](../howto/custom_linops.md) - When to subclass vs. use SimpleLinop
