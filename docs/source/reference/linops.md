# Linear Operators

```{eval-rst}
.. currentmodule:: torchlinops
```

## Core Operators

Fundamental linear operators that form the building blocks for more complex operations.

```{eval-rst}
.. autosummary::
    :toctree: generated/linops
    :template: linop_template.rst
    :nosignatures:

    NamedLinop
    Identity
    Zero
    Scalar
    Diagonal
    Dense
```

## Composition Operators

Operators that combine or compose other linear operators.

```{eval-rst}
.. autosummary::
    :toctree: generated/linops
    :template: linop_template.rst
    :nosignatures:

    Add
    Chain
    Concat
    Stack
```

## Transform Operators

Operators that apply mathematical transforms to data.

```{eval-rst}
.. autosummary::
    :toctree: generated/linops
    :template: linop_template.rst
    :nosignatures:

    FFT
    NUFFT
    Interpolate
    Sampling
    Rearrange
    SumReduce
    Repeat
```

## Structural Operators

Operators that manipulate the structure and shape of data.

```{eval-rst}
.. autosummary::
    :toctree: generated/linops
    :template: linop_template.rst
    :nosignatures:

    ArrayToBlocks
    PadLast
    Truncate
    PadDim
    ShapeSpec
```

## Device & Batching Operators

Operators for device management and batch processing.

```{eval-rst}
.. autosummary::
    :toctree: generated/linops
    :template: linop_template.rst
    :nosignatures:

    ToDevice
    Batch
    MPBatch
```

## Special Operators

Special-purpose operators for debugging and unique use cases.

```{eval-rst}
.. autosummary::
    :toctree: generated/linops
    :template: linop_template.rst
    :nosignatures:

    BreakpointLinop
```

## Named Dimensions

Classes for handling named dimensions and shape specifications.

```{eval-rst}
.. autosummary::
    :toctree: generated/linops
    :template: linop_template.rst
    :nosignatures:

    NamedDimCollection
    NamedShape
```