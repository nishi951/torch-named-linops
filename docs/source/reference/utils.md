# Utility Functions

```{eval-rst}
.. currentmodule:: torchlinops.utils
```

## FFT Functions

Centered Fast Fourier Transform utilities.

```{eval-rst}
.. autosummary::
    :toctree: generated/utils
    :nosignatures:

    cfftn
    cifftn
    cfft
    cifft
    cfft2
    cifft2
```

## Device and Memory Management

Functions for managing device placement and memory efficiently.

```{eval-rst}
.. autosummary::
    :toctree: generated/utils
    :nosignatures:

    get_device
    device_ordinal
    cdata
    same_storage
    tensor_memory_span
    memory_aware_to
    memory_aware_deepcopy
    MemReporter
    ModuleMemoryMap
```

## Array Conversion and Interoperability

Utilities for converting between array libraries and data structures.

```{eval-rst}
.. autosummary::
    :toctree: generated/utils
    :nosignatures:

    to_pytorch
    from_pytorch
    numpy2torch
```

## Data Structure Utilities

Functions for working with complex data structures and nested containers.

```{eval-rst}
.. autosummary::
    :toctree: generated/utils
    :nosignatures:

    apply_struct
    print_shapes
    NDList
```

## Batching and Iteration

Utilities for batch processing and iteration.

```{eval-rst}
.. autosummary::
    :toctree: generated/utils
    :nosignatures:

    ceildiv
    batch_iterator
    batch_tqdm
    dict_product
```

## Mathematical Utilities

Mathematical operations and testing functions.

```{eval-rst}
.. autosummary::
    :toctree: generated/utils
    :nosignatures:

    inner
    is_adjoint
```

## Padding

Padding operations for tensors.

```{eval-rst}
.. autosummary::
    :toctree: generated/utils
    :nosignatures:

    end_pad_with_zeros
```

## Default Value Handling

Utilities for handling default values and dictionary merging.

```{eval-rst}
.. autosummary::
    :toctree: generated/utils
    :nosignatures:

    default_to
    default_to_dict
```

## Function Dispatch and Signature Checking

Tools for function validation and dispatch.

```{eval-rst}
.. autosummary::
    :toctree: generated/utils
    :nosignatures:

    check_signature
```

## Logging and Formatting

Logging utilities and output formatting tools.

```{eval-rst}
.. autosummary::
    :toctree: generated/utils
    :nosignatures:

    setup_console_logger
    Indenter
    INDENT
```

## CUDA Events

CUDA event management utilities.

```{eval-rst}
.. autosummary::
    :toctree: generated/utils
    :nosignatures:

    RepeatedEvent
```



## Triton Compatibility

Compatibility utilities for environments without Triton.

```{eval-rst}
.. autosummary::
    :toctree: generated/utils
    :nosignatures:

    fake_triton
    fake_tl
```