# API Reference

## Linear Operators


### Core Operators

Fundamental linear operators that form the building blocks for more complex operations.

- [NamedLinop](torchlinops/linops/namedlinop.md)
- [Identity](torchlinops/linops/identity.md#torchlinops.linops.identity.Identity)
- [Zero](torchlinops/linops/identity.md#torchlinops.linops.identity.Zero)
- [Scalar](linops/Scalar.md)
- [Diagonal](linops/Diagonal.md)
- [Dense](linops/Dense.md)

### Composition Operators

Operators that combine or compose other linear operators.

- [Add](linops/Add.md)
- [Chain](linops/Chain.md)
- [Concat](linops/Concat.md)
- [Stack](linops/Stack.md)

### Transform Operators

Operators that apply mathematical transforms to data.

- [FFT](linops/FFT.md)
- [NUFFT](linops/NUFFT.md)
- [Interpolate](linops/Interpolate.md)
- [Sampling](linops/Sampling.md)
- [Rearrange](linops/Rearrange.md)
- [SumReduce](linops/SumReduce.md)
- [Repeat](linops/Repeat.md)

### Structural Operators

Operators that manipulate the structure and shape of data.

- [ArrayToBlocks](linops/ArrayToBlocks.md)
- [PadLast](linops/PadLast.md)
- [Truncate](linops/Truncate.md)
- [PadDim](linops/PadDim.md)
- [ShapeSpec](linops/ShapeSpec.md)

### Device & Batching Operators

Operators for device management and batch processing.

- [ToDevice](linops/ToDevice.md)
- [Batch](linops/Batch.md)
- [MPBatch](linops/MPBatch.md)

### Special Operators

Special-purpose operators for debugging and unique use cases.

- [BreakpointLinop](linops/BreakpointLinop.md)

### Named Dimensions

Classes for handling named dimensions and shape specifications.

- [NamedDimCollection](linops/NamedDimCollection.md)
- [NamedShape](linops/NamedShape.md)

## Utility Functions


### FFT Functions

Centered Fast Fourier Transform utilities.

- [cfftn](utils/cfftn.md)
- [cifftn](utils/cifftn.md)
- [cfft](utils/cfft.md)
- [cifft](utils/cifft.md)
- [cfft2](utils/cfft2.md)
- [cifft2](utils/cifft2.md)

### Device and Memory Management

Functions for managing device placement and memory efficiently.

- [get_device](utils/get_device.md)
- [device_ordinal](utils/device_ordinal.md)
- [cdata](utils/cdata.md)
- [same_storage](utils/same_storage.md)
- [tensor_memory_span](utils/tensor_memory_span.md)
- [memory_aware_to](utils/memory_aware_to.md)
- [memory_aware_deepcopy](utils/memory_aware_deepcopy.md)
- [MemReporter](utils/MemReporter.md)
- [ModuleMemoryMap](utils/ModuleMemoryMap.md)

### Array Conversion and Interoperability

Utilities for converting between array libraries and data structures.

- [to_pytorch](utils/to_pytorch.md)
- [from_pytorch](utils/from_pytorch.md)
- [numpy2torch](utils/numpy2torch.md)

### Data Structure Utilities

Functions for working with complex data structures and nested containers.

- [apply_struct](utils/apply_struct.md)
- [print_shapes](utils/print_shapes.md)
- [NDList](utils/NDList.md)

### Batching and Iteration

Utilities for batch processing and iteration.

- [ceildiv](utils/ceildiv.md)
- [batch_iterator](utils/batch_iterator.md)
- [batch_tqdm](utils/batch_tqdm.md)
- [dict_product](utils/dict_product.md)

### Mathematical Utilities

Mathematical operations and testing functions.

- [inner](utils/inner.md)
- [is_adjoint](utils/is_adjoint.md)

### Padding

Padding operations for tensors.

- [end_pad_with_zeros](utils/end_pad_with_zeros.md)

### Default Value Handling

Utilities for handling default values and dictionary merging.

- [default_to](utils/default_to.md)
- [default_to_dict](utils/default_to_dict.md)

### Function Dispatch and Signature Checking

Tools for function validation and dispatch.

- [check_signature](utils/check_signature.md)

### Logging and Formatting

Logging utilities and output formatting tools.

- [setup_console_logger](utils/setup_console_logger.md)
- [Indenter](utils/Indenter.md)
- [INDENT](utils/INDENT.md)

### CUDA Events

CUDA event management utilities.

- [RepeatedEvent](utils/RepeatedEvent.md)


### Triton Compatibility

Compatibility utilities for environments without Triton.

- [fake_triton](utils/fake_triton.md)
- [fake_tl](utils/fake_tl.md)

## Algorithms

### Iterative Solvers

Methods for solving linear systems iteratively.

- [conjugate_gradients](algorithms/conjugate_gradients.md)

### Eigenvalue Methods

Methods for finding eigenvalues and eigenvectors.

- [power_method](algorithms/power_method.md)

### Preconditioning

Methods for creating and applying preconditioners.

- [polynomial_preconditioner](algorithms/polynomial_preconditioner.md)

