# Utility Functions


## FFT Functions

Centered Fast Fourier Transform utilities.

- [cfftn](generated/utils/torchlinops.utils.cfftn.md)
- [cifftn](generated/utils/torchlinops.utils.cifftn.md)
- [cfft](generated/utils/torchlinops.utils.cfft.md)
- [cifft](generated/utils/torchlinops.utils.cifft.md)
- [cfft2](generated/utils/torchlinops.utils.cfft2.md)
- [cifft2](generated/utils/torchlinops.utils.cifft2.md)

## Device and Memory Management

Functions for managing device placement and memory efficiently.

- [get_device](generated/utils/torchlinops.utils.get_device.md)
- [device_ordinal](generated/utils/torchlinops.utils.device_ordinal.md)
- [cdata](generated/utils/torchlinops.utils.cdata.md)
- [same_storage](generated/utils/torchlinops.utils.same_storage.md)
- [tensor_memory_span](generated/utils/torchlinops.utils.tensor_memory_span.md)
- [memory_aware_to](generated/utils/torchlinops.utils.memory_aware_to.md)
- [memory_aware_deepcopy](generated/utils/torchlinops.utils.memory_aware_deepcopy.md)
- [MemReporter](generated/utils/torchlinops.utils.MemReporter.md)
- [ModuleMemoryMap](generated/utils/torchlinops.utils.ModuleMemoryMap.md)

## Array Conversion and Interoperability

Utilities for converting between array libraries and data structures.

- [to_pytorch](generated/utils/torchlinops.utils.to_pytorch.md)
- [from_pytorch](generated/utils/torchlinops.utils.from_pytorch.md)
- [numpy2torch](generated/utils/torchlinops.utils.numpy2torch.md)

## Data Structure Utilities

Functions for working with complex data structures and nested containers.

- [apply_struct](generated/utils/torchlinops.utils.apply_struct.md)
- [print_shapes](generated/utils/torchlinops.utils.print_shapes.md)
- [NDList](generated/utils/torchlinops.utils.NDList.md)

## Batching and Iteration

Utilities for batch processing and iteration.

- [ceildiv](generated/utils/torchlinops.utils.ceildiv.md)
- [batch_iterator](generated/utils/torchlinops.utils.batch_iterator.md)
- [batch_tqdm](generated/utils/torchlinops.utils.batch_tqdm.md)
- [dict_product](generated/utils/torchlinops.utils.dict_product.md)

## Mathematical Utilities

Mathematical operations and testing functions.

- [inner](generated/utils/torchlinops.utils.inner.md)
- [is_adjoint](generated/utils/torchlinops.utils.is_adjoint.md)

## Padding

Padding operations for tensors.

- [end_pad_with_zeros](generated/utils/torchlinops.utils.end_pad_with_zeros.md)

## Default Value Handling

Utilities for handling default values and dictionary merging.

- [default_to](generated/utils/torchlinops.utils.default_to.md)
- [default_to_dict](generated/utils/torchlinops.utils.default_to_dict.md)

## Function Dispatch and Signature Checking

Tools for function validation and dispatch.

- [check_signature](generated/utils/torchlinops.utils.check_signature.md)

## Logging and Formatting

Logging utilities and output formatting tools.

- [setup_console_logger](generated/utils/torchlinops.utils.setup_console_logger.md)
- [Indenter](generated/utils/torchlinops.utils.Indenter.md)
- [INDENT](generated/utils/torchlinops.utils.INDENT.md)

## CUDA Events

CUDA event management utilities.

- [RepeatedEvent](generated/utils/torchlinops.utils.RepeatedEvent.md)



## Triton Compatibility

Compatibility utilities for environments without Triton.

- [fake_triton](generated/utils/torchlinops.utils.fake_triton.md)
- [fake_tl](generated/utils/torchlinops.utils.fake_tl.md)
