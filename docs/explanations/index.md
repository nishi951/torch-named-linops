# Explanations

This section contains discussion about the overall design and implementation of this package.

- [**Named Abstractions**](named_abstractions.md) -- The core abstraction stack: `NamedDimension`, `NamedDimCollection`, `NamedShape`, and `NamedLinop`.
- [**How Named Dimensions Propagate**](named_dims_propagation.md) -- How named dimensions are inferred, renamed, resolved, and incremented as operators are constructed, composed, adjointed, and normalized.
- [**Configuration Options**](config.md) -- The `torchlinops.config` flags: `reduce_identity_in_normal`, `cache_adjoint_normal`, `log_device_transfers`, `shape_inference`, and the `config.using()` context manager.
- [**Why This Package?**](why_this_package.md) -- Motivation: the problems with traditional linops and how named dimensions solve them.
- [**Design Notes**](design_notes.md) -- Key design decisions and tradeoffs (staticmethods, caching in lists, pickle-ability, shape matching).
- [**Copying Linops**](copying_linops.md) -- How shallow copy and memory-aware deep copy work, and why they matter.
- [**Multi-GPU Execution**](multi_gpu.md) -- Splitting operators across devices and synchronizing parallel execution: `BatchSpec`, `ToDevice`, `SyncContext`, streams, and events.
- [**FAQ**](faq.md) -- Frequently asked questions.
