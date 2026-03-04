# Explanations

This section contains discussion about the overall design and implementation of this package.

- [**Named Abstractions**](named_abstractions.md) -- The core abstraction stack: `NamedDimension`, `NamedDimCollection`, `NamedShape`, and `NamedLinop`.
- [**Why This Package?**](why_this_package.md) -- Motivation: the problems with traditional linops and how named dimensions solve them.
- [**Design Notes**](design_notes.md) -- Key design decisions and tradeoffs (staticmethods, caching in lists, pickle-ability, shape matching).
- [**Copying Linops**](copying_linops.md) -- How shallow copy and memory-aware deep copy work, and why they matter.
- [**Multi-GPU Splitting**](multi_gpu.md) -- Splitting operators across devices: `BatchSpec`, `ToDevice`, streams, and events.
- [**FAQ**](faq.md) -- Frequently asked questions.
