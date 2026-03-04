# Copying Linops

Copying linops is a surprisingly difficult thing to achieve in a memory-efficient manner. Linear operators often hold large tensors (interpolation kernels, diagonal weights, dense matrices), and many derived operators (adjoints, normals, splits) share the same underlying data. Getting the copy semantics wrong can lead to silent data corruption, excessive memory usage, or both.

## Basic shallow copy

`copy.copy()` of a `NamedLinop` is the workhorse of the library. It is used internally by `adjoint()`, `normal()`, and many `split_forward()` implementations.

A shallow copy:

1. **Shares parameter and buffer data.** The new linop references the same `nn.Parameter` and buffer tensors as the original. No tensor data is duplicated.

2. **Gets its own PyTorch module dictionaries.** The internal `_parameters`, `_modules`, and `_buffers` dicts are copied (shallowly), so adding or removing entries in the copy does not affect the original's registry.

3. **Resets derived operator caches.** The `_adjoint` and `_normal` caches are set to `None`. This is necessary because the copy may have a different shape or different function bindings, making the original's cached adjoint/normal invalid.

4. **Deep-copies the shape.** The `_shape` (`NamedShape`) is deep-copied so that mutating the copy's dimensions does not affect the original.

This is the mechanism behind adjoint creation:

```python
def adjoint(self):
    adj = copy(self)              # Shallow copy, shares data
    adj._shape = adj._shape.H    # Flip input/output dims
    adj.fn, adj.adj_fn = adj.adj_fn, adj.fn  # Swap functions
    return adj
```

The result is a new `NamedLinop` that represents $A^H$ but uses the exact same weight tensors as $A$. Modifying the weights of one will affect the other -- this is intentional and desirable, as the adjoint should always reflect the current state of the operator.

!!! note
    Many concrete linops override `split_forward` to call their own constructor via `type(self)(...)` rather than using `copy()`. This is because PyTorch's `copy` does not properly isolate `nn.Parameter` objects -- modifications to a shallow-copied parameter can propagate back to the original. Calling the constructor creates truly independent parameters that happen to reference the same tensor data.

## Memory-aware deepcopy

When you do need an independent copy of a linop (e.g., for placing copies on different GPUs), a naive `copy.deepcopy()` can be dangerous:

- **View relationships are lost.** If multiple parameters in a linop are views into the same underlying storage (a common optimization), `deepcopy` will allocate separate storage for each, potentially doubling or tripling memory usage.
- **GPU memory pressure.** Large linops on GPU can easily cause OOM if data is carelessly duplicated.

`NamedLinop` overrides `__deepcopy__` to use a **memory-aware** strategy via the `ModuleMemoryMap` utility:

### How it works

1. **Analyze storage topology.** `ModuleMemoryMap` walks the module tree and groups all parameters/buffers by their underlying storage pointer (`cdata`). This identifies which tensors share memory.

2. **Allocate one new buffer per storage group.** For each group of tensors sharing the same storage, a single contiguous buffer is allocated that spans the full memory range (from minimum to maximum offset + size).

3. **Recreate tensors as views.** Each parameter/buffer is recreated as a view (`as_strided`) into the new buffer, preserving the original size, stride, and storage offset relationships.

4. **Copy data.** The new buffers contain copies of the original data, so the new linop is fully independent.

The result is a true deep copy where:

- All tensor data is duplicated (the copy is independent of the original).
- View relationships are preserved (tensors that shared storage still share storage in the copy).
- Memory usage is minimal (no redundant allocations).

### Memory-aware device transfer

The same `ModuleMemoryMap` machinery powers `memory_aware_to(device)`, which moves a linop to a new device while preserving storage topology. This is used by the [multi-GPU splitting](multi_gpu.md) system to efficiently place sub-linops on target devices without unnecessary memory overhead.

```python
# Standard PyTorch .to() -- may break view relationships
linop_gpu = linop.to("cuda:0")

# Memory-aware -- preserves view topology
linop_gpu = linop.to("cuda:0", memory_aware=True)
```
