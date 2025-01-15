# Tutorials

### Custom Linops
Custom linops should satisfy the following:
- `@staticmethod` `.fn()`, `.adj_fn()`, and `.normal_fn()`
  - Allows for swapping of functions rather than `bound_methods` in `adjoint()`.
- Calling the constructor of itself via `type(self)` inside `.split_forward()`.
  - This is because of how pytorch handles `copy`-ing parameters. We call the
    constructor rather than deepcopy to avoid making an extra copy of tensors.
    However, `copy` does not propery isolate the new parameter - modifications
    to the parameter will propagate up to the calling function.
- Use of `super().__init__(NS(ishape, oshape))` in the constructor.
  - This initializes the linop's shape
  - May change later to not require `NS`

