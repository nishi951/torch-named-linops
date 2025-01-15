# Explanations

## The Problem

## Shapes
Currently implemented as `NamedDim`

`NamedShape`

## Linops



# Notes
### Splitting Linops and copy/deepcopy
After some frustration, it turns out that the easiest way to slice a linop is to
`copy.deepcopy` it and then replace the copy's Parameters with split versions of the
original parameters. Basically, the trifecta of:
1. `nn.Module + nn.Parameter` for easy device management and tensor registration,
2. The need to copy non-tensor/non-parameter attributes exactly, without
   invoking the constructor, and
3. The need to reuse memory.
all make this a tricky problem to solve.

Because of this, I will probably go back to using the constructor (via
`type(self)(*args, **kwargs)``).
