# Linear Operators


## Core Operators

Fundamental linear operators that form the building blocks for more complex operations.

- [NamedLinop](generated/linops/torchlinops.NamedLinop.md)
- [Identity](generated/linops/torchlinops.Identity.md)
- [Zero](generated/linops/torchlinops.Zero.md)
- [Scalar](generated/linops/torchlinops.Scalar.md)
- [Diagonal](generated/linops/torchlinops.Diagonal.md)
- [Dense](generated/linops/torchlinops.Dense.md)

## Composition Operators

Operators that combine or compose other linear operators.

- [Add](generated/linops/torchlinops.Add.md)
- [Chain](generated/linops/torchlinops.Chain.md)
- [Concat](generated/linops/torchlinops.Concat.md)
- [Stack](generated/linops/torchlinops.Stack.md)

## Transform Operators

Operators that apply mathematical transforms to data.

- [FFT](generated/linops/torchlinops.FFT.md)
- [NUFFT](generated/linops/torchlinops.NUFFT.md)
- [Interpolate](generated/linops/torchlinops.Interpolate.md)
- [Sampling](generated/linops/torchlinops.Sampling.md)
- [Rearrange](generated/linops/torchlinops.Rearrange.md)
- [SumReduce](generated/linops/torchlinops.SumReduce.md)
- [Repeat](generated/linops/torchlinops.Repeat.md)

## Structural Operators

Operators that manipulate the structure and shape of data.

- [ArrayToBlocks](generated/linops/torchlinops.ArrayToBlocks.md)
- [PadLast](generated/linops/torchlinops.PadLast.md)
- [Truncate](generated/linops/torchlinops.Truncate.md)
- [PadDim](generated/linops/torchlinops.PadDim.md)
- [ShapeSpec](generated/linops/torchlinops.ShapeSpec.md)

## Device & Batching Operators

Operators for device management and batch processing.

- [ToDevice](generated/linops/torchlinops.ToDevice.md)
- [Batch](generated/linops/torchlinops.Batch.md)
- [MPBatch](generated/linops/torchlinops.MPBatch.md)

## Special Operators

Special-purpose operators for debugging and unique use cases.

- [BreakpointLinop](generated/linops/torchlinops.BreakpointLinop.md)

## Named Dimensions

Classes for handling named dimensions and shape specifications.

- [NamedDimCollection](generated/linops/torchlinops.NamedDimCollection.md)
- [NamedShape](generated/linops/torchlinops.NamedShape.md)
