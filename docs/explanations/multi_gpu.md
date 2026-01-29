# Multi-GPU Splitting

Linops should be able to take advantage of multi-GPU systems to leverage the larger total GPU memory available and to gain increased speed from parallelization across separate devices.

We assume CUDA devices with peer-to-peer memory access. 

In the following

A summary of the design decisions:
- Compute happens on default stream of each device
- Per-linop repeated event signals the ot

Limitations/Future work:
- Running computations on distributed GPU nodes on multiple servers is possible
  in principle and may be supported via normal PyTorch APIs, but no simple API
  is available within this library.
- 

