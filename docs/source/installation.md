# Installation

## Anaconda (recommended)

[mamba](https://mamba.readthedocs.io/en/latest/) is the recommended conda-based installer.

1. Install pytorch, torchvision, and torchaudio. Refer to [https://pytorch.org](https://pytorch.org) for the most up-to-date info.
2. Install torchlinops with:

```console
$ pip install torchlinops[cuda11] # Cuda 11 versions of things
$ pip install torchlinops[cuda12] # Cuda 12 versions of things

Make sure to install the version of torchlinops that corresponds to your cuda version.
```

## Pip

The pip install uses the Pypy versions of pytorch.

1. Install torchlinops with
   .. code-block:: console

   > \$ pip install torchlinops\[cuda11\] # Cuda 11 versions of things
   > \$ pip install torchlinops\[cuda12\] # Cuda 12 versions of things

:::{note}
Cupy doesn't like `nccl` dependencies installed as wheels from pip. Importing
cupy the first time will present an import error with a python command that can
be used to manually install the nccl library, e.g. (for cuda 12.x - replace with
relevant cuda version)

```console
$ python -m cupyx.tools.install_library --library nccl --cuda 12.x
```

For more up-to-date info, can follow the issue [here](https://github.com/cupy/cupy/issues/8227).
:::
