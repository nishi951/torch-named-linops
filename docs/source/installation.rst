Installation
============

The following packages should be installed prior to using torchlinops.

* ``pytorch >= 2.1``
* ``torchvision``
* ``torchaudio``

Then, install torchlinops with:

.. code-block:: console

   $ pip install torchlinops[cuda11] # Cuda 11 versions of things
   $ pip install torchlinops[cuda12] # Cuda 12 versions of things

.. note ::

   Cupy doesn't like ``nccl`` dependencies installed as wheels from pip. Importing
   cupy the first time will present an import error with a python command that can
   be used to manually install the nccl library, e.g. (for cuda 12.x - replace with
   relevant cuda version)

   .. code-block:: console

      $ python -m cupyx.tools.install_library --library nccl --cuda 12.x

   For more up-to-date info, can follow the issue `here <https://github.com/cupy/cupy/issues/8227>`_.
