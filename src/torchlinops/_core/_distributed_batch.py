from torch import Tensor

from itertools import cycle

import torch
from tqdm import tqdm

from ._batch import Batch

__all__ = ["DistributedBatch"]


class DistributedBatch(Batch):
    def __init__(self, *args, devices: list[torch.device], **kwargs):
        """
        Parameters
        ----------
        Same as Batch.

        Additional parameters:
        devices : list[torch.device]
            A list of devices to parallelize the linop over
            DistributedBatch will attempt to parallelize the work evenly over
            all the devices specified.


        """
        super().__init__(*args, **kwargs)
        self.devices = devices

    def forward(self, x: Tensor):
        # Complete the size specifications
        for dim, total in zip(self.ishape, x.shape):
            self.sizes[dim] = total

        # For holding input and output batches
        xs = []
        ys = []

        # Distribute linops and inputs
        for linop, in_batch, device in zip(
            self._linops, self._input_batches, cycle(self.devices)
        ):
            linop.to(device)
            xs.append(x[in_batch].clone().to(device))

        # Run linops on inputs
        for linop, xbatch in tqdm(
            zip(self._linops, xs),
            total=len(self._linops),
            desc=f"DistributedBatch({self.name}: {self.batch_sizes})",
            disable=(not self.pbar),
        ):
            ybatch = linop(xbatch)
            ys.append(ybatch)

        # Gather outputs
        y = torch.zeros(
            tuple(self.sizes[dim] for dim in self.oshape),
            dtype=self.output_dtype,
            device=self.output_device,
        )
        for ybatch, out_batch in zip(ys, self._output_batches):
            y[out_batch] += ybatch.to(self.output_device)
        return y
