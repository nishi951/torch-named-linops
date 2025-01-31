from torch import Tensor

from itertools import cycle

import torch
from tqdm import tqdm

from .batch import Batch

__all__ = ["MPBatch"]


class MPBatch(Batch):
    def __init__(self, *args, devices: list[torch.device], **kwargs):
        """
        Parameters
        ----------
        Same as Batch.

        Additional parameters:
        devices : list[torch.device]
            A list of devices to parallelize the linop over
            MPBatch will attempt to parallelize the work evenly over
            all the devices specified.


        """
        self.devices = devices
        super().__init__(*args, **kwargs)

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
            xs.append(x[in_batch].to(device))

        # Run linops on inputs
        for linop, xbatch in tqdm(
            zip(self._linops, xs),
            total=len(self._linops),
            desc=f"MPBatch({self.repr_name}: {self.batch_sizes})",
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

    def adjoint(self):
        batch_sizes = {str(k): v for k, v in self.batch_sizes.items()}
        adj = type(self)(
            linop=self.linop.H,
            input_device=self.output_device,
            output_device=self.input_device,
            input_dtype=self.output_dtype,
            output_dtype=self.input_dtype,
            name=self.name + ".H",
            pbar=self.pbar,
            devices=self.devices,
            post_batch_hook=self.post_batch_hook,
            **batch_sizes,
        )
        return adj

    def normal(self, inner=None):
        normal_linop = self.linop.N
        # Collect shape updates from computing the normal
        shape_updates = getattr(normal_linop, "_shape_updates", {})
        for d, nd in shape_updates.items():
            if d in self.batch_sizes:
                self.batch_sizes[shape_updates[d]] = self.batch_sizes[d]
        batch_size_kwargs = {str(k): v for k, v in self.batch_sizes.items()}
        normal = type(self)(
            linop=self.linop.N,
            input_device=self.input_device,
            output_device=self.input_device,
            input_dtype=self.input_dtype,
            output_dtype=self.input_dtype,
            name=self.name + ".N",
            pbar=self.pbar,
            devices=self.devices,
            post_batch_hook=self.post_batch_hook,
            **batch_size_kwargs,
        )
        return normal
