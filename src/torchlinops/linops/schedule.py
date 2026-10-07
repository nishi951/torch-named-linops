"""Parallel execution for composite linops.

Provides ``parallel_execute()``, which runs child linops either sequentially
or in a ``ThreadPoolExecutor``, passing a shared ``SyncContext`` to each child.
Used by ``Add``, ``Concat``, and ``Stack`` to coordinate parallel execution
of their direct children.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from torch import Tensor

from torchlinops.utils import batch_iterator

__all__ = ["parallel_execute"]


def thread_initializer():
    """Create a tensor to warm up the cuda context."""
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")


def parallel_execute(
    linops,
    inputs,
    context,
    reduce_fn,
    threaded=False,
    num_workers=None,
    accumulate=True,
    accumulate_fn=None,
):
    """Execute a set of linops, possibly with threading-based concurrency.

    Note that under the GIL, threading-based concurrency only helps when the actual compute is
    not written in python (e.g. numpy functions, GPU kernels).

    Written as a "map-reduce" operation.

    Parameters
    ----------
    linops : list[NamedLinop]
        The linops to execute in parallel.
    inputs : list[Tensor]
        The corresponding list of inputs, one for each linop in `linops`.
    context : SyncContext
        The context object used to synchronize torch linop calls across multiple GPUs.
    reduce_fn : Callable[[list[Tensor]], Tensor]
        Function that combines the individual outputs of each linop to give the final output.
        Must be linear!
    threaded : bool, default False
        Whether to run the linops in separate threads.
    num_workers : int, optional
        The maximum number of workers to use in the threaded case.
        Doubles as the batch size for threaded=False and accumulate=True
    accumulate : bool
        Accumulate the results of the computations to limit memory usage.
        Uses num_workers as the batch size.
    accumulate_fn : Callable[[Tensor, Tensor], Tensor], optional
        If accumulate is True, the function to use to accumulate the output incrementally.

    Returns
    -------
    Tensor
        The output tensor.
    """
    if len(linops) != len(inputs):
        raise ValueError(
            f"linops and inputs must have same length but got linops: {len(linops)} != inputs: {len(inputs)}"
        )
    if len(linops) == 0:
        # TODO: decide if this is correct
        raise ValueError(f"linops must have length greater than or equal to 1.")

    if not accumulate:
        return _execute(linops, inputs, context, reduce_fn, threaded, num_workers)

    if accumulate_fn is None:

        def accumulate_fn(x, y):
            return reduce_fn([x, y])

    output = None
    job_batch_size = num_workers if num_workers is not None else 1
    for start_job, end_job in batch_iterator(len(linops), job_batch_size):
        linops_batch = linops[start_job:end_job]
        inputs_batch = inputs[start_job:end_job]
        output_batch = _execute(
            linops_batch, inputs_batch, context, reduce_fn, threaded, num_workers
        )

        if output is None:
            output = output_batch
        else:
            output = accumulate_fn(output, output_batch)  # type: ignore
    return output


def _execute(linops, inputs, context, reduce_fn, threaded, num_workers):
    """Helper function that executes all linops on all inputs."""

    if not threaded:
        return reduce_fn([linop(x, context) for linop, x in zip(linops, inputs)])

    def worker(idx: int):
        linop = linops[idx]
        x = inputs[idx]
        results[idx] = linop(x, context)

    num_workers = num_workers if num_workers is not None else len(linops)
    results: list[Optional[Tensor]] = [None] * len(linops)

    idxs = range(len(linops))
    with ThreadPoolExecutor(
        max_workers=num_workers, initializer=thread_initializer
    ) as pool:
        list(pool.map(worker, idxs))
    return reduce_fn(results)
