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

__all__ = ["parallel_execute"]


def thread_initializer():
    """Create a tensor to warm up the cuda context."""
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")


def parallel_execute(
    linops, inputs, context, reduce_fn, threaded=False, num_workers=None
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

    if not threaded:
        return reduce_fn([linop(x, context) for linop, x in zip(linops, inputs)])

    num_workers = num_workers if num_workers is not None else len(linops)
    results: list[Optional[Tensor]] = [None] * len(linops)

    def worker(idx: int):
        linop = linops[idx]
        x = inputs[idx]
        results[idx] = linop(x, context)

    with ThreadPoolExecutor(
        max_workers=num_workers, initializer=thread_initializer
    ) as pool:
        list(pool.map(worker, range(len(linops))))
    return reduce_fn(results)
