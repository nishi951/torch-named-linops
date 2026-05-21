"""Parallel execution for composite linops.

Provides ``parallel_execute()``, which runs child linops either sequentially
or in a ``ThreadPoolExecutor``, passing a shared ``SyncContext`` to each child.
Used by ``Add``, ``Concat``, and ``Stack`` to coordinate parallel execution
of their direct children.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal, Optional

import torch
from torch import Tensor

__all__ = ["parallel_execute"]

# Type definitions
LinopId = int | Literal["parent"]

# A linop-event pair of the form (linop, [start|end]_event)
# e.g. ("parent", "start_event"), or (1, "end_event")
LinopEvent = tuple[LinopId, Literal['start_event", "end_event"']]

# A mapping from the linop id to a list of events the linop must wait for
# before starting.
Dependencies = dict[LinopId, list[LinopEvent]]


def thread_initializer():
    """Create a tensor to warm up the cuda context."""
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")


def parallel_execute(
    linops, inputs, context, reduce_fn, parent=None, threaded=False, num_workers=None
):
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
