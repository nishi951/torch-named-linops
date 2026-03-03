import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable

import torch
from torch import Tensor

__all__ = ["Threadable"]


class Threadable:
    """Mixin to enable parallel execution of sub-linops using Python threads.

    When ``threaded=True``, the linop's ``fn`` and ``adj_fn`` methods will run
    each sub-linop in parallel using a ThreadPoolExecutor. This is useful when
    sub-linops are I/O bound or release the GIL (e.g., PyTorch tensor operations).

    Attributes
    ----------
    linops : list[NamedLinop]
        The list of linops to run in parallel.
    threaded : bool
        Whether to run sub-linops in parallel. Default is True.
    num_workers : int | None
        Number of worker threads. If None, defaults to the number of sub-linops.
    """

    def __init__(
        self, *args, threaded: bool = True, num_workers: Optional[int] = None, **kwargs
    ):
        """
        Parameters
        ----------
        threaded : bool, optional
            Whether to run sub-linops in parallel. Default is True.
        num_workers : int | None, optional
            Number of worker threads. If None, defaults to len(self.linops).
        """
        super().__init__(*args, **kwargs)
        self.threaded = threaded
        self.num_workers = num_workers
        self.linops = []  # Placeholder

    def _setup_events(self):
        for linop in self.linops:
            linop.input_listener = (self, "input_listener")

    def _setup_defaults(self, x, num_workers):
        if not hasattr(self, "linops") or len(self.linops) == 0:
            raise AttributeError("Threadable class must have `linops` attribute.")
        xs = list(x) if isinstance(x, (list, tuple)) else [x]
        if num_workers is None:
            num_workers = max(len(self.linops), len(xs))
        return xs, num_workers

    def threaded_apply_sum_reduce(
        self, x: Tensor | list[Tensor], num_workers: Optional[int] = None
    ) -> Tensor:
        """Wrapper around _threaded_apply_sum_reduce."""
        xs, num_workers = self._setup_defaults(x, num_workers)
        return _threaded_apply_sum_reduce(self.linops, xs, num_workers)

    def threaded_apply(
        self, x: Tensor | list[Tensor], num_workers: Optional[int] = None
    ):
        """Wrapper around _threaded_apply"""
        xs, num_workers = self._setup_defaults(x, num_workers)
        return _threaded_apply(self.linops, xs, num_workers)


def _threaded_apply_sum_reduce(linops, xs, num_workers):
    """Apply linops in parallel using ThreadPoolExecutor.

    Returns the sum of the outputs

    Parameters
    ----------
    linops : list[NamedLinop]
    x : Tensor
        Input tensor.
    apply_fn : callable
        Function to apply to each (linop, x) pair. Should return a tensor.

    Returns
    -------
    Tensor
        The sum of the outputs of each sub linop
    """
    lock = threading.Lock()
    output: list[Tensor | float] = [0.0]  # Use list for indirection

    def worker_fn(linop_x: tuple):
        linop, x = linop_x
        y = linop(x)
        with lock:
            output[0] += y

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(worker_fn, zip(linops, xs)))
    return torch.as_tensor(output[0])


def _threaded_apply(linops, xs, num_workers):
    """Apply linops in parallel using ThreadPoolExecutor.

    Parameters
    ----------
    linop : NamedLinop
        The linop containing sub-linops.
    x : Tensor
        Input tensor.
    apply_fn : callable
        Function to apply to each (linop, x) pair. Should return a tensor.

    Returns
    -------
    list[Tensor]
        List of results from each sub-linop.
    """

    def worker_fn(linop, x):
        y = linop(x)
        return y

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_fn, linop, x) for linop, x in zip(linops, xs)]
        results = [future.result() for future in futures]

    return results
