import threading
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["Threadable"]


class Threadable:
    """Mixin to enable parallel execution of sub-linops using Python threads.

    When ``threaded=True``, the linop's ``fn`` and ``adj_fn`` methods will run
    each sub-linop in parallel using a ThreadPoolExecutor. This is useful when
    sub-linops are I/O bound or release the GIL (e.g., PyTorch tensor operations).

    The mixin manages sub-linops through the ``linops`` property, which automatically
    creates shallow copies of each linop when assigned. This ensures that shared
    linops (e.g., ``Add(A, A)``) have independent identities for threading while
    still sharing tensor data.

    Attributes
    ----------
    linops : nn.ModuleList
        The list of linops to run in parallel. Setting this property triggers
        automatic shallow copying and input listener setup.
    threaded : bool
        Whether to run sub-linops in parallel. Default is True.
    num_workers : int | None
        Number of worker threads. If None, defaults to the number of sub-linops.
    settings : dict
        Dictionary with ``threaded`` and ``num_workers`` keys for easy copying
        of threading configuration.
    """

    def __init__(
        self,
        *args,
        threaded: bool = True,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        threaded : bool, optional
            Whether to run sub-linops in parallel. Default is True.
        num_workers : int | None, optional
            Number of worker threads. If None, defaults to the number of
            sub-linops when ``threaded_apply`` or ``threaded_apply_sum_reduce``
            is called.
        linops : list[NamedLinop], optional
            The list of linops to run in parallel. If assigned via the
            ``linops`` property, input listeners will be set up automatically.
        """
        super().__init__(*args, **kwargs)
        self.threaded = threaded
        self.num_workers = num_workers

    @property
    def linops(self):
        """The list of sub-linops managed by this Threadable.

        This is a property rather than a direct attribute to intercept assignment
        and perform automatic housekeeping whenever linops are set. The setter
        creates shallow copies of each linop (preserving tensor data sharing)
        and sets up input listeners for event coordination.

        Returns
        -------
        nn.ModuleList
            The list of sub-linops.
        """
        return self._linops

    @linops.setter
    def linops(self, new_linops):
        """Set sub-linops with automatic copying and event setup.

        When linops are assigned, this setter:
        1. Creates shallow copies of each linop using ``copy()``, ensuring
           shared linops have independent identities (for threading safety)
           while still sharing tensor data.
        2. Sets up input listeners on each copied linop.

        Parameters
        ----------
        new_linops : list[NamedLinop]
            The linops to manage.
        """
        self._linops = new_linops
        self._setup_events()

    def __setattr__(self, name, value):
        """Set attribute, with special handling for ``linops``.

        PyTorch's ``nn.Module.__setattr__`` intercepts attribute assignment and
        performs special handling for modules, parameters, and buffers. This
        override ensures that ``linops`` assignment goes through the property
        descriptor rather than being intercepted by PyTorch's logic.

        Parameters
        ----------
        name : str
            Attribute name.
        value : Any
            Attribute value.
        """
        if name == "linops":
            type(self).linops.fset(self, value)
        else:
            super().__setattr__(name, value)

    def _setup_events(self):
        """Set up input listeners on all sub-linops.

        This method is called automatically when ``linops`` is assigned via
        the property setter. It performs two operations:

        1. Creates shallow copies of each linop using ``copy()``, ensuring that
           linops shared by identity (e.g., ``Add(A, A)``) become independent
           objects while still sharing tensor data. This prevents race conditions
           when the same linop appears multiple times in a threaded context.

        2. Attaches an input listener to each linop, enabling coordination
           between the parent Threadable and its sub-linops.
        """
        self._linops = nn.ModuleList([copy(linop) for linop in self._linops])
        for linop in self._linops:
            linop.input_listener = (self, "input_listener")

    def _apply_defaults(self, x, num_workers):
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
        xs, num_workers = self._apply_defaults(x, num_workers)
        return _threaded_apply_sum_reduce(self.linops, xs, num_workers)

    def threaded_apply(
        self, x: Tensor | list[Tensor], num_workers: Optional[int] = None
    ):
        """Wrapper around _threaded_apply"""
        xs, num_workers = self._apply_defaults(x, num_workers)
        return _threaded_apply(self.linops, xs, num_workers)

    @property
    def settings(self):
        """Get threading settings as a dictionary.

        Returns
        -------
        dict
            Dictionary with ``threaded`` and ``num_workers`` keys.
        """
        return {"threaded": self.threaded, "num_workers": self.num_workers}

    @settings.setter
    def settings(self, new_settings):
        """Set threading settings from a dictionary.

        Parameters
        ----------
        new_settings : dict
            Dictionary with ``threaded`` and ``num_workers`` keys.
        """
        self.threaded = new_settings["threaded"]
        self.num_workers = new_settings["num_workers"]


def _threaded_apply_sum_reduce(linops, xs, num_workers):
    """Apply linops in parallel and sum the results.

    This function uses a ThreadPoolExecutor to apply each linop to its
    corresponding input in parallel, with thread-safe accumulation of results.

    Parameters
    ----------
    linops : list[NamedLinop]
        The list of linops to apply.
    xs : list[Tensor]
        The list of input tensors, one per linop.
    num_workers : int
        The number of worker threads. More workers increase parallelism
        but also memory footprint.

    Returns
    -------
    Tensor
        The sum of the outputs from each sub-linop.
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
    """Apply linops in parallel and return the list of results.

    This function uses a ThreadPoolExecutor to apply each linop to its
    corresponding input in parallel.

    Parameters
    ----------
    linops : list[NamedLinop]
        The list of linops to apply.
    xs : list[Tensor]
        The list of input tensors, one per linop.
    num_workers : int
        The number of worker threads. More workers increase parallelism
        but also memory footprint.

    Returns
    -------
    list[Tensor]
        List of outputs from each sub-linop, in the same order as inputs.
    """

    def worker_fn(linop, x):
        y = linop(x)
        return y

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_fn, linop, x) for linop, x in zip(linops, xs)]
        results = [future.result() for future in futures]
    return results
