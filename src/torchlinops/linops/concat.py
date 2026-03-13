import logging
from copy import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

import torchlinops.config as config
from torchlinops.utils import INDENT

from ..nameddim import (
    ANY,
    ELLIPSES,
    NamedDimension as ND,
    NamedShape as NS,
    iscompatible,
    isequal,
    max_shape,
    standardize_shapes,
)
from .add import Add
from .device import ToDevice
from .identity import Zero
from .namedlinop import NamedLinop
from .threadable import Threadable, _threaded_apply, _threaded_apply_sum_reduce

__all__ = ["Concat"]

logger = logging.getLogger("torchlinops")


def _log_transfer(msg):
    if config.log_device_transfers:
        logger.info(msg)


class Concat(Threadable, NamedLinop):
    """Concatenate some linops along an existing dimension.

    Linops need not output tensors of the same size, but they should
    output tensors of the same number of dimensions.

    Stacking type depends on dimensions provided:

    Horizontal stacking (stacking along an input dimension)::

        A B C

    Vertical stacking (stacking along an output dimension)::

        A
        B
        C

    Diagonal stacking (stacking along separate input and output dimensions)::

        A . .
        . B .
        . . C

    Inherits from ``Threadable`` to support parallel execution of sub-linops.
    When ``threaded=True`` (default), each sub-linop is executed in parallel
    using a ThreadPoolExecutor.

    Note that shared linops (e.g., ``Concat(A, A, idim="x")``) are automatically
    shallow-copied to ensure independent identity for threading, while still
    sharing tensor data. See ``Threadable`` for details.

    Attributes
    ----------
    linops : nn.ModuleList
        The list of linops being concatenated.
    threaded : bool
        Whether to run sub-linops in parallel. Default is True.
    num_workers : int | None
        Number of worker threads. If None, defaults to the number of sub-linops.
    idim : ND | None
        Input dimension along which to concatenate.
    odim : ND | None
        Output dimension along which to concatenate.
    """

    def __init__(
        self,
        *linops,
        idim: Optional[ND | str] = None,
        odim: Optional[ND | str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        *linops : NamedLinop
            The linops to concatenate.
        idim : str or ND, optional
            Input dimension along which to concatenate. If ``None``, the input
            is not concatenated (all linops receive the same input).
        odim : str or ND, optional
            Output dimension along which to concatenate. If ``None``, the output
            is not concatenated (outputs are summed).
        """
        self._check_linop_compatibility(linops)
        super().__init__(NS(linops[0].ishape, linops[0].oshape), **kwargs)
        self.linops = nn.ModuleList(list(linops))
        self._setup_indices(idim, odim)

    @staticmethod
    def fn(concat, x):
        return concat._fn(
            x,
            concat.linops,
            concat.idim_idx,
            concat.odim_idx,
            concat.islices,
            concat.oslices,
            concat.threaded,
            concat.num_workers,
        )

    @staticmethod
    def adj_fn(concat, x):
        adj_linops = [linop.H for linop in concat.linops]
        return concat._fn(
            x,
            adj_linops,
            concat.odim_idx,
            concat.idim_idx,
            concat.oslices,
            concat.islices,
            concat.threaded,
            concat.num_workers,
        )

    @staticmethod
    def _fn(
        x: Tensor,
        linops,
        idim_idx,
        odim_idx,
        islices,
        oslices,
        threaded: bool = False,
        num_workers: int | None = None,
    ):
        """Unifies forward and adjoint functionality for stacked linops"""
        if idim_idx is not None:  # Diagonal, Horizontal
            if islices[-1] != x.shape[idim_idx]:
                raise ValueError(
                    f"Concat Linop expecting input of size {islices[-1]} got input of size {x.shape} with non-matching concat size {x.shape[idim_idx]}"
                )
            xs = x.tensor_split(islices, dim=idim_idx)[:-1]  # Omit final slice
        else:  # Vertical
            xs = [x] * len(oslices)

        if odim_idx is not None:  # Diagonal, Vertical
            if threaded:
                ys = _threaded_apply(list(linops), xs, num_workers)
            else:
                ys = [linop(xi) for xi, linop in zip(xs, linops)]
            return torch.concatenate(ys, dim=odim_idx)

        # Horizontal
        if threaded:
            y = _threaded_apply_sum_reduce(list(linops), xs, num_workers)
        else:
            y = 0.0
            for xi, linop in zip(xs, linops):
                y += linop(xi)
        return y

    def size(self, dim):
        if dim == self.idim:
            return sum(self.isizes)
        elif dim == self.odim:
            return sum(self.osizes)
        else:
            for linop in self.linops:
                if linop.size(dim) is not None:
                    return linop.size(dim)

    def split_forward(self, ibatch, obatch):
        """Split concat linop, making a new concat linop if necessary"""
        ibatches = self.subslice(ibatch, self.idim_idx, self.islices, len(self.linops))
        obatches = self.subslice(obatch, self.odim_idx, self.oslices, len(self.linops))

        output_linop_idxs = ibatches.keys() & obatches.keys()
        output_linop_idxs = sorted(list(output_linop_idxs))
        if len(output_linop_idxs) == 0:
            # No linops satisfy this slice (diagonal stacking)
            return Zero(self.ishape, self.oshape)
        elif len(output_linop_idxs) == 1:
            # Singleton linop
            linop_idx = output_linop_idxs.pop()
            linop = self.linops[linop_idx]
            ibatch, obatch = ibatches[linop_idx], obatches[linop_idx]
            return linop.split_forward(ibatch, obatch)
        else:
            output_linop_idxs = sorted(list(output_linop_idxs))
            output_linops = []
            for i in output_linop_idxs:
                linop = self.linops[i]
                ibatch, obatch = ibatches[i], obatches[i]
                output_linops.append(linop.split_forward(ibatch, obatch))
            return self.spinoff(output_linops, idim=self.idim, odim=self.odim)

    @staticmethod
    def subslice(batch: list[slice], dim_idx: Optional[int], slices, num_linops):
        """Given a slice over some dims of a concat linop,
        return a mapping from the linop index to the relevant sub-slice for that linop.
        """
        linops_batch = {}
        if dim_idx is not None:
            slc = batch[dim_idx]
            slice_partition = slices.detach().cpu().numpy().tolist()
            slice_partition.insert(0, 0)
            sub_linop_slices = partition_slices(slice_partition, slc)
            for i, slc in sub_linop_slices:
                sub_linop_batch = copy(batch)
                sub_linop_batch[dim_idx] = slc
                linops_batch[i] = sub_linop_batch
        else:
            for i in range(num_linops):
                linops_batch[i] = batch
        return linops_batch

    def adjoint(self):
        adj_linops = [linop.H for linop in self.linops]
        adj_shape = adj_linops[0].shape
        return self.spinoff(
            linops=adj_linops,
            shape=adj_shape,
            idim=self.odim,
            odim=self.idim,
        )

    def normal(self, inner=None):
        if inner is None:
            # Standardize on this shape
            max_ishape = max_shape([linop.N.ishape for linop in self.linops])
            max_oshape = max_shape([linop.N.oshape for linop in self.linops])
            new_shape = NS(max_ishape, max_oshape)
            if self.idim is None:  # Vertical (inner product)
                linops = [linop.N for linop in self.linops]
                linops = standardize_shapes(linops, new_shape)
                new = Add(*linops)
                new.settings = self.settings  # Copy Threadable settings
                return new
            elif self.odim is None:  # Horizontal (outer product)
                rows = []
                new_idim, new_odim = self._get_new_normal_io_dims(new_shape, self.idim)
                for linop_left in self.linops:
                    row = []
                    for linop_right in self.linops:
                        if linop_left == linop_right:
                            new_linop = linop_right.N
                        else:
                            new_linop = linop_left.H @ linop_right
                        row.append(new_linop)
                        row = standardize_shapes(row, new_shape)
                    rows.append(
                        self.spinoff(
                            linops=row, shape=new_shape, idim=new_idim, odim=None
                        )
                    )
                # rows = standardize_shapes(rows, new_shape)
                return self.spinoff(rows, shape=new_shape, idim=None, odim=new_odim)
            else:  # Diagonal
                diag = []
                new_idim, new_odim = self._get_new_normal_io_dims(new_shape, self.idim)
                for linop in self.linops:
                    diag.append(linop.N)
                diag = standardize_shapes(diag, new_shape)
                return self.spinoff(diag, shape=new_shape, idim=new_idim, odim=new_odim)
        return super().normal(inner)

    @staticmethod
    def _get_new_normal_io_dims(new_shape, dim) -> tuple:
        i = new_shape.ishape.index(dim)
        new_idim = new_shape.ishape[i]
        new_odim = new_shape.oshape[i]
        return new_idim, new_odim

    @staticmethod
    def _check_linop_compatibility(linops: list[NamedLinop]):
        """Ensure linops can actually be concatenated along the requested dimension"""
        target_shape = linops[0].shape
        for linop in linops:
            if not (
                isequal(target_shape.ishape, linop.ishape)
                and isequal(target_shape.oshape, linop.oshape)
            ):
                raise ValueError(
                    f"Incompatible linops being stacked. Target shape: {target_shape} but got linop shape: {linop.shape}"
                )

    def _setup_indices(self, idim, odim):
        ishape = self.linops[0].ishape
        oshape = self.linops[0].oshape
        self.idim, self.isizes, self.islices = self._setup_dim(idim, ishape)
        self.odim, self.osizes, self.oslices = self._setup_dim(odim, oshape)

        if self.idim is None and self.odim is None:
            raise ValueError(f"At least one of idim and odim cannot be None.")

        self.idim_idx = self._infer_dim_idx(self.idim, ishape)
        self.odim_idx = self._infer_dim_idx(self.odim, oshape)

    def _setup_dim(self, dim, shape):
        if dim is not None:
            _dim = ND.infer(dim)
            if any(linop.size(_dim) is None for linop in self.linops):
                raise ValueError(
                    f"Found linop with undefined size for dim {_dim} when attempting concat."
                )
            _sizes = [linop.size(_dim) for linop in self.linops]
            _slices = torch.tensor(_sizes).cumsum(0)  # Keep on CPU
        else:
            _dim = None
            _sizes = None
            _slices = None
        return _dim, _sizes, _slices

    @staticmethod
    def _infer_dim_idx(dim: ND, shape: tuple[ND, ...]) -> int:
        """Get index of dim within requested shape tuple

        Tries to infer index in the presence of ellipses "..." shapes
        Returns positive int if possible
        Otherwise, tries to return negative int
        Fails if neither is possible.

        """
        if dim is None:
            return None
        if dim not in shape:
            raise ValueError(
                f"Provided concat dimension {dim} not found in shape {shape}"
            )
        shape_list = [str(s) for s in shape]
        dim_idx = shape_list.index(str(dim))
        pre, post = shape_list[:dim_idx], shape_list[dim_idx + 1 :]
        if ELLIPSES in pre:
            if ELLIPSES in post:
                raise ValueError(
                    f"Cannot infer concat dimension for dim {dim} from shape {shape}"
                )
            else:
                return -(len(post) + 1)
        return len(pre)

    def __getitem__(self, idx):
        linops = self.linops[idx]
        if isinstance(linops, NamedLinop):
            return linops
        return self.spinoff(linops, idim=self.idim, odim=self.odim)

    def spinoff(self, linops=None, shape=None, idim=None, odim=None):
        """Helper function for creating a new linop using the provided inputs.

        Preserves settings from the original linop.
        """
        linops = linops if linops is not None else self.linops
        shape = shape if shape is not None else self.shape
        new = copy(self)
        new.shape = shape
        new.linops = nn.ModuleList(linops)
        new._setup_indices(idim=idim, odim=odim)
        return new

    def __len__(self):
        return len(self.linops)

    def __repr__(self):
        output = ""
        output += INDENT.indent(self.repr_name + f"({self._shape}\n")
        with INDENT:
            for linop in self.linops:
                output += repr(linop) + "\n"
            output += INDENT.indent(f"idim = {self.idim}, odim = {self.odim}\n")
        output += INDENT.indent(")")
        return output


def get_slice_start_stop(slice_obj, array_length) -> tuple[int, int]:
    """Calculate the start and stop indices of a slice object for an array of a given length.

    Parameters
    ----------
    slice_obj : slice
        The slice object to interpret.
    array_length : int
        The length of the array being indexed.

    Returns
    -------
    tuple of int
        A tuple containing the start and stop indices as integers.

    Examples
    --------
    >>> s = slice(-3, 7)
    >>> length = 10
    >>> get_slice_start_stop(s, length)
    (7, 7)

    >>> s = slice(None, -2)
    >>> length = 5
    >>> get_slice_start_stop(s, length)
    (0, 3)
    """
    # Handle default values for start and stop
    start = slice_obj.start if slice_obj.start is not None else 0
    stop = slice_obj.stop if slice_obj.stop is not None else array_length

    # Adjust negative indices
    if start < 0:
        start += array_length
    if stop < 0:
        stop += array_length

    # Clamp values to be within array bounds
    start = max(0, min(start, array_length))
    stop = max(0, min(stop, array_length))

    return start, stop


def partition_slices(partition, slc):
    """Generate sub-slices for a given slice and partition.

    Parameters
    ----------
    partition : list
        A list of integers representing the partition boundaries [0, a1, a2, ..., ak, N].
        Includes 0 and ending
    slc : slice
        A slice object representing the range [start, stop).

    Returns
    -------
    list
        A list of tuples
            First entry is the index of the interval
            Second entry is sub-slice object corresponding to that interval.

    Examples
    --------
    >>> partition_slices([0, 5, 10, 15, 20], slice(3, 6))
    [(0, slice(3, 5, None)), (1, slice(0, 1, None))]

    >>> partition_slices([0, 5, 10, 15, 20], slice(3, 11))
    [(0, slice(3, 5, None)), (1, slice(0, 5, None)), (2, slice(0, 1, None))]

    """
    # Validate the partition
    if partition[0] != 0:
        raise ValueError("The first boundary in the partition must be 0.")
    if not all(partition[i] <= partition[i + 1] for i in range(len(partition) - 1)):
        raise ValueError("Partition boundaries must be non-decreasing.")
    if slc.stop is not None and slc.stop > partition[-1]:
        raise ValueError(
            f"Slice {slc} is out of range for array of length {partition[-1]}"
        )
    if slc.step is not None and slc.step != 1:
        raise NotImplementedError(
            "Partition slicing with step != 1 is not currently supported."
        )

    start, stop = get_slice_start_stop(slc, partition[-1])
    result = []

    for i in range(len(partition) - 1):
        # Interval boundaries
        interval_start = partition[i]
        interval_end = partition[i + 1]

        # Check overlap of slice with the current interval
        if interval_end <= start:  # Slice starts after the interval
            continue
        if interval_start >= stop:  # Slice ends before the interval
            break

        # Calculate sub-slice within the current interval
        sub_start = max(start, interval_start)
        sub_stop = min(stop, interval_end)
        result.append((i, slice(sub_start - interval_start, sub_stop - interval_start)))

    return result


def dynamic_slice(t: Tensor, dim: int, slc: slice | int):
    """Access a tensor at dimension dim with slice slc

    Examples
    --------
    >>> dynamic_slice(torch.randn(2, 3, 4), 2, slc=slice(0, 3)).shape
    torch.Size([2, 3, 3])

    >>> dynamic_slice(torch.randn(2, 3, 4), 2, slc=1).shape
    torch.Size([2, 3])

    """
    if isinstance(slc, int):
        slc = torch.tensor(slc)
    indices = [slice(None)] * t.ndim
    indices[dim] = slc
    return t[indices]
