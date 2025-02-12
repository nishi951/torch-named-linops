from typing import Optional
from jaxtyping import Integer
from torch import Tensor

from copy import copy

import torch
import torch.nn as nn

from .namedlinop import NamedLinop
from .add import Add
from .identity import Zero
from .nameddim import NS, isequal, ELLIPSES, NDorStr, ND

from torchlinops.functional import slice2range

from torchlinops.utils import default_to, INDENT

__all__ = ["Stack"]


class Stack(NamedLinop):
    """Concatenate some linops along a new dimension

    Linops need not output tensors of the same size, but they should
    output tensors of the same number of dimensions

    Stacking type depends on dimensions provided

    Horizontal stacking
    stacking along an input dimension:

    A B C

    Vertical stacking
    stacking along an output dimension:

    A
    B
    C

    Diagonal stacking:
    stacking along a separate input and output dimensions

    A . .
    . B .
    . . C


    """

    def __init__(
        self,
        *linops,
        idim_and_idx: Optional[tuple[NDorStr, int]] = (None, None),
        odim_and_idx: Optional[tuple[NDorStr, int]] = (None, None),
    ):
        """
        stack_input_dim / stack_output_dim : int
            If not None, inputs will be stacked (rather than concatenated) along the requested
            dimension. idim / odim must NOT be present if the respective stack_* flag is set.
        """
        self._check_linop_compatibility(linops)

        self.idim, self.idim_idx, ishape = self._get_dim_and_idx(
            *idim_and_idx, linops[0].ishape
        )
        self.odim, self.odim_idx, oshape = self._get_dim_and_idx(
            *odim_and_idx, linops[0].oshape
        )

        # Initialize parent class
        super().__init__(NS(ishape, oshape))
        self.linops = nn.ModuleList(list(linops))

    @staticmethod
    def _get_dim_and_idx(dim, idx, shape):
        if dim is not None:
            dim = ND.infer(dim)
            if dim in shape:
                raise ValueError(
                    f"Stack linop attempting to add dim {dim} to shape {shape} but shape already contains {dim}"
                )
            shape = list(shape)
            shape.insert(idx, dim)
        else:
            dim = None
            idx = None
        return dim, idx, shape

    def forward(self, x):
        return self.fn(self, x)

    @staticmethod
    def fn(stack, x):
        return stack._fn(
            x,
            stack.linops,
            stack.idim_idx,
            stack.odim_idx,
        )

    @staticmethod
    def adj_fn(stack, x):
        adj_linops = [linop.H for linop in stack.linops]
        return stack._fn(
            x,
            adj_linops,
            stack.odim_idx,
            stack.idim_idx,
        )

    @staticmethod
    def _fn(x: Tensor, linops, idim_idx, odim_idx):
        """Unifies forward and adjoint functionality for stacked linops"""
        # Split inputs
        if idim_idx is not None:  # Diagonal, Horizontal
            if len(linops) != x.shape[idim_idx]:
                raise ValueError(
                    f"Stack Linop expecting input of size {len(linops)} at dim {idim_idx} got input of size {x.shape} with non-matching stack size {x.shape[idim_idx]}"
                )
            xs = x.tensor_split(len(linops), idim_idx)
            xs = [xi.squeeze(idim_idx) for xi in xs]
        else:  # Vertical
            xs = [x] * len(linops)

        # Compute linop(x) for all xs
        ys = []
        for xi, linop in zip(xs, linops):
            ys.append(linop(xi))

        # Combine outputs
        if odim_idx is not None:  # Diagonal, Vertical
            return torch.stack(ys, dim=odim_idx)
        # Horizontal
        return sum(ys)

    @staticmethod
    def normal_fn(stack, x):
        return stack.adj_fn(stack, stack.fn(stack, x))

    def size(self, dim):
        return self.size_fn(dim)

    def size_fn(self, dim, /):
        if dim == self.idim or dim == self.odim:
            return len(self.linops)
        else:
            return self.linops[0].size(dim)

    def split_forward(self, ibatch, obatch):
        """Split stack linop"""
        linop_idxs = set(range(len(self.linops)))
        for i, slc in enumerate(ibatch):
            if i == self.idim_idx:
                linop_idxs &= set(slice2range(slc, len(self.linops)))
        for i, slc in enumerate(obatch):
            if i == self.odim_idx:
                linop_idxs &= set(slice2range(slc, len(self.linops)))

        if len(linop_idxs) == 0:
            # No linops satisfy this slice (diagonal stacking)
            return Zero(self.ishape, self.oshape)
            # elif len(output_linop_idxs) == 1:
            # else:
        linop_idxs = sorted(list(linop_idxs))
        output_linops = []
        # Remove stack dims from slice batch
        if self.idim_idx is not None:
            ibatch.pop(self.idim_idx)
        if self.odim_idx is not None:
            obatch.pop(self.odim_idx)

        # Slice each sub-linop
        for i in linop_idxs:
            linop = self.linops[i]
            output_linops.append(linop.split_forward(ibatch, obatch))
        return type(self)(
            *output_linops,
            idim_and_idx=(self.idim, self.idim_idx),
            odim_and_idx=(self.odim, self.odim_idx),
        )

    def split_forward_fn(self, ibatch, obatch, data_list):
        """Split stack linop, making a new stack linop if necessary

        Parameters
        ----------
        data_list : list, same length as linops
            List of data for each linop in this stack linop

        """
        linop_idxs = set(range(len(self.linops)))
        for i, slc in enumerate(ibatch):
            if i == self.idim_idx:
                linop_idxs &= set(slice2range(slc, len(self.linops)))
        for i, slc in enumerate(obatch):
            if i == self.odim_idx:
                linop_idxs &= set(slice2range(slc, len(self.linops)))

        if len(linop_idxs) == 0:
            # No linops satisfy this slice (diagonal stacking)
            return 0.0  # TODO is this ok
        linop_idxs = sorted(list(linop_idxs))
        output_linop_data = []
        # Remove stack dims from slice batch
        if self.idim_idx is not None:
            ibatch = copy(ibatch)
            ibatch.pop(self.idim_idx)
        if self.odim_idx is not None:
            obatch = copy(obatch)
            obatch.pop(self.odim_idx)

        # Slice each sub-linop
        for i in linop_idxs:
            linop = self.linops[i]
            data = data_list[i]
            output_linop_data.append(linop.split_forward_fn(ibatch, obatch, data))
        return output_linop_data

    def adjoint(self):
        adj_linops = [linop.H for linop in self.linops]
        return type(self)(
            *adj_linops,
            idim_and_idx=(self.odim, self.odim_idx),
            odim_and_idx=(self.idim, self.idim_idx),
        )

    def normal(self, inner=None):
        if inner is None:
            if self.idim is None:  # Vertical (inner product)
                # self.odim is not None
                return Add(*(linop.N for linop in self.linops))
            elif self.odim is None:  # Horizontal (outer product)
                # self.idim is not None
                new_idim, new_odim = self._get_new_normal_io_dims(
                    self._shape, self.idim
                )
                rows = []
                new_shape = self.linops[0].shape.N
                for linop_left in self.linops:
                    row = []
                    for linop_right in self.linops:
                        if linop_left == linop_right:
                            new_linop = linop_right.N
                        else:
                            new_linop = linop_left.H @ linop_right
                            new_linop.ishape = new_shape.ishape
                            new_linop.oshape = new_shape.oshape
                        row.append(new_linop)
                    row = type(self)(
                        *row,
                        idim_and_idx=(new_idim, self.idim_idx),
                        odim_and_idx=(None, None),
                    )
                    rows.append(row)
                return type(self)(
                    *rows,
                    idim_and_idx=(None, None),
                    odim_and_idx=(new_odim, self.idim_idx),
                )
            else:  # Diagonal
                # self.idim and self.odim are not None
                diag = []
                new_idim, new_odim = self._get_new_normal_io_dims(
                    self._shape, self.idim
                )
                for linop in self.linops:
                    diag.append(linop.N)
                return type(self)(
                    *diag,
                    idim_and_idx=(new_idim, self.idim_idx),
                    odim_and_idx=(new_odim, self.odim_idx),
                )
        return super().normal(inner)

    @staticmethod
    def _get_new_normal_io_dims(shape, dim) -> tuple:
        new_shape = shape.N
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

    def __getitem__(self, idx):
        linops = self.linops[idx]
        if isinstance(linops, NamedLinop):
            return linops
        return type(self)(
            *linops,
            idim_and_idx=(self.idim, self.idim_idx),
            odim_and_idx=(self.odim, self.odim_idx),
        )

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
