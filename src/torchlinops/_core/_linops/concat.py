from typing import Optional
from torch import Tensor

import torch
import torch.nn as nn

from .namedlinop import NamedLinop
from .add import Add
from .nameddim import NS, isequal, ELLIPSES
from .nameddim._nameddimcollection import NDorStr, ND

from torchlinops.utils import default_to

__all__ = ["Concat"]


class Concat(NamedLinop):
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
        idim: Optional[NDorStr] = None,
        odim: Optional[NDorStr] = None,
    ):
        self._check_linop_compatibility(linops)

        super().__init__(NS(linops[0].ishape, linops[0].oshape))

        self.linops = nn.ModuleList(list(linops))

        # Handle ishape
        ishape = linops[0].ishape
        if idim is not None:
            self.idim = ND.infer(idim)
            if any(linop.size(self.idim) is None for linop in linops):
                raise ValueError(
                    f"Found linop with undefined size for dim {self.odim} when attempting concat."
                )
            self.isizes = [linop.size(self.idim) for linop in linops]
            self.islices = torch.tensor(self.isizes).cumsum(0)
        else:
            self.idim = None
            self.isizes = None
            self.islices = None

        # Handle oshape
        oshape = linops[0].oshape
        if odim is not None:
            self.odim = ND.infer(odim)
            if any(linop.size(self.odim) is None for linop in linops):
                raise ValueError(
                    f"Found linop with undefined size for dim {self.odim} when attempting concat."
                )
            self.osizes = [linop.size(self.odim) for linop in linops]
            self.oslices = torch.tensor(self.osizes).cumsum(0)
        else:
            self.odim = None
            self.osizes = None
            self.oslices = None

        self.idim_idx = self._infer_dim_idx(self.idim, ishape)
        self.odim_idx = self._infer_dim_idx(self.odim, oshape)
        # if self.idim in ishape and self.odim in oshape:
        #     # self.concat_type = "diagonal"
        #     self._idim = self._infer_dim_idx(self.dim, ishape)
        #     self._odim = self._infer_dim_idx(self.dim, oshape)
        # elif self.idim in ishape:
        #     # self.concat_type = "horizontal"
        #     self._idim = self._infer_dim_idx(self.dim, ishape)
        #     self._odim = None
        # elif self.odim in oshape:
        #     # self.concat_type = "vertical"
        #     self._idim = None
        #     self._odim = self._infer_dim_idx(self.dim, oshape)
        # else:
        #     raise ValueError(
        #         f"Attempted to stack linops along dim {self.dim} but dimension was not found in linop[0] with shape {linops[0].shape}"
        #     )

    def forward(self, x):
        return self.fn(self, x)

    @staticmethod
    def fn(concat, x):
        return concat._fn(
            x,
            concat.linops,
            concat.idim_idx,
            concat.odim_idx,
            concat.islices,
            concat.oslices,
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
        )

    @staticmethod
    def _fn(x: Tensor, linops, idim_idx, odim_idx, islices, oslices):
        """Unifies forward and adjoint functionality for stacked linops"""
        # Split inputs
        if idim_idx is not None:  # Diagonal, Horizontal
            # if sum(sizes) != x.shape[idim_idx]:
            if islices[-1] != x.shape[idim_idx]:
                raise ValueError(
                    f"Concat Linop expecting input of size {islices[-1]} got input of size {x.shape} with non-matching concat size {x.shape[idim_idx]}"
                )
            xs = x.tensor_split(islices, dim=idim_idx)[:-1]  # Omit final slice
        else:  # Vertical
            xs = [x] * len(oslices)

        # Compute linop(x) for all xs
        ys = []
        for xi, linop in zip(xs, linops):
            ys.append(linop(xi))

        # Combine outputs
        if odim_idx is not None:  # Diagonal, Vertical
            return torch.concatenate(ys, dim=odim_idx)
        # Horizontal
        return sum(ys)

    @staticmethod
    def normal_fn(concat, x):
        return concat.adj_fn(concat, concat.fn(concat, x))

    def size(self, dim):
        return self.size_fn(dim)

    def size_fn(self, dim, /):
        if dim == self.idim:
            return sum(self.isizes)
        elif dim == self.odim:
            return sum(self.osizes)
        else:
            return self.linops[0].size(dim)

    def adjoint(self):
        adj_linops = [linop.H for linop in self.linops]
        return type(self)(*adj_linops, self.odim, self.idim)

    def normal(self, inner=None):
        if inner is None:
            if self.idim is None:  # Vertical (inner product)
                return Add(linop.N for linop in self.linops)
            elif self.odim is None:  # Horizontal (outer product)
                new_idim, new_odim = self._get_new_normal_io_dims(
                    self.linops[0].shape, self.idim
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
                            new_linop.shape = new_shape
                        row.append(new_linop)
                        row = type(self)(*row, new_idim, None)
                    rows.append(row)
                return type(self)(*rows, None, new_odim)
            else:  # Diagonal
                diag = []
                new_idim, new_odim = self._get_new_normal_io_dims(
                    self.linops[0].shape, self.idim
                )
                for linop in self.linops:
                    diag.append(linop.N)
                return type(self)(*diag, new_idim, new_odim)
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
        return type(self)(*linops, self.idim, self.odim)

    def __len__(self):
        return len(self.linops)

    def __repr__(self):
        linop_chain = "\n\t".join(repr(linop) for linop in self.linops)
        return f"{self.__class__.__name__}(\n\t{linop_chain}\n\tidim = {self.idim}, odim = {self.odim}\n)"


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
