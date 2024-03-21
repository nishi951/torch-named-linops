from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Union

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn

from .block import Block


__all__ = [
    "LocallyLowRankConfig",
    "LocallyLowRank",
    "ScheduledLLR",
]


@dataclass
class LocallyLowRankConfig:
    block_size: Union[Tuple[int, int], Tuple[int, int, int]]
    block_stride: Union[Tuple[int, int], Tuple[int, int, int]]
    threshold: float
    shift_increment: Union[int, str] = 1
    """Int or 'random' """


class LocallyLowRank(nn.Module):
    """Version of LLR mimicking Sid's version in Sigpy

    Language based on spatiotemporal blocks
    """

    def __init__(
        self,
        input_size: Tuple,
        config: LocallyLowRankConfig,
        input_type: Optional[Callable] = None,
    ):
        super().__init__()
        self.input_type = input_type if input_type is not None else torch.complex64
        self.config = config

        # Derived
        self.block = Block(self.config.block_size, self.config.block_stride)
        self.block_weights = nn.Parameter(
            self.block.precompute_normalization(input_size).type(self.input_type),
            requires_grad=False,
        )

        # Bookkeeping
        self.shift = (0,) * len(self.config.block_size)

    def forward(self, x: torch.Tensor):
        """
        x: [N A H W [D]]
          - N: Batch dim
          - A: Temporal (subspace) dim
          - H, W, [D]: spatial dims

        """
        assert x.dim() >= 4

        # Roll in each axis by some shift amount
        x = torch.roll(x, self.shift, dims=tuple(range(-len(self.shift), 0)))

        # Extract Blocks
        x, nblocks = self.block(x)

        # Combine within-block dimensions
        # Move temporal dimension to be last
        unblocked_shape = x.shape  # Save block shape for later
        x = rearrange(x, "n a b ... -> n b (...) a")

        # Take SVD
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        # Threshold
        S = S - self.config.threshold
        S[S < 0] = 0.0
        S = S.type(U.dtype)

        # Recompose blocks
        x = U @ torch.diag_embed(S) @ Vh

        # Unblock and normalize
        x = rearrange(x, "n b x a -> n a b x")
        x = x.reshape(*unblocked_shape)
        x = self.block.adjoint(x, nblocks, norm_weights=self.block_weights)

        # Undo the roll in each shift direction
        x = torch.roll(
            x, tuple(-i for i in self.shift), dims=tuple(range(-len(self.shift), 0))
        )

        # Update the shift amount
        if isinstance(self.config.shift_increment, int):
            self.shift = tuple(i + self.config.shift_increment for i in self.shift)
        elif self.config.shift_increment == "random":
            self.shift = tuple(np.random.randint(0, self.config.block_stride))
        else:
            raise ValueError(f"Invalid shift increment: {self.config.shift_increment}")

        # Return the thresholded input
        return x

    def forward_mrf(self, x: torch.Tensor):
        """Simple wrapper that fixes dimensions
        x: [A H W [D]]

        Adds batch dim
        """
        assert x.dim() == 3 or x.dim() == 4
        x = x[None, ...]
        x = self(x)
        x = x[0, ...]
        return x


class ScheduledLLR(nn.Module):
    def __init__(
        self,
        llr_module: LocallyLowRank,
        schedule: Optional[Callable] = None,
    ):
        """
        schedule: Function that, when given an iteration and the
            current LLR object, returns the threshold to use.
        """
        super().__init__()
        self.llr = llr_module
        self.schedule = schedule
        self._iteration = 0

    def forward(self, x: torch.Tensor):
        self.llr.config.threshold = self.schedule(
            self.llr.config.threshold, self._iteration
        )
        x = self.llr(x)
        self._iteration += 1
        return x

    def forward_mrf(self, x: torch.Tensor):
        """Simple wrapper that fixes dimensions
        x: [A H W [D]]

        Adds batch dim
        """
        assert x.dim() == 3 or x.dim() == 4
        x = x[None, ...]
        x = self(x)
        x = x[0, ...]
        return x
