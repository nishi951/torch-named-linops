"""Shift-free sampling NUFFT (folds FFT centering into gather indices + phase).

See GitHub issue #207.
"""

import math
from functools import lru_cache
from typing import ClassVar

import torch
from jaxtyping import Integer
from torch import Tensor, nn

from torchlinops import config

from ...nameddim import NamedShape as NS
from ..diagonal import Diagonal
from ..fft import FFT
from ..pad_last import Pad
from ..sampling import Sampling
from .sampling_nufft import SamplingNUFFT

__all__ = ["FastSamplingNUFFT"]


class FastSamplingNUFFT(SamplingNUFFT):
    """Non-uniform FFT (type II, sampling mode) without the FFT shift sandwich.

    ``SamplingNUFFT`` builds ``[Pad, FFT(centered=True), Sampling]``. The centered
    FFT copies the full grid 3x per axis (ifftshift/fftshift rolls), in forward
    and again through autograd's backward. This subclass applies the shift
    theorem instead: it runs a plain ``fftn`` on the padded grid and folds the
    centering into the gather indices plus a per-sample complex phase,
    precomputed once per trajectory:

        fftshift(fftn(ifftshift(x)))[l] = S(k) * fftn(x)[k],
        k = (l - N//2) mod N,
        S(k) = exp(-2j*pi * sum_d k_d * ((N_d + 1)//2) / N_d)

    For all-even padded sizes ``S`` reduces to the parity sign
    ``(-1)**sum_d k_d``. Computes the same operator as ``SamplingNUFFT`` (to
    complex64 rounding); inherits ``prep_locs``, ``device``, and options.

    The centering phase lives at either end of the gather, selectable via the
    ``phase_placement`` option (issue #210):

    - ``"samples"`` — phase ``Diagonal`` after ``Sampling`` (per-sample
      phase): ``O(nlocs)`` multiplies forward+backward.
    - ``"grid"`` — per-axis broadcast ``Diagonal``s on the uncentered fftn
      grid before ``Sampling``: ``O(prod padded_size)`` multiplies
      forward+backward, ``O(sum N_d)`` storage.
    - ``"auto"`` (default) — ``"grid"`` iff the per-batch-element sample count
      (leading ``locs`` dims not named in ``batch_shape``) exceeds
      ``prod(padded_size)``; falls back to the full ``locs`` leading product
      when the positional alignment with ``output_shape`` is ambiguous;
      resolved at build time, recorded as ``self.phase_placement_resolved``.

    Both placements compute the same operator: the unit-modulus phase
    commutes with the gather.
    """

    default_options: ClassVar[dict] = {
        **SamplingNUFFT.default_options,
        "phase_placement": "auto",
    }

    @staticmethod
    def resolve_phase_placement(
        phase_placement: str, nlocs: int, padded_size: tuple[int, ...]
    ) -> str:
        """Resolve the ``phase_placement`` option to a concrete placement.

        Parameters
        ----------
        phase_placement : str
            One of ``"samples"``, ``"grid"``, ``"auto"``.
        nlocs : int
            Number of gathered sample points, ``FastSamplingNUFFT.sample_count``.
        padded_size : tuple[int, ...]
            Oversampled grid size, one entry per spatial axis.

        Returns
        -------
        str
            ``"grid"`` iff ``nlocs > prod(padded_size)`` when ``"auto"``
            (ties resolve to ``"samples"``); otherwise the literal
            ``"samples"``/``"grid"``.

        Raises
        ------
        ValueError
            If ``phase_placement`` is unrecognized.
        """
        if phase_placement == "auto":
            return "grid" if nlocs > math.prod(padded_size) else "samples"
        if phase_placement in ("samples", "grid"):
            return phase_placement
        raise ValueError(
            f"phase_placement must be 'samples', 'grid', or 'auto' but got "
            f"{phase_placement!r}"
        )

    def build(self):
        ndim = len(self.grid_size)
        padded_size = tuple(int(i * self.options["oversamp"]) for i in self.grid_size)
        self.phase_placement_resolved = self.resolve_phase_placement(
            self.options["phase_placement"],
            math.prod(self.locs.shape[:-1]),
            padded_size,
        )

        # Inherited, cached locs preparation: l_p = centered gather indices
        if config.cache_nufft_parameters:
            locs_prepared = self.prep_locs(self.locs, self.grid_size, padded_size)
        else:
            locs_prepared = self.prep_locs.__wrapped__(
                self.locs, self.grid_size, padded_size
            )

        pad = Pad(
            padded_size,
            self.grid_size,
            in_shape=self.input_shape,
            batch_shape=self.batch_shape,
        )

        # Plain (uncentered) FFT: no shift sandwich anywhere in the chain
        fft = FFT(
            ndim=ndim,
            centered=False,
            batch_shape=self.batch_shape,
            grid_shapes=(pad.out_im_shape, self.input_kshape),
        )

        grid_shape = fft._shape.output_grid_shape
        # Uncentered gather indices: idx = (l_p - N//2) mod N, identical for
        # both placements; for "samples" the phase comes from fold_centering.
        size = torch.tensor(
            padded_size, dtype=locs_prepared.dtype, device=locs_prepared.device
        )
        idx = FastSamplingNUFFT.ifftshift_locs(locs_prepared, size)
        sampling = Sampling.from_stacked_idx(
            idx,
            dim=-1,
            input_size=padded_size,
            output_shape=self.output_shape,
            input_shape=grid_shape,
            batch_shape=self.batch_shape,
        )

        if self.phase_placement_resolved == "samples":
            # Cached fold: per-sample centering phase on the gathered output;
            # broadcasts over batch dims. Unit modulus -> adjoint/normal exact.
            if config.cache_nufft_parameters:
                phase = FastSamplingNUFFT.fold_centering(locs_prepared, padded_size)
            else:
                phase = FastSamplingNUFFT.fold_centering.__wrapped__(
                    locs_prepared, padded_size
                )
            oshape = NS(self.batch_shape) + NS(self.output_shape)
            phase_diag = Diagonal(phase, oshape.ishape)
            phase_diag.name = "FoldedCenteringPhase"
            return [pad, fft, sampling, phase_diag]
        # Grid placement: the same phase applied on the full uncentered fftn
        # grid as D per-axis broadcast Diagonals BEFORE the gather
        phase_diags = FastSamplingNUFFT.grid_phase_diags(
            padded_size,
            grid_shape,
            self.batch_shape,
            device=locs_prepared.device,
        )
        return [pad, fft, *phase_diags, sampling]

    @staticmethod
    def grid_phase_ramps(padded_size: tuple[int, ...], device=None) -> list[Tensor]:
        """Per-axis phase ramps for the uncentered fftn output grid.

        Parameters
        ----------
        padded_size : tuple[int, ...]
            Oversampled grid size, one entry per spatial axis.
        device : torch.device, optional

        Returns
        -------
        list[Tensor]
            D complex64 tensors of shape ``(N_d,)``:
            ``R_d[m] = exp(-2j*pi*m*rolls_d/N_d)``, ``rolls_d = (N_d+1)//2``.
            Parity sign ``(-1)**m`` on even axes. Unit modulus.
            Stored factored: the full-grid outer product ``R_1 * ... * R_D``
            (``prod(padded_size)`` entries) is never materialized; applying
            the ramps in sequence yields it exactly.
        """
        ramps = []
        for n in padded_size:
            rolls = (n + 1) // 2
            m = torch.arange(n, dtype=torch.float32, device=device)
            ramps.append(torch.exp(-2j * torch.pi * m * rolls / n))
        return ramps

    @staticmethod
    def grid_phase_diags(padded_size, grid_shape, batch_shape, device=None):
        """Broadcast Diagonals applying the grid phase, one per axis."""
        ndim = len(padded_size)
        oshape = NS(batch_shape) + NS(grid_shape)
        diags = []
        ramps = FastSamplingNUFFT.grid_phase_ramps(padded_size, device=device)
        for d, (ramp, n) in enumerate(zip(ramps, padded_size)):
            # Broadcast over all other grid axes (and any batch dims); torch
            # broadcasting keeps this a per-axis multiply, never O(prod N).
            weight = ramp.view((1,) * d + (n,) + (1,) * (ndim - 1 - d))
            broadcast_dims = tuple(grid_shape[i] for i in range(ndim) if i != d)
            diag = Diagonal(weight, oshape.ishape, broadcast_dims=broadcast_dims)
            diag.name = f"GridPhase{d}"
            diags.append(diag)
        return diags

    @staticmethod
    @lru_cache(maxsize=64)
    def fold_centering(
        locs_prepared: Integer[Tensor, "... D"],
        padded_size: tuple[int, ...],
    ) -> Tensor:
        """Fold the centered-FFT sandwich into plain-FFT gather indices + phase.

        Parameters
        ----------
        locs_prepared : Integer[Tensor, "... D"]
            Output of ``SamplingNUFFT.prep_locs`` (centered gather indices in
            ``[0, N_d - 1]``), dtype int64.
        padded_size : tuple[int, ...]
            Oversampled grid size, one entry per spatial axis.

        Returns
        -------
        idx : Tensor
            int64 tensor, same shape as ``locs_prepared``: gather indices into
            the *uncentered* ``fftn`` output grid.
        phase : Tensor
            complex64 tensor of shape ``locs_prepared.shape[:-1]``: per-sample
            unit-modulus phase ``S(k)``.
        """
        size = torch.tensor(
            padded_size, dtype=locs_prepared.dtype, device=locs_prepared.device
        )
        idx = FastSamplingNUFFT.ifftshift_locs(locs_prepared, size)
        rolls = (size + 1) // 2
        angle = (
            idx.to(torch.float32) * rolls.to(torch.float32) / size.to(torch.float32)
        ).sum(dim=-1)
        phase = torch.exp(-2j * torch.pi * angle)
        return phase

    @staticmethod
    def ifftshift_locs(locs_prepared, size):
        return torch.remainder(locs_prepared - size // 2, size)
