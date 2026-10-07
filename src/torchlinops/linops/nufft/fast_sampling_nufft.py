"""Shift-free sampling NUFFT (folds FFT centering into gather indices + phase).

See GitHub issue #207.
"""

from functools import lru_cache

import torch
from jaxtyping import Integer
from torch import Tensor

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
    """

    def build(self):
        ndim = len(self.grid_size)
        padded_size = tuple(int(i * self.options["oversamp"]) for i in self.grid_size)

        # Inherited, cached locs preparation: l_p = centered gather indices
        if config.cache_nufft_parameters:
            locs_prepared = self.prep_locs(self.locs, self.grid_size, padded_size)
            idx, phase = self.fold_centering(locs_prepared, padded_size)
        else:
            locs_prepared = self.prep_locs.__wrapped__(
                self.locs, self.grid_size, padded_size
            )
            idx, phase = self.fold_centering.__wrapped__(locs_prepared, padded_size)

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
        sampling = Sampling.from_stacked_idx(
            idx,
            dim=-1,
            input_size=padded_size,
            output_shape=self.output_shape,
            input_shape=grid_shape,
            batch_shape=self.batch_shape,
        )

        # Per-sample centering phase on the gathered output; broadcasts over batch
        # dims. Unit modulus, so adjoint/normal stay exact.
        oshape = NS(self.batch_shape) + NS(self.output_shape)
        phase_diag = Diagonal(phase, oshape.ishape)
        phase_diag.name = "FoldedCenteringPhase"

        return [pad, fft, sampling, phase_diag]

    def post_init_hook(self):
        # Inherited hook sets .pad/.fft/.interp (identical positions in our chain)
        super().post_init_hook()
        self.phase_diag = self.linops[3]

    @staticmethod
    @lru_cache(maxsize=64)
    def fold_centering(
        locs_prepared: Integer[Tensor, "... D"],
        padded_size: tuple[int, ...],
    ) -> tuple[Tensor, Tensor]:
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
        idx = torch.remainder(locs_prepared - size // 2, size)
        rolls = (size + 1) // 2
        angle = (
            idx.to(torch.float32) * rolls.to(torch.float32) / size.to(torch.float32)
        ).sum(dim=-1)
        phase = torch.exp(-2j * torch.pi * angle)
        return idx, phase
