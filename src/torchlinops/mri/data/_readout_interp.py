"""
https://github.com/danielabrahamgit/igrog/blob/main/src/igrog/readout_interp.py
Retrieved 25 March 2024
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import sigpy as sp
from scipy.signal import resample_poly
from torchaudio.functional import resample
# from mr_recon.utils.func import np_to_torch, torch_to_np


def upsample1d(
    x: torch.Tensor,
    oversamp: float,
    width: int = 120,
    resamp_method: str = "sinc_interp_kaiser",
):
    """
    x : torch.Tensor
        Tensor to upsample, of shape [..., time]
    oversamp : float
        Oversampling ratio
    width : int
        Width of the lowpass filter
    resamp_method : str
        Resampling method to use. See https://pytorch.org/audio/stable/generated/torchaudio.functional.resample.html#torchaudio.functional.resample for options

    """

    def _resample(x):
        return resample(
            x,
            orig_freq=1,
            new_freq=oversamp,
            lowpass_filter_width=width,
            resampling_method=resamp_method,
        )

    if torch.is_complex(x):
        x_real = _resample(x.real)
        x_imag = _resample(x.imag)
        return x_real + 1j * x_imag
    return _resample(x)


@dataclass
class ReadoutInterpolatorConfig:
    oversamp: float = 4.0
    width: int = 120
    resamp_method: str = "sinc_interp_kaiser"


class ReadoutInterpolator(nn.Module):
    """
    Performs 1D readout interpolation
    """

    def __init__(self, ksp: torch.Tensor, config: ReadoutInterpolatorConfig):
        """
        Parameters:
        -----------
        ksp - torch tensor <complex> | CPU
            k-space data with shape (..., nro), where nro is the readout dimension
        oversamp - int
            oversampling rate along readout direction
        device - torch.device
            GPU device
        """
        super().__init__()
        self.config = config

        # upsample
        if self.config.oversamp != 1:
            ksp = upsample1d(
                ksp,
                self.config.oversamp,
                self.config.width,
                self.config.resamp_method,
            )
        self.ksp = nn.Parameter(ksp, requires_grad=False)

    def forward(self, c):
        """
        Interpolates input to new_inds.

        Parameters:
        -----------
        c : torch.Tensor
            Coordinates in which to interpolate the readout
            Shape [..., 1] for 1D interpolation along readout

        Returns:
        --------
        torch.Tensor
            The interpolated points at coordinates c
        """
        assert c.shape[-1] == 1, "Readout Interpolation is 1D only"
        c = c * self.config.oversamp  # Adjust for oversampling
        c_sp = sp.from_pytorch(c)
        dev = sp.get_device(c_sp)
        xp = dev.xp
        ksp_th = torch.stack((self.ksp.real, self.ksp.imag), dim=-1)
        ksp_sp = sp.from_pytorch(ksp_th, iscomplex=True)
        with dev:
            ksp_interp = sp.interpolate(ksp_sp, c_sp)
        ksp_interp = sp.to_pytorch(ksp_interp)
        ksp_interp = torch.view_as_complex(ksp_interp.contiguous())
        return ksp_interp
