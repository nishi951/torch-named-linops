import torch
import numpy as np
import sigpy as sp
import os

from sigpy.fourier import _get_oversamp_shape, _apodize, _scale_coord
from dataclasses import dataclass, field, asdict
from typing import Optional, Mapping
from einops import rearrange

__all__ = [
    'time_segment_helper'
    'CalibRegion',
    'CalibHparams',
]


@dataclass
class CalibHparams:
    time_oversamp: float = 10.
    """Factor above the maximum b0 frequency that temporal planes are sampled.
    2 = nyquist
    10 = linear
    """

class CalibRegion:
    def __init__(
            self,
            ksp_cal, # Spatial
            hparams: CalibHparams,
            b0_map: Optional[np.ndarray] = None, # [Hz]
            dt: Optional[float] = 4e-6,
            segment_length: Optional[int] = None, # Length of time segment, in number of samples
            device_idx: Optional[int] = -1
            # Eddy
            # Motion
            # ...
    ):
        """
        ksp_cal: [nc, *cal_size]
        b0_map: [*im_size]
        """

        # Useful consts
        self.hparams = hparams
        self.ksp_cal = ksp_cal
        self.b0_map = b0_map
        self.dim = len(ksp_cal.shape[1:])
        self.nc = ksp_cal.shape[0]
        self.cal_size = self.ksp_cal.shape[1:]

        # GPU
        self.device = sp.Device(device_idx)

        if b0_map is not None:
            assert segment_length is not None, 'Must provide segment_length when b0 calibration is requested.'
            self.segment_length = segment_length
            im_size = b0_map.shape
            assert len(self.cal_size) == len(im_size), 'Calibration region and B0 map have different dimensionality'

            # Reshape calibratoin to size of b0 map
            ksp_cal_rs = sp.resize(self.ksp_cal, (self.nc, *b0_map.shape))
            img_cal = sp.ifft(ksp_cal_rs, axes=range(-self.dim, 0))

            # Create T calibrations, where T number of samples along time
            max_b0_Hz = np.abs(b0_map).max()
            sampling_rate = (max_b0_Hz * self.hparams.time_oversamp).astype(int) + 1

            segment_time = self.segment_length * dt
            T = int(np.ceil(segment_time * sampling_rate))
            self.dt_max = (segment_time + 1e-7) / 2
            self.ts = np.linspace(-self.dt_max, self.dt_max, T)
            self.delta_t = self.ts[1] - self.ts[0]
            self.ts = sp.to_device(self.ts, sp.get_device(b0_map))
            temporal_img_cal = self._combine_calib_with_b0(img_cal, b0_map, self.ts)

            # Preproccesing steps for KB interpolation later
            self.temporal_ksp_cal_pre_KB = self._FT_pre_KB(img_cal=temporal_img_cal,
                                                           spatial_oshape=ksp_cal_rs.shape[1:])
            self.temporal_orig_shape = temporal_img_cal.shape
            self.temporal_ksp_cal_pre_KB = sp.to_device(self.temporal_ksp_cal_pre_KB, self.device)

        # Kinda silly but its ok
        with self.device:
            self.img_cal = sp.to_device(sp.ifft(self.ksp_cal, axes=range(-self.dim, 0)), self.device)
        self.orig_shape = self.img_cal.shape
        self.ksp_cal_pre_KB = self._FT_pre_KB(img_cal=self.img_cal,
                                              spatial_oshape=self.ksp_cal.shape[1:])

    def __call__(
            self,
            loc: np.ndarray,
            dts: Optional[np.ndarray] = None,
            #motion: Optional[np.ndarray] = None,
    ):
        """
        D = 2 or 3: spatial ksp dimension

        loc: [batch_size, D] in [-N//2, N//2]
            - N is the calib size
        dts: [batch_size]
        motion: TODO

        Returns
        -------
        ksp: [batch_size, nc]
        """
        if dts is not None:
            assert loc.shape[0] == dts.shape[0], 'times and locations must have same batch size'
            assert dts.min() >= -self.dt_max and dts.max() <= self.dt_max, 'times must be within the segment duration'

            # Consts
            batch_size = loc.shape[0]
            nc = self.temporal_ksp_cal_pre_KB.shape[1]

            # Linear interpolate time, KB interp in space
            with self.device:
                xp = self.device.xp

                # Group time indices
                sampling_time = self.ts[1] - self.ts[0]
                lower_idxs = xp.floor((dts - self.ts[0]) / sampling_time).astype(int)
                lower_idx_set = xp.unique(lower_idxs).tolist()

                ksp = self.device.xp.zeros((batch_size, nc), dtype=xp.complex64)
                for idx in lower_idx_set:

                    # Select corresponsing times/locs to idx
                    mask = lower_idxs == idx
                    loc_interp = loc[mask]
                    dt_interp  = dts[mask]

                    # spatial KB interp
                    ksp_cal_idx = sp.to_device(self.temporal_ksp_cal_pre_KB[idx], self.device)
                    nufft_lower = self._kb_interp(ksp_cal_pre_KB=ksp_cal_idx,
                                                  orig_shape=self.temporal_orig_shape,
                                                  loc=loc_interp)
                    ksp_cal_idx = sp.to_device(self.temporal_ksp_cal_pre_KB[idx+1], self.device)
                    nufft_upper = self._kb_interp(ksp_cal_pre_KB=ksp_cal_idx,
                                                  orig_shape=self.temporal_orig_shape,
                                                  loc=loc_interp)

                    # temporal linear interp
                    t_lower = self.ts[idx]
                    t_upper = self.ts[idx+1]
                    alpha = (dt_interp - t_upper) / (t_lower - t_upper)
                    nufft = alpha * nufft_lower + (1 - alpha) * nufft_upper # [nc, npoints]
                    ksp[mask, :] = nufft.T
                return ksp
        else:
            # Regular NUFFT
            # ksp = sp.nufft(self.img_cal, loc)
            ksp = self._kb_interp(ksp_cal_pre_KB=self.ksp_cal_pre_KB,
                                  orig_shape=self.orig_shape,
                                  loc=loc)
            return ksp.T

    def _kb_interp(self,
                   ksp_cal_pre_KB: np.ndarray,
                   orig_shape: tuple,
                   loc: np.ndarray,
                   width: Optional[int] = 4,
                   oversamp: Optional[float] = 1.25):
        """
        Does the FFT step in NUFTT with apodization and oversampling.

        Parameters:
        -----------
        ksp_cal_pre_KB : np.ndarray
            the *pre-proccesed* calib k-space with shape (..., N, N, (N))
        orig_shape : tuple
            this is the shape of the image before any processing
            i.e. the shape of the image input into _FT_pre_KB
        loc : np.ndarray
            (batch_size, d) where d is number of spatial dims
        width : int
            width of KB kernel for NUFFT
        oversamp : float
            oversampling ratio for NUFFT

        Returns:
        --------
        ksp_interp : np.ndarray
            the k-space interpolated samples with shape (..., batch_size)
        """

        # Consts
        B, d = loc.shape
        beta = np.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
        assert sp.get_device(loc) == self.device

        # KB Interp
        loc_rescaled = _scale_coord(loc, orig_shape, oversamp)
        with self.device:
            ksp_interp = sp.interp.interpolate(
                ksp_cal_pre_KB, loc_rescaled, kernel="kaiser_bessel", width=width, param=beta
            )
        ksp_interp /= width**d

        return ksp_interp

    def _FT_pre_KB(self,
                   img_cal: np.ndarray,
                   spatial_oshape: tuple,
                   oversamp: Optional[float] = 1.25,
                   width: Optional[int] = 4,
                   fft_batch_size: int = 3,
                   ):
        """
        Does the FFT step in NUFTT with apodization and oversampling.

        Parameters:
        -----------
        img_cal : np.ndarray
            the calibration image with shape (..., N, N, (N))
        spatial_oshape : tuple
            the spatial output shape, ex: (os1, os2, (os3))
        width : int
            width of KB kernel for NUFFT
        oversamp : float
            oversampling ratio for NUFFT

        Returns:
        --------
        ksp_cal_pre_KB : np.ndarray
            the k-space calibration ready for KB interpolation with shape (..., *spatial_oshape)
        """

        # FFT part of NUFFT (copied from sigpy)
        d = len(spatial_oshape)
        os_shape = _get_oversamp_shape(img_cal.shape, d, oversamp)
        output = sp.to_device(img_cal).copy()

        # Default oshape
        if spatial_oshape is None:
            spatial_oshape = os_shape[-d:]
        else:
            spatial_oshape = _get_oversamp_shape(spatial_oshape, d, oversamp)

        # Apodize
        beta = np.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
        _apodize(output, d, oversamp, width, beta)

        # Zero-pad
        output /= sp.util.prod(img_cal.shape[-d:]) ** 0.5
        output = sp.util.resize(output, os_shape)

        # FFT
        with self.device:
            output_rs = output.reshape((-1, *output.shape[-d:]))
            ksp_cal_pre_KB = self.device.xp.zeros((output_rs.shape[0], *spatial_oshape), dtype=np.complex64)
            for i, j in batch_iterator(output_rs.shape[0], fft_batch_size):
                full_ksp = sp.fft(sp.to_device(output_rs[i:j], self.device), axes=range(-d, 0), norm=None)
                ksp_cal_pre_KB[i:j] = sp.resize(full_ksp, (full_ksp.shape[0], *spatial_oshape))
            ksp_cal_pre_KB = ksp_cal_pre_KB.reshape((*output.shape[:-d], *spatial_oshape))
            return ksp_cal_pre_KB

    @staticmethod
    def _combine_calib_with_b0(img_cal, b0_map, ts):
        """
        img_cal: [nc, *im_size]
        b0_map: [*im_size]
        ts: array of times [nt], units of [s]
        """
        dev = sp.get_device(b0_map)
        xp = dev.xp
        d = len(b0_map.shape)
        with dev:
            b0_gpu = sp.to_device(b0_map, dev)
            ts_gpu = sp.to_device(ts, dev)
            img_cal_gpu = sp.to_device(img_cal, dev)
            phase_maps = xp.exp(2j * np.pi * b0_gpu[..., None] * ts_gpu)
            phase_maps = rearrange(phase_maps, '... t -> t ...') # [nt *im_size]
            phase_maps = phase_maps[:, None] * img_cal_gpu # [nt nc *im_size]
        return phase_maps
