"""Data generation for implicit grog
Inputs (Train):
- Calibration region (fully sampled)
- Sampling distribution
  - Or noncartesian trajectory.
- Implicit grog hyperparameters
1. Sample points from calibration region according to sampling distribution
2. Create MLP according to hyperparams
3. Train MLP


"""

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from einops import rearrange
from torch.utils.data import Dataset

from .calibration import CalibRegion
from .trj_utils import oversample


@dataclass
class TimeSegmentationParams:
    nseg: int
    """Number of segments"""
    dt: float
    """Time per sample"""


@dataclass
class ImplicitGROGDatasetConfig:
    num_kpoints: int
    """Number of ksp points per grog kernel"""
    readout_spacing: float
    """Spacing of kernel points along readout, arbitrary units"""
    calib_buffer: int
    """Width of margin to discard from cartesian calibration region"""
    oversample_readout: float
    """Factor by which to upsample the readout"""
    oversample_grid: float
    """Factor by which to oversample the grid"""
    tseg: Optional[TimeSegmentationParams] = None
    """Time segmentation config"""


class ImplicitGROGDataset(Dataset):
    def __init__(
        self,
        acs: CalibRegion,
        trj: np.ndarray,
        params: ImplicitGROGDatasetConfig,
        transform: Optional[Callable] = None,
    ):
        """
        trj: [..., nro, dim]
        """
        super().__init__()
        self.acs = acs
        self.trj = trj
        self.params = params

        # Derived
        self.valid_coords = acs.coords_with_buffer(width=self.params.calib_buffer)
        self.dk = self.sample_dk()
        self.use_b0 = self.params.tseg is not None
        if self.use_b0:
            assert (
                self.acs.b0_map is not None
            ), "Time segmentation requested but no b0 map provided"
            self.times = np.arange(self.params.tseg.nseg) * self.params.tseg.dt
            self.times -= self.times[len(self.times) // 2]

    def __getitem__(self, idx):
        # Sample a random point from the calibration region
        source = {}
        target = {}
        kcenter = np.random.choice(self.valid_coords)
        if self.use_b0:
            k_idx = idx // len(self.times)
            t_idx = idx % len(self.times)

            # Grab the ith block of orientations
            dk = self.dk[k_idx]
            # Grab the jth time offset
            dt = self.times[t_idx]
            # Interpolate the
            source["dk"] = dk
            source["dt"] = dt
            source["ksp"] = self.acs(kcenter + dk, time)
            target["ksp"] = self.acs(kcenter)
        else:
            # Grab the ith block of orientations
            dk = self.dk[idx]
            # Grab the jth time offset
            source["ksp"] = self.acs(kcenter + dk)
            target["ksp"] = self.acs(kcenter)
        if self.transform is not None:
            source, target = self.transform(source, target)
        return source, target

    def __len__(self):
        if self.use_b0:
            return len(self.times) * len(self.valid_coords)
        return len(self.valid_coords)

    def sample_dk(self):
        """
        Parameters
        ----------
        trj: [..., nro, dim] np.ndarray
        Kspace trajectory in [-N//2, N//2] (sigpy-style)
        num_kpoints: int
        Number of trj points to use to grid the data

        Returns
        -------
        Array of dk vectors: [nro, num_kpoints, dim]


        Reference:
        - `precompute_orientations``
        """
        trj = self.trj.copy()  # Preserve original trj
        if self.params.oversamp_readout != 1.0:
            trj = oversample(trj, axis=-2, factor=self.params.oversamp_readout)

        # Reshape trj
        trj_batch_shape = trj.shape[:-2]
        nro = trj.shape[-2]
        dim = trj.shape[-1]
        trj = rearrange(trj, "... K D -> (...) K D")

        # Precompute some stuff
        d_idx = np.arange(
            -(self.params.num_kpoints // 2), self.params.num_kpoints // 2 + 1
        )
        dk = np.zeros((*trj.shape, self.params.num_kpoints))  # [... nro, d, num_points]

        # Walk along (all) trajectories
        for trj_idx in tqdm(range(trj.shape[0])):
            # Velocity along this trajectory
            v = np.linalg.norm(
                np.diff(trj[trj_idx], dim=-2), dim=-1
            )  # Velocity along trajectory
            v = np.append(v, v[:, -1], axis=-1)
            trj_interp = make_interp_spline(np.arange(nro), trj[trj_idx], k=1, axis=-2)
            for k_idx in tqdm(range(trj.shape[1]), "Precompute Orientations"):
                # Identify source off-grid point
                kcenter = trj[trj_idx, k_idx]  # [D]
                # Identify target grid point
                ktarget = np.round(kcenter * self.params.oversample_grid)

                # Convert readout spacing to arbitrary/index units
                spacing_idx = self.params.readout_spacing / v[k_idx]

                # Get samples along the readout in both directions
                k_idx_along_readout = np.clip(
                    k_idx + d_idx * spacing_idx, a_min=0.0, a_max=nro - 1
                )

                # Interpolate to find trj points
                ksources = trj_interp(k_idx_along_readout)
                ksources = rearrange(ksources, "npts D -> D npts")

                # Compute orientation vectors
                dk[t_idx, k_idx] = ksources - ktarget[..., None]
        dk = np.reshape(dk, (*trj_batch_shape, nro, -1, self.params.num_kpoints))
        return dk
