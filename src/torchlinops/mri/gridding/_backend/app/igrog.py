from typing import Optional, Callable

import torch
from torch.utils.data import Dataset

__all__ = [
    'VanillaGROGL1Loss',
    'VanillaDataset',
    'VanillaCalib',
    'VanillaGParams',
    'VanillaImplicitGROG',
]


class VanillaGROGL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        B: Batch size
        C: Coil dimension
        npts: number of kernel points
        """
        # kernel = pred['kernel'] # [B C npts]
        # source_ksp = target['source_ksp'] # [B C npts]
        # pred_ksp = torch.tensordot(kernel.conj(), source_ksp, dims=-1)
        pred_ksp = pred['ksp']
        target_ksp = target['target_ksp'] # [B C]
        diff = pred_ksp - target_ksp
        return torch.mean(torch.abs(diff.real) + torch.abs(diff.imag))


class VanillaDataset(Dataset):
    def __init__(
            self,
            orientations: np.ndarray,
            input_ksp: np.ndarray,
            target_ksp: Optional[np.ndarray] = None,
            transform: Optional[Callable] = None):
        self.orientations = orientations
        self.input_ksp = input_ksp
        self.target_ksp = target_ksp
        self.transform = transform

    def __getitem__(self, i):
        features = {
            'dk': self.orientations[i],
            'source_ksp': self.input_ksp[i],
        }
        target = {
            'target_ksp': self.target_ksp[i] if self.target_ksp is not None else None
        }
        if self.transform is not None:
            features, target = self.transform(features, target)
        return features, target

    def __len__(self):
        return len(self.input_kpoints)


class VanillaCalib:
    def __init__(self, ksp_cal, buffer: int = 0):
        """
        ksp_cal: [nc, *cal_size]
        buffer:
        """

        self.ksp_cal = ksp_cal
        self.buffer = buffer
        self.dim = len(ksp_cal.shape[1:])
        self.nc = ksp_cal.shape[0]
        self.cal_size = self.ksp_cal.shape[1:]
        self.orig_shape = self.img_cal.shape

    def __call__(self, loc: np.ndarray):
        """
        D = 2 or 3: spatial ksp dimension

        loc: [batch_size, D] in [-N//2, N//2]
            - N is the calib size
        Returns
        -------
        ksp: [batch_size, nc]
        """
        # Regular NUFFT
        # ksp = sp.nufft(self.img_cal, loc)
        ksp = self._kb_interp(ksp_cal_pre_KB=self.ksp_cal_pre_KB,
                                orig_shape=self.orig_shape,
                                loc=loc)
        return ksp.T

    @property
    def valid_coords(self):
        if self._valid_coords is None:
            coords = tuple((np.arange(w-self.buffer) - (w-self.buffer) // 2)
                           for w in self.cal_size)
            coords = np.stack(np.meshgrid(*coords), axis=-1)
            coords = rearrange(coords, '... d -> (...) d')
            self._valid_coords = coords
        return self._valid_coords

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

        # KB Interp
        loc_rescaled = _scale_coord(loc, orig_shape, oversamp)
        with device:
            ksp_interp = sp.interp.interpolate(
                ksp_cal_pre_KB, loc_rescaled, kernel="kaiser_bessel", width=width, param=beta
            )
        ksp_interp /= width**d

        return ksp_interp



@dataclass
class VanillaGParams:
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


class VanillaImplicitGROG(ImplicitGROG):
    """Original Implicit GROG implementation
    Trains ksp points
    """

    def preprocess(self, data: Mapping, device):
        trj, calib = data['trj'], data['calib']
        dks, _, ro_idxs = self.extract_features(trj, data['gparams'])
        # Create training dataset
        # Randomly choose calib ksp points
        source_ksp = []
        target_ksp = []
        for dk in dks:
            i = np.random.randint(calib.valid_coords.shape[0])
            ktarget = calib.valid_coords[i]
            source_ksp.append(calib(ktarget + dk))
            target_ksp.append(calib(ktarget))
        dataset = VanillaDataset(
            dks, source_ksp, target_ksp
        )
        # Create loss function
        loss_fn = VanillaGROGL1Loss()

        # Save for later
        data['dks'] = dks
        data['ktargets'] = ktargets
        data['ro_idxs'] = ro_idxs
        return dataset, loss_fn, data

    def apply_model(self, data, model, device):
        trj, ksp = data['trj'], data['ksp']
        dks, ktargets, ro_idxs = data['dks'], data['ktargets'], data['ro_idxs']

        # Gridded trj is just the target ksp indices
        trj_grd = ktargets
        trj_batch_shape = trj.shape[:-2]

        # Create val dataset
        dks = rearrange(dks, '... K D N -> (...) K D N')
        npts = dks.shape[-1]
        ro_idxs = rearrange(ro_idxs, '... K N -> (...) K N')
        ksp = rearrange(ksp, 'C ... K -> (...) C K')
        #ksp_grd = np.zeros_like(ksp)

        source_ksp = np.zeros((*ksp.shape, npts), dtype=ksp.dtype)
        for t_idx in range(ksp.shape[0]):
            ro_idx = ro_idxs[t_idx]
            for c_idx in range(ksp.shape[1]):
                k_interp = readout_interp_1d(ksp[t_idx, c_idx])
                for n_idx in range(ro_idx.shape[-1]):
                    source_ksp[t_idx, c_idx, :, n_idx] = k_interp(ro_idx[:, n_idx])
        dataset = VanillaDataset(dks, source_ksp)

        # Run eval with model
        preds = self.eval(dataset, model)
        preds = torch2np(preds)
        ksp_grd = np.stack([p['ksp'] for p in preds])
        ksp_grd = ksp_grd.reshape(*ksp.shape)
        ksp_grd = rearrange(ksp_grd, 'T C K -> C T K')
        ksp_grd = ksp_grd.reshape(ksp_grd.shape[0], *trj_batch_shape, ksp_grd.shape[-1])

        # Return gridded data
        data['trj_grd'], data['ksp_grd'] = trj_grd, ksp_grd
        return data

    def extract_features(self, trj, gparams: VanillaGParams):
        """
        trj: [... K D]

        Returns
        -------
        For each trajectory point:

        kcenter: [... K D]
            location of closest grid point, sigpy-style coords
        dk: [... K D npts]
            orientations of readout points relative to closest grid point
        ro_idx: [... K npts]
            relative index of readout points along the readout (index coordinates)

        """
        if self.params.oversamp_readout != 1.0:
            trj = oversample(trj, axis=-2, factor=gparams.oversamp_readout)

        # Reshape trj
        trj_batch_shape = trj.shape[:-2]
        nro = trj.shape[-2]
        dim = trj.shape[-1]
        trj = rearrange(trj, '... K D -> (...) K D')

        # Precompute some stuff
        d_idx = np.arange(-(gparams.num_kpoints // 2), gparams.num_kpoints//2 + 1)
        dks = np.zeros((*trj.shape, gparams.num_kpoints)) # [... nro, d, num_points]
        ktargets = np.zeros_like(trj) # [... nro, d]
        ro_idxs = np.zeros((*trj_batch_shape, nro, gparams.num_kpoints), dtype=int)

        # Walk along (all) trajectories
        for trj_idx in tqdm(range(trj.shape[0])):
            # Velocity along this trajectory
            v = np.linalg.norm(np.diff(trj[trj_idx], dim=-2), dim=-1) # Velocity along trajectory
            v = np.append(v, v[:, -1], axis=-1)
            trj_interp = make_interp_spline(np.arange(nro), trj[trj_idx], k=1, axis=-2)
            for k_idx in tqdm(range(trj.shape[1]), 'Precompute Orientations'):
                # Identify source off-grid point
                kcenter = trj[trj_idx, k_idx] # [D]

                # Identify target grid point
                ktarget = np.round(kcenter * gparams.oversample_grid)

                # Convert readout spacing to arbitrary/index units
                spacing_idx = gparams.readout_spacing / v[k_idx]

                # Get samples along the readout in both directions
                ro_idx = np.clip(k_idx + d_idx * spacing_idx,
                                 a_min=0., a_max=nro-1)

                # Interpolate to find trj points
                ksources = trj_interp(ro_idx)
                ksources = rearrange(ksources, 'npts D -> D npts')

                # Compute orientation vectors
                dks[t_idx, k_idx] = ksources - ktarget[..., None]
                ktargets[t_idx, k_idx] = ktarget
                ro_idxs[t_idx, k_idx] = ro_idx
        dks = np.reshape(dks, (*trj_batch_shape, nro, -1, gparams.num_kpoints))
        ktargets = np.reshape(ktargets, (*trj_batch_shape, nro, -1))
        ro_idxs = np.reshape(ro_idxs, (*trj_batch_shape, nro, gparams.num_kpoints))
        return dks, ktargets, ro_idxs
