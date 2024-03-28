from torch.utils.data import Dataset
import numpy as np

__all__ = ["CalibRegion"]


class CalibRegion:
    def __init__(self, ksp_cal: np.ndarray, buffer: int = 0):
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
        ksp = self._kb_interp(
            ksp_cal_pre_KB=self.ksp_cal_pre_KB, orig_shape=self.orig_shape, loc=loc
        )
        return ksp.T

    @property
    def valid_coords(self):
        if self._valid_coords is None:
            coords = tuple(
                (np.arange(w - self.buffer) - (w - self.buffer) // 2)
                for w in self.cal_size
            )
            coords = np.stack(np.meshgrid(*coords), axis=-1)
            coords = rearrange(coords, "... d -> (...) d")
            self._valid_coords = coords
        return self._valid_coords

    def _kb_interp(
        self,
        ksp_cal_pre_KB: np.ndarray,
        orig_shape: tuple,
        loc: np.ndarray,
        width: Optional[int] = 4,
        oversamp: Optional[float] = 1.25,
    ):
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
                ksp_cal_pre_KB,
                loc_rescaled,
                kernel="kaiser_bessel",
                width=width,
                param=beta,
            )
        ksp_interp /= width**d

        return ksp_interp


class CalibrationDataset(Dataset):
    def __init__(
        self,
        orientations: np.ndarray,
        calib: CalibRegion,
    ):
        self.orientations = orientations
        self.calib = calib

    def __getitem__(self, i):
        dk = self.orientations[i]
        source_ksp, target_ksp = self.randomize_center_point(dk)
        features = {
            "dk": self.orientations[i],
            "source_ksp": source_ksp,
        }
        target = {
            "target_ksp": target_ksp,
        }
        return features, target

    def __len__(self):
        return len(self.orientations)

    def randomize_center_point(self, dk):
        i = np.random.randint(self.calib.valid_coords.shape[0])
        ktarget = calib.valid_coords[i]
        source_ksp.append(calib(ktarget + dk))
        target_ksp.append(calib(ktarget))
        return source_ksp, target_ksp
