import torch
import sigpy as sp

from torchlinops.mri.gridding._default.engine.basic._datamodule import CalibrationDataset
from torchlinops.mri.app import Calib

def test_calibration_dataset(spiral_synthetic_dataset):
    img, trj, mps, ksp = spiral_synthetic_dataset.simulate()

    im_size = img.shape

    # Get calibration region
    mps_recon, kgrid = Calib(
        trj,
        ksp,
        im_size=im_size,
        calib_width=24,
        kernel_width=7,
        device=sp.Device(-1),
    ).run()

    # Create CalibrationDataset
    calib = CalibRegion(kgrid, buffer=1)

    # Try sampling from it
    loc = np.array([[0., 0.,], [0., 1.], [-1, 0.]])
    gt = np.array([kgrid[32, 32], kgrid[32, 33], kgrid[31, 32]])
    kcal = calib(loc)
    assert np.isclose(kcal, gt)
