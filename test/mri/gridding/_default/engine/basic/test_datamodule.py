from torchlinops.mri.gridding._default.engine.basic._datamodule import CalibrationDataset

def test_calibration_dataset(spiral_small):
    img, trj, mps = spiral_small
