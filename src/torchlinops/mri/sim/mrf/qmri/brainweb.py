from pathlib import Path
import numpy as np
from einops import rearrange

import gdown
from scipy.io import loadmat


__all__ = [
    "brainweb_phantom",
    "MRF_FISP",
]

data_id = "1aLT0-Hf38MKXpCbHCXpO-pVsPPGGqB0z"
data_filepath = Path(__file__).parent / "quantitative_phantom"


def brainweb_phantom():
    if not (data_filepath / "brainweb_phantom.npz").is_file():
        gdown.download_folder(
            id=data_id, output=str(data_filepath), quiet=False, use_cookies=False
        )
    filename = Path(data_filepath) / "brainweb_phantom.npz"
    # Transpose stuff to make it nicer
    data = dict(np.load(filename))
    for k, img in data.items():
        img = rearrange(img, "z y x -> x y z")
        img = np.flip(img, axis=(0, 1, 2))
        data[k] = img.copy()
    # axial slice is [x y]
    # sagittal slice is [x z]
    # coronal slice is [y z]
    return data


def MRF_FISP():
    """
    Download and correct the flip angles and TRs for MRF simulation
    """
    if not (data_filepath / "mrf_sequence.mat").is_file():
        gdown.download_folder(
            id=data_id, output=str(data_filepath), quiet=False, use_cookies=False
        )
    filename = Path(data_filepath) / "mrf_sequence.mat"
    data = loadmat(str(filename))
    # Fix some stuff
    data["TR_init"][:, -1] = 15
    return data["FA_init"][0].astype(np.float32), data["TR_init"][0].astype(np.float32)
