from pathlib import Path
import numpy as np

import gdown
from scipy.io import loadmat


__all__ = [
    'brainweb_phantom',
    'mrf_trajectory',
]

data_id = '1aLT0-Hf38MKXpCbHCXpO-pVsPPGGqB0z'
data_filepath = Path(__file__).parent/'quantitative_phantom'

def brainweb_phantom():
    if not (data_filepath/'brainweb_phantom.npz').is_file():
        gdown.download_folder(id=data_id,
                              output=str(data_filepath),
                              quiet=False,
                              use_cookies=False)
    filename = Path(data_filepath)/'brainweb_phantom.npz'
    return np.load(filename)

def MRF_FISP():
    """
    Download and correct the flip angles and TRs for MRF simulation
    """
    if not (data_filepath/'mrf_sequence.mat').is_file():
        gdown.download_folder(id=data_id,
                              output=str(data_filepath),
                              quiet=False,
                              use_cookies=False)
    filename = Path(data_filepath)/'mrf_sequence.mat'
    data = loadmat(str(filename))
    # Fix some stuff
    data['TR_init'][:, -1] = 15
    return data['FA_init'][0].astype(np.float32), data['TR_init'][0].astype(np.float32)
