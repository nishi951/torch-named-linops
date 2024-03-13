from pathlib import Path
import numpy as np
import gdown

__all__ = [
    'brainweb_phantom',
    'mrf_trajectory',
]

data_id = '1aLT0-Hf38MKXpCbHCXpO-pVsPPGGqB0z'
data_filepath = Path(__file__).parent/'quantitative_phantom'

def brainweb_phantom():
    if not data_filepath.is_dir():
        gdown.download_folder(id=data_id,
                              output=str(data_filepath),
                              quiet=False,
                              use_cookies=False)
    filename = Path(data_filepath)/'brainweb_phantom.npz'
    return np.load(filename)

def mrf_trajectory():
    if not data_filepath.is_dir():
        gdown.download_folder(id=data_id,
                              output=str(data_filepath),
                              quiet=False,
                              use_cookies=False)
    filename = Path(data_filepath)/'brainweb_phantom.npz'
    return np.load(filename)
