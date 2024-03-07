from torch.utils.data import DataLoader, Dataset

import numpy as np

from .. import AbstractDataModule

from ._calibration import CalibrationDataset

@dataclass
class BasicDataModuleConfig:
    ...

class BasicDataModule(AbstractDataModule):
    def __init__(self, config):
        self.config = config

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.config.train_dataloader_kwargs)
