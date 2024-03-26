from torch.utils.data import DataLoader


from .. import AbstractDataModule
from ._calibration import CalibRegion, CalibrationDataset


@dataclass
class KCalDataModuleConfig:
    buffer: int = 0
    """Amount of buffer to give to edges of kspace calibration region"""
    train_dataloader_kwargs: Mapping = field(default_factory=lambda: {
        'num_workers': 2,
        'shuffle': True,
        'batch_size': 256,
    }),
    test_dataloader_kwargs: Mapping = field(default_factory=lambda: {
        'num_workers': 1,
        'shuffle': False,
        'batch_size': 256,
    }),


class KCalDataModule(AbstractDataModule):
    def __init__(self, dks: torch.Tensor, kcal: torch.Tensor, config: KCalDataModuleConfig):
        """
        dks : torch.Tensor
        """
        self.dks = dks
        self.kcal = kcal
        self.config = config
        self.calib_region = CalibRegion(self.kcal.detach().cpu().numpy(), buffer=self.config.buffer)
        self.train_dataset = CalibrationDataset(self.dks.detach().cpu().numpy(), self.calib_region)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.config.train_dataloader_kwargs)

    def test_dataloader(self, trj, ksp):
        test_dataset =
        ...

class OrientationDataset(Dataset):
    def __init__(
        self,
        orientations: np.ndarray,
        ksp: np.ndarray,
    ):
        self.orientations = orientations
        self.ksp = ksp

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
