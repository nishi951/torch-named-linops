
@dataclass
class KCalDataModuleConfig:
    ...

class KCalDataModule(AbstractDataModule):
    def __init__(self, kcal, config):
        self.kcal = kcal
        self.config = config

        self.calib =

    def train_dataloader(self):
