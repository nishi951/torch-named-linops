@dataclass
class KCalConfig:
    ...


class KCalGridding(GriddingBase):
    def __init__(self, kcal: torch.Tensor, config: KCalConfig):
        """
        kcal : torch.Tensor
            [num_coils, *calib_size] kspace calibration region, fully sampled
        """
        self.config = config
        self.kcal = kcal

    @property
    def datamodule(self):
        return KCalDataModule(self.kcal, self.config.datamodule_config)

    @property
    def trainer(self):
        return KCalTrainer(self.config.trainer_config)

    @property
    def inferencer(self):
        return KCalInferencer(self.config.inferencer_config)

    @property
    def loss_fn(self):
        return KCalLoss()

    @property
    def model(self):
        return KCalModel(self.config.model_config)

    def grid(self, test_results, trj, ksp):
        ...
