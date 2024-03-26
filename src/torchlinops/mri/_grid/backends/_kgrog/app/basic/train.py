import torchlinops.mri.gridding._default.engine.basic as basic


class BasicIGROGApp:
    def __init__(self, config: basic.BasicIGROGAppConfig):
        self.config = config

    def grogify(trj: torch.Tensor, ksp: torch.Tensor, ksp_cal: torch.Tensor):
        # Make datamodule
        # Make trainer
        # Make inferencer
        # Run training
        # Run inference
        # Unpack

        return trj_grd, ksp_grd
