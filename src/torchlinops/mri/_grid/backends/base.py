import torch


class GriddingBase:
    def datamodule(self):
        raise NotImplementedError()

    def trainer(self):
        raise NotImplementedError()

    def inferencer(self):
        raise NotImplementedError()

    def loss_fn(self):
        raise NotImplementedError()

    def model(self):
        raise NotImplementedError()

    def grid(self, test_results, trj, ksp):
        raise NotImplementedError()

    def grogify(self, trj, ksp) -> Tuple[torch.Tensor, torch.Tensor]:
        train_dataloader = self.datamodule.train_dataloader()
        self.model = self.trainer.train(self.model, train_dataloader, self.loss_fn)
        test_dataloader = self.datamodule.test_dataloader(trj, ksp)
        trj_grd = self.grid_trj(trj)
        ksp_grd = self.inferencer.test(self.model, test_dataloader)
        return trj_grd, ksp_grd

    def grid_trj(self, trj):
        raise NotImplementedError()
