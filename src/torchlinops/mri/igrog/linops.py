

class ImplicitGROG:
    """Mega-class for implicit grog functionality"""
    def __init__(self, gparams):
        self.gparams = gparams

    def make_training_data(self, trj, ksp, calib)


    def grogify(self, trj, ksp, calib, model, loss_fn):
        train_data = self.make_training_data(trj, ksp, calib)
        test_data = self.make_test_data(trj, ksp)
        model = self.calib_train(train_data, loss_fn, model)
        results = self.grid(test_data, model)
        trj_grd = self.trj_update(trj, test_data, results)
        ksp_grd = self.ksp_update(ksp, test_data, results)
        return trj_grd, ksp_grd


    def calib_train(self, train_data, loss_fn, model):
        """
        train_data: Calibration data
        val_data: Full data to evaluate
        loss_fn: Evaluates GRAPPA loss
        model: Maps features to gridded data
        """
        trainer = Trainer(self.config.train_hparams)
        model = trainer.train(train_data, loss_fn)
        return model

    @torch.no_grad
    def grid(self, test_data, model):
        model.eval()
        test_dataloader = Dataloader(test_data, shuffle=False, batch_size=1)
        results = []
        for features, _ in test_dataloader:
            target = model(features)
            results.append(target)
        return results


def convert_linop(nufft_linop, ksp, calib, gparams):
    igrog = ImplicitGROG(gparams)
    model =
    trj_grd, ksp_grd = igrog.grogify(trj, ksp, calib, model, loss_fn)
    F = ImplicitGROGLinop(trj_grd)
    return F, ksp_grd





class ImplicitGROGLinop(NamedLinop):
    def __init__(
            self,
            trj_grd: torch.Tensor,
    ):
        self.trj = nn.Parameter(trj_grd, requires_grad=False)

    def fn(self, x, /, trj):
        ...
