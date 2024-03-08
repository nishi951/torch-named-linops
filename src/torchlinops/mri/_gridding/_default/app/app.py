#!/usr/bin/env python3

class VanillaImplicitGROG:
    def __init__(self,
                 data: Mapping,
                 datamodule_hparams,
                 model_hparams,
                 train_hparams,
                 inference_hparams,
    ):
        self.data = data
        self.datamodule_hparams = datamodule_hparams
        self.model_hparams = model_hparams
        self.train_hparams = train_hparams
        self.inference_hparams = inference_hparams

    def run(self, device: torch.device = 'cpu'):
        # Make dataset
        model = ImplicitGROGMLP(**asdict(self.model_hparams))
        datamodule = DataModule(self.datamodule_hparams)
        datamodule = datamodule.preprocess(self.data)

        # Generic (do not edit)
        train_dataloader = datamodule.train_dataloader()
        test_dataloader = datamodule.test_dataloader()
        trainer = Trainer(self.train_hparams)
        model.to(device)
        model = trainer.train(model, loss_fn, train_dataloader)
        inferencer = Inferencer(self.inference_hparams)
        results = inferencer.test(model, test_dataloader)

        datamodule = datamodule.postprocess(results)
        return datamodule.data
