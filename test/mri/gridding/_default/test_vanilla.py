
def test_vanilla():
    # Make dataset
    datamodule = DataModule(data, datamodule_hparams) # Includes preprocessing
    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()

    model = MLP(...)
    trainer = Trainer(train_hparams)
    model.to(device)
    trainer.to(device)
    model = trainer.train(model, loss_fn, train_dataloader)

    inferencer = Inferencer(inference_hparams)
    results = inferencer.val(model, test_dataloader)

    data_grd = datamodule.postprocess(results)
    return data_grd


def test_vanilla_app():
    data_grd = VanillaImplicitGROG(data).run()
    ...

    du
