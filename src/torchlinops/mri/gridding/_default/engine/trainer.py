
class Trainer(nn.Module):
    def __init__(self, hparams):
        self.hparams = hparams

    def train(self, model, loss_fn, train_dataloader):
        return model
