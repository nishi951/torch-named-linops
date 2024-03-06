
class VanillaImplicitGROGMLP(nn.Module):
    """The model trained for implicit grog
    """
    ...

@dataclass
class VanillaDataHparams:
    num_points: int = 3

class VanillaDataModule(AbstractDataModule):
    """The data processing module used for vanilla implicit grog
    """
    def preprocess(self, data)

class VanillaLoss(nn.Module):
    """The loss function used to train the implicit grog
    """
    ...
