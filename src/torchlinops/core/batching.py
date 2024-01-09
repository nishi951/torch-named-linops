
class BatchedLinop(nn.Module):
    def __init__(self, linop, names):
        self.linop = linop
        self.names = names

    def forward(self, x: torch.Tensor):
        ...
