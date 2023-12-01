from copy import copy

class LinOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """matrix-vector multiply"""
        return NotImplemented

    def compose(self, L: LinOp) -> LinOpChain:
        """i.e. matrix-matrix multiply"""
        return NotImplemented

    @property
    def H(self) -> LinOp:
        if self.adj is None:
            self.adj = copy(self) # shallow
            self.adj.forward = self.adjoint
            self.adj.adjoint = self.forward
        return self.adj

    @property
    def N(self) -> LinOp:
        """matrix-vector multiply"""
        if self.normal is None:
            self.normal = self.H @ self
        return self.normal

    def __matmul__(self, V: LinOp):
        return self.compose(V)

    def __rmatmul__(self, U: LinOp):
        return U.compose(self)


class LinOpChain(LinOp):
    def __init__(self, linops):
        super(LinOp, self).__init__()
        self._linops = linops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linop in reversed(self._linops):
            x = linop(x)
        return x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        for linop in self._linops:
            y = linop.H(y)
        return y

    @property
    def N(self) -> LinOpChain:
        return
