from einops import rearrange
import torch

class Batch(NamedLinop):
    def __init__(self, linop, **batch_sizes):
        self.linop = linop
        self.batch_sizes = batch_sizes


        isplit


    def forward(self, x: torch.Tensor):

        ...

