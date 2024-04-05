from typing import Optional

import torch.nn as nn

__all__ = ["MLP"]


class MLP(nn.Module):
    """A simple multilayered perceptron"""

    def __init__(
        self,
        in_chans,
        out_chans,
        hidden_chans: Optional[int],
        num_layers: int,
        bias: bool = True,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        assert num_layers >= 1
        channels = [in_chans] + [hidden_chans] * (num_layers - 1) + [out_chans]

        layers = []
        for i, (in_c, out_c) in enumerate(zip(channels[:-1], channels[1:])):
            layers.append(
                nn.Linear(
                    in_c,
                    out_c,
                    bias=bias,
                ),
            )
            if i < num_layers - 1:
                layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
