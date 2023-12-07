import torch

from torchlinops.mrf import SenseMapsLinop

def test_mps():
    mps = torch.randn((6, 64, 64), dtype=torch.complex64)

    S = SenseMapsLinop(mps)
    print(S)

    S_slc = S[1:3]
    print(S_slc)
    breakpoint()
