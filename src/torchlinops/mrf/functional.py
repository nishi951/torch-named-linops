
def mrf_forward(
        trj,
        phi,
        mps,
):
    ...

def mps_forward(mps, x):
    return mps * x

def mps_adjoint(mps, x):
    return torch.conj(mps) * x
