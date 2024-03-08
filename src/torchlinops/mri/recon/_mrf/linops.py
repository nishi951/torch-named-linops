def subspace_linop(trj, mps, phi, dcf, gridded: bool = False):
    P = Dense(phi)
    S = SENSE()
    F = NUFFT()
    A = F @ S @ P
    if dcf is not None:
        D = Diagonal()
        A = D @ A
    return A


A = Batch(subspace_linop(), ...)
A.N  # Should be batched here
A.H  # Should be batched here too
