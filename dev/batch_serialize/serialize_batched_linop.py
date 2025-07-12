import torch
from torchlinops import (
    BatchSpec,
    Dense,
    Diagonal,
    create_batched_linop,
    Dim,
)

from torchlinops.utils import MemReporter


def main():
    """Requires CUDA"""
    device = torch.device("cuda:0")
    B, C, X, Y = 6, 5, 64, 64
    input_ = torch.randn(X, Y)
    S = Dense(torch.randn(C, X, Y), Dim("CXY"), ishape=Dim("XY"), oshape=Dim("CXY"))
    F = Dense(torch.randn(B, 1, X, Y), Dim("B"), ishape=Dim("CXY"), oshape=Dim("BCXY"))

    # Make linop
    A = F @ S
    A.to(device)
    print("Not batched")
    MemReporter().report(A)
    A = create_batched_linop(A, [BatchSpec({"C": 1}), BatchSpec({"B": 2})])
    print(A)

    # Print memory usage
    print("Batched")
    MemReporter().report(A)

    # Serialize
    torch.save(A, "A.pt")

    A2 = torch.load("A.pt", weights_only=False)

    # Print memory usage
    print("Deserialized")
    MemReporter().report(A2)

    # Everything seems ok?


if __name__ == "__main__":
    main()
