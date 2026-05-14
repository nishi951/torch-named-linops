from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn
from torchlinops import BatchSpec, Dense, Diagonal, Dim, create_batched_linop
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
    print(A)
    print("Not batched (GPU)")
    MemReporter().report(A)
    A = create_batched_linop(A, [BatchSpec({"C": 1}), BatchSpec({"B": 2})])
    print(A)

    # Print memory usage
    print("Batched (GPU)")
    MemReporter().report(A)

    # Serialize
    torch.save(A, "A.pt")
    A2 = torch.load("A.pt", weights_only=False)

    # Print memory usage
    print("Deserialized (GPU -> GPU)")
    MemReporter().report(A2)

    # Deepcopy
    # Memory expands?
    A_copy = deepcopy(A)
    print("A_copy")
    MemReporter().report(A_copy)
    breakpoint()
    print("A_copy is different: ")
    breakpoint()

    # CPU
    # Memory drastically expands again...
    A_copy = deepcopy(A)
    MemReporter().report(A_copy)
    A3 = A_copy.to("cpu")
    print("CPU")
    MemReporter().report(A3)

    # Preserve references
    A4 = A.to("cpu")
    print("CPU Attempt #2")
    MemReporter().report(A4)


def test_deepcopy():
    """Requires CUDA"""
    device = torch.device("cuda:0")
    B, C, X, Y = 6, 5, 64, 64
    input_ = torch.randn(X, Y)
    S = Dense(torch.randn(C, X, Y), Dim("CXY"), ishape=Dim("XY"), oshape=Dim("CXY"))
    F = Dense(torch.randn(B, 1, X, Y), Dim("B"), ishape=Dim("CXY"), oshape=Dim("BCXY"))

    # Make linop
    A = F @ S
    A.to(device)
    A = create_batched_linop(A, [BatchSpec({"C": 1}), BatchSpec({"B": 2})])

    # deepcopy
    # Memory drastically expands again...
    A_copy = deepcopy(A)
    MemReporter().report(A_copy)

    assert id(A[0][0][0].weight) != id(A_copy[0][0][0].weight)


if __name__ == "__main__":
    main()
    test_deepcopy()
