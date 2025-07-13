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

    # CPU
    # Memory drastically expands again...
    A3 = deepcopy(A).to("cpu")
    print("CPU")
    MemReporter().report(A3)

    # Preserve references
    A4 = move_module_preserve_sharing(A, "cpu")
    print("CPU Attempt #2")
    MemReporter().report(A4)


def move_module_preserve_sharing(module: nn.Module, device):
    # Map from original storage ID to list of (tensor, name)
    storage_tensors = defaultdict(list)

    def collect(m):
        for name, t in list(m.named_parameters(recurse=False)) + list(
            m.named_buffers(recurse=False)
        ):
            if t is None:
                continue
            key = t.untyped_storage()._cdata
            storage_tensors[key].append(t)
        for child in m.children():
            collect(child)

    collect(module)

    # Move unique storage blocks
    storage_map = {}
    for key, tensors in storage_tensors.items():
        # Find max offset + size for any tensor sharing this storage
        max_offset = 0
        dtype = None
        device_orig = None
        for t in tensors:
            size_bytes = t.element_size() * max_storage_size(t)
            offset_bytes = t.storage_offset() * t.element_size()
            max_offset = max(max_offset, offset_bytes + size_bytes)
            dtype = t.dtype
            device_orig = t.device

        # Create a flat tensor that spans the full required storage
        base_tensor = torch.empty(
            (max_offset // torch.tensor([], dtype=dtype).element_size(),),
            dtype=dtype,
            device=device_orig,
        )
        storage_map[key] = base_tensor.to(device)

    # Replace parameters/buffers
    def remap(m):
        for name, t in m._parameters.items():
            if t is not None:
                new_t = as_view_on_moved(t, storage_map)
                m._parameters[name] = nn.Parameter(new_t, requires_grad=t.requires_grad)
        for name, t in m._buffers.items():
            if t is not None:
                m._buffers[name] = as_view_on_moved(t, storage_map)
        for child in m.children():
            remap(child)

    remap(module)
    return module


def max_storage_size(tensor):
    """Compute the number of elements spanned by a strided tensor."""
    if tensor.numel() == 0:
        return 0
    size = 0
    for i, (dim, stride) in enumerate(zip(tensor.size(), tensor.stride())):
        size += (dim - 1) * stride
    return size + 1


def as_view_on_moved(tensor, storage_map):
    storage = tensor.untyped_storage()
    key = storage._cdata
    base = storage_map[key]
    return base.as_strided(tensor.size(), tensor.stride(), tensor.storage_offset())


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        base = torch.empty(4 * 64 * 64)
        self.shared_view = nn.Parameter(
            base.as_strided((2, 1, 64, 64), (4096, 4096, 64, 1), 8192)
        )
        self.same_view = nn.Parameter(self.shared_view)
        self.empty = nn.Parameter(torch.tensor([], dtype=torch.float32))
        self.zero_shape = nn.Parameter(torch.empty((0, 3, 224)))
        self.noncontig = nn.Parameter(
            torch.arange(12).view(3, 4).t(), requires_grad=False
        )  # transposed (non-contig)


if __name__ == "__main__":
    main()

    model = TestModule()
    model_cuda = move_module_preserve_sharing(model, torch.device("cuda"))

    # Assertions
    assert (
        model_cuda.shared_view.untyped_storage()._cdata
        == model_cuda.same_view.untyped_storage()._cdata
    )
    assert model_cuda.empty.numel() == 0 and model_cuda.empty.device.type == "cuda"
    assert model_cuda.zero_shape.shape == (0, 3, 224)
    assert not model_cuda.noncontig.is_contiguous()
