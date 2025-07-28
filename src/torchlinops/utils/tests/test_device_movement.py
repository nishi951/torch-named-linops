import pytest
import torch
import torch.nn as nn
from torchlinops import NUFFT, Dense, Diagonal, Dim, Stack, split_linop
from torchlinops.utils import memory_aware_to, MemReporter, tensor_memory_span

PYTEST_GPU_MARKS = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU is required but not available"
    ),
]


class SampleModule(nn.Module):
    def __init__(self):
        super().__init__()
        base = torch.arange(4 * 64 * 64).float()
        self.shared_view = nn.Parameter(
            base.as_strided((2, 1, 64, 64), (4096, 4096, 64, 1), 8192)
        )
        self.same_view = nn.Parameter(self.shared_view)
        self.empty = nn.Parameter(torch.tensor([], dtype=torch.float32))
        self.zero_shape = nn.Parameter(torch.empty((0, 3, 224)))
        self.noncontig = nn.Parameter(
            torch.arange(12).view(3, 4).t(), requires_grad=False
        )  # transposed (non-contig)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_model_storage_after_movement():
    model = SampleModule()
    model_cuda = memory_aware_to(model, torch.device("cuda"))
    assert (
        model_cuda.shared_view.untyped_storage()._cdata
        == model_cuda.same_view.untyped_storage()._cdata
    )
    assert model_cuda.empty.numel() == 0 and model_cuda.empty.device.type == "cuda"
    assert model_cuda.zero_shape.shape == (0, 3, 224)
    assert not model_cuda.noncontig.is_contiguous()

    assert (model.shared_view == model_cuda.shared_view).all()


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_model_storage_duplicate_submodules():
    weight = torch.randn(3, 4)
    P = Dense(weight, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
    A = Stack(P, P, P, odim_and_idx=("B", 0))
    memory_aware_to(A, torch.device("cuda"))
    assert A[0].weight.is_cuda
    assert id(A[0].weight) == id(A[1].weight)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_model_storage_parameterlist():
    class ModuleWithParameterList(nn.Module):
        def __init__(self, *tensor_list):
            super().__init__()
            self.weights = nn.ParameterList(tensor_list)

    class ModuleWithDuplicateSubmodules(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.modules = nn.ModuleList([module, module])

    base_tensor = torch.randn(6, 3)
    modulea = ModuleWithParameterList(
        base_tensor[..., 0], base_tensor[..., 1], base_tensor[..., 2]
    )
    moduleb = ModuleWithDuplicateSubmodules(modulea)
    memory_aware_to(moduleb, torch.device("cpu"))


def make_linop(trj, dcf, mps, nufft_width, nufft_oversamp, nufft_mode):
    """Sample linop with some complexity reflecting real-world use"""
    im_size = mps.shape[1:]
    P = trj.shape[0]
    linops = []
    S = Dense(
        mps,
        weightshape=Dim("CNxNyNz"),
        ishape=Dim("NxNyNz"),
        oshape=Dim("CNxNyNz"),
    )
    for p in range(P):
        # DCF
        Dp = Diagonal(dcf[p], ioshape=Dim("CTK"), broadcast_dims=Dim("C"))
        Fp = NUFFT(
            trj[p],
            im_size,
            output_shape=Dim("TK"),
            oversamp=nufft_oversamp,
            mode=nufft_mode,
        )
        Ap = (Dp ** (1 / 2)) @ Fp @ S
        linops.append(Ap)
    A = Stack(*linops, idim_and_idx=("P", 0), odim_and_idx=("P", 0))
    return A


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_full_linop():
    C, Nx, Ny, Nz = 5, 8, 8, 8
    T, K = 7, 11
    trj = Nx * torch.rand(T, K, 3) - Nx // 2
    trj = torch.round(trj)
    dcf = torch.randn(T, K)
    mps = torch.randn(C, Nx, Ny, Nz, dtype=torch.complex64)

    A = make_linop(
        trj,
        dcf,
        mps,
        nufft_width=4,
        nufft_oversamp=1.25,
        nufft_mode="sampling",
    )
    memory_aware_to(A, torch.device("cuda"))
    assert True


def check_span(t):
    min_off, max_off = tensor_memory_span(t)
    flat = t._base.view(-1) if t._base is not None else t.view(-1)
    used = t.clone().contiguous().view(-1)
    span = flat[min_off : max_off + 1]
    # Ensure the span includes the tensor's flattened values
    for v in used:
        assert (span == v).any(), f"Value {v.item()} not found in span"


@pytest.mark.parametrize("shape", [(10,), (4, 5), (2, 3, 4)])
def test_contiguous_tensor(shape):
    t = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    min_off, max_off = tensor_memory_span(t)
    assert min_off == t.storage_offset()
    assert max_off == t.storage_offset() + t.numel() - 1
    check_span(t)


def test_strided_tensor():
    t = torch.arange(20).reshape(4, 5)
    v = t[::2, ::2]  # non-contiguous
    min_off, max_off = tensor_memory_span(v)
    check_span(v)


def test_negative_stride_1d():
    t = torch.arange(10)
    # v = t[::-1]
    v = t.flip(0)
    min_off, max_off = tensor_memory_span(v)
    assert min_off == 0
    assert max_off == 9
    check_span(v)


def test_negative_stride_2d():
    t = torch.arange(16).reshape(4, 4)
    # v = t[::-1, ::-1]
    v = t.flip(0, 1)
    min_off, max_off = tensor_memory_span(v)
    assert min_off == 0
    assert max_off == 15
    check_span(v)


def test_partial_negative_stride():
    t = torch.arange(24).reshape(2, 3, 4)
    # v = t[:, ::-1, ::2]  # only flip dim 1
    v = t.flip(1)[..., ::2]
    min_off, max_off = tensor_memory_span(v)
    check_span(v)


def test_storage_offset():
    t = torch.arange(100)
    v = t[10:20]
    min_off, max_off = tensor_memory_span(v)
    assert min_off == v.storage_offset()
    assert max_off == v.storage_offset() + v.numel() - 1
    check_span(v)


def test_subview_of_view():
    t = torch.arange(100)
    v1 = t[10:50]
    v2 = v1[::2]  # subview of a view
    min_off, max_off = tensor_memory_span(v2)
    check_span(v2)


def test_nontrivial_stride():
    t = torch.arange(3 * 5 * 7).reshape(3, 5, 7)
    # v = t[::2, ::-1, ::3]
    v = t.flip(1)[::2, :, ::3]
    min_off, max_off = tensor_memory_span(v)
    check_span(v)


def test_scalar_tensor():
    t = torch.tensor(42)
    min_off, max_off = tensor_memory_span(t)
    assert min_off == t.storage_offset()
    assert max_off == t.storage_offset()


def test_reconstruct_from_span_no_negative_stride():
    base = torch.arange(60).reshape(3, 4, 5)
    # Slice with positive strides only
    t = base[1::2, 1:3, ::2]  # shape (1, 2, 3), all strides positive

    min_off, max_off = tensor_memory_span(t)
    flat = base.view(-1)
    block = flat[min_off : max_off + 1].clone()

    # Compute offset relative to min_off (no adjustment needed for positive strides)
    new_offset = t.storage_offset() - min_off

    reconstructed = torch.as_strided(
        block,
        size=t.size(),
        stride=t.stride(),
        storage_offset=new_offset,
    )

    # Check that reconstruction matches original tensor exactly
    assert torch.equal(reconstructed, t), "Reconstructed tensor does not match original"


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required but not available"
)
def test_split_movement():
    """Messing around with block sizes"""
    M, N = 100, 10
    weight = torch.randn(M, N)

    # weight_half = weight[:, :5]
    # weight_half = weight_half.to(torch.device("cuda"))
    # breakpoint()

    # weight = weight.to(torch.device("cuda"))
    # breakpoint()

    A = Dense(weight, weightshape=Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))

    # linops, _, _ = split_linop(A, {"N": N // 2})
    linops, _, _ = split_linop(A, {"M": M // 2})
    linop_gpu = memory_aware_to(linops[1], torch.device("cuda"))
    # linop_gpu_2 = linops[1].to(torch.device("cuda"))
    linop_cpu = linops[0]

    MemReporter().report(linop_gpu)
    MemReporter().report(linop_cpu)
