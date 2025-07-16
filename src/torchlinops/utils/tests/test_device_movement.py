import pytest
import torch
import torch.nn as nn
from torchlinops import NUFFT, Dense, Diagonal, Dim, Stack
from torchlinops.utils import memory_aware_to

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


def make_linop(
    trj,
    dcf,
    mps,
    nufft_width,
    nufft_oversamp,
    nufft_mode,
):
    im_size = mps.shape[1:]
    P = trj.shape[0]
    linops = []
    # SENSE
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
    breakpoint()
