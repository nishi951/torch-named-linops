import pytest

from einops import rearrange, einsum
import torch

from torchlinops.mri import SpatialBasis, TemporalBasis, NUFFT


@pytest.fixture
def spatiotemporal_basis():
    M, N = 32, 32
    L = 8
    K = 321
    B = torch.randn(M * N, K, dtype=torch.complex64)
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    spatial_basis = torch.reshape(U[:, :L], (M, N, L))
    spatial_basis = rearrange(spatial_basis, "M N L -> L M N")
    temporal_basis = Vh[:L, :]
    return spatial_basis, temporal_basis


def test_spatiotemporal(spatiotemporal_basis):
    spatial, temporal = spatiotemporal_basis

    img = torch.randn(3, *spatial.shape[1:], dtype=spatial.dtype)
    trj = 2 * torch.pi * (torch.rand(321, 2) - 0.5)

    F = NUFFT(
        trj,
        im_size=spatial.shape[1:],
        in_batch_shape=("A", "L"),
        out_batch_shape=("K",),
        backend="fi",
    )
    Sb = SpatialBasis(spatial, in_batch_shape=("A",))
    Tb = TemporalBasis(temporal, out_batch_shape=("K",), in_batch_shape=("A",))

    TS = (Tb @ F @ Sb)(img)
    x = einsum(img, spatial, "A M N, L M N -> A L M N")
    x = F(x)
    x = einsum(x, temporal, "A L K, L K -> A K")
    assert torch.isclose(TS, x).all()
