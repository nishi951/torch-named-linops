import pytest
import torch

from torchlinops import Sampling
from torchlinops.tests.test_base import BaseNamedLinopTests
import torchlinops.functional as F


class TestSampling(BaseNamedLinopTests):
    equality_check = "approx"

    isclose_kwargs = dict(rtol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        N = 64
        ndim = 2
        R, K = 13, 17
        B = 3  # Batch
        idx = torch.randint(0, N - 1, (R, K, ndim))
        x = torch.randn(B, N, N)
        y = torch.randn(B, R, K)
        linop = Sampling.from_stacked_idx(idx, (N, N), ("R", "K"))
        return linop, x, y

    def test_size(self, linop_input_output):
        A, x, y = linop_input_output
        assert A.size("R") == 13
        assert A.size("K") == 17
        assert A.size("N") is None

    def test_split(self, linop_input_output):
        A, x, y = linop_input_output
        tile = {"R": slice(2, 5), "K": slice(4, 10)}
        A_split = A.split(A, tile)
        Ax = A(x)
        Ax_split = A_split(x)
        assert (Ax[:, 2:5, 4:10] == Ax_split).all()


def test_sampling_slc():
    N = 64
    ndim = 2
    R, K = 13, 17
    B = 3  # Batch
    idx = torch.randint(0, N - 1, (R, K, ndim))
    x = torch.randn(B, N, N)

    linop = Sampling.from_stacked_idx(idx, (N, N), ("R", "K"))

    linop_split = linop.split(linop, {"R": slice(2, 5), "K": slice(4, 10)})
    Ax = linop(x)
    Ax_split = linop_split(x)
    assert (Ax[:, 2:5, 4:10] == Ax_split).all()


def test_sampling_from_mask():
    """from_mask should produce the same result as from_stacked_idx with the equivalent indices."""
    N = 8
    mask = torch.zeros(N, N, dtype=torch.bool)
    mask[2, 3] = True
    mask[5, 6] = True
    linop = Sampling.from_mask(mask, input_size=(N, N))
    x = torch.randn(N, N)
    result = linop(x)
    assert result.shape[0] == 2  # Two True entries in the mask
    assert result[0].item() == pytest.approx(x[2, 3].item())
    assert result[1].item() == pytest.approx(x[5, 6].item())


def test_sampling_out_of_bounds_index_raises():
    """Sampling should raise ValueError if any index exceeds the input size."""
    N = 8
    idx = (torch.tensor([0, N]),)  # N is out-of-range for size N (valid: 0..N-1)
    with pytest.raises(ValueError, match="range"):
        Sampling(idx, input_size=(N,))


def test_sampling_locs_property_shape():
    """The locs property should return a tensor of shape (*output_shape, ndim)."""
    N = 16
    ndim = 2
    K = 10
    idx = tuple(torch.randint(0, N - 1, (K,)) for _ in range(ndim))
    linop = Sampling(idx, input_size=(N, N), output_shape=("K",))
    locs = linop.locs
    assert locs.shape == (K, ndim)


def test_sampling_register_shape():
    """register_shape should add a named shape to the linop's _shape collection."""
    N, K = 8, 5
    idx = (torch.randint(0, N - 1, (K,)),)
    linop = Sampling(idx, input_size=(N,), output_shape=("K",))
    linop.register_shape("custom", ("A", "B"))
    looked_up = linop._shape["custom"]
    assert len(looked_up) == 2
