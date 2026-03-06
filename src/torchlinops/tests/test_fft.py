import pytest
import torch

from torchlinops.linops.fft import FFT
from torchlinops.tests.test_base import BaseNamedLinopTests
from torchlinops.utils import inner


class TestFFT(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        F = FFT(ndim=2, centered=True, norm="ortho")
        x = torch.randn(8, 10, dtype=torch.complex64)
        y = torch.randn(8, 10, dtype=torch.complex64)
        return F, x, y

    def test_normal_is_identity(self, linop_input_output):
        from torchlinops import Identity

        F, x, y = linop_input_output
        N = F.normal()
        assert isinstance(N, Identity)

    def test_batch_shape(self, linop_input_output):
        F, x, y = linop_input_output
        assert F.batch_shape is not None


def test_fft_invalid_grid_shapes():
    with pytest.raises(ValueError):
        FFT(ndim=2, centered=True, norm="ortho", grid_shapes=(("A",), ("B",), ("C",)))


class TestFFTNotCentered(BaseNamedLinopTests):
    """Test that FFT with centered=False also satisfies the adjoint property."""

    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        F = FFT(ndim=2, centered=False, norm="ortho")
        x = torch.randn(8, 10, dtype=torch.complex64)
        y = torch.randn(8, 10, dtype=torch.complex64)
        return F, x, y


def test_fft_split_forward():
    """split_forward should return an FFT with identical behaviour."""
    F = FFT(ndim=2, centered=True, norm="ortho")
    x = torch.randn(8, 10, dtype=torch.complex64)
    F_split = F.split_forward(
        [slice(None), slice(None)],
        [slice(None), slice(None)],
    )
    assert isinstance(F_split, FFT)
    assert torch.allclose(F(x), F_split(x))


def test_fft_normal_with_inner():
    """FFT.normal(inner=D) should equal F.H @ D @ F numerically."""
    from torchlinops import Diagonal

    F = FFT(ndim=1, centered=True, norm="ortho")
    N = 16
    d = Diagonal(torch.randn(N, dtype=torch.complex64), ("Kx",))
    normal = F.normal(inner=d)
    x = torch.randn(N, dtype=torch.complex64)
    expected = F.H(d(F(x)))
    assert torch.allclose(normal(x), expected, rtol=1e-5)
