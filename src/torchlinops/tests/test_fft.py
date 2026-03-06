import pytest
import torch

from torchlinops.linops.fft import FFT
from torchlinops.tests.test_base import BaseNamedLinopTests


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
