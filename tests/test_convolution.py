import pytest
import torch

from torchlinops import Dim
from torchlinops.linops.convolution import Convolution, FFTConvolution
from torchlinops.testing import BaseNamedLinopTests


class TestConvolution1D(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class", params=[torch.float32, torch.complex64])
    def linop_input_output(self, request):
        dtype = request.param
        weight = torch.randn(5, dtype=dtype)  # (out_c, in_c, kx)
        conv = Convolution(
            weight,
            in_grid_shape=("x",),
            out_grid_shape=("x",),
        )
        x = torch.randn(3, 16, dtype=dtype)  # (c_in, x)
        y = torch.randn(3, 16, dtype=dtype)  # (c_out, x)
        return conv, x, y


class TestConvolution2D(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class", params=[torch.float32, torch.complex64])
    def linop_input_output(self, request):
        dtype = request.param
        weight = torch.randn(3, 3, dtype=dtype)  # (kx, ky)
        conv = Convolution(
            weight,
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        x = torch.randn(3, 8, 8, dtype=dtype)  # (c_in, x, y)
        y = torch.randn(3, 8, 8, dtype=dtype)  # (c_out, x, y)
        return conv, x, y


class TestConvolution3D(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class", params=[torch.float32, torch.complex64])
    def linop_input_output(self, request):
        dtype = request.param
        weight = torch.randn(3, 3, 3, dtype=dtype)  # (out_c, in_c, kx, ky, kz)
        conv = Convolution(
            weight,
            in_grid_shape=("x", "y", "z"),
            out_grid_shape=("x", "y", "z"),
        )
        x = torch.randn(3, 8, 8, 8, dtype=dtype)  # (c_in, x, y, z)
        y = torch.randn(3, 8, 8, 8, dtype=dtype)  # (c_out, x, y, z)
        return conv, x, y


class TestConvolutionNormalModes:
    """Test different normal operator modes."""

    def test_normal_default_mode(self):
        weight = torch.randn(3, 3)
        conv_default = Convolution(
            weight,
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
            padding_mode="zeros",
            normal_mode=None,
        )
        x = torch.randn(1, 5, 5)
        assert torch.allclose(
            conv_default.N(x), conv_default.H(conv_default(x)), rtol=1e-4
        )

    def test_normal_conv_mode(self):
        """Composed convolution mode should match default."""
        weight = torch.randn(2, 3)
        conv_default = Convolution(
            weight,
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
            padding_mode="circular",
            normal_mode=None,
        )
        conv_conv = Convolution(
            weight,
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
            padding_mode="circular",
            normal_mode="conv",
        )
        x = torch.randn(1, 5, 2)
        assert torch.allclose(conv_default.N(x), conv_conv.N(x), rtol=1e-4)

    def test_normal_fft_mode_circular(self):
        """FFT mode should work for circular padding_mode."""
        weight = torch.randn(3, 3)
        conv_fft = Convolution(
            weight,
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
            padding_mode="circular",
            normal_mode="fft",
        )
        x = torch.randn(3, 11, 8)
        result = conv_fft.N(x)
        assert result.shape == x.shape

    def test_normal_fft_mode_error_non_circular(self):
        """FFT mode should raise error for non-circular padding_mode."""
        weight = torch.randn(3, 3)
        conv = Convolution(
            weight,
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
            padding_mode="zeros",
            normal_mode="fft",
        )
        x = torch.randn(3, 8, 8)
        with pytest.raises(ValueError, match="FFT normal mode only"):
            conv.N(x)


class TestConvolutionBatched(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(3, 3)
        conv = Convolution(
            weight,
            batch_shape=("batch", "channel"),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        x = torch.randn(2, 3, 8, 8)  # (batch, c_in, x, y)
        y = torch.randn(2, 3, 8, 8)  # (batch, c_out, x, y)
        return conv, x, y


class TestConvolutionCircular(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(3, 3, 3)
        conv = Convolution(
            weight,
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
            padding_mode="circular",
        )
        x = torch.randn(3, 8, 8)
        y = torch.randn(3, 8, 8)
        return conv, x, y


class TestFFTConvolution1D(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class", params=[torch.float32, torch.complex64])
    def linop_input_output(self, request):
        dtype = request.param
        weight = torch.randn(5, dtype=dtype)
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x",),
            out_grid_shape=("x",),
        )
        x = torch.randn(3, 16, dtype=dtype)
        y = torch.randn(3, 16, dtype=dtype)
        return fft_conv, x, y


class TestFFTConvolution2D(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class", params=[torch.float32, torch.complex64])
    def linop_input_output(self, request):
        dtype = request.param
        weight = torch.randn(3, 3, dtype=dtype)
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        x = torch.randn(3, 8, 8, dtype=dtype)
        y = torch.randn(3, 8, 8, dtype=dtype)
        return fft_conv, x, y


class TestFFTConvolution3D(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class", params=[torch.float32, torch.complex64])
    def linop_input_output(self, request):
        dtype = request.param
        weight = torch.randn(3, 3, 3, dtype=dtype)
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y", "z"),
            out_grid_shape=("x", "y", "z"),
        )
        x = torch.randn(3, 8, 8, 8, dtype=dtype)
        y = torch.randn(3, 8, 8, 8, dtype=dtype)
        return fft_conv, x, y


class TestFFTConvolutionBatched(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(3, 3)
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("batch", "channel"),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        x = torch.randn(2, 3, 8, 8)
        y = torch.randn(2, 3, 8, 8)
        return fft_conv, x, y


class TestFFTConvolutionBehavior:
    """Test FFTConvolution-specific behaviors not covered by BaseNamedLinopTests."""

    def test_forward_matches_circular_convolution_float(self):
        """FFT convolution forward should match circular Convolution forward for real kernels."""
        torch.manual_seed(0)
        weight = torch.randn(3, 3)
        circ_conv = Convolution(
            weight,
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
            padding_mode="circular",
        )
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        x = torch.randn(3, 8, 8)
        assert torch.allclose(circ_conv(x), fft_conv(x), rtol=1e-4, atol=1e-4)

    def test_forward_matches_circular_convolution_complex(self):
        """FFT convolution forward should match circular Convolution forward for complex kernels."""
        torch.manual_seed(0)
        weight = torch.randn(3, 3, dtype=torch.complex64)
        circ_conv = Convolution(
            weight,
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
            padding_mode="circular",
        )
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        x = torch.randn(3, 8, 8, dtype=torch.complex64)
        assert torch.allclose(circ_conv(x), fft_conv(x), rtol=1e-4, atol=1e-4)

    def test_adjoint_matches_circular_convolution(self):
        """FFT convolution adjoint should match circular Convolution adjoint."""
        torch.manual_seed(0)
        weight = torch.randn(3, 3)
        circ_conv = Convolution(
            weight,
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
            padding_mode="circular",
        )
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        y = torch.randn(3, 8, 8)
        assert torch.allclose(circ_conv.H(y), fft_conv.H(y), rtol=1e-4, atol=1e-4)

    def test_normal_matches_circular_convolution(self):
        """FFT convolution normal should match circular Convolution normal (composed)."""
        torch.manual_seed(0)
        weight = torch.randn(3, 3)
        circ_conv = Convolution(
            weight,
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
            padding_mode="circular",
        )
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        x = torch.randn(3, 8, 8)
        assert torch.allclose(circ_conv.N(x), fft_conv.N(x), rtol=1e-4, atol=1e-4)

    def test_fourier_kernel_caching(self):
        """Fourier kernel should be cached after first computation."""
        weight = torch.randn(3, 3)
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        grid_size = (8, 8)
        assert len(fft_conv._fourier_kernel_cache) == 0
        fk1 = fft_conv._get_fourier_kernel(grid_size)
        assert grid_size in fft_conv._fourier_kernel_cache
        fk2 = fft_conv._get_fourier_kernel(grid_size)
        assert fk1 is fk2

    def test_fourier_kernel_different_grid_sizes(self):
        """Different grid sizes should produce different cached Fourier kernels."""
        weight = torch.randn(3, 3)
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        x1 = torch.randn(1, 8, 8)
        x2 = torch.randn(1, 16, 16)
        fft_conv(x1)
        fft_conv(x2)
        assert (8, 8) in fft_conv._fourier_kernel_cache
        assert (16, 16) in fft_conv._fourier_kernel_cache
        assert len(fft_conv._fourier_kernel_cache) == 2

    def test_kernel_larger_than_grid_raises(self):
        """Should raise ValueError when kernel dimension exceeds grid dimension."""
        weight = torch.randn(5, 5)
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        with pytest.raises(ValueError, match="grid_sizes"):
            fft_conv._get_fourier_kernel((3, 3))

    def test_size_returns_none(self):
        """size() should always return None for FFTConvolution."""
        weight = torch.randn(3, 3)
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        for dim in fft_conv.dims:
            assert fft_conv.size(dim) is None

    def test_real_output_dtype(self):
        """Real kernel + real input should produce real output."""
        weight = torch.randn(3, 3)
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        x = torch.randn(3, 8, 8)
        y = fft_conv(x)
        assert not torch.is_complex(y)

    def test_complex_kernel_real_input_produces_complex_output(self):
        """Complex kernel + real input should produce complex output."""
        weight = torch.randn(3, 3, dtype=torch.complex64)
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        x = torch.randn(3, 8, 8)
        y = fft_conv(x)
        assert torch.is_complex(y)

    def test_real_kernel_complex_input_produces_complex_output(self):
        """Real kernel + complex input should produce complex output."""
        weight = torch.randn(3, 3)
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x", "y"),
            out_grid_shape=("x", "y"),
        )
        x = torch.randn(3, 8, 8, dtype=torch.complex64)
        y = fft_conv(x)
        assert torch.is_complex(y)

    def test_1d_forward_matches_circular(self):
        """1D FFT convolution should match 1D circular convolution."""
        torch.manual_seed(0)
        weight = torch.randn(5)
        circ_conv = Convolution(
            weight,
            in_grid_shape=("x",),
            out_grid_shape=("x",),
            padding_mode="circular",
        )
        fft_conv = FFTConvolution(
            weight,
            batch_shape=("...",),
            in_grid_shape=("x",),
            out_grid_shape=("x",),
        )
        x = torch.randn(3, 16)
        assert torch.allclose(circ_conv(x), fft_conv(x), rtol=1e-4, atol=1e-4)
