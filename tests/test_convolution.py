import pytest
import torch

from torchlinops import Dim
from torchlinops.linops.convolution import Convolution
from torchlinops.testing import BaseNamedLinopTests


class TestConvolution2D(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(4, 3, 3, 3)  # (out_c, in_c, kx, ky)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=("c_in", "x", "y"),
            out_grid_shape=("c_out", "x", "y"),
        )
        x = torch.randn(3, 8, 8)  # (c_in, x, y)
        y = torch.randn(4, 8, 8)  # (c_out, x, y)
        return conv, x, y


class TestConvolutionCorrectness:
    """Verify Convolution matches PyTorch's native operations."""

    def test_forward_matches_pytorch(self):
        """Forward should match F.conv2d."""
        import torch.nn.functional as F

        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=("c_in", "x", "y"),
            out_grid_shape=("c_out", "x", "y"),
            padding="zeros",
        )
        x = torch.randn(3, 8, 8)

        # Our implementation
        result = conv(x)

        # PyTorch native
        padding = tuple(k // 2 for k in weight.shape[2:])
        expected = F.conv2d(x.unsqueeze(0), weight, padding=padding).squeeze(0)

        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_adjoint_matches_pytorch(self):
        """Adjoint should match F.conv_transpose2d."""
        import torch.nn.functional as F

        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=("c_in", "x", "y"),
            out_grid_shape=("c_out", "x", "y"),
            padding="zeros",
        )
        y = torch.randn(4, 8, 8)

        # Our implementation
        result = conv.H(y)

        # PyTorch native
        padding = tuple(k // 2 for k in weight.shape[2:])
        expected = F.conv_transpose2d(y.unsqueeze(0), weight, padding=padding).squeeze(0)

        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_forward_1d_matches_pytorch(self):
        """1D forward should match F.conv1d."""
        import torch.nn.functional as F

        weight = torch.randn(4, 3, 5)
        conv = Convolution(
            weight,
            ndim=1,
            in_grid_shape=("c_in", "x"),
            out_grid_shape=("c_out", "x"),
        )
        x = torch.randn(3, 16)

        result = conv(x)
        padding = tuple(k // 2 for k in weight.shape[2:])
        expected = F.conv1d(x.unsqueeze(0), weight, padding=padding).squeeze(0)

        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_forward_3d_matches_pytorch(self):
        """3D forward should match F.conv3d."""
        import torch.nn.functional as F

        weight = torch.randn(4, 3, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=3,
            in_grid_shape=("c_in", "x", "y", "z"),
            out_grid_shape=("c_out", "x", "y", "z"),
        )
        x = torch.randn(3, 8, 8, 8)

        result = conv(x)
        padding = tuple(k // 2 for k in weight.shape[2:])
        expected = F.conv3d(x.unsqueeze(0), weight, padding=padding).squeeze(0)

        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_forward_with_stride(self):
        """Forward with stride should match F.conv2d."""
        import torch.nn.functional as F

        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=("c_in", "x", "y"),
            out_grid_shape=("c_out", "x2", "y2"),
            stride=2,
        )
        x = torch.randn(3, 8, 8)

        result = conv(x)
        padding = tuple(k // 2 for k in weight.shape[2:])
        expected = F.conv2d(x.unsqueeze(0), weight, stride=2, padding=padding).squeeze(0)

        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)


class TestConvolutionNormalModes:
    """Test different normal operator modes."""

    def test_normal_conv_mode(self):
        """Composed convolution mode should match default."""
        weight = torch.randn(4, 3, 3, 3)
        conv_default = Convolution(
            weight,
            ndim=2,
            in_grid_shape=("c_in", "x", "y"),
            out_grid_shape=("c_out", "x", "y"),
            padding="zeros",
        )
        conv_conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=("c_in", "x", "y"),
            out_grid_shape=("c_out", "x", "y"),
            padding="zeros",
            normal_mode="conv",
        )
        x = torch.randn(3, 8, 8)
        assert torch.allclose(conv_default.N(x), conv_conv.N(x), rtol=1e-4)

    def test_normal_fft_mode_circular(self):
        """FFT mode should work for circular padding."""
        weight = torch.randn(4, 3, 3, 3)
        conv_fft = Convolution(
            weight,
            ndim=2,
            in_grid_shape=("c_in", "x", "y"),
            out_grid_shape=("c_out", "x", "y"),
            padding="circular",
            normal_mode="fft",
        )
        x = torch.randn(3, 8, 8)
        result = conv_fft.N(x)
        assert result.shape == x.shape

    def test_normal_fft_mode_error_non_circular(self):
        """FFT mode should raise error for non-circular padding."""
        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=("c_in", "x", "y"),
            out_grid_shape=("c_out", "x", "y"),
            padding="zeros",
            normal_mode="fft",
        )
        x = torch.randn(3, 8, 8)
        with pytest.raises(ValueError, match="FFT normal mode only supports circular padding"):
            conv.N(x)


class TestConvolution1D(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(4, 3, 5)  # (out_c, in_c, kx)
        conv = Convolution(
            weight,
            ndim=1,
            in_grid_shape=("c_in", "x"),
            out_grid_shape=("c_out", "x"),
        )
        x = torch.randn(3, 16)  # (c_in, x)
        y = torch.randn(4, 16)  # (c_out, x)
        return conv, x, y


class TestConvolution3D(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(4, 3, 3, 3, 3)  # (out_c, in_c, kx, ky, kz)
        conv = Convolution(
            weight,
            ndim=3,
            in_grid_shape=("c_in", "x", "y", "z"),
            out_grid_shape=("c_out", "x", "y", "z"),
        )
        x = torch.randn(3, 8, 8, 8)  # (c_in, x, y, z)
        y = torch.randn(4, 8, 8, 8)  # (c_out, x, y, z)
        return conv, x, y


class TestConvolutionBatched(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            batch_shape=("batch",),
            in_grid_shape=("c_in", "x", "y"),
            out_grid_shape=("c_out", "x", "y"),
        )
        x = torch.randn(2, 3, 8, 8)  # (batch, c_in, x, y)
        y = torch.randn(2, 4, 8, 8)  # (batch, c_out, x, y)
        return conv, x, y


class TestConvolutionStride(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=("c_in", "x", "y"),
            out_grid_shape=("c_out", "x2", "y2"),
            stride=2,
        )
        x = torch.randn(3, 8, 8)  # (c_in, x, y)
        y = torch.randn(4, 4, 4)  # (c_out, x/2, y/2)
        return conv, x, y


class TestConvolutionCircular(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-4, atol=1e-4)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        weight = torch.randn(4, 3, 3, 3)
        conv = Convolution(
            weight,
            ndim=2,
            in_grid_shape=("c_in", "x", "y"),
            out_grid_shape=("c_out", "x", "y"),
            padding="circular",
        )
        x = torch.randn(3, 8, 8)
        y = torch.randn(4, 8, 8)
        return conv, x, y
