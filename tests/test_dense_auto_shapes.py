import pytest
import torch
from torchlinops import Dense


class TestDenseAutoShapes:
    def test_dense_2d_auto_shapes(self):
        """Dense(mat) should work for 2D matrices."""
        mat = torch.randn(5, 3)
        A = Dense(mat)
        x = torch.randn(3)
        y = A(x)
        expected = mat @ x
        assert torch.allclose(y, expected)

    def test_dense_3d_auto_shapes(self):
        """Dense(mat) should work for batched 3D matrices."""
        mat = torch.randn(2, 5, 3)  # (batch, M, N)
        A = Dense(mat)
        x = torch.randn(2, 3)  # (batch, N)
        y = A(x)
        expected = torch.matmul(mat, x.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(y, expected)

    def test_dense_explicit_shapes_still_work(self):
        """Original explicit API should still work."""
        mat = torch.randn(5, 3)
        A = Dense(mat, ("M", "N"), ("N",), ("M",))
        x = torch.randn(3)
        y = A(x)
        expected = mat @ x
        assert torch.allclose(y, expected)

    def test_dense_partial_shapes_error(self):
        """Providing some but not all shapes should raise error."""
        mat = torch.randn(5, 3)
        with pytest.raises(ValueError, match="all must be None"):
            Dense(mat, weightshape=("M", "N"))

    def test_dense_1d_error(self):
        """1D tensor should raise error."""
        mat = torch.randn(5)
        with pytest.raises(ValueError, match="at least 2D"):
            Dense(mat)

    def test_dense_auto_shapes_property(self):
        """Auto-inferred shapes should use ordinal ANYs."""
        mat = torch.randn(5, 3)
        A = Dense(mat)
        # Check that shapes are ordinal ANYs
        assert A.weightshape[0].name == "()"
        assert A.weightshape[1].name == "()"
        assert A.ishape[0].name == "()"
        assert A.oshape[0].name == "()"


class TestEinstrSanitization:
    def test_einstr_sanitization(self):
        """Ordinal ANYs should be sanitized for einops."""
        mat = torch.randn(5, 3)
        A = Dense(mat)
        
        # Internal representation uses parentheses
        assert A.weightshape[0].name == "()"
        assert A.weightshape[1].name == "()"
        
        # Einsum string uses sanitized names
        assert "any0" in A.forward_einstr
        assert "any1" in A.forward_einstr
        
        # Should work with einops
        x = torch.randn(3)
        y = A(x)
        assert y.shape == (5,)
