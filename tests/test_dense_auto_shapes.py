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


class TestDenseOperators:
    def test_dense_adjoint_auto_shapes(self):
        """Adjoint should work with auto-inferred shapes."""
        mat = torch.randn(5, 3, dtype=torch.complex64)
        A = Dense(mat)

        # Adjoint
        AH = A.H
        y = torch.randn(5, dtype=torch.complex64)
        z = AH(y)
        expected = mat.conj().T @ y
        assert torch.allclose(z, expected)

    def test_dense_normal_auto_shapes(self):
        """Normal operator should work with auto-inferred shapes."""
        mat = torch.randn(5, 3, dtype=torch.complex64)
        A = Dense(mat)

        # Normal
        AN = A.N
        x = torch.randn(3, dtype=torch.complex64)
        y = AN(x)
        expected = mat.conj().T @ mat @ x
        assert torch.allclose(y, expected)

    def test_dense_with_named_linops(self):
        """Dense with auto shapes should compose with named linops."""
        from torchlinops import Diagonal

        mat = torch.randn(5, 3)
        A = Dense(mat)
        d = torch.randn(5)
        B = Diagonal(d, ioshape=("M",))

        # Composition should work
        C = B @ A
        x = torch.randn(3)
        y = C(x)
        expected = d * (mat @ x)
        assert torch.allclose(y, expected)
