import pytest
import torch

from torchlinops import Add, Concat, Dense, Dim, Stack
from torchlinops.linops.concat import partition_slices
from torchlinops.tests.test_base import BaseNamedLinopTests
from torchlinops.utils import inner


class TestConcatHorizontal(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        N, P, Q = 1, 3, 4
        A = Dense(
            torch.randn(N, P, Q, dtype=torch.complex64),
            ("N", "P", "Q"),
            ("N", "Q"),
            ("N", "P"),
        )
        B = Dense(
            torch.randn(N, P, Q, dtype=torch.complex64),
            ("N", "P", "Q"),
            ("N", "Q"),
            ("N", "P"),
        )
        C = Concat(A, B, idim="Q")
        x = torch.randn(N, 2 * Q, dtype=torch.complex64)
        y = torch.randn(N, P, dtype=torch.complex64)
        return C, x, y


class TestConcatVertical(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        N, P, Q = 1, 3, 4
        A = Dense(
            torch.randn(N, P, Q, dtype=torch.complex64),
            ("N", "P", "Q"),
            ("N", "Q"),
            ("N", "P"),
        )
        B = Dense(
            torch.randn(N, P, Q, dtype=torch.complex64),
            ("N", "P", "Q"),
            ("N", "Q"),
            ("N", "P"),
        )
        C = Concat(A, B, odim="P")
        x = torch.randn(N, Q, dtype=torch.complex64)
        y = torch.randn(N, 2 * P, dtype=torch.complex64)
        return C, x, y


class TestConcatDiagonal(BaseNamedLinopTests):
    equality_check = "approx"
    isclose_kwargs = dict(rtol=1e-5, atol=1e-5)

    @pytest.fixture(scope="class")
    def linop_input_output(self):
        N, P, Q = 1, 3, 4
        A = Dense(
            torch.randn(N, P, Q, dtype=torch.complex64),
            ("N", "P", "Q"),
            ("N", "Q"),
            ("N", "P"),
        )
        B = Dense(
            torch.randn(N, P, Q, dtype=torch.complex64),
            ("N", "P", "Q"),
            ("N", "Q"),
            ("N", "P"),
        )
        C = Concat(A, B, idim="Q", odim="P")
        x = torch.randn(N, 2 * Q, dtype=torch.complex64)
        y = torch.randn(N, 2 * P, dtype=torch.complex64)
        return C, x, y


def test_concat_shape_inference():
    from torchlinops import ND

    shape = (ND("A"), ND("B"), ND("..."), ND("C"), ND("D"), ND("E"))
    d1 = Concat._infer_dim_idx("B", shape)
    assert d1 == 1
    d2 = Concat._infer_dim_idx("C", shape)
    assert d2 == -3


def test_concat_forward_horizontal():
    N, P, Q = 1, 3, 4
    A = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    B = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    C = Concat(A, B, idim="Q")
    assert C.size("Q") == 2 * Q
    x = torch.randn(C.size("N"), C.size("Q"))
    Cx = C(x)
    xs = x.tensor_split(C.islices, dim=C.idim_idx)[:-1]
    Cx_ref = A(xs[0]) + B(xs[1])
    assert Cx.allclose(Cx_ref)


def test_concat_split():
    N, P, Q = 1, 3, 4
    A = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    B = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    C = Concat(A, B, odim="P")
    C1 = C.split(C, {"P": slice(0, 3)})
    assert isinstance(C1, Dense)

    C2 = C.split(C, {"P": slice(2, 6)})
    assert isinstance(C2, Concat)


def test_concat_getitem():
    N, P, Q = 1, 3, 4
    A = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    B = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    C = Concat(A, B, odim="P")
    assert len(C) == 2
    assert repr(C)


def test_partition_slices_step_raises():
    """partition_slices should raise NotImplementedError for step != 1."""
    with pytest.raises(NotImplementedError, match="step"):
        partition_slices([0, 5, 10], slice(0, 8, 2))


def test_partition_slices_first_boundary_nonzero_raises():
    """partition_slices should raise ValueError if partition[0] != 0."""
    with pytest.raises(ValueError, match="first boundary"):
        partition_slices([1, 5, 10], slice(0, 5))


def test_partition_slices_non_decreasing_raises():
    """partition_slices should raise ValueError for non-monotone boundaries."""
    with pytest.raises(ValueError, match="non-decreasing"):
        partition_slices([0, 10, 5], slice(0, 5))


def test_partition_slices_stop_exceeds_total_raises():
    """partition_slices should raise ValueError when slice.stop > partition[-1]."""
    with pytest.raises(ValueError, match="out of range"):
        partition_slices([0, 5, 10], slice(0, 15))


def test_concat_no_dims_raises():
    """Concat with neither idim nor odim should raise ValueError."""
    N, P, Q = 1, 3, 4
    A = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    B = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    with pytest.raises(ValueError, match="At least one"):
        Concat(A, B)


def test_concat_size_none_raises():
    """Concat should raise ValueError when a sub-linop has unknown size for concat dim."""
    from torchlinops import Identity

    # Identity has size() == None for all dims
    A = Identity(("N",))
    B = Identity(("N",))
    with pytest.raises(ValueError, match="undefined size"):
        Concat(A, B, odim="N")


def test_concat_infer_dim_idx_ellipses_both_sides_raises():
    """_infer_dim_idx should raise ValueError when ELLIPSES appears on both sides of the dim."""
    from torchlinops import ND

    shape = (ND("..."), ND("B"), ND("..."))
    with pytest.raises(ValueError, match="Cannot infer"):
        Concat._infer_dim_idx("B", shape)


def test_concat_getitem_slice():
    """Concat[0:1] should return a new Concat with the sliced sub-linops."""
    N, P, Q = 1, 3, 4
    A = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    B = Dense(torch.randn(N, P, Q), ("N", "P", "Q"), ("N", "Q"), ("N", "P"))
    C_3 = Concat(A, B, A, odim="P")
    sub = C_3[0:2]
    assert isinstance(sub, Concat)
    assert len(sub) == 2
