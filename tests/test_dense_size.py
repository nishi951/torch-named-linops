import pytest
import torch

from torchlinops import Dense


def test_size_returns_weight_extent_without_ellipsis():
    weight = torch.randn(5, 3)
    linop = Dense(weight, ("M", "N"), ("N",), ("M",))

    assert linop.size("M") == 5
    assert linop.size("N") == 3
    assert linop.size("Z") is None


def test_size_returns_weight_extent_with_one_axis_ellipsis():
    weight = torch.randn(2, 5, 3)
    linop = Dense(weight, ("...", "M", "N"), ("...", "N"), ("...", "M"))

    assert linop.size("M") == 5
    assert linop.size("N") == 3
    assert linop.size("...") is None


def test_size_returns_weight_extent_with_multi_axis_ellipsis():
    weight = torch.randn(2, 7, 5, 3)
    linop = Dense(weight, ("...", "M", "N"), ("...", "N"), ("...", "M"))

    assert linop.size("M") == 5
    assert linop.size("N") == 3


def test_size_returns_weight_extent_with_ellipsis_between_named_dims():
    weight = torch.randn(5, 2, 7, 3)
    linop = Dense(weight, ("M", "...", "N"), ("...", "N"), ("...", "M"))

    assert linop.size("M") == 5
    assert linop.size("N") == 3


def test_size_handles_auto_any_dims():
    weight = torch.randn(5, 3)
    linop = Dense(weight)

    assert linop.size(linop.weightshape[0]) == 5
    assert linop.size(linop.weightshape[1]) == 3


def test_size_returns_none_for_broadcast_dims():
    weight = torch.randn(2, 3, 4)
    linop = Dense(
        weight,
        ("B", "P", "Q"),
        ("B", "Q"),
        ("B", "P"),
        broadcast_dims=["B"],
    )

    assert linop.size("B") is None
    assert linop.size("P") == 3
    assert linop.size("Q") == 4


def test_size_tracks_dim_renames_in_weightshape():
    weight = torch.randn(5, 3)
    linop = Dense(weight, ("M", "N"), ("N",), ("M",))

    linop.oshape = ("M1",)

    assert linop.size("M1") == 5
    assert linop.size("M") is None


def test_check_input_shape_uses_multi_axis_ellipsis_sizes():
    weight = torch.randn(2, 7, 5, 3)
    linop = Dense(weight, ("...", "M", "N"), ("...", "M"), ("...", "N"))

    linop.check_input_shape(torch.randn(2, 7, 5))
    with pytest.raises(ValueError, match="expected dim M with size 5"):
        linop.check_input_shape(torch.randn(2, 7, 4))


def test_incompatible_weightshape_raises_at_construction():
    with pytest.raises(ValueError, match="not compatible"):
        Dense(torch.randn(5, 3), ("M", "N", "K"), ("N",), ("M",))


@pytest.mark.parametrize(
    "weightshape",
    [
        ("...", "M", "...", "N"),
        ("...", "...", "M", "N"),
    ],
)
def test_multiple_ellipses_in_weightshape_raises(weightshape):
    with pytest.raises(ValueError, match="at most one ellipsis"):
        Dense(torch.randn(2, 3, 4, 5), weightshape, ("...", "N"), ("...", "M"))
