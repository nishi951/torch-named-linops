import pytest

from torchlinops import NamedDimCollection, ANY


def test_overwrite_any():
    old_shape = ("A", ANY, "C", ANY)
    new_shape = ("A", "B", "C", "D")
    ndc = NamedDimCollection(shape=old_shape)
    ndc.update("shape", new_shape)
    assert ndc.shape == new_shape
