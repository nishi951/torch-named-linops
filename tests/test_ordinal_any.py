import pytest
from torchlinops.nameddim import AnyDim, NamedDimension


class TestAnyDim:
    def test_anydim_base(self):
        """AnyDim() should create base ANY dimension."""
        d = AnyDim()
        assert d.name == "()"
        assert d.i == 0
        assert repr(d) == "()"
        assert d == "()"

    def test_anydim_ordinal_1(self):
        """AnyDim(1) should create ordinal ANY dimension."""
        d = AnyDim(1)
        assert d.name == "()"
        assert d.i == 1
        assert repr(d) == "(1)"
        assert d == "(1)"

    def test_anydim_ordinal_2(self):
        """AnyDim(2) should create ordinal ANY dimension."""
        d = AnyDim(2)
        assert d.name == "()"
        assert d.i == 2
        assert repr(d) == "(2)"
        assert d == "(2)"

    def test_anydim_equality(self):
        """Ordinal ANYs should be distinct."""
        assert AnyDim() != AnyDim(1)
        assert AnyDim(1) != AnyDim(2)
        assert AnyDim() == "()"
        assert AnyDim(1) == "(1)"

    def test_anydim_hash(self):
        """Ordinal ANYs should be hashable and distinct."""
        d1 = AnyDim()
        d2 = AnyDim(1)
        d3 = AnyDim(1)
        assert hash(d1) != hash(d2)
        assert hash(d2) == hash(d3)
        assert len({d1, d2, d3}) == 2
