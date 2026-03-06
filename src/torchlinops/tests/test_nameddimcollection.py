import pytest

from torchlinops import ANY, NamedDimCollection
from torchlinops.nameddim._matching import max_shape


def test_overwrite_any():
    old_shape = ("A", ANY, "C", ANY)
    new_shape = ("A", "B", "C", "D")
    ndc = NamedDimCollection(shape=old_shape)
    ndc["shape"] = new_shape
    assert ndc.shape == new_shape
    ndc["another_shape"] = ("E", "F")
    assert ndc.another_shape == ("E", "F")


def test_ndc_update():
    """update() should add all shapes from the dict."""
    ndc = NamedDimCollection(shape1=("A", "B"))
    ndc.update({"shape2": ("C", "D")})
    assert ndc.shape2 == ("C", "D")


def test_ndc_add_duplicate_raises():
    """_add should raise ValueError if the shape name already exists."""
    ndc = NamedDimCollection(shape=("A", "B"))
    with pytest.raises(ValueError, match="already in index"):
        ndc._add("shape", ("C", "D"))


def test_ndc_singleton_update():
    """Updating a singleton shape with a new ND should replace it."""
    from torchlinops import ND

    ndc = NamedDimCollection()
    ndc._add("single", "A")
    ndc["single"] = ND("B")
    assert str(ndc.single) == "B"


def test_ndc_mapping_shape():
    """_add should support Mapping-typed data (dict keys → ND)."""
    ndc = NamedDimCollection()
    ndc._add("axes", {"A": 3, "B": 4})
    looked_up = ndc.axes
    # Should return a dict keyed by ND
    assert isinstance(looked_up, dict)


def test_ndc_repr():
    """__repr__ should not raise and should contain the class name."""
    ndc = NamedDimCollection(shape=("X", "Y"))
    r = repr(ndc)
    assert "NamedDimCollection" in r


def test_ndc_incompatible_update_raises():
    """Updating a shape with an incompatible new shape should raise ValueError."""
    ndc = NamedDimCollection(shape=("A", "B"))
    with pytest.raises(ValueError):
        ndc["shape"] = ("C", "D", "E")  # length 3 vs 2, no ellipsis


def test_ndc_singleton_update_non_nd_raises():
    """Updating a singleton shape with a non-singleton value should raise ValueError."""
    from torchlinops import ND

    ndc = NamedDimCollection()
    ndc._add("single", "A")
    # Passing a tuple instead of a single ND should raise
    with pytest.raises(ValueError, match="non-singleton"):
        ndc["single"] = ("B", "C")


# --- max_shape tests ---


def test_max_shape_prefers_higher_index():
    """max_shape should pick the dim with the larger numeric index."""
    result = max_shape([("N", "Q"), ("N", "Q1")])
    assert str(result[1]) == "Q1"


def test_max_shape_ellipses_loses():
    """max_shape should prefer a concrete shape over an ellipsis."""
    result = max_shape([("...",), ("A", "B")])
    names = [str(d) for d in result]
    assert "A" in names and "B" in names


def test_max_shape_incompatible_raises():
    """max_shape should raise ValueError for incompatible shapes."""
    with pytest.raises(ValueError, match="incompatib"):
        max_shape([("A", "B"), ("C",)])
