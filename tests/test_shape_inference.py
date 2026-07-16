import torchlinops.config as config
from torchlinops.nameddim import resolve_wildcards


def test_shape_inference_default_false():
    """shape_inference should default to False."""
    assert config.shape_inference is False


def test_shape_inference_context_manager():
    """shape_inference should be toggleable via context manager."""
    assert config.shape_inference is False
    
    with config.using(shape_inference=True):
        assert config.shape_inference is True
    
    assert config.shape_inference is False


def test_resolve_wildcards_fully_wildcard():
    """resolve_wildcards should handle fully-wildcard shapes."""
    result = resolve_wildcards(("...",), ("A", "B", "C"))
    assert result == ("A", "B", "C")


def test_resolve_wildcards_partially_wildcard():
    """resolve_wildcards should handle partially-wildcard shapes."""
    result = resolve_wildcards(("C", "..."), ("C", "Kx", "Ky"))
    assert result == ("C", "Kx", "Ky")


def test_resolve_wildcards_no_wildcards():
    """resolve_wildcards should return original shape if no wildcards."""
    result = resolve_wildcards(("A", "B"), ("A", "B"))
    assert result == ("A", "B")


def test_resolve_wildcards_incompatible():
    """resolve_wildcards should return original shape if incompatible."""
    result = resolve_wildcards(("X", "..."), ("A", "B"))
    assert result == ("X", "...")
