import pytest
import torch

import torchlinops.config as config
from torchlinops import FFT, Sampling, Diagonal, Chain
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


def test_chain_inference_mri_forward_model():
    """Chain should infer shapes through FFT -> Sampling -> Diagonal pipeline.
    
    This is a realistic MRI forward model where:
    - FFT: image space -> k-space (NxNy -> KxKy)
    - Sampling: k-space -> undersampled k-space (KxKy -> K)
    - Diagonal: apply mask/weights (K -> K)
    """
    # Create sampling indices (simulate undersampled k-space)
    num_samples = 100
    idx = (torch.randint(0, 64, (num_samples,)), torch.randint(0, 64, (num_samples,)))
    
    # FFT with default grid shapes (NxNy -> KxKy)
    F = FFT(ndim=2, batch_shape=("C",))
    
    # Sampling with partially-wildcard ishape
    S = Sampling(idx, input_size=(64, 64), batch_shape=("C",), output_shape=("K",))
    
    # Diagonal with fully-wildcard shape
    weights = torch.randn(num_samples, dtype=torch.complex64)
    D = Diagonal(weights)
    
    # Without inference, shapes should remain as wildcards
    chain_no_infer = Chain(F, S, D)
    assert "..." in str(S.ishape) or "()" in str(S.ishape)  # Still has wildcards
    assert "..." in str(D.ishape) or "()" in str(D.ishape)  # Still has wildcards
    
    # Reset for next test
    S2 = Sampling(idx, input_size=(64, 64), batch_shape=("C",), output_shape=("K",))
    D2 = Diagonal(weights)
    
    # With inference, shapes should propagate automatically
    with config.using(shape_inference=True):
        chain = Chain(F, S2, D2)
        # Verify shapes were inferred (no wildcards remain)
        assert S2.ishape == F.oshape  # S.input_shape inferred from FFT output
        assert D2.ishape == S2.oshape  # D.ioshape inferred from Sampling output
        assert "..." not in str(S2.ishape) and "()" not in str(S2.ishape)
        assert "..." not in str(D2.ishape) and "()" not in str(D2.ishape)
