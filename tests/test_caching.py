"""Tests for caching behavior of adjoint and normal operators.

These tests verify that accessing .H and .N properties returns consistent
cached objects and that internal operations don't corrupt the cache.
"""

import copy

import pytest
import torch

from torchlinops import Add, Chain, Concat, Dense, Diagonal, Stack


def test_adjoint_caching():
    """Test that .H property returns the same cached object."""
    N = 5
    weight = torch.randn(N, dtype=torch.complex64)
    A = Diagonal(weight, ("N",))

    # First access creates the cache
    adj1 = A.H
    # Second access should return the same cached object
    adj2 = A.H

    assert adj1 is adj2, "Adjoint should be cached and return the same object"


def test_normal_caching():
    """Test that .N property returns the same cached object."""
    N = 5
    weight = torch.randn(N, dtype=torch.complex64)
    A = Diagonal(weight, ("N",))

    # First access creates the cache
    norm1 = A.N
    # Second access should return the same cached object
    norm2 = A.N

    assert norm1 is norm2, "Normal should be cached and return the same object"


def test_adjoint_of_adjoint_returns_original():
    """Test that A.H.H returns the original operator A."""
    N = 5
    weight = torch.randn(N, dtype=torch.complex64)
    A = Diagonal(weight, ("N",))

    # Get adjoint twice
    adj1 = A.H
    adj2 = A.H.H

    # Should return the original operator
    assert adj2 is A, "A.H.H should return the original operator A"


def test_stack_horizontal_normal_does_not_mutate_cache():
    """Test that Stack.normal() doesn't mutate cached adjoints/normals."""
    P, Q = 3, 4
    A = Dense(torch.randn(P, Q, dtype=torch.complex64), ("P", "Q"), ("Q",), ("P",))
    B = Dense(torch.randn(P, Q, dtype=torch.complex64), ("P", "Q"), ("Q",), ("P",))

    # Prime the cache
    A_H = A.H
    A_N = A.N
    B_H = B.H
    B_N = B.N

    # Create horizontal stack and compute normal
    stack = Stack(A, B, idim_and_idx=("N", 0))
    _ = stack.N

    # Verify cached objects weren't mutated
    assert A.H is A_H, "A.H cache was corrupted"
    assert A.N is A_N, "A.N cache was corrupted"
    assert B.H is B_H, "B.H cache was corrupted"
    assert B.N is B_N, "B.N cache was corrupted"


def test_stack_vertical_normal_does_not_mutate_cache():
    """Test that Stack.normal() with vertical stacking doesn't mutate cache."""
    P, Q = 3, 4
    A = Dense(torch.randn(P, Q, dtype=torch.complex64), ("P", "Q"), ("Q",), ("P",))
    B = Dense(torch.randn(P, Q, dtype=torch.complex64), ("P", "Q"), ("Q",), ("P",))

    # Prime the cache
    A_H = A.H
    A_N = A.N
    B_H = B.H
    B_N = B.N

    # Create vertical stack and compute normal
    stack = Stack(A, B, odim_and_idx=("M", 1))
    _ = stack.N

    # Verify cached objects weren't mutated
    assert A.H is A_H, "A.H cache was corrupted"
    assert A.N is A_N, "A.N cache was corrupted"
    assert B.H is B_H, "B.H cache was corrupted"
    assert B.N is B_N, "B.N cache was corrupted"


def test_stack_diagonal_normal_does_not_mutate_cache():
    """Test that Stack.normal() with diagonal stacking doesn't mutate cache."""
    P, Q = 3, 4
    A = Dense(torch.randn(P, Q, dtype=torch.complex64), ("P", "Q"), ("Q",), ("P",))
    B = Dense(torch.randn(P, Q, dtype=torch.complex64), ("P", "Q"), ("Q",), ("P",))

    # Prime the cache
    A_H = A.H
    A_N = A.N
    B_H = B.H
    B_N = B.N

    # Create diagonal stack and compute normal
    stack = Stack(A, B, idim_and_idx=("N", 0), odim_and_idx=("M", 1))
    _ = stack.N

    # Verify cached objects weren't mutated
    assert A.H is A_H, "A.H cache was corrupted"
    assert A.N is A_N, "A.N cache was corrupted"
    assert B.H is B_H, "B.H cache was corrupted"
    assert B.N is B_N, "B.N cache was corrupted"


def test_add_normal_does_not_mutate_cache():
    """Test that Add.normal() doesn't mutate cached adjoints/normals."""
    A = Dense(torch.randn(5, 5, dtype=torch.complex64), ("M", "N"), ("N",), ("M",))
    B = Dense(torch.randn(5, 5, dtype=torch.complex64), ("M", "N"), ("N",), ("M",))

    # Prime the cache
    A_H = A.H
    A_N = A.N
    B_H = B.H
    B_N = B.N

    # Create Add and compute normal
    add = Add(A, B)
    _ = add.N

    # Verify cached objects weren't mutated
    assert A.H is A_H, "A.H cache was corrupted"
    assert A.N is A_N, "A.N cache was corrupted"
    assert B.H is B_H, "B.H cache was corrupted"
    assert B.N is B_N, "B.N cache was corrupted"


def test_concat_normal_does_not_mutate_cache():
    """Test that Concat.normal() doesn't mutate cached adjoints/normals."""
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

    # Prime the cache
    A_H = A.H
    A_N = A.N
    B_H = B.H
    B_N = B.N

    # Create Concat and compute normal
    concat = Concat(A, B, idim="Q")
    _ = concat.N

    # Verify cached objects weren't mutated
    assert A.H is A_H, "A.H cache was corrupted"
    assert A.N is A_N, "A.N cache was corrupted"
    assert B.H is B_H, "B.H cache was corrupted"
    assert B.N is B_N, "B.N cache was corrupted"


def test_adjoint_creation_does_not_mutate_cache():
    """Test that creating adjoints doesn't mutate cached linops."""
    weight = torch.randn(5, dtype=torch.complex64)
    A = Diagonal(weight, ("N",))

    # Prime the cache
    A_H = A.H
    A_N = A.N

    # Access adjoint again
    _ = A.H

    # Verify cached objects weren't mutated
    assert A.H is A_H, "A.H cache was corrupted"
    assert A.N is A_N, "A.N cache was corrupted"
