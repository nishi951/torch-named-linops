import torch

from torchlinops import (
    Chain,
    Dense,
    Diagonal,
    NamedShape,
    NUFFT,
    NS,
    Dim,
    get_nd_shape,
)


def test_dense():
    M, N = 9, 3
    weight = torch.randn(M, N, dtype=torch.complex64)
    weightshape = ("M", "N")
    x = torch.randn(N, dtype=torch.complex64)
    ishape = ("N",)
    # y = torch.randn(M)
    oshape = ("M",)
    A = Dense(weight, weightshape, ishape, oshape)
    assert torch.isclose(A.N(x), A.H(A(x))).all()
    # Make sure dense's normal doesn't create a chain (unnecessary)
    # If desired, just make the linop explicitly
    assert not isinstance(A.N, Chain)
    assert A.N.ishape == ("N",)
    assert A.N.oshape == ("N1",)


def test_diagonal_normal():
    M = 10
    N, P = 5, 7
    weight = torch.randn(M, 1, 1, dtype=torch.complex64)
    # weightshape = ("M",)
    x = torch.randn(M, N, P, dtype=torch.complex64)
    ioshape = ("M", "N", "P")
    A = Diagonal(weight, ioshape)
    assert torch.isclose(A.N(x), A.H(A(x))).all()
    assert not isinstance(A.N, Chain)


def test_normal_shape_propagation():
    M, N, P = 3, 5, 7

    B = Dense(torch.randn(M, P), ("M", "P"), ishape=("M",), oshape=("P",))
    C = Diagonal(torch.randn(M), ("...",))
    D = Dense(torch.randn(M, N), ("M", "N"), ishape=("N",), oshape=("M",))

    A = B @ C @ D
    AN = A.N

    required_shape = NamedShape(("M1",), ("N1",))

    assert AN[-1].ishape == required_shape.ishape
    assert AN[-1].oshape == required_shape.oshape


def test_composed_normal_nufft_dense():
    """Regression test: composed operator's .N with NUFFT @ Dense.

    When computing A.N for A = F @ S where F is NUFFT and S is Dense with
    a coil dimension, the normal operator should produce the same result
    as A.H(A(x)).
    See: https://github.com/nishi951/torch-named-linops/issues/205
    """
    im_size = (32, 32)
    num_coils = 2
    num_shots = 4
    torch.manual_seed(42)
    mps = torch.randn(num_coils, *im_size, dtype=torch.complex64)
    trj = torch.rand(num_shots, 256, 2, dtype=torch.float32) - 0.5

    im_shape = get_nd_shape(2)
    shape = NS(None) + NS(tuple(), ("C",)) + NS(im_shape)
    S = Dense(
        mps,
        weightshape=("C", *im_shape),
        ishape=shape.ishape,
        oshape=shape.oshape,
        broadcast_dims=None,
    )
    F = NUFFT(trj, im_size, output_shape=Dim("RK"), mode="interpolate", oversamp=1.25)

    x = torch.randn(im_size, dtype=torch.complex64)

    # Individual .N should work
    assert F.N(x).shape == x.shape
    assert S.N(x).shape == x.shape

    # Composed operator
    A = F @ S

    # A.H(A(x)) should work
    expected = A.H(A(x))
    assert expected.shape == x.shape

    # A.N(x) should produce the same result
    actual = A.N(x)
    assert actual.shape == x.shape
    assert torch.allclose(actual, expected, rtol=1e-4)


def test_normal_fallback_drop_and_rename():
    """When a component's normal both drops a dim and renames another, the
    chain should degrade to the naive A^H A rather than raise or corrupt.
    """
    im_shape = get_nd_shape(2)  # ("Nx", "Ny")
    C, R = 2, 7

    shape = NS(None) + NS(tuple(), ("C",)) + NS(im_shape)
    S = Dense(
        torch.randn(C, 4, 5, dtype=torch.complex64),
        weightshape=("C", *im_shape),
        ishape=shape.ishape,
        oshape=shape.oshape,
    )
    # B's normal renames Nx->Nx1, Ny->Ny1 while dropping the C batch dim
    B = Dense(
        torch.randn(R, 4, 5, dtype=torch.complex64),
        weightshape=("R", *im_shape),
        ishape=NS(None).ishape + tuple(im_shape),
        oshape=NS(None).ishape + ("R",),
    )

    x = torch.randn(4, 5, dtype=torch.complex64)
    A = B @ S
    assert torch.allclose(A.N(x), A.H(A(x)), rtol=1e-4)
