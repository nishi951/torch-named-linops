"""Polynomial preconditioning for iterative solvers.

Adapted from https://github.com/sidward/ppcs
(DOI: https://zenodo.org/badge/latestdoi/452385092I).
"""

from typing import Literal

import numpy as np
from scipy.special import binom
from sympy import (
    Float,
    Interval,
    Poly,
    diff,
    integrate,
    simplify,
    stationary_points,
    symbols,
)

from torchlinops.linops import Identity, NamedLinop

__all__ = ["polynomial_preconditioner"]


def polynomial_preconditioner(
    T: NamedLinop,
    degree: int,
    norm: Literal["l_2", "l_inf", "ifista"] = "l_2",
    lower_eig: float = 0.0,
    upper_eig: float = 1.0,
) -> NamedLinop:
    """Construct a polynomial preconditioner $P(T)$ for a linear operator $T$.

    Given eigenvalue bounds, computes optimal polynomial coefficients and
    returns a ``NamedLinop`` that applies $P(T) = \\sum_k c_k T^k$.

    Parameters
    ----------
    T : NamedLinop
        The operator to precondition.
    degree : int
        Polynomial degree. Use ``-1`` for no preconditioning, ``0+`` for
        polynomial preconditioning.
    norm : ``{"l_2", "l_inf", "ifista"}``, default ``"l_2"``
        Optimization criterion for computing the polynomial coefficients:

        - ``"l_2"`` -- Minimizes $\\int |1 - x p(x)|^2 dx$.
        - ``"l_inf"`` -- Minimizes $\\sup |1 - x p(x)|$ (Chebyshev).
        - ``"ifista"`` -- Coefficients from DOI:10.1137/140970537.
    lower_eig : float, default 0.0
        Lower bound on the eigenvalue spectrum of $T$.
    upper_eig : float, default 1.0
        Upper bound on the eigenvalue spectrum of $T$.

    Returns
    -------
    NamedLinop
        The polynomial preconditioner applied to *T*.
    """
    Id: NamedLinop = Identity()  # Fixing shapes
    if degree < 0:
        return Id

    if norm == "l_2":
        c, _ = l_2_opt(degree, lower_eig, upper_eig)
    elif norm == "l_inf":
        c, _ = l_inf_opt(degree, lower_eig, upper_eig)
    elif norm == "ifista":
        c = ifista_coeffs(degree)
    else:
        raise ValueError(f"Unknown polynomial preconditioning norm option: {norm}")

    def phelper(c) -> NamedLinop:
        if c.size == 1:
            return c[0] * Id
        L = c[0] * Id  # ... -> ...
        R = phelper(c[1:]) @ T @ Id  # ... -> ...
        return L + R  # ... -> ...

    P = phelper(c)
    return P


def l_inf_opt(degree, lower=0, upper=1, verbose=True):
    """Compute the $L_\\infty$-optimal polynomial minimizing $\\sup|1 - x p(x)|$.

    Uses Chebyshev polynomials following Equation 50 of Shewchuk,
    "An introduction to the conjugate gradient method without the agonizing
    pain, Edition 1 1/4."

    Parameters
    ----------
    degree : int
        Degree of the polynomial.
    lower : float, default 0
        Lower bound of the eigenvalue interval.
    upper : float, default 1
        Upper bound of the eigenvalue interval.
    verbose : bool, default True
        Print diagnostic information.

    Returns
    -------
    coeffs : np.ndarray
        Polynomial coefficients (lowest degree first).
    polyexpr : sympy.Expr
        The resulting polynomial as a SymPy expression.

    References
    ----------
    Chebyshev package: https://github.com/mlazaric/Chebyshev/
    (DOI: 10.5281/zenodo.5831845).
    """
    from Chebyshev.chebyshev import polynomial as chebpoly

    assert degree >= 0

    if verbose:
        print("L-infinity optimized polynomial.")
        print("> Degree:   %d" % degree)
        print("> Spectrum: [%0.2f, %0.2f]" % (lower, upper))

    T = chebpoly.get_nth_chebyshev_polynomial(degree + 1)

    y = symbols("y")
    P = T((upper + lower - 2 * y) / (upper - lower))
    P = P / P.subs(y, 0)
    P = simplify((1 - P) / y)

    if verbose:
        print("> Resulting polynomial: %s" % repr(P))

    if degree > 0:
        points = stationary_points(P, y, Interval(lower, upper))
        vals = np.array(
            [P.subs(y, point) for point in points]
            + [P.subs(y, lower)]
            + [P.subs(y, upper)]
        )
        assert np.abs(vals).min() > 1e-8, "Polynomial not injective."

    c = Poly(P).all_coeffs()[::-1] if degree > 0 else (Float(P),)
    return (np.array(c, dtype=np.float32), P)


def l_2_opt(degree, lower=0, upper=1, weight=1, verbose=True):
    """Compute the $L_2$-optimal polynomial minimizing $\\int w(x)(1 - x p(x))^2 dx$.

    The weight function $w(x)$ can be used to emphasize regions of the
    eigenvalue spectrum.

    Parameters
    ----------
    degree : int
        Degree of the polynomial.
    lower : float, default 0
        Lower bound of the eigenvalue interval.
    upper : float, default 1
        Upper bound of the eigenvalue interval.
    weight : sympy.Expr or float, default 1
        Weight function $w(x)$ for the $L_2$ integrand.
    verbose : bool, default True
        Print diagnostic information.

    Returns
    -------
    coeffs : np.ndarray
        Polynomial coefficients (lowest degree first), as float32.
    polyexpr : sympy.Expr
        The resulting polynomial as a SymPy expression.

    References
    ----------
    Johnson, Micchelli, and Paul, "Polynomial Preconditioners for Conjugate
    Gradient Calculations", DOI: 10.1137/0720025.
    """
    if verbose:
        print("L-2 optimized polynomial.")
        print("> Degree:   %d" % degree)
        print("> Spectrum: [%0.2f, %0.2f]" % (lower, upper))

    c = symbols("c0:%d" % (degree + 1))
    x = symbols("x")

    p = sum([(c[k] * x**k) for k in range(degree + 1)])
    f = weight * (1 - x * p) ** 2
    J = integrate(f, (x, lower, upper))

    mat = [[0] * (degree + 1) for _ in range(degree + 1)]
    vec = [0] * (degree + 1)

    for edx in range(degree + 1):
        eqn = diff(J, c[edx])
        tmp = eqn.copy()
        # Coefficient index
        for cdx in range(degree + 1):
            mat[edx][cdx] = float(Poly(eqn, c[cdx]).coeffs()[0])
            tmp = tmp.subs(c[cdx], 0)
        vec[edx] = float(-tmp)

    mat = np.array(mat, dtype=np.double)
    vec = np.array(vec, dtype=np.double)
    res = np.array(np.linalg.pinv(mat) @ vec, dtype=np.float32)

    poly = sum([(res[k] * x**k) for k in range(degree + 1)])
    if verbose:
        print("> Resulting polynomial: %s" % repr(poly))

    if degree > 0:
        points = stationary_points(poly, x, Interval(lower, upper))
        vals = np.array(
            [poly.subs(x, point) for point in points]
            + [poly.subs(x, lower)]
            + [poly.subs(x, upper)]
        )
        assert vals.min() > 1e-8, "Polynomial is not positive."

    return (res, poly)


def ifista_coeffs(degree):
    """Compute polynomial coefficients from the improved FISTA algorithm.

    Parameters
    ----------
    degree : int
        Degree of the polynomial.

    Returns
    -------
    coeffs : np.ndarray
        Binomial-based polynomial coefficients.

    References
    ----------
    Bhotto, Ahmad, and Swamy, "An Improved Fast Iterative Shrinkage
    Thresholding Algorithm for Image Deblurring", DOI: 10.1137/140970537.
    """
    c = []
    for k in range(degree + 1):
        c.append(binom(degree + 1, k + 1) * ((-1) ** (k)))
    return np.array(c)
