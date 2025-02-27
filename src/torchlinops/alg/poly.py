"""Copied from https://github.com/sidward/ppcs
DOI: https://zenodo.org/badge/latestdoi/452385092I
"""

from typing import Callable, Literal

from dataclasses import dataclass
import logging

import numpy as np
from scipy.special import binom
from sympy import (
    symbols,
    simplify,
    stationary_points,
    Interval,
    Poly,
    Float,
    diff,
    integrate,
)

from torchlinops.linops import NamedLinop, Identity

__all__ = ["polynomial_preconditioner"]


def polynomial_preconditioner(
    T: NamedLinop,
    degree: int,
    norm: Literal["l_2", "l_inf", "ifista"] = "l_2",
    lower_eig: float = 0.0,
    upper_eig: float = 1.0,
) -> NamedLinop:
    """Apply polynomial preconditioning to a linop.
    norm : Type of preconditioner.
        - "l_2"    = l_2 optimized polynomial.
        - "l_inf"  = l_inf optimized polynomial.
        - "ifista" = from DOI: 10.1137/140970537.
    degree : int, >= -1
        Degree of polynomial to use
        -1 = no preconditioning
        0+ = preconditioning
    lower_eig, upper_eig : float
        Eigenvalue bounds for coefficient optimization
    Returns
    -------
    P : NamedLinop
        Polynomial preconditioned version of T.
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
    """
    (coeffs, polyexpr) = l_inf_opt(degree, l=0, L=1, verbose=True)

    Calculate polynomial p(x) that minimizes the supremum of |1 - x p(x)|
    over (l, L).

    Based on Equation 50 of:
       Shewchuk, J. R.
       An introduction to the conjugate gradient method without the agonizing
       pain, Edition 1¼.

    Uses the following package:
      https://github.com/mlazaric/Chebyshev/
      DOI: 10.5281/zenodo.5831845

    Inputs:
      degree (Int): Degree of polynomial to calculate.
      lower (Float): Lower bound of interval.
      upper (Float): Upper bound of interval.
      verbose (Bool): Print information.

    Returns:
      coeffs (Array): Coefficients of optimized polynomial.
      polyexpr (SymPy): Resulting polynomial as a SymPy expression.
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
    """
    (coeffs, polyexpr) = l_2_opt(degree, l=0, L=1, verbose=True)

    Calculate polynomial p(x) that minimizes the following:

    ..math:
      \int_l^l w(x) (1 - x p(x))^2 dx

    To incorporate priors, w(x) can be used to weight regions of the
    interval (l, L) of the expression above.

    Based on:
      Polynomial Preconditioners for Conjugate Gradient Calculations
      Olin G. Johnson, Charles A. Micchelli, and George Paul
      DOI: 10.1137/0720025

    Inputs:
      degree (Int): Degree of polynomial to calculate.
      l (Float): Lower bound of interval.
      L (Float): Upper bound of interval.
      weight (SymPy): Sympy expression to include prior weight.
      verbose (Bool): Print information.

    Returns:
      coeffs (Array): Coefficients of optimized polynomial.
      polyexpr (SymPy): Resulting polynomial as a SymPy expression.
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
    """
    coeffs = ifista_coeffs(degree)

    Returns coefficients from:
      An Improved Fast Iterative Shrinkage Thresholding Algorithm for Image
      Deblurring
      Md. Zulfiquar Ali Bhotto, M. Omair Ahmad, and M. N. S. Swamy
      DOI: 10.1137/140970537

    Inputs:
      degree (Int): Degree of polynomial to calculate.

    Returns:
      coeffs (Array): Coefficients of optimized polynomial.
    """
    c = []
    for k in range(degree + 1):
        c.append(binom(degree + 1, k + 1) * ((-1) ** (k)))
    return np.array(c)
