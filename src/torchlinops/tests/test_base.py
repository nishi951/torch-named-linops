from abc import ABC, abstractmethod
from typing import Literal

import pytest
import torch

from torchlinops.utils import inner


class BaseNamedLinopTests(ABC):
    """Abstract base class for all tests that any linop should satisfy.

    To test a new linop:
    1. Create a class that inherits from this one
    2. Override the `linop_input_output` method to return a parameterized linop, and a test input and output`.
    3. If desired, add additional methods to the class (beginning with `test_`) to test additional functionality.
    """

    equality_check: Literal["exact", "approx"] = "exact"
    isclose_kwargs: dict = {}

    @pytest.fixture
    @abstractmethod
    def linop_input_output(self):
        """Create and return:
        1. A linop to test
        2. A tensor with the same shape as the linop's inputs
        3. A tensor with the same shape as the linop's outputs
        Ideally these are randomized in some fashion
        """
        pass

    def test_adjoint(self, linop_input_output):
        A, x, y = linop_input_output
        yAx = inner(y, A(x))
        xAHy = inner(A.H(y), x)
        if self.equality_check == "exact":
            assert yAx == xAHy
        elif self.equality_check == "approx":
            assert torch.isclose(yAx, xAHy, **self.isclose_kwargs).all()
        else:
            raise ValueError(f"Unrecognized equality_check mode: {self.equality_check}")

    def test_normal(self, linop_input_output):
        A, x, _ = linop_input_output
        AHAx = A.H(A(x))
        ANx = A.N(x)
        if self.equality_check == "exact":
            assert AHAx == ANx
        elif self.equality_check == "approx":
            assert torch.isclose(AHAx, ANx, **self.isclose_kwargs).all()
        else:
            raise ValueError(f"Unrecognized equality_check mode: {self.equality_check}")

    def test_adjoint_level2(self, linop_input_output):
        A, x, y = linop_input_output
        self.test_adjoint((A.H, y, x))
        self.test_normal((A.H, y, y))

    def test_normal_level2(self, linop_input_output):
        A, x, y = linop_input_output
        self.test_adjoint((A.N, x, x))
        self.test_normal((A.N, x, x))

    def test_split(self, linop_input_output):
        A, x, y = linop_input_output
        for dim in set(A.ishape + A.oshape):
            tile = {dim: slice(0, 2)}
            try:
                A_split = A.split(A, tile)
                assert A_split is not None
            except (KeyError, ValueError, NotImplementedError):
                pass

    def test_size(self, linop_input_output):
        A, x, y = linop_input_output
        for dim in A.dims:
            size = A.size(dim)
            assert size is None or isinstance(size, int)

    def test_adj_fn(self, linop_input_output):
        A, x, y = linop_input_output
        AHx = A.H(y)
        adj_fn_result = A.adj_fn(A, y)
        assert torch.isclose(AHx, adj_fn_result, **self.isclose_kwargs).all()

    def test_normal_fn(self, linop_input_output):
        A, x, y = linop_input_output
        ANx = A.N(x)
        normal_fn_result = A.normal_fn(A, x.clone())
        assert torch.isclose(ANx, normal_fn_result, **self.isclose_kwargs).all()

    def test_backprop(self, linop_input_output):
        A, x, y = linop_input_output

        x = x.clone().requires_grad_(True)
        out = A.apply(x)
        if torch.is_complex(out):
            out.real.sum().backward()
        else:
            out.sum().backward()
        grad_out = torch.ones_like(out)
        assert x.grad.allclose(A.H.apply(grad_out), rtol=1e-3)
