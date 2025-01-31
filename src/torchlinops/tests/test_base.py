from typing import Literal
import pytest

from abc import ABC, abstractmethod

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
