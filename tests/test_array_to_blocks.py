from itertools import product
from typing import ClassVar

import pytest
import torch

import torchlinops.functional as F
from torchlinops import ArrayToBlocks
from torchlinops.testing import BaseNamedLinopTests

PYTEST_GPU_MARKS = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU is required but not available"
    ),
]


class TestArrayToBlocks(BaseNamedLinopTests):
    equality_check = "approx"

    # Fixture/test parameterization
    devices = ["cpu", "cuda"]
    instances = ["small3d", "small4d_nomask"]
    # Complex64 fold accumulates up to 81 overlapping blocks with CUDA atomics,
    # so results vary by a few ulps between runs.
    isclose_kwargs: ClassVar[dict] = {"rtol": 1e-3, "atol": 1e-4}

    @pytest.fixture(
        scope="class",
        params=[
            pytest.param((spec, dev), marks=PYTEST_GPU_MARKS)
            if dev == "cuda"
            else (spec, dev)
            for spec, dev in product(instances, devices)
        ],
    )
    @classmethod
    def linop_input_output(cls, request):
        spec, dev = request.param
        spec = request.getfixturevalue(spec)
        device = torch.device(dev)
        dtype = torch.complex64

        batch_size = spec["N"]
        im_size = spec["shape"]
        block_size = spec["block_size"]
        stride = spec["stride"]
        mask = spec["mask"]
        nblocks = F.get_nblocks(im_size, block_size, stride)
        ishape = (*batch_size, *im_size)
        if mask is not None:
            oshape = (*batch_size, *nblocks, int(mask.sum()))
        else:
            oshape = (*batch_size, *nblocks, *block_size)

        linop = ArrayToBlocks(im_size, block_size, stride, mask)
        x = torch.randn(ishape, dtype=dtype, device=device)

        y = torch.randn(oshape, dtype=dtype, device=device)

        return linop, x, y

    @pytest.fixture(scope="class")
    @classmethod
    def small3d(cls, request):
        spec = {
            "N": (1,),
            "shape": (15, 15, 15),
            "block_size": (3, 3, 3),
            "stride": (1, 1, 1),
            "mask": torch.tensor(
                [
                    [
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0],
                    ],
                    [
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                    ],
                ],
                dtype=bool,
            ),
        }
        return spec

    @pytest.fixture(scope="class")
    @classmethod
    def small4d_nomask(cls, request):
        spec = {
            "N": (1,),
            "shape": (15, 15, 15, 15),
            "block_size": (3, 3, 3, 3),
            "stride": (1, 1, 1, 1),
            "mask": None,
        }
        return spec

    def test_size(self, linop_input_output):
        A, _x, _y = linop_input_output
        for dim in A.dims:
            if dim != "...":
                size = A.size(dim)
                assert size is None or isinstance(size, int)

    def test_compile(self, linop_input_output):
        """Use a tiny fixture for the compile smoke test.

        The class's correctness fixture is 3D with stride 1, which produces
        13^3 = 2197 blocks. The CPU fallback unrolls that loop into a graph
        that takes Inductor minutes to schedule (and can exceed Python's
        recursion limit), so compile a single-block 2D operator instead.
        """
        op = ArrayToBlocks((2, 2), (2, 2), (2, 2))
        x = torch.randn(2, 2, 2, dtype=torch.complex64)
        y = op(x).clone()

        torch._dynamo.reset()
        torch.testing.assert_close(torch.compile(op)(x), op(x))

        H = op.H
        torch._dynamo.reset()
        torch.testing.assert_close(torch.compile(H)(y), H(y))

        N = op.N
        torch._dynamo.reset()
        torch.testing.assert_close(torch.compile(N)(x), N(x))
