import pytest
from itertools import product

from math import prod

import torch

import torchlinops.functional as F
from torchlinops import ArrayToBlocks

from torchlinops.utils import is_adjoint
from torchlinops.tests.test_base import BaseNamedLinopTests


PYTEST_GPU_MARKS = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU is required but not available"
    ),
]


class TestArrayToBlocks(BaseNamedLinopTests):
    equality_check = "approx"

    # Fixture/test parameterization
    devices = ["cpu"]
    instances = ["small3d"]

    @pytest.fixture(
        scope="class",
        params=product(instances, devices),
    )
    def linop_input_output(self, request):
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
    def small3d(self, request):
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
