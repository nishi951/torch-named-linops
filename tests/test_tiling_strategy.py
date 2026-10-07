"""Unit tests for TilingStrategy and create_batched_linop_v2 (issue #201).

Tests the declarative API for device placement/batching:
- TilingStrategy construction and properties
- Placement API (place_all, distribute_across_devices, copy_role)
- Query API (device_of, tiles_on, transfers)
- Pipeline (split, wrap, schedule, apply)
- create_batched_linop_v2 correctness
- Equivalence with create_batched_linop + BatchSpec
"""

import numpy as np
import pytest
import torch

from torchlinops import BatchSpec, Dense, create_batched_linop
from torchlinops.linops.split import (
    TilingStrategy,
    DeviceRole,
    create_batched_linop_v2,
)

CPU = torch.device("cpu")


def _make_dense(B=10, M=6, N=8, seed=42):
    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(B, M, N, generator=g).to(torch.complex64)
    return Dense(weight, ("B", "M", "N"), ("B", "N"), ("B", "M"))


# ---------------------------------------------------------------- construction


class TestConstruction:
    def test_basic_properties(self):
        bounds = {"N": [0, 3, 8], "M": [0, 2, 4, 6]}
        strategy = TilingStrategy(bounds, [CPU])
        assert strategy.axes == ["N", "M"]
        assert strategy.bounds == bounds
        assert strategy.devices == [CPU]
        assert strategy.shape == (2, 3)
        assert strategy.tiles.shape == (2, 3, 3)
        assert strategy.tiles.dtype == int

    def test_single_axis(self):
        strategy = TilingStrategy({"N": [0, 4, 8]}, [CPU])
        assert strategy.axes == ["N"]
        assert strategy.shape == (2,)
        assert strategy.tiles.shape == (2, 3)

    def test_three_axes(self):
        strategy = TilingStrategy(
            {"A": [0, 5, 10], "B": [0, 3, 6], "C": [0, 2, 4]},
            [CPU],
        )
        assert strategy.shape == (2, 2, 2)
        assert strategy.tiles.shape == (2, 2, 2, 3)


# ---------------------------------------------------------------- placement API


class TestPlacementAPI:
    def test_place_all_default_roles(self):
        strategy = TilingStrategy({"N": [0, 3, 8]}, [CPU])
        result = strategy.place_all(CPU)
        assert result is strategy
        assert np.all(strategy.tiles[..., 0] == 0)
        assert np.all(strategy.tiles[..., 1] == 0)
        assert np.all(strategy.tiles[..., 2] == 0)

    def test_place_all_single_role(self):
        strategy = TilingStrategy({"N": [0, 3, 8]}, [CPU])
        strategy.tiles[:] = -1
        strategy.place_all(CPU, roles=(DeviceRole.COMPUTE,))
        assert np.all(strategy.tiles[..., 0] == -1)
        assert np.all(strategy.tiles[..., 1] == 0)
        assert np.all(strategy.tiles[..., 2] == -1)

    def test_distribute_across_devices_1d(self):
        dev0, dev1 = torch.device("cpu"), torch.device("cuda:0")
        strategy = TilingStrategy({"N": [0, 2, 4, 6, 8]}, [dev0, dev1])
        strategy.distribute_across_devices("N")
        expected = np.array([0, 1, 0, 1])
        for role in DeviceRole:
            assert np.array_equal(strategy.tiles[..., role], expected)

    def test_distribute_across_devices_2d_axis0(self):
        dev0, dev1 = torch.device("cpu"), torch.device("cuda:0")
        strategy = TilingStrategy(
            {"N": [0, 2, 4, 6], "M": [0, 3, 6]},
            [dev0, dev1],
        )
        strategy.distribute_across_devices("N")
        expected_col0 = [0, 1, 0]
        expected_col1 = [0, 1, 0]
        for role in DeviceRole:
            assert np.array_equal(strategy.tiles[..., role], [[0, 0], [1, 1], [0, 0]])

    def test_distribute_across_devices_2d_axis1(self):
        dev0, dev1 = torch.device("cpu"), torch.device("cuda:0")
        strategy = TilingStrategy(
            {"N": [0, 2, 4], "M": [0, 3, 6, 9]},
            [dev0, dev1],
        )
        strategy.distribute_across_devices("M")
        for role in DeviceRole:
            assert np.array_equal(strategy.tiles[..., role], [[0, 1, 0], [0, 1, 0]])

    def test_copy_role(self):
        dev0, dev1 = torch.device("cpu"), torch.device("cuda:0")
        strategy = TilingStrategy({"N": [0, 2, 4, 6]}, [dev0, dev1])
        strategy.place_all(dev0)
        strategy.distribute_across_devices("N", roles=(DeviceRole.COMPUTE,))
        strategy.copy_role(DeviceRole.COMPUTE, DeviceRole.OUT)
        assert np.array_equal(
            strategy.tiles[..., DeviceRole.OUT],
            strategy.tiles[..., DeviceRole.COMPUTE],
        )
        assert not np.array_equal(
            strategy.tiles[..., DeviceRole.IN],
            strategy.tiles[..., DeviceRole.COMPUTE],
        )

    def test_fluent_api_chaining(self):
        dev0, dev1 = torch.device("cpu"), torch.device("cuda:0")
        strategy = TilingStrategy(
            {"N": [0, 2, 4], "M": [0, 3, 6]},
            [dev0, dev1],
        )
        result = (
            strategy.place_all(dev0)
            .distribute_across_devices("N", roles=(DeviceRole.COMPUTE,))
            .copy_role(DeviceRole.COMPUTE, DeviceRole.OUT)
        )
        assert result is strategy


# ---------------------------------------------------------------- query API


class TestQueryAPI:
    def test_device_of_single_device(self):
        strategy = TilingStrategy({"N": [0, 3, 8]}, [CPU])
        strategy.place_all(CPU)
        assert strategy.device_of((0,), DeviceRole.IN) == CPU
        assert strategy.device_of((1,), DeviceRole.COMPUTE) == CPU

    def test_device_of_multi_device(self):
        dev0, dev1 = torch.device("cpu"), torch.device("cuda:0")
        strategy = TilingStrategy({"N": [0, 2, 4, 6]}, [dev0, dev1])
        strategy.distribute_across_devices("N")
        assert strategy.device_of((0,), DeviceRole.IN) == dev0
        assert strategy.device_of((1,), DeviceRole.IN) == dev1
        assert strategy.device_of((2,), DeviceRole.IN) == dev0

    def test_tiles_on_single_device(self):
        strategy = TilingStrategy({"N": [0, 3, 8]}, [CPU])
        strategy.place_all(CPU)
        result = strategy.tiles_on(CPU)
        assert DeviceRole.IN in result
        assert DeviceRole.COMPUTE in result
        assert DeviceRole.OUT in result
        assert (0,) in result[DeviceRole.IN]
        assert (1,) in result[DeviceRole.IN]

    def test_tiles_on_multi_device(self):
        dev0, dev1 = torch.device("cpu"), torch.device("cuda:0")
        strategy = TilingStrategy({"N": [0, 2, 4, 6, 8]}, [dev0, dev1])
        strategy.distribute_across_devices("N")
        result0 = strategy.tiles_on(dev0, role=DeviceRole.IN)
        assert DeviceRole.IN in result0
        assert set(result0[DeviceRole.IN]) == {(0,), (2,)}
        result1 = strategy.tiles_on(dev1, role=DeviceRole.IN)
        assert set(result1[DeviceRole.IN]) == {(1,), (3,)}

    def test_tiles_on_empty(self):
        dev0, dev1 = torch.device("cpu"), torch.device("cuda:0")
        strategy = TilingStrategy({"N": [0, 2, 4]}, [dev0, dev1])
        strategy.place_all(dev0)
        result = strategy.tiles_on(dev1, role=DeviceRole.IN)
        assert result[DeviceRole.IN] == []

    def test_transfers_all_same_device(self):
        strategy = TilingStrategy({"N": [0, 3, 8]}, [CPU])
        strategy.place_all(CPU)
        assert strategy.transfers() == []

    def test_transfers_with_different_roles(self):
        dev0, dev1 = torch.device("cpu"), torch.device("cuda:0")
        strategy = TilingStrategy({"N": [0, 2, 4, 6, 8]}, [dev0, dev1])
        strategy.place_all(dev0)
        strategy.distribute_across_devices("N", roles=(DeviceRole.COMPUTE,))
        # 4 tiles, 2 devices: COMPUTE = [0, 1, 0, 1]
        # IN=[0,0,0,0], COMPUTE=[0,1,0,1], OUT=[0,0,0,0]
        # IN != COMPUTE at tiles 1 and 3
        transfers_in_to_compute = strategy.transfers(DeviceRole.IN, DeviceRole.COMPUTE)
        assert len(transfers_in_to_compute) == 2
        for tile_idx, src, dst in transfers_in_to_compute:
            assert src == dev0
            assert dst == dev1
        transfers_compute_to_out = strategy.transfers(
            DeviceRole.COMPUTE, DeviceRole.OUT
        )
        # COMPUTE != OUT at tiles 1 and 3
        assert len(transfers_compute_to_out) == 2


# ---------------------------------------------------------------- split / wrap


class TestSplitAndWrap:
    def test_split_returns_correct_shapes(self):
        strategy = TilingStrategy({"N": [0, 3, 8], "M": [0, 2, 4, 6]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense()
        linops, pre, post = strategy.split(linop)
        assert linops.shape == (2, 3)
        assert pre.shape == (2, 3)
        assert post.shape == (2, 3)

    def test_split_no_device_transfers_on_single_device(self):
        strategy = TilingStrategy({"N": [0, 3, 8]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense()
        _, pre, post = strategy.split(linop)
        for idx in np.ndindex(pre.shape):
            assert pre[idx] is None
            assert post[idx] is None

    def test_wrap_with_no_pre_post(self):
        strategy = TilingStrategy({"N": [0, 3, 8]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense()
        linops, pre, post = strategy.split(linop)
        wrapped = strategy.wrap(linops, pre, post)
        assert wrapped.shape == linops.shape
        for idx in np.ndindex(wrapped.shape):
            assert wrapped[idx] is linops[idx]

    def test_wrap_shape_mismatch_raises(self):
        strategy = TilingStrategy({"N": [0, 3, 8]}, [CPU])
        linops = np.empty((2,), dtype=object)
        pre = np.empty((3,), dtype=object)
        post = np.empty((2,), dtype=object)
        with pytest.raises(ValueError, match="same shape"):
            strategy.wrap(linops, pre, post)


# ---------------------------------------------------------------- schedule


class TestSchedule:
    def test_schedule_reduces_all_axes(self):
        strategy = TilingStrategy({"N": [0, 3, 8], "M": [0, 2, 4, 6]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense()
        linops, pre, post = strategy.split(linop)
        wrapped = strategy.wrap(linops, pre, post)
        result = strategy.schedule(wrapped)
        assert result is not None
        x = torch.randn(10, 8, dtype=torch.complex64)
        y = result(x)
        assert y.shape == (10, 6)


# ---------------------------------------------------------------- apply


class TestApply:
    def test_apply_single_device_matches_original(self):
        strategy = TilingStrategy({"N": [0, 3, 8], "M": [0, 2, 4, 6]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense()
        batched = strategy.apply(linop)
        for _ in range(5):
            x = torch.randn(10, 8, dtype=torch.complex64)
            assert torch.allclose(batched(x), linop(x), rtol=1e-5, atol=1e-6)

    def test_apply_adjoint_matches_original(self):
        strategy = TilingStrategy({"N": [0, 4, 8], "M": [0, 3, 6]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense()
        batched = strategy.apply(linop)
        for _ in range(5):
            y = torch.randn(10, 6, dtype=torch.complex64)
            assert torch.allclose(batched.H(y), linop.H(y), rtol=1e-5, atol=1e-6)

    def test_apply_single_tile_is_identity_split(self):
        linop = _make_dense()
        strategy = TilingStrategy(
            {"N": [0, 8], "M": [0, 6]},
            [CPU],
        )
        strategy.place_all(CPU)
        batched = strategy.apply(linop)
        x = torch.randn(10, 8, dtype=torch.complex64)
        assert torch.allclose(batched(x), linop(x), rtol=1e-5, atol=1e-6)

    def test_apply_1d_split(self):
        strategy = TilingStrategy({"N": [0, 2, 5, 8]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense()
        batched = strategy.apply(linop)
        x = torch.randn(10, 8, dtype=torch.complex64)
        assert torch.allclose(batched(x), linop(x), rtol=1e-5, atol=1e-6)

    def test_apply_ragged_tiles(self):
        strategy = TilingStrategy({"N": [0, 3, 8]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense(B=10, M=6, N=7)
        batched = strategy.apply(linop)
        x = torch.randn(10, 7, dtype=torch.complex64)
        assert torch.allclose(batched(x), linop(x), rtol=1e-5, atol=1e-6)

    def test_apply_with_options(self):
        strategy = TilingStrategy({"N": [0, 3, 8], "M": [0, 2, 4, 6]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense()
        batched = strategy.apply(linop, accumulate=True)
        x = torch.randn(10, 8, dtype=torch.complex64)
        assert torch.allclose(batched(x), linop(x), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- create_batched_linop_v2


class TestCreateBatchedLinopV2:
    def test_empty_strategies_returns_original(self):
        linop = _make_dense()
        result = create_batched_linop_v2(linop, [])
        assert result is linop

    def test_single_strategy_matches_original(self):
        strategy = TilingStrategy({"N": [0, 3, 8], "M": [0, 2, 4, 6]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense()
        batched = create_batched_linop_v2(linop, [strategy])
        for _ in range(5):
            x = torch.randn(10, 8, dtype=torch.complex64)
            assert torch.allclose(batched(x), linop(x), rtol=1e-5, atol=1e-6)

    def test_single_strategy_adjoint(self):
        strategy = TilingStrategy({"N": [0, 4, 8], "M": [0, 3, 6]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense()
        batched = create_batched_linop_v2(linop, [strategy])
        for _ in range(5):
            y = torch.randn(10, 6, dtype=torch.complex64)
            assert torch.allclose(batched.H(y), linop.H(y), rtol=1e-5, atol=1e-6)

    def test_multiple_strategies(self):
        s1 = TilingStrategy({"N": [0, 3, 8]}, [CPU])
        s1.place_all(CPU)
        s2 = TilingStrategy({"M": [0, 2, 4, 6]}, [CPU])
        s2.place_all(CPU)
        linop = _make_dense()
        batched = create_batched_linop_v2(linop, [s1, s2])
        for _ in range(5):
            x = torch.randn(10, 8, dtype=torch.complex64)
            assert torch.allclose(batched(x), linop(x), rtol=1e-5, atol=1e-6)

    def test_with_options(self):
        strategy = TilingStrategy({"N": [0, 3, 8], "M": [0, 2, 4, 6]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense()
        batched = create_batched_linop_v2(linop, [strategy], accumulate=True)
        x = torch.randn(10, 8, dtype=torch.complex64)
        assert torch.allclose(batched(x), linop(x), rtol=1e-5, atol=1e-6)

    def test_apply_matches_create_batched_linop_v2(self):
        strategy = TilingStrategy({"N": [0, 3, 8], "M": [0, 2, 4, 6]}, [CPU])
        strategy.place_all(CPU)
        linop = _make_dense()
        a = strategy.apply(linop)
        b = create_batched_linop_v2(linop, [strategy])
        for _ in range(5):
            x = torch.randn(10, 8, dtype=torch.complex64)
            assert torch.allclose(a(x), b(x), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- equivalence with BatchSpec API


class TestEquivalenceWithBatchSpec:
    """Test that TilingStrategy API produces same results as BatchSpec API."""

    def test_single_axis_split_matches_batchspec(self):
        """TilingStrategy with N bounds [0,2,4,6,8] should match BatchSpec({N: 2})."""
        linop = _make_dense()

        # BatchSpec: uniform chunks of size 2
        batched_old = create_batched_linop(linop, BatchSpec({"N": 2}))

        # TilingStrategy: explicit bounds for same chunks
        strategy = TilingStrategy({"N": [0, 2, 4, 6, 8]}, [CPU])
        strategy.place_all(CPU)
        batched_new = strategy.apply(linop)

        for _ in range(5):
            x = torch.randn(10, 8, dtype=torch.complex64)
            assert torch.allclose(batched_new(x), batched_old(x), rtol=1e-5, atol=1e-6)
            assert torch.allclose(
                batched_new.H(x[:10, :6]),
                batched_old.H(x[:10, :6]),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_two_axis_split_matches_batchspec(self):
        """TilingStrategy with N=[0,2,4,6,8], M=[0,2,4,6] should match BatchSpec({N:2, M:2})."""
        linop = _make_dense()

        # BatchSpec: uniform chunks
        batched_old = create_batched_linop(linop, BatchSpec({"N": 2, "M": 2}))

        # TilingStrategy: explicit bounds for same chunks
        strategy = TilingStrategy({"N": [0, 2, 4, 6, 8], "M": [0, 2, 4, 6]}, [CPU])
        strategy.place_all(CPU)
        batched_new = strategy.apply(linop)

        for _ in range(5):
            x = torch.randn(10, 8, dtype=torch.complex64)
            assert torch.allclose(batched_new(x), batched_old(x), rtol=1e-5, atol=1e-6)

    def test_ragged_split_matches_batchspec(self):
        """TilingStrategy with ragged bounds [0,3,8] should match BatchSpec({N: 3})."""
        linop = _make_dense()

        # BatchSpec: chunks of size 3 → [0,3), [3,6), [6,8)
        batched_old = create_batched_linop(linop, BatchSpec({"N": 3}))

        # TilingStrategy: explicit ragged bounds
        strategy = TilingStrategy({"N": [0, 3, 6, 8]}, [CPU])
        strategy.place_all(CPU)
        batched_new = strategy.apply(linop)

        for _ in range(5):
            x = torch.randn(10, 8, dtype=torch.complex64)
            assert torch.allclose(batched_new(x), batched_old(x), rtol=1e-5, atol=1e-6)

    def test_create_batched_linop_v2_matches_v1(self):
        """create_batched_linop_v2 with single strategy should match create_batched_linop."""
        linop = _make_dense()

        # v1 API
        batched_v1 = create_batched_linop(linop, BatchSpec({"N": 2, "M": 3}))

        # v2 API: bounds for N=8 with chunk 2, M=6 with chunk 3
        strategy = TilingStrategy({"N": [0, 2, 4, 6, 8], "M": [0, 3, 6]}, [CPU])
        strategy.place_all(CPU)
        batched_v2 = create_batched_linop_v2(linop, [strategy])

        for _ in range(5):
            x = torch.randn(10, 8, dtype=torch.complex64)
            assert torch.allclose(batched_v2(x), batched_v1(x), rtol=1e-5, atol=1e-6)

    def test_equivalence_with_accumulate_options(self):
        """Both APIs should produce same results when accumulate=True."""
        linop = _make_dense()

        # v1 with accumulate
        batched_v1 = create_batched_linop(linop, BatchSpec({"N": 2}), accumulate=True)

        # v2 with accumulate
        strategy = TilingStrategy({"N": [0, 2, 4, 6, 8]}, [CPU])
        strategy.place_all(CPU)
        batched_v2 = create_batched_linop_v2(linop, [strategy], accumulate=True)

        for _ in range(5):
            x = torch.randn(10, 8, dtype=torch.complex64)
            assert torch.allclose(batched_v2(x), batched_v1(x), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- GPU tests: ToDevice wrapping


GPU = torch.device("cuda")
GPU_AVAILABLE = torch.cuda.is_available()


class TestToDeviceWrapping:
    """Test that TilingStrategy correctly wraps tiles with ToDevice when devices differ."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")
    def test_split_creates_todevice_when_in_ne_compute(self):
        """When IN device != COMPUTE device, split() should create ToDevice in pre."""
        strategy = TilingStrategy({"N": [0, 4, 8]}, [CPU, GPU])
        strategy.place_all(CPU)
        strategy.distribute_across_devices("N", roles=(DeviceRole.COMPUTE,))
        # Now IN=[CPU, CPU], COMPUTE=[GPU, GPU] (alternating but let's check)

        linop = _make_dense()
        linops, pre, post = strategy.split(linop)

        # Check that pre contains ToDevice objects where IN != COMPUTE
        for idx in np.ndindex(pre.shape):
            in_dev = strategy.device_of(idx, DeviceRole.IN)
            compute_dev = strategy.device_of(idx, DeviceRole.COMPUTE)
            if in_dev != compute_dev:
                assert pre[idx] is not None, (
                    f"pre[{idx}] should be ToDevice when IN != COMPUTE"
                )
                assert hasattr(pre[idx], "ispec"), "pre should contain ToDevice objects"
                assert pre[idx].ispec.device == in_dev
                assert pre[idx].ospec.device == compute_dev

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")
    def test_split_creates_todevice_when_compute_ne_out(self):
        """When COMPUTE device != OUT device, split() should create ToDevice in post."""
        strategy = TilingStrategy({"N": [0, 4, 8]}, [CPU, GPU])
        strategy.place_all(CPU)
        strategy.distribute_across_devices("N", roles=(DeviceRole.COMPUTE,))
        strategy.copy_role(DeviceRole.IN, DeviceRole.OUT)
        # Now IN=[CPU, CPU], COMPUTE=[GPU, GPU], OUT=[CPU, CPU]

        linop = _make_dense()
        linops, pre, post = strategy.split(linop)

        # Check that post contains ToDevice objects where COMPUTE != OUT
        for idx in np.ndindex(post.shape):
            compute_dev = strategy.device_of(idx, DeviceRole.COMPUTE)
            out_dev = strategy.device_of(idx, DeviceRole.OUT)
            if compute_dev != out_dev:
                assert post[idx] is not None, (
                    f"post[{idx}] should be ToDevice when COMPUTE != OUT"
                )
                assert hasattr(post[idx], "ispec"), (
                    "post should contain ToDevice objects"
                )
                assert post[idx].ispec.device == compute_dev
                assert post[idx].ospec.device == out_dev

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")
    def test_wrap_creates_chain_with_todevice(self):
        """wrap() should create Chain objects when pre or post are non-None."""
        from torchlinops import Chain

        strategy = TilingStrategy({"N": [0, 4, 8]}, [CPU, GPU])
        strategy.place_all(CPU)
        strategy.distribute_across_devices("N", roles=(DeviceRole.COMPUTE,))
        strategy.copy_role(DeviceRole.IN, DeviceRole.OUT)

        linop = _make_dense()
        linops, pre, post = strategy.split(linop)
        wrapped = strategy.wrap(linops, pre, post)

        # Check that wrapped contains Chain objects where pre or post are non-None
        for idx in np.ndindex(wrapped.shape):
            has_pre = pre[idx] is not None
            has_post = post[idx] is not None
            if has_pre or has_post:
                assert isinstance(wrapped[idx], Chain), (
                    f"wrapped[{idx}] should be Chain when pre or post are non-None"
                )

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")
    def test_split_places_linops_on_compute_device(self):
        """split() should move tiled linops to their COMPUTE device."""
        strategy = TilingStrategy({"N": [0, 4, 8]}, [CPU, GPU])
        strategy.place_all(CPU)
        strategy.distribute_across_devices("N", roles=(DeviceRole.COMPUTE,))

        linop = _make_dense()
        linops, pre, post = strategy.split(linop)

        # Check that each linop is on its COMPUTE device
        for idx in np.ndindex(linops.shape):
            compute_dev = strategy.device_of(idx, DeviceRole.COMPUTE)
            # Check that the linop's weight is on the compute device
            assert linops[idx].weight.device == compute_dev, (
                f"linop[{idx}] should be on {compute_dev} but is on {linops[idx].weight.device}"
            )

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")
    def test_apply_with_gpu_compute_correctness(self):
        """End-to-end test: apply() with GPU compute should produce correct results."""
        strategy = TilingStrategy({"N": [0, 2, 4, 6, 8]}, [CPU, GPU])
        strategy.place_all(CPU)
        strategy.distribute_across_devices("N", roles=(DeviceRole.COMPUTE,))
        strategy.copy_role(DeviceRole.IN, DeviceRole.OUT)

        linop = _make_dense()
        batched = strategy.apply(linop)

        # Test forward pass
        for _ in range(5):
            x = torch.randn(10, 8, dtype=torch.complex64)
            result = batched(x)
            expected = linop(x)
            assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6), (
                "Forward pass with GPU compute should match original"
            )

        # Test adjoint pass
        for _ in range(5):
            y = torch.randn(10, 6, dtype=torch.complex64)
            result = batched.H(y)
            expected = linop.H(y)
            assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6), (
                "Adjoint pass with GPU compute should match original"
            )

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")
    def test_apply_with_gpu_compute_and_output(self):
        """Test with IN=CPU, COMPUTE=GPU, OUT=CPU (output moves back to CPU)."""
        strategy = TilingStrategy({"N": [0, 4, 8]}, [CPU, GPU])
        strategy.place_all(CPU)
        strategy.distribute_across_devices("N", roles=(DeviceRole.COMPUTE,))
        strategy.copy_role(DeviceRole.IN, DeviceRole.OUT)
        # Now IN=[CPU, CPU], COMPUTE=[GPU, GPU], OUT=[CPU, CPU]

        linop = _make_dense()
        batched = strategy.apply(linop)

        # The final output should be on CPU
        x = torch.randn(10, 8, dtype=torch.complex64)
        result = batched(x)
        expected = linop(x)
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)
        # Output should be on CPU
        assert result.device == CPU

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")
    def test_create_batched_linop_v2_with_gpu(self):
        """create_batched_linop_v2 should work with GPU compute devices."""
        strategy = TilingStrategy({"N": [0, 2, 4, 6, 8], "M": [0, 3, 6]}, [CPU, GPU])
        strategy.place_all(CPU)
        strategy.distribute_across_devices("N", roles=(DeviceRole.COMPUTE,))
        strategy.copy_role(DeviceRole.IN, DeviceRole.OUT)

        linop = _make_dense()
        batched = create_batched_linop_v2(linop, [strategy])

        for _ in range(5):
            x = torch.randn(10, 8, dtype=torch.complex64)
            result = batched(x)
            expected = linop(x)
            assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")
    def test_transfers_with_gpu(self):
        """transfers() should correctly identify transfers to/from GPU."""
        strategy = TilingStrategy({"N": [0, 2, 4, 6, 8]}, [CPU, GPU])
        strategy.place_all(CPU)
        strategy.distribute_across_devices("N", roles=(DeviceRole.COMPUTE,))
        strategy.copy_role(DeviceRole.IN, DeviceRole.OUT)

        transfers_in_to_compute = strategy.transfers(DeviceRole.IN, DeviceRole.COMPUTE)
        # Should have transfers from CPU to GPU
        assert len(transfers_in_to_compute) > 0
        for tile_idx, src, dst in transfers_in_to_compute:
            assert src == CPU
            assert dst.type == "cuda"  # Check device type, not exact device

        transfers_compute_to_out = strategy.transfers(
            DeviceRole.COMPUTE, DeviceRole.OUT
        )
        # Should have transfers from GPU to CPU
        assert len(transfers_compute_to_out) > 0
        for tile_idx, src, dst in transfers_compute_to_out:
            assert src.type == "cuda"  # Check device type
            assert dst == CPU

    @pytest.mark.gpu
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")
    def test_tiles_on_gpu(self):
        """tiles_on() should correctly identify tiles assigned to GPU."""
        strategy = TilingStrategy({"N": [0, 2, 4, 6, 8]}, [CPU, GPU])
        strategy.place_all(CPU)
        strategy.distribute_across_devices("N", roles=(DeviceRole.COMPUTE,))

        gpu_tiles = strategy.tiles_on(GPU, role=DeviceRole.COMPUTE)
        # Should have some tiles on GPU
        assert len(gpu_tiles[DeviceRole.COMPUTE]) > 0
        # Should have some tiles on CPU
        cpu_tiles = strategy.tiles_on(CPU, role=DeviceRole.COMPUTE)
        assert len(cpu_tiles[DeviceRole.COMPUTE]) > 0
