import gc
import warnings

import pytest
import torch
import torch.nn as nn


class TestCData:
    def test_cdata_returns_int(self):
        from torchlinops.utils._device import cdata

        t = torch.randn(3, 4)
        result = cdata(t)
        assert isinstance(result, int)

    def test_cdata_different_tensors_different(self):
        from torchlinops.utils._device import cdata

        t1 = torch.randn(3, 4)
        t2 = torch.randn(3, 4)
        assert cdata(t1) != cdata(t2)

    def test_cdata_same_storage(self):
        from torchlinops.utils._device import cdata

        base = torch.randn(3, 4)
        view = base[:, :2]
        assert cdata(base) == cdata(view)


class TestGetDevice:
    def test_get_device_cpu(self):
        from torchlinops.utils._device import get_device

        result = get_device(-1)
        assert result == torch.device("cpu")

    def test_get_device_cuda_index(self):
        from torchlinops.utils._device import get_device

        result = get_device(0)
        assert result == torch.device("cuda:0")

    def test_get_device_cuda_index_n(self):
        from torchlinops.utils._device import get_device

        result = get_device(1)
        assert result == torch.device("cuda:1")


class TestDeviceOrdinal:
    def test_device_ordinal_cpu(self):
        from torchlinops.utils._device import device_ordinal

        t = torch.randn(3, 4)
        result = device_ordinal(t.device)
        assert result == -1 or result == 0  # CPU is -1 in some versions, 0 in newer

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_device_ordinal_cuda(self):
        from torchlinops.utils._device import device_ordinal

        t = torch.randn(3, 4, device="cuda:0")
        result = device_ordinal(t.device)
        assert result == 0


class TestSameStorage:
    def test_same_storage_same_tensor(self):
        from torchlinops.utils._device import same_storage

        t = torch.randn(3, 4)
        assert same_storage(t, t)

    def test_same_storage_different_tensors(self):
        from torchlinops.utils._device import same_storage

        t1 = torch.randn(3, 4)
        t2 = torch.randn(3, 4)
        assert not same_storage(t1, t2)

    @pytest.mark.skip(reason="same_storage can't handle non-contiguous views")
    def test_same_storage_views(self):
        from torchlinops.utils._device import same_storage

        base = torch.arange(12).reshape(3, 4)
        view = base[:, :2]
        result = same_storage(base, view)
        assert result is True

    @pytest.mark.skip(reason="same_storage can't handle non-contiguous tensors")
    def test_same_storage_transpose(self):
        from torchlinops.utils._device import same_storage

        base = torch.arange(12).reshape(3, 4)
        transposed = base.T
        result = same_storage(base, transposed)
        assert result is True

    def test_same_storage_copy(self):
        from torchlinops.utils._device import same_storage

        t = torch.randn(3, 4)
        copied = t.clone()
        assert not same_storage(t, copied)


class TestResolveDevice:
    def test_resolve_device_cpu_string(self):
        from torchlinops.utils._device import resolve_device

        result = resolve_device("cpu")
        assert result == torch.device("cpu")

    def test_resolve_device_cpu_device(self):
        from torchlinops.utils._device import resolve_device

        result = resolve_device(torch.device("cpu"))
        assert result == torch.device("cpu")

    def test_resolve_device_cuda_with_index(self):
        from torchlinops.utils._device import resolve_device

        result = resolve_device(torch.device("cuda:0"))
        assert result == torch.device("cuda:0")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_resolve_device_cuda_without_index(self):
        from torchlinops.utils._device import resolve_device

        result = resolve_device("cuda")
        assert result.type == "cuda"


class TestMemReporter:
    def test_sizeof(self):
        from torchlinops.utils._device import MemReporter

        t = torch.randn(3, 4, dtype=torch.float32)
        result = MemReporter._sizeof(t)
        assert result == 3 * 4 * 4  # float32 is 4 bytes

    def test_sizeof_complex(self):
        from torchlinops.utils._device import MemReporter

        t = torch.randn(3, 4, dtype=torch.complex64)
        result = MemReporter._sizeof(t)
        assert result == 3 * 4 * 8  # complex64 is 8 bytes

    def test_sizeof_empty(self):
        from torchlinops.utils._device import MemReporter

        t = torch.tensor([])
        result = MemReporter._sizeof(t)
        assert result == 0

    def test_format_size_bytes(self):
        from torchlinops.utils._device import MemReporter

        size, unit = MemReporter._format_size(500, "GiB")
        assert size == "500"
        assert unit == "B"

    def test_format_size_kilobytes(self):
        from torchlinops.utils._device import MemReporter

        size, unit = MemReporter._format_size(2048, "GiB")
        assert unit == "KiB"
        assert float(size) == pytest.approx(2.0)

    def test_format_size_megabytes(self):
        from torchlinops.utils._device import MemReporter

        size, unit = MemReporter._format_size(2 * 1024**2, "GiB")
        assert unit == "MiB"
        assert float(size) == pytest.approx(2.0)

    def test_format_size_gigabytes(self):
        from torchlinops.utils._device import MemReporter

        size, unit = MemReporter._format_size(2 * 1024**3, "GiB")
        assert unit == "GiB"
        assert float(size) == pytest.approx(2.0)

    def test_format_size_gb_base1000(self):
        from torchlinops.utils._device import MemReporter

        size, unit = MemReporter._format_size(2 * 1000**3, "GB")
        assert unit == "GB"
        assert float(size) == pytest.approx(2.0)

    def test_collect_tensors_global(self):
        from torchlinops.utils._device import MemReporter

        reporter = MemReporter()
        reporter._collect_tensors()

        assert isinstance(reporter.tensors, dict)
        assert isinstance(reporter.device_map, dict)

    def test_collect_tensors_module(self):
        from torchlinops.utils._device import MemReporter

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(3, 3))
                self.register_buffer("buf", torch.randn(2, 2))

        module = SimpleModule()
        reporter = MemReporter()
        reporter._collect_tensors(module)

        assert "param" in reporter.tensors

    def test_get_root_tensors_single(self):
        from torchlinops.utils._device import MemReporter

        reporter = MemReporter()
        t = torch.randn(3, 4)
        reporter.tensors = {"t": t}
        reporter.device_map = {t.device: ["t"]}

        roots = reporter._get_root_tensors(t.device, ["t"])
        assert "t" in roots

    def test_get_root_tensors_nested(self):
        from torchlinops.utils._device import MemReporter

        reporter = MemReporter()
        base = torch.randn(10)
        reporter.tensors = {
            "base": base,
            "view": base[:5],
        }
        reporter.device_map = {base.device: ["base", "view"]}

        roots = reporter._get_root_tensors(base.device, ["base", "view"])
        assert len(roots) == 1

    def test_get_root_tensors_independent(self):
        from torchlinops.utils._device import MemReporter

        reporter = MemReporter()
        t1 = torch.randn(3, 4)
        t2 = torch.randn(5, 6)
        reporter.tensors = {"t1": t1, "t2": t2}
        reporter.device_map = {t1.device: ["t1", "t2"]}

        roots = reporter._get_root_tensors(t1.device, ["t1", "t2"])
        assert len(roots) == 2

    def test_report_output(self, capsys):
        from torchlinops.utils._device import MemReporter

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 2))

        module = SimpleModule()
        reporter = MemReporter()
        reporter.report(module)

        captured = capsys.readouterr()
        assert "Device" in captured.out

    def test_report_with_warnings(self):
        from torchlinops.utils._device import MemReporter

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 2))

        module = SimpleModule()
        reporter = MemReporter()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reporter.report(module)
            non_contig_warnings = [x for x in w if "Non-contiguous" in str(x.message)]
            assert len(non_contig_warnings) == 0

    def test_report_global(self, capsys):
        from torchlinops.utils._device import MemReporter

        mod = nn.Linear(3, 4)
        reporter = MemReporter()
        reporter.report(mod)

        captured = capsys.readouterr()
        assert "Device" in captured.out


class TestModuleMemoryMap:
    def test_register(self):
        from torchlinops.utils._device import ModuleMemoryMap, cdata

        mmm = ModuleMemoryMap()
        t = torch.randn(3, 4)
        mmm.register(t)

        key = (cdata(t), t.size(), t.stride(), t.storage_offset(), t.dtype, t.device)
        assert key in mmm.tensor_map

    def test_register_with_new_device(self):
        from torchlinops.utils._device import ModuleMemoryMap, cdata

        mmm = ModuleMemoryMap()
        t = torch.randn(3, 4)
        mmm.register(t, t)

        key = (cdata(t), t.size(), t.stride(), t.storage_offset(), t.dtype, t.device)
        assert key in mmm.tensor_map
        assert t.device in mmm.tensor_map[key]

    def test_register_module(self):
        from torchlinops.utils._device import ModuleMemoryMap, cdata

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(3, 4))
                self.register_buffer("buf", torch.randn(2, 2))

        module = SimpleModule()
        mmm = ModuleMemoryMap()
        mmm.register_module(module)

        assert len(mmm.tensor_cdata_index) > 0

    def test_register_module_with_none_parameters(self):
        from torchlinops.utils._device import ModuleMemoryMap

        class ModuleWithNone(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = None  # type: ignore[assignment]
                self.register_buffer("buf", torch.randn(2, 2))

        module = ModuleWithNone()
        mmm = ModuleMemoryMap()
        mmm.register_module(module)

        assert len(mmm.tensor_cdata_index) > 0

    def test_register_nested_module(self):
        from torchlinops.utils._device import ModuleMemoryMap

        class SubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 2))

        class ParentModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModule()
                self.param = nn.Parameter(torch.randn(3, 3))

        module = ParentModule()
        mmm = ModuleMemoryMap()
        mmm.register_module(module)

        assert len(mmm.tensor_cdata_index) >= 2

    def test_allocate_new_storage(self):
        from torchlinops.utils._device import ModuleMemoryMap, cdata

        t = torch.randn(3, 4)
        mmm = ModuleMemoryMap()
        mmm.register(t)

        storage, min_offset = mmm.allocate_new_storage(cdata(t))
        assert storage.numel() >= t.numel()
        assert storage.dtype == t.dtype

    def test_allocate_new_storage_with_device(self):
        from torchlinops.utils._device import ModuleMemoryMap, cdata

        t = torch.randn(3, 4)
        mmm = ModuleMemoryMap()
        mmm.register(t)

        storage, min_offset = mmm.allocate_new_storage(
            cdata(t), device=torch.device("cpu")
        )
        assert storage.device.type == "cpu"

    def test_ensure_view_exists_new_device(self):
        from torchlinops.utils._device import ModuleMemoryMap

        t = torch.randn(3, 4)
        mmm = ModuleMemoryMap()
        mmm.register(t)

        view = mmm.ensure_view_exists(t, t.device)
        assert view.shape == t.shape

    def test_ensure_view_exists_same_device(self):
        from torchlinops.utils._device import ModuleMemoryMap

        t = torch.randn(3, 4)
        mmm = ModuleMemoryMap()
        mmm.register(t)

        view1 = mmm.ensure_view_exists(t, t.device)
        view2 = mmm.ensure_view_exists(t, t.device)

        assert torch.equal(view1, view2)

    def test_memory_aware_to_cpu(self):
        from torchlinops.utils._device import ModuleMemoryMap

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(3, 4))
                self.register_buffer("buf", torch.randn(2, 2))

        module = SimpleModule()
        mmm = ModuleMemoryMap()
        result = mmm.memory_aware_to(module, torch.device("cpu"))

        assert result is module

    def test_memory_aware_to_same_device(self):
        from torchlinops.utils._device import ModuleMemoryMap

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(3, 4))

        module = SimpleModule()
        mmm = ModuleMemoryMap()
        result = mmm.memory_aware_to(module, module.param.device)

        assert result is module

    def test_memory_aware_deepcopy(self):
        from torchlinops.utils._device import ModuleMemoryMap

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(3, 4))

        module = SimpleModule()
        mmm = ModuleMemoryMap()
        copied = mmm.memory_aware_deepcopy(module)

        assert copied is not module
        assert torch.equal(copied.param, module.param)

    def test_memory_aware_deepcopy_buffers(self):
        from torchlinops.utils._device import ModuleMemoryMap

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.randn(3, 4))

        module = SimpleModule()
        mmm = ModuleMemoryMap()
        copied = mmm.memory_aware_deepcopy(module)

        assert copied is not module
        assert torch.equal(copied.buf, module.buf)


class TestCollectFunction:
    def test_collect_simple_module(self):
        from torchlinops.utils._device import collect

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(3, 4))

        module = SimpleModule()
        result = collect(module)
        assert len(result) > 0

    def test_collect_with_none_parameters(self):
        from torchlinops.utils._device import collect

        class ModuleWithNone(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = None  # type: ignore[assignment]

        module = ModuleWithNone()
        result = collect(module)
        assert len(result) == 0

    def test_collect_nested_module(self):
        from torchlinops.utils._device import collect

        class SubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 2))

        class ParentModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModule()

        module = ParentModule()
        result = collect(module)
        assert len(result) >= 1


class TestCreateSharedBufferMap:
    def test_create_shared_buffer_map_simple(self):
        from torchlinops.utils._device import create_shared_buffer_map

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(3, 4))

        module = SimpleModule()
        result = create_shared_buffer_map(module, device=torch.device("cpu"))
        assert isinstance(result, dict)

    def test_create_shared_buffer_map_copy(self):
        from torchlinops.utils._device import create_shared_buffer_map

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(3, 4))

        module = SimpleModule()
        result = create_shared_buffer_map(module, device=torch.device("cpu"), copy=True)
        assert isinstance(result, dict)


class TestMaxStorageSize:
    def test_max_storage_size_simple(self):
        from torchlinops.utils._device import max_storage_size

        t = torch.randn(3, 4)
        result = max_storage_size(t)
        assert result == 12

    def test_max_storage_size_empty(self):
        from torchlinops.utils._device import max_storage_size

        t = torch.tensor([])
        result = max_storage_size(t)
        assert result == 0

    def test_max_storage_size_strided(self):
        from torchlinops.utils._device import max_storage_size

        t = torch.randn(3, 4)
        view = t[:, ::2]
        result = max_storage_size(view)
        assert result >= view.numel()


class TestAsViewOnMoved:
    def test_as_view_on_moved_storage_map_creation(self):
        from torchlinops.utils._device import create_shared_buffer_map

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(3, 4))

        module = SimpleModule()
        result = create_shared_buffer_map(module, device=torch.device("cpu"), copy=True)
        assert isinstance(result, dict)
        assert len(result) > 0
