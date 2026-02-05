import gc
import itertools
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Optional, TypeVar
from warnings import warn

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "cdata",
    "tensor_memory_span",
    "get_device",
    "device_ordinal",
    "same_storage",
    "MemReporter",
    "ModuleMemoryMap",
    "memory_aware_to",
    "memory_aware_deepcopy",
]

T = TypeVar("T")
logger = logging.getLogger("torchlinops.utils")

TensorKey = tuple[int, int, tuple, int, torch.dtype, torch.device]
CData = int


def resolve_device(dev):
    """Resolve the device to be used for tensors.

    Parameters
    ----------
    dev : str or torch.device
        The device string or torch.device object to resolve.

    Returns
    -------
    torch.device
        A device object representing the resolved device.
    """
    d = torch.device(dev)
    if d.type == "cuda" and d.index is None:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return d


def memory_aware_to(module, device):
    """Move the given module to the specified device while being memory aware.

    Parameters
    ----------
    module : torch.nn.Module
        The module to move to the specified device.
    device : str or torch.device
        The device to which the module should be moved.

    Returns
    -------
    torch.nn.Module
        The module moved to the specified device.
    """
    return ModuleMemoryMap().memory_aware_to(module, device)


def memory_aware_deepcopy(module):
    """Create a deep copy of the given module while being memory aware.

    Parameters
    ----------
    module : torch.nn.Module
        The module to be deep copied.

    Returns
    -------
    torch.nn.Module
        A deep copy of the module.
    """
    return ModuleMemoryMap().memory_aware_deepcopy(module)


def get_key(t: Tensor):
    return (cdata(t), t.size(), t.stride(), t.storage_offset(), t.dtype, t.device)


@dataclass
class ModuleMemoryMap:
    """Remembers module and submodule memory layout."""

    tensor_cdata_index: dict[CData, list[Tensor]] = field(
        default_factory=lambda: defaultdict(list)
    )
    """Maps cdata to a list of tensors associated with that buffer"""

    storage_map: dict[tuple[CData, torch.device], dict[torch.device, Tensor]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    """Maps (cdata, original device) -> map from (device) to (buffer storage)"""

    tensor_map: dict[TensorKey, dict[torch.device, Tensor]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def register(self, t: Tensor, new_t: Optional[Tensor] = None):
        device = t.device if new_t is None else new_t.device
        self.tensor_map[get_key(t)][device] = t
        self.tensor_cdata_index[cdata(t)].append(t)

    def register_module(self, module):
        """Construct the smallest set of tensors that can be used to hold all the parameters in a module."""

        def _collect(m):
            """Recursively collect all parameters and buffers in the module."""
            for name, t in list(m.named_parameters(recurse=False)) + list(
                m.named_buffers(recurse=False)
            ):
                if t is None:
                    continue
                # key = get_key(t)
                # key = cdata(t)
                self.register(t.data)

            for child in m.children():
                _collect(child)

        _collect(module)

    def allocate_new_storage(self, cdata_t, device=None):
        tensors = self.tensor_cdata_index[cdata_t]
        # Find max offset + size for any tensor sharing this storage
        min_offset = float("inf")
        max_offset = float("-inf")
        dtype = None
        device_orig = None
        for t in tensors:
            new_min_offset, new_max_offset = tensor_memory_span(t)
            # size_bytes = t.element_size() * max_storage_size(t)
            # offset_bytes = t.storage_offset() * t.element_size()
            min_offset = min(min_offset, new_min_offset)
            max_offset = max(max_offset, new_max_offset)
            dtype = t.dtype
            device_orig = t.device

        # Create a flat tensor that spans the full required storage
        new_tensor = torch.empty(
            max_offset - min_offset + 1, dtype=dtype, device=device_orig
        )
        new_device = device if device is not None else device_orig
        return new_tensor.to(new_device), min_offset

    def ensure_view_exists(self, t, device):
        # If storage has not been allocated, make new storage
        device_storage_map = self.storage_map[(cdata(t), t.device)]
        if device not in device_storage_map:
            # Allocate new storage
            storage, min_offset = self.allocate_new_storage(cdata(t), device)
            device_storage_map[device] = (storage, min_offset)
        else:
            storage, min_offset = device_storage_map[device]

        # If storage exists, get a view on it
        if (view := self.tensor_map[get_key(t)].get(device)) is None:
            view = storage.as_strided(
                t.size(),
                t.stride(),
                t.storage_offset() - min_offset,
            ).copy_(t)
            self.tensor_map[get_key(t)][device] = view
        return view

    def memory_aware_to(self, module: nn.Module, device: torch.device):
        """Move a module to a device, without unnecessary memory overhead."""
        device = resolve_device(device)
        # storage_tensors = collect(module)
        # if id(module) in self.memo[device]:
        #     return module

        # storage_map = self.storage_maps[device]
        # create_shared_buffer_map(
        #     module, device, copy=False, storage_map=self.tensor_map
        # )

        self.register_module(module)

        # Remember which modules were visited
        def remap(m, level=0):
            # Need memoization at the buffer level
            # if id(m) in self.memo[device]:
            #     return
            logger.debug("\t" * level + f"{type(m).__name__}")
            for name, t in m._parameters.items():
                if t is not None and t.device != device:
                    view = self.ensure_view_exists(t, device)
                    m._parameters[name] = nn.Parameter(
                        view, requires_grad=t.requires_grad
                    )

            for name, t in m._buffers.items():
                if t is not None and t.device != device:
                    view = self.ensure_view_exists(t, device)
                    m._buffers[name] = view
            for child in m.children():
                remap(child, level + 1)
            # self.memo[device].add(id(m))

        remap(module)

        return module

    def memory_aware_deepcopy(self, module):
        """Deepcopy a module, without unnecessary memory overhead."""
        # storage_map = create_shared_buffer_map(module, copy=True)
        self.register_module(module)
        for cdata_t in self.tensor_cdata_index.keys():
            self.allocate_new_storage(cdata_t)
        # Make copies of every buffer
        self.storage_map = deepcopy(self.storage_map)

        # Recursively
        def copy_memory_aware(m: nn.Module):
            cls = type(m)
            new = cls.__new__(cls)
            new.__dict__ = m.__dict__.copy()
            new._parameters = dict()
            new._buffers = dict()
            new._modules = dict()

            for name, t in m._parameters.items():
                if t is not None:
                    new_t = self.ensure_view_exists(t, t.device)
                    # new_t = as_view_on_moved(t, storage_map)
                    new._parameters[name] = nn.Parameter(
                        new_t, requires_grad=t.requires_grad
                    )
            for name, t in m._buffers.items():
                if t is not None:
                    new_t = self.ensure_view_exists(t, t.device)
                    # new_t = as_view_on_moved(t, storage_map)
                    new._buffers[name] = new_t
            for module_name, child_module in m._modules.items():
                new._modules[module_name] = copy_memory_aware(child_module)
            return new

        new_module = copy_memory_aware(module)
        return new_module


def collect(module) -> dict:
    """Construct the smallest set of tensors that can be used to hold all the parameters in a module."""
    storage_tensors = defaultdict(list)

    def _collect(m):
        """Recursively collect all parameters and buffers in the module."""
        for name, t in list(m.named_parameters(recurse=False)) + list(
            m.named_buffers(recurse=False)
        ):
            if t is None:
                continue
            key = cdata(t)
            storage_tensors[key].append(t)
        for child in m.children():
            _collect(child)

    _collect(module)

    return storage_tensors


def create_shared_buffer_map(module, device=None, copy: bool = False):
    storage_tensors = collect(module)

    # Maps from cdata to tensor on target device
    storage_map = {}
    for key, tensors in storage_tensors.items():
        if all(tensor.device == device for tensor in tensors) and not copy:
            # No new memory need be allocated for this block.
            continue

        # Find max offset + size for any tensor sharing this storage
        min_offset = float("inf")
        max_offset = float("-inf")
        dtype = None
        device_orig = None
        for t in tensors:
            new_min_offset, new_max_offset = tensor_memory_span(t)
            # size_bytes = t.element_size() * max_storage_size(t)
            # offset_bytes = t.storage_offset() * t.element_size()
            min_offset = min(min_offset, new_min_offset)
            max_offset = max(max_offset, new_max_offset)
            dtype = t.dtype
            device_orig = t.device

        # Create a flat tensor that spans the full required storage
        new_tensor = torch.empty(
            max_offset - min_offset + 1, dtype=dtype, device=device_orig
        )
        new_device = device if device is not None else device_orig
        storage_map[key] = (new_tensor.to(new_device), min_offset)
    return storage_map


def tensor_memory_span(t: torch.Tensor):
    """Compute the min and max memory offsets of a tensor.

    Supports negative strides (even though PyTorch doesn't.)

    """
    shape = t.shape
    strides = t.stride()
    offset = t.storage_offset()

    # For each dimension, pick the index that causes min and max access
    # If stride >= 0: (0, size-1), else (size-1, 0)
    index_ranges = [
        (0, s - 1) if stride >= 0 else (s - 1, 0) for s, stride in zip(shape, strides)
    ]

    # Generate all corners
    corners = itertools.product(*index_ranges)

    # Compute linear memory offset for each corner
    mem_offsets = [offset + sum(i * s for i, s in zip(idx, strides)) for idx in corners]

    return min(mem_offsets), max(mem_offsets)


def cdata(t: Tensor):
    """Get a pointer to the block of memory that the tensor is part of."""
    return t.untyped_storage()._cdata


def max_storage_size(tensor):
    """Compute the number of elements spanned by a strided tensor."""
    if tensor.numel() == 0:
        return 0
    size = 0
    for i, (dim, stride) in enumerate(zip(tensor.size(), tensor.stride())):
        size += (dim - 1) * stride
    return size + 1


def as_view_on_moved(tensor, storage_map):
    """Copy the data from a tensor to the actual target location in the target allocated storage."""
    key = cdata(tensor)
    base, min_offset = storage_map[key]
    view = base.as_strided(
        tensor.size(), tensor.stride(), tensor.storage_offset() - min_offset
    ).copy_(tensor)
    return view


def get_device(device_idx: int = -1):
    return torch.device(f"cuda:{device_idx}" if device_idx >= 0 else "cpu")


def device_ordinal(device: torch.device):
    return torch.zeros(1, device=device).get_device()


def same_storage(x, y):
    """Determine if tensors share the same storage or not"""
    x_ptrs = set(e.data_ptr() for e in x.view(-1))
    y_ptrs = set(e.data_ptr() for e in y.view(-1))
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)


class MemReporter:
    """A utility class for reporting memory usage of PyTorch tensors by device.

    Features:
        - Tracks tensors in Python scope (excluding C++-managed buffers)
        - Reports memory in GB (base 1024) or GiB (base 1000) format
        - Identifies root tensors to avoid double-counting overlapping memory
        - Supports module-specific analysis or global tensor tracking

    Parameters
    ----------
    format_mode : Literal["GB", "GiB"]
        Memory unit format (GB=base1024, GiB=base1000)

    Attributes
    ----------
    tensors : dict
        Name-to-tensor mapping for all collected tensors
    device_map : defaultdict
        Maps devices to tensor names

    Examples
    --------
    >>> reporter = MemReporter(format_mode="GiB")
    >>> reporter.report()  # Analyze all tracked tensors
    >>> reporter.report(module=my_model)  # Analyze tensors in a specific module

    Notes
    -----
    - Does not track:
        * Tensors allocated in C++ (e.g. backward pass buffers)
        * Gradient tensors (.grad attributes)
    - Non-contiguous tensors may have inefficient pointer calculations
    """

    def __init__(self, format_mode: Literal["GB", "GiB"] = "GiB"):
        self.format_mode = format_mode
        self.tensors = {}
        self.device_map = defaultdict(list)

    @staticmethod
    def _sizeof(tensor):
        return tensor.element_size() * tensor.nelement()

    @staticmethod
    def _format_size(size_B, mode: Literal["GB", "GiB"] = "GB"):
        if mode == "GB":
            base = 1000
            prefix = ["K", "M", "G"]
        elif mode == "GiB":
            base = 1024
            prefix = ["Ki", "Mi", "Gi"]
        if size_B < base:
            return f"{size_B}", "B"
        elif size_B < base**2:
            return f"{size_B / base:.2f}", f"{prefix[0]}B"
        elif size_B < base**3:
            return f"{size_B / (base**2):.2f}", f"{prefix[1]}B"
        else:
            return f"{size_B / (base**3):.2f}", f"{prefix[2]}B"

    def _collect_tensors(self, module: Optional[nn.Module] = None):
        """Collect all tensor objects tracked by python

        NOTICE:
            - the buffers for backward which is implemented in C++ are
            not tracked by python's reference counting.
            - the gradients(.grad) of Parameters is not collected, and
            I don't know why.
        """
        gc.collect()
        if module is None:
            objects = gc.get_objects()
            n = len(self.tensors)  # Track number of tensors
            for obj in objects:
                if isinstance(obj, Tensor):
                    name = f"Tensor{n}"
                    self.tensors[name] = obj
                    self.device_map[obj.device].append(name)
                    n += 1
        else:
            for name, t in module.named_parameters():
                self.tensors[name] = t
                self.device_map[t.device].append(name)
        # print(f"{len(self.tensors)} tensor(s) collected")

    def _get_root_tensors(self, device, names):
        ptrs = {}
        for name, t in self.tensors.items():
            if device == t.device and name in names:
                # Get range of start and end pointers
                if not t.is_contiguous():
                    warn(f"Non-contiguous tensor {name} cannot be indexed efficiently")
                # (start, end)
                # ptrs[name] = (t.view(-1)[0].data_ptr(), t.view(-1)[-1].data_ptr())
                ptrs[name] = (
                    t.data_ptr(),
                    t.data_ptr() + t.nelement() * t.element_size() - 1,
                )
        # print(f"Collected {len(self.ptrs)} tensors")
        # Create a dependency graph of inclusion
        roots = []
        for name, this_ptrs in ptrs.items():
            counted = False
            new_roots = []
            for root in roots:
                root_ptrs = ptrs[root]
                if this_ptrs[0] >= root_ptrs[0] and this_ptrs[1] <= root_ptrs[1]:
                    # root supercedes this
                    new_roots.append(root)
                    counted = True
                elif this_ptrs[0] <= root_ptrs[0] and this_ptrs[1] >= root_ptrs[1]:
                    # this supercedes root
                    new_roots.append(name)
                    counted = True
                else:
                    # Other options that we didn't deal with...
                    # Independent
                    new_roots.append(root)
                    pass

            if not counted:
                new_roots.append(name)
            else:
                pass
            roots = new_roots
        return roots

    def report(self, module: Optional[nn.Module] = None):
        self._collect_tensors(module)
        for dev, names in self.device_map.items():
            total_size = 0
            roots = self._get_root_tensors(dev, names)
            print(f"Device {dev}")
            print("=" * 20)
            for name in names:
                if name in roots:
                    tensor = self.tensors[name]
                    shape = tuple(tensor.shape)
                    tsize = self._sizeof(tensor)
                    size, NB = self._format_size(tsize, self.format_mode)
                    print(f"{name}\t{shape}\t\t{size} {NB}")
                    total_size += tsize
            size, NB = self._format_size(total_size, self.format_mode)
            print(f"Total: {size} {NB}")
