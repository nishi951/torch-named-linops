#!/usr/bin/env python
"""Print comprehensive CUDA/PyTorch system configuration.

Run with: uv run torchlinops-diag
"""

import sys
import subprocess
import torch
import numpy as np


def _run(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def main():
    print("=" * 60)
    print("TORCH-NAMED-LINOPS SYSTEM CONFIGURATION")
    print("=" * 60)

    # Python
    print(f"\n--- Python ---")
    print(f"Version: {sys.version}")
    print(f"Executable: {sys.executable}")

    # PyTorch
    print(f"\n--- PyTorch ---")
    print(f"Version: {torch.__version__}")
    print(f"Build type: {'Debug' if torch.version.debug else 'Release'}")
    print(f"Git version: {torch.version.git_version}")

    # CUDA
    print(f"\n--- CUDA ---")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (compiled): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

    # NCCL
    print(f"\n--- NCCL ---")
    try:
        from torch.distributed import is_nccl_available

        print(f"NCCL available: {is_nccl_available()}")
    except ImportError:
        print("NCCL available: Unknown (torch.distributed not importable)")
    try:
        nccl_ver = torch.cuda.nccl.version()
        if isinstance(nccl_ver, tuple):
            print(f"NCCL version: {'.'.join(str(x) for x in nccl_ver)}")
        else:
            major = nccl_ver // 1000
            minor = (nccl_ver % 1000) // 10
            patch = nccl_ver % 10
            print(f"NCCL version: {major}.{minor}.{patch}")
    except AttributeError:
        print("NCCL version: Not accessible via torch.cuda.nccl")

    # NVIDIA Driver
    print(f"\n--- NVIDIA Driver ---")
    if torch.cuda.is_available():
        print(f"CUDA runtime version: {torch.version.cuda}")
        driver = _run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        if driver:
            print(f"Driver version (nvidia-smi): {driver.split()[0]}")
        else:
            print("Driver version: Could not query nvidia-smi")

    # GPU details
    print(f"\n--- GPU Devices ---")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Multi-processor count: {props.multi_processor_count}")
            print(f"  Memory clock rate: {props.memory_clock_rate / 1e6:.1f} GHz")
            print(f"  Core clock rate: {props.clock_rate / 1e6:.1f} GHz")
            print(f"  Memory bus width: {props.memory_bus_width} bits")
            print(f"  L2 cache size: {props.L2_cache_size / 1e6:.1f} MB")
            for attr in [
                "concurrent_kernels",
                "ECC_enabled",
                "unified_addressing",
                "managed_memory",
                "is_multi_gpu_board",
            ]:
                val = getattr(props, attr, None)
                if val is not None:
                    print(f"  {attr}: {val}")

        # P2P access
        if torch.cuda.device_count() >= 2:
            print(f"\n--- Peer-to-Peer Access ---")
            for i in range(torch.cuda.device_count()):
                for j in range(torch.cuda.device_count()):
                    if i != j:
                        can = torch.cuda.can_device_access_peer(i, j)
                        print(f"  GPU {i} -> GPU {j}: {'Yes' if can else 'No'}")

    # Key dependencies
    print(f"\n--- Key Dependencies ---")
    deps = {"numpy": np.__version__}
    for name, ver in deps.items():
        print(f"{name}: {ver}")

    try:
        import importlib.metadata

        for pkg in ["einops", "scipy", "numba"]:
            try:
                print(f"{pkg}: {importlib.metadata.version(pkg)}")
            except Exception:
                pass
    except Exception:
        pass

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
