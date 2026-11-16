"""Allocatable CUDA memory, including reclaimable RAM on integrated devices."""

from __future__ import annotations

from pathlib import Path


def _linux_available_memory() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            fields = line.split()
            if fields and fields[0] == "MemAvailable:":
                if len(fields) != 3 or fields[2] != "kB":
                    return None
                value = int(fields[1])
                return value * 1024 if value >= 0 else None
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def cuda_memory_info(device=None) -> tuple[int, int]:
    import torch

    free, total = torch.cuda.mem_get_info(device)
    properties = torch.cuda.get_device_properties(device)
    if getattr(properties, "is_integrated", False):
        # NVIDIA UMA guidance: CUDA's immediately-free count excludes
        # reclaimable page cache. MemAvailable includes it, without swap.
        # https://nvidia.custhelp.com/app/answers/detail/a_id/5728
        available = _linux_available_memory()
        if available is not None:
            free = min(available, total)
    return free, total
