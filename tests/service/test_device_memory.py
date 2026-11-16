from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unflatten.service import worker_cli

MIB = 2**20


@pytest.mark.parametrize(
    "integrated,available,expected",
    [
        (False, 45000, 2000),
        (True, 45000, 45000),
        (True, 200000, 120000),
    ],
)
def test_device_facts_use_reclaimable_ram_only_on_integrated_cuda(
    monkeypatch, integrated, available, expected
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (2000 * MIB, 120000 * MIB))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 1))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(is_integrated=int(integrated)),
    )
    read_text = Path.read_text

    def meminfo(path, *args, **kwargs):
        if path == Path("/proc/meminfo"):
            return f"MemAvailable: {available * 1024} kB\nSwapFree: 999999999 kB\n"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", meminfo)
    facts = worker_cli._measure_own_device_facts("cuda:0")
    assert facts["free_vram_mib"] == expected
    assert facts["total_vram_mib"] == 120000


@pytest.mark.parametrize(
    "payload,expected",
    [
        ("MemAvailable: 123 kB\nSwapFree: 999999999 kB", 123 * 1024),
        ("MemAvailable: 0 kB", 0),
        ("SwapFree: 999999999 kB", None),
        ("MemAvailable: -1 kB", None),
        ("MemAvailable: nope kB", None),
        ("MemAvailable: 300 MB", None),
    ],
)
def test_available_memory_parser_never_counts_swap_or_invalid_measurements(
    monkeypatch, payload, expected
):
    from unflatten.service.device_memory import _linux_available_memory

    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: payload)
    assert _linux_available_memory() == expected


def test_integrated_device_keeps_cuda_measurement_when_proc_is_unreadable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (2000 * MIB, 120000 * MIB))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 1))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(is_integrated=1)
    )

    def unavailable(*args, **kwargs):
        raise OSError("not available on this platform")

    monkeypatch.setattr(Path, "read_text", unavailable)
    assert worker_cli._measure_own_device_facts("cuda:0")["free_vram_mib"] == 2000
