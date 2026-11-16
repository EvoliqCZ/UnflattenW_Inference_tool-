"""Deterministic OPC/ZIP package writer (RFC002 "Determinism guarantees").

Fixes every axis of ZIP non-determinism the spec calls out: entry
order (caller-supplied, never reordered here), timestamp
(1980-01-01 00:00:00 — the DOS epoch, the earliest value `zipfile`
supports), compression method/level (DEFLATE, level 6), external
file attributes, and UTF-8 entry names.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_UNIX_REGULAR_FILE_PERMS = 0o100644
_DEFLATE_LEVEL = 6


@dataclass(frozen=True)
class PackagePart:
    """One OPC package part: its package-relative name and raw bytes."""

    name: str
    data: bytes


def build_package(parts: list[PackagePart]) -> bytes:
    """Write `parts` into a deterministic ZIP/OPC package, in given order.

    Raises `ValueError` on duplicate part names — the ZIP builder must
    never silently allow shadowing entries (RFC002 "Security").
    """
    seen: set[str] = set()
    for part in parts:
        if part.name in seen:
            raise ValueError(f"build_package: duplicate part name {part.name!r}")
        seen.add(part.name)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for part in parts:
            info = zipfile.ZipInfo(part.name, date_time=_FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # unix, so external_attr unix-perms bits apply
            info.external_attr = _UNIX_REGULAR_FILE_PERMS << 16
            zf.writestr(
                info, part.data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=_DEFLATE_LEVEL
            )
    return buffer.getvalue()
