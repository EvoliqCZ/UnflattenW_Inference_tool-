"""Registry of isolated LibreOffice profile directories.

Every headless call path must own a distinct profile namespace.  Keeping the
registry in one place makes adding a call path visible to the collision test.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path.home() / "unflatten-probe"
REGISTERED_CALL_PATHS = ("renderer", "m2b", "synthetic", "import_rasterize", "diagnostic_preview")


def profile_dir(call_path: str, *, process_id: int | None = None, slot: int = 0) -> Path:
    """Return the isolated profile path for a registered call path."""
    if call_path not in REGISTERED_CALL_PATHS:
        raise ValueError(f"unregistered LibreOffice call path: {call_path}")
    pid = os.getpid() if process_id is None else process_id
    if call_path == "renderer":
        return _ROOT / f"profile-{pid}"
    if call_path == "m2b":
        return _ROOT / "m2b-profiles" / f"profile-{slot}"
    if call_path == "synthetic":
        return _ROOT / "synthetic-profiles" / f"profile-{pid}"
    if call_path == "import_rasterize":
        return _ROOT / "import-rasterize-profiles" / f"profile-{pid}"
    # DEC-044 Dodatok D (C): the diagnostic service's own docx-preview render
    # (its own trusted output, never untrusted input) -- one more call path,
    # never sharing a profile with the other three.
    return _ROOT / "diagnostic-preview-profiles" / f"profile-{pid}"


def registered_profile_dirs(*, process_id: int, slot: int = 0) -> dict[str, Path]:
    """Materialize the registry for collision checks and diagnostics."""
    return {
        name: profile_dir(name, process_id=process_id, slot=slot)
        for name in REGISTERED_CALL_PATHS
    }
