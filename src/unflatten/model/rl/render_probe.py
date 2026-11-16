"""D1/D6/D8 -- LibreOffice render probe: STATUS ONLY, outside every hash.

The probe answers one question about a candidate DOCX ("does LibreOffice open it
and produce pages?") and records the answer in an UNHASHED sibling record keyed by
``candidate_evaluation_hash``.  It is deliberately not importable into the sealed
body of ``reward.py``: a transient LibreOffice timeout must never move a hash.
Raster bytes are counted and dropped here; they never reach a scorer
(``pixel_component`` stays ``off``).

Profile isolation (D8): the m2b render worker keys its profile on ``os.getpid()``,
so any small fixed slot range can equal a live pid.  The RL slot is
``pid_max + 1 + k`` -- never a pid, so never an m2b profile -- under the existing
``"m2b"`` call path (the call-path registry is not edited).

PROPOSED, dormant: nothing calls this until the composer wires it in.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from unflatten.dataset.migration import m2b
from unflatten.model.rl.policies import RENDER_PROBE_MODES

PID_MAX_PATH: Final = Path("/proc/sys/kernel/pid_max")
PID_MAX_FALLBACK: Final = 4_194_304
_INT32_LIMIT: Final = 2**31
_HASH_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_TAG_RE: Final = re.compile(r"^[A-Za-z0-9_.-]+$")

STATUS_RENDERED: Final = "rendered"
STATUS_FAILED: Final = "render_failed"
STATUS_MISSING: Final = "render_missing"
STATUS_OFF: Final = "probe_off"


class RenderSlotInvalid(ValueError):
    """The requested LibreOffice profile slot could collide with a pid or overflow."""


@dataclass(frozen=True)
class RenderProbeRecord:
    """Unhashed sibling of ``rl_candidate_evaluation``; carries status only."""

    candidate_evaluation_hash: str
    status: str
    libreoffice_version: str | None
    slot: int | None
    tag: str
    page_count: int | None


def read_pid_max(path: Path = PID_MAX_PATH) -> int:
    """Kernel pid ceiling; the fallback is used only when the file is unreadable."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return PID_MAX_FALLBACK


def rl_render_slot(k: int, *, pid_max: int | None = None) -> int:
    """``pid_max + 1 + k``: a profile slot that can never equal a process id."""
    ceiling = read_pid_max() if pid_max is None else pid_max
    if isinstance(ceiling, bool) or not isinstance(ceiling, int) or not 0 < ceiling < _INT32_LIMIT:
        raise RenderSlotInvalid(f"pid_max {ceiling!r} is not a positive int32")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise RenderSlotInvalid(f"slot index k must be an int >= 0, got {k!r}")
    return check_slot(ceiling + 1 + k, ceiling)


def check_slot(slot: int, ceiling: int) -> int:
    """The D8 invariant: ``slot > pid_max`` and never the live pid, int32-safe."""
    if slot >= _INT32_LIMIT:
        raise RenderSlotInvalid(f"slot {slot} overflows int32")
    pid = os.getpid()
    if pid > ceiling:
        raise RenderSlotInvalid(f"live pid {pid} exceeds pid_max {ceiling}: pid_max is wrong")
    # a real raise, not `assert`: it must survive `python -O`
    if slot <= ceiling or slot == pid:
        raise RenderSlotInvalid(f"slot {slot} is not above pid_max {ceiling}")
    return slot


def _record(
    eval_hash: str, status: str, tag: str, *, version=None, slot=None, pages=None
) -> RenderProbeRecord:
    return RenderProbeRecord(eval_hash, status, version, slot, tag, pages)


def probe_render(
    candidate_docx: bytes | Path,
    *,
    candidate_evaluation_hash: str,
    policy_render_probe: str,
    slot_k: int,
    tag: str,
    pid_max: int | None = None,
) -> RenderProbeRecord:
    """Open one candidate DOCX in LibreOffice and report status (never pixels)."""
    if policy_render_probe not in RENDER_PROBE_MODES:
        raise ValueError(f"render_probe {policy_render_probe!r} not in {RENDER_PROBE_MODES}")
    if not isinstance(candidate_evaluation_hash, str) or not _HASH_RE.match(
        candidate_evaluation_hash
    ):
        raise ValueError(f"malformed candidate_evaluation_hash {candidate_evaluation_hash!r}")
    if not isinstance(tag, str) or not _TAG_RE.match(tag):
        raise ValueError(f"tag {tag!r} must match {_TAG_RE.pattern}")
    if policy_render_probe == "off":
        return _record(candidate_evaluation_hash, STATUS_OFF, tag)
    slot = rl_render_slot(slot_k, pid_max=pid_max)
    if shutil.which("libreoffice") is None or shutil.which("pdftoppm") is None:
        return _record(candidate_evaluation_hash, STATUS_MISSING, tag, slot=slot)
    docx = candidate_docx.read_bytes() if isinstance(candidate_docx, Path) else candidate_docx
    pdf = m2b._libreoffice_pdf(docx, tag, profile_slot=slot)
    pages = m2b._pdf_pages_png(pdf, tag) if pdf is not None else []
    count = len(pages)
    del pages  # raster bytes are dropped here: status and a page count only
    version = m2b._libreoffice_version()
    status = STATUS_RENDERED if count > 0 else STATUS_FAILED
    return _record(
        candidate_evaluation_hash,
        status,
        tag,
        version=version,
        slot=slot,
        pages=count if count > 0 else None,
    )
