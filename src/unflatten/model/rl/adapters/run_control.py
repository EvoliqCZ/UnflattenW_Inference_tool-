"""Run control for the S4.5 loop: stop file and progress file (torch-free, no signals).

Mirrors ``s4_train``'s control surface: a launcher asks for a stop by creating a
file; the loop polls it between updates and checkpoints. Progress is one small
JSON object with a closed key set, written atomically. No clock is read here:
``written_at`` appears only when the caller injects a ``clock``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

PROGRESS_KEYS = (
    "step",
    "update_step",
    "pages_seen",
    "ledger_records",
    "rollouts",
    "gate_failed_count",
)


class ProgressRecordInvalid(ValueError):
    """The progress record has a foreign key, a missing key or a non-count value."""


def make_stop_requested(stop_file: Path) -> Callable[[], bool]:
    """True iff ``stop_file`` exists at the moment of the call."""
    target = Path(stop_file)

    def stop_requested() -> bool:
        return target.exists()

    return stop_requested


def write_progress(
    path: Path, record: Mapping[str, Any], *, clock: Callable[[], str] | None = None
) -> None:
    """Atomically write ``record`` (exactly :data:`PROGRESS_KEYS`) as JSON."""
    keys: set[str] = set(record)
    expected: set[str] = set(PROGRESS_KEYS)
    if keys != expected:
        raise ProgressRecordInvalid(
            f"progress keys must be exactly {PROGRESS_KEYS}; "
            f"missing {sorted(expected - keys)}, extra {sorted(keys - expected)}"
        )
    for key in PROGRESS_KEYS:
        value = record[key]
        if type(value) is not int or value < 0:
            raise ProgressRecordInvalid(f"progress {key!r} must be an int >= 0, got {value!r}")
    body: dict[str, Any] = {key: record[key] for key in PROGRESS_KEYS}
    if clock is not None:
        body["written_at"] = clock()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(body, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
