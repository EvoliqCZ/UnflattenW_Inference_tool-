"""M2 acceptance ledger (assembler design §8).

One append-only, fixed per-release ledger of frozen-`assembly_policy_hash`
runs against an acceptance split (`val_iid` / `val_template_ood`; `test` is
never accepted, design §8). Contract, ratified by
`docs/plans/spec-s4-assembler-m1-results.md` §9 item 6:

- ``ledger_path(release_dir)`` is fixed --
  ``release_dir.parent / "assembly-acceptance" / release_dir.name / "ledger.jsonl"``
  -- independent of whatever ``--out`` the eval command was given.
- One entry per ``(policy_hash, split)``. A second entry for the same pair
  is refused: re-tuning a policy after acceptance must mint a new policy
  hash, so the ledger shows both entries, never a silent overwrite.
- The file is read strictly: a line that is not a JSON object with exactly
  the required keys is a corrupted or foreign file, and the whole ledger
  is refused rather than the bad line being skipped (fail closed).
- This module records MEASUREMENTS only. Whether a measurement clears the
  RFC005 §12.2 G-A1 floors/margins is a separate, not-yet-frozen owner
  decision (design §10) -- no function here renders a PASS/FAIL verdict.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AcceptanceLedgerError(RuntimeError):
    """A corrupted/foreign ledger file, or a disallowed append."""


_REQUIRED_KEYS = frozenset(
    {
        "policy_hash",
        "release_root_hash",
        "split",
        "summary",
        "summary_hash",
        "report_sha256",
        "code_commit",
        "timestamp",
    }
)


@dataclass(frozen=True, slots=True)
class AcceptanceEntry:
    """One M2 acceptance measurement, ready to append.

    ``summary`` is the full `harness.compute_summary` output for the run
    (the measured per-class G-A1 table and hard zeros); ``summary_hash``
    and ``report_sha256`` let a reader verify it against the raw per-
    document ledger without re-running the harness. ``timestamp`` is
    passed in by the caller, never generated inside this dataclass.
    """

    policy_hash: str
    release_root_hash: str
    split: str
    summary: Mapping[str, Any]
    summary_hash: str
    report_sha256: str
    code_commit: str
    timestamp: str

    def to_json(self) -> dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "release_root_hash": self.release_root_hash,
            "split": self.split,
            "summary": self.summary,
            "summary_hash": self.summary_hash,
            "report_sha256": self.report_sha256,
            "code_commit": self.code_commit,
            "timestamp": self.timestamp,
        }


def ledger_path(release_dir: Path) -> Path:
    """The fixed per-release acceptance ledger path (design §8)."""
    return release_dir.parent / "assembly-acceptance" / release_dir.name / "ledger.jsonl"


def read_entries(path: Path) -> list[dict[str, Any]]:
    """Strict read: a missing file is an empty ledger; any other content
    must be one JSON object per line carrying exactly the required keys,
    or the file is refused as corrupted/foreign."""
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AcceptanceLedgerError(
                f"{path}:{lineno}: not valid JSON, ledger refused: {exc}"
            ) from exc
        if not isinstance(row, dict) or set(row) != _REQUIRED_KEYS:
            raise AcceptanceLedgerError(
                f"{path}:{lineno}: does not carry exactly the acceptance-ledger "
                f"keys {sorted(_REQUIRED_KEYS)}; corrupted or foreign file, refused"
            )
        entries.append(row)
    return entries


def has_entry(path: Path, policy_hash: str, split: str) -> bool:
    """Whether the ledger already holds an entry for this pair. Raises on
    a corrupted/foreign ledger, same as :func:`read_entries`."""
    return any(
        e["policy_hash"] == policy_hash and e["split"] == split for e in read_entries(path)
    )


def append_entry(path: Path, entry: AcceptanceEntry) -> None:
    """Append one entry. Refuses if the ledger is corrupted/foreign, or
    if an entry for the same ``(policy_hash, split)`` already exists."""
    if has_entry(path, entry.policy_hash, entry.split):
        raise AcceptanceLedgerError(
            f"{path}: an entry for policy_hash={entry.policy_hash!r} "
            f"split={entry.split!r} already exists; a re-tuned policy needs "
            "a new policy hash, never a silent overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(entry.to_json(), sort_keys=True) + "\n")
