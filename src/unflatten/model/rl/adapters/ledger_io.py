"""Atomic JSON persistence of the S4.5 exposure ledger (torch-free at import).

``write_ledger_json`` verifies the ledger's own hash chain before a single byte
is written and replaces the target atomically (tmp file in the same directory,
``fsync``, ``os.replace``): a crash anywhere leaves the previous ledger intact.
``open_ledger_json`` re-derives the whole ledger from the file through
``RlExposureLedger.from_document`` and refuses a corrupt, foreign-run or
foreign-policy file; a missing file yields a fresh ledger only when the caller
says this is a fresh run (``fresh_run=True``) -- never by default, so a lost
ledger on a resume is a refusal, not a silent restart with no exposure history.

``unflatten.model.rl.exposure`` pulls ``torch`` in through the rollout module,
so it is imported lazily inside :func:`open_ledger_json` only.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from unflatten.ast.canonical_json import to_canonical_json

if TYPE_CHECKING:
    from unflatten.model.rl.exposure import RlExposureLedger


class LedgerFileRefused(ValueError):
    """The ledger file is missing (on a resume), unreadable, or not this run's."""


def write_ledger_json(ledger: Any, path: Path) -> None:
    """Atomically persist ``ledger`` as canonical JSON at ``path``.

    ``ledger.verify()`` runs first (an unverifiable ledger is never written).
    """
    ledger.verify()
    payload = to_canonical_json(ledger.to_document())
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(target.parent)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def open_ledger_json(
    path: Path,
    *,
    run_id: str,
    dataset_manifest: Mapping[str, Any],
    reward_policy_hash: str,
    fresh_run: bool,
) -> RlExposureLedger:
    """The ledger stored at ``path``, or a fresh one iff ``fresh_run``."""
    from unflatten.model.rl.exposure import RlExposureLedger

    target = Path(path)
    if not target.exists():
        if not fresh_run:
            raise LedgerFileRefused(
                f"ledger file {target} is missing on a run that is not fresh -- refused "
                "rather than restarting with no exposure history"
            )
        return RlExposureLedger(
            run_id, dataset_manifest=dataset_manifest, reward_policy_hash=reward_policy_hash
        )
    if fresh_run:
        raise LedgerFileRefused(
            f"ledger file {target} already exists on a fresh run -- refused rather than "
            "overwriting a run's exposure history"
        )
    try:
        doc = json.loads(target.read_bytes())
    except (OSError, ValueError) as exc:
        raise LedgerFileRefused(f"ledger file {target} is unreadable: {exc}") from exc
    if not isinstance(doc, dict):
        raise LedgerFileRefused(f"ledger file {target} is not a JSON object")
    if doc.get("run_id") != run_id:
        raise LedgerFileRefused(
            f"ledger file {target} belongs to run {doc.get('run_id')!r}, not {run_id!r}"
        )
    if doc.get("reward_policy_hash") != reward_policy_hash:
        raise LedgerFileRefused(
            f"ledger file {target} was written under reward policy "
            f"{doc.get('reward_policy_hash')!r}, not {reward_policy_hash!r}"
        )
    return RlExposureLedger.from_document(doc, dataset_manifest=dataset_manifest)


def make_ledger_ports(
    path: Path, *, fresh_run: bool
) -> tuple[
    Callable[[str, Mapping[str, Any], str], RlExposureLedger],
    Callable[[RlExposureLedger], None],
]:
    """``(open_ledger, write_ledger)`` bound to one file, as ``RealRunPorts`` wants."""

    def open_ledger(
        run_id: str, dataset_manifest: Mapping[str, Any], reward_policy_hash: str
    ) -> RlExposureLedger:
        return open_ledger_json(
            path,
            run_id=run_id,
            dataset_manifest=dataset_manifest,
            reward_policy_hash=reward_policy_hash,
            fresh_run=fresh_run,
        )

    def write_ledger(ledger: RlExposureLedger) -> None:
        write_ledger_json(ledger, path)

    return open_ledger, write_ledger
