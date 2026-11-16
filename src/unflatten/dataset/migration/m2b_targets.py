"""M2b-targets — backfill target bundles under today's derivation code
(DEC-043 §1 hyperlink flattening).

Rows in ``renders.jsonl`` written before commit 2e51d48 carry a
``targets.target_bundle`` derived while ``hyperlink`` was still an
opening/closing structural node instead of a transparent container —
their stored bundle still emits ``HYPERLINK_OPEN``/``HYPERLINK_CLOSE``
events that today's ``derive_event_streams`` no longer produces. This
backfill closes that gap without re-rendering anything: the target
bundle (+ its ``single_page`` fragments) is a pure function of the
stored canonical AST and the stored Observation Sidecar, so no compile,
no LibreOffice and no PDF are involved — it calls the SAME
``m2b._build_targets`` the render pipeline itself uses, never a
reimplementation.

``renders.jsonl`` is NEVER rewritten (append-only, and a long m2b run
may hold it). Results land in a NEW ledger ``target-bundles.jsonl``, one
line per bundle::

    {"document_bundle_id", "target_bundle", "fragment_index",
     "fragment_count", "window_policy", "ast_hash", "derivation_hash"}

joined on ``document_bundle_id`` the same way ``migration.m2c`` joins
``mask-projections.jsonl`` (see ``m2c._read_mask_ledger`` /
``m2c._with_mask``).

Unlike the font-mask backfill, EVERY rendered bundle with an observation
sidecar is a candidate — there is no "already has one" exemption,
because every bundle stored before this backfill exists is stale by
construction (DEC-043 §1 is a semantics fix, not a coverage fix).
Resume: bundle ids already present in ``target-bundles.jsonl`` are
skipped. A derivation failure appends nothing (it stays pending and is
retried on the next run) and is reported as a typed code in the
summary. Parallelism: workers only read the store and compute; the
parent process is the single appender, guarded by a pid lockfile.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from unflatten.ast.canonicalizer import CanonicalizationError, canonicalize
from unflatten.ast.hashing import ast_hash
from unflatten.dataset.gates.store import StoreError, read_object
from unflatten.dataset.migration.m2b import (
    M2bError,
    _acquire_lock,
    _build_targets,
    derivation_hash,
)

BACKFILL_VERSION = "0.1.0"
LEDGER_NAME = "target-bundles.jsonl"


def _last_wins_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Last line per ``document_bundle_id`` (m2b/m2c resume semantic)."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                rows[row["document_bundle_id"]] = row
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
    return rows


def _index_envelopes(path: Path) -> dict[str, dict[str, Any]]:
    envelopes: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                envelope = json.loads(line)["envelope"]
                envelopes[envelope["document_bundle_id"]] = envelope
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
    return envelopes


def _needs_target(row: dict[str, Any]) -> bool:
    """Whether ``row`` is a candidate for re-derivation.

    Every rendered row that carries an Observation Sidecar is eligible —
    unlike the font-mask backfill there is no "already carries one"
    exemption, because the fix this backfill delivers (DEC-043 §1) makes
    every bundle stored before it exists semantically stale, not merely
    absent."""
    if row.get("render_status") != "rendered":
        return False
    observation = row.get("observation")
    return isinstance(observation, dict) and bool(observation.get("sidecar"))


def _backfill_one(
    args: tuple[Path, str, dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Worker: derive + store one target bundle. Returns a ledger row, or
    a ``{"error": code}`` marker the parent counts and does not append."""
    store_dir, bundle_id, envelope, row = args
    try:
        stored_ast = json.loads(read_object(store_dir, envelope["ast"]))
        sidecar = json.loads(read_object(store_dir, row["observation"]["sidecar"]))
    except StoreError as exc:
        return {"document_bundle_id": bundle_id, "error": exc.code}
    except (KeyError, TypeError, json.JSONDecodeError):
        return {"document_bundle_id": bundle_id, "error": "LOAD_RECORD_MALFORMED"}
    try:
        canonical = canonicalize(stored_ast)
    except CanonicalizationError:
        return {"document_bundle_id": bundle_id, "error": "CANONICALIZATION_FAILED"}
    artifacts: dict[str, Any] = {}
    targets, codes = _build_targets(
        canonical,
        sidecar,
        store_dir,
        artifacts,
        document_bundle_id=bundle_id,
        ast_file_hash=envelope["ast"]["sha256"],
    )
    if targets is None:
        code = codes[0] if codes else "TARGET_DERIVATION_FAILED"
        return {"document_bundle_id": bundle_id, "error": code}
    return {
        "document_bundle_id": bundle_id,
        "target_bundle": targets["target_bundle"],
        "fragment_index": targets["fragment_index"],
        "fragment_count": targets["fragment_count"],
        "window_policy": targets["window_policy"],
        "ast_hash": ast_hash(canonical),
        # m2c's stale-ledger guard (mirroring `_refuse_stale_mask_ledger`)
        # pins THIS, not a schema version: `_build_targets` carries no
        # schema field of its own the way mask projections do, but the
        # source of `_DERIVATION_MODULES` is exactly what decided the
        # content below — the same signal m2b itself uses to know
        # whether a stored render is derived by current code.
        "derivation_hash": derivation_hash(),
    }


def _iter_results(
    tasks: list[tuple[Path, str, dict[str, Any], dict[str, Any]]], worker_count: int
) -> Iterator[dict[str, Any]]:
    if worker_count <= 1:
        for task in tasks:
            yield _backfill_one(task)
        return
    pool = ProcessPoolExecutor(max_workers=worker_count)
    try:
        yield from pool.map(_backfill_one, tasks, chunksize=8)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def backfill_targets(
    store_dir: Path,
    *,
    limit: int | None = None,
    workers: int | None = None,
    force: bool = False,
    progress_every: int = 200,
) -> dict[str, Any]:
    """Backfill target bundles + fragments for rendered rows under today's
    derivation code (DEC-043 §1).

    ``force`` re-derives regardless of whether the bundle already has a
    row in this tool's own ``target-bundles.jsonl`` ledger — a bundle
    already present is derived again rather than skipped as ``done``; the
    ledger is last-wins, so the new line simply supersedes the old one.
    Omitting it reproduces exactly today's resume behaviour."""
    store_dir = Path(store_dir)
    index_path = store_dir / "index.jsonl"
    renders_path = store_dir / "renders.jsonl"
    if not index_path.is_file() or not renders_path.is_file():
        raise M2bError(f"m2b-targets: {store_dir} has no index.jsonl + renders.jsonl")

    ledger_path = store_dir / LEDGER_NAME
    done = set(_last_wins_rows(ledger_path))
    envelopes = _index_envelopes(index_path)
    rows = _last_wins_rows(renders_path)
    candidates = [bundle_id for bundle_id, row in rows.items() if _needs_target(row)]
    skipped_already_done = 0 if force else sum(1 for b in candidates if b in done)
    missing_envelope = [
        b for b in candidates if (force or b not in done) and b not in envelopes
    ]
    pending = [b for b in candidates if (force or b not in done) and b in envelopes]
    if limit is not None:
        pending = pending[:limit]

    tasks = [(store_dir, bundle_id, envelopes[bundle_id], rows[bundle_id]) for bundle_id in pending]

    lock_path = ledger_path.with_name(LEDGER_NAME + ".lock")
    _acquire_lock(lock_path)
    started = time.monotonic()
    written = 0
    failure_codes: dict[str, int] = {}
    try:
        with open(ledger_path, "a", encoding="utf-8") as ledger:
            for position, result in enumerate(_iter_results(tasks, workers or 1), start=1):
                error = result.get("error")
                if error is not None:
                    failure_codes[error] = failure_codes.get(error, 0) + 1
                else:
                    ledger.write(json.dumps(result, sort_keys=True) + "\n")
                    ledger.flush()
                    written += 1
                if progress_every and position % progress_every == 0:
                    print(
                        f"[m2b-targets] {position}/{len(tasks)} "
                        f"(written={written}, failed={sum(failure_codes.values())})",
                        flush=True,
                    )
    finally:
        lock_path.unlink(missing_ok=True)

    elapsed = time.monotonic() - started
    return {
        "backfill_version": BACKFILL_VERSION,
        "store": str(store_dir),
        "ledger": str(ledger_path),
        "candidates": len(candidates),
        "processed": len(tasks),
        "written": written,
        "failed": sum(failure_codes.values()),
        "skipped_already_done": skipped_already_done,
        "missing_index_envelope": len(missing_envelope),
        "failure_codes": dict(sorted(failure_codes.items(), key=lambda kv: -kv[1])),
        "force": force,
        "seconds_total": round(elapsed, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.dataset.migration.m2b_targets", description=__doc__
    )
    parser.add_argument("store_dir", type=Path, help="M2a/M2b store root")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--force-rederive",
        action="store_true",
        help="re-derive target bundles for every eligible row, including ones "
        "already present in this tool's own ledger — NOT the default",
    )
    args = parser.parse_args()
    summary = backfill_targets(
        args.store_dir,
        limit=args.limit,
        workers=args.workers,
        force=args.force_rederive,
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
