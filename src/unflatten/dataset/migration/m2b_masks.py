"""M2b-masks — backfill the RFC003 §29 font-substitution mask projection.

M2b rows written before the mask existed carry a ``targets.target_bundle``
but no ``mask_projection``, so ``/style/font_family`` on every substituted
run is still implicitly supervised. This backfill closes that gap without
re-rendering anything: the mask is a pure function of the canonical AST
and the pinned ``FontPolicy``, so no compile, no LibreOffice and no PDF
are involved.

``renders.jsonl`` is NEVER rewritten (it is append-only, and a long m2b
run may hold it). Results land in a NEW ledger ``mask-projections.jsonl``,
one line per bundle::

    {"document_bundle_id", "mask_projection", "masked_font_runs", "ast_hash"}

joined on ``document_bundle_id`` the same way m2c joins the other ledgers.

Resume: bundle ids already present in ``mask-projections.jsonl`` are
skipped. A derivation failure appends nothing (it stays pending and is
retried on the next run) and is reported as a typed code in the summary.
Parallelism: workers only read the store and compute; the parent process
is the single appender, guarded by a pid lockfile.
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

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.ast.canonicalizer import CanonicalizationError, canonicalize
from unflatten.dataset.gates.store import StoreError, read_object
from unflatten.dataset.migration.m2b import M2bError, _acquire_lock, _store_ref
from unflatten.dataset.targets.masks import (
    build_mask_projection,
    projection_is_stale,
    validate_mask_projection,
)
from unflatten.renderer.api import RendererOptions
from unflatten.renderer.manifest import compute_font_pack_hash

BACKFILL_VERSION = "0.1.0"
LEDGER_NAME = "mask-projections.jsonl"

#: The row was rendered under a font policy other than the one this
#: backfill derives with — its mask would be wrong in both directions.
POLICY_MISMATCH_CODE = "MASK_POLICY_MISMATCH"
#: The row does not record `render_profile.font_pack_hash`, so the policy
#: it was rendered under cannot be proven. Unprovable is not "fine".
POLICY_UNKNOWN_CODE = "MASK_POLICY_UNKNOWN"


def _backfill_options() -> RendererOptions:
    """The renderer configuration the backfill derives masks against —
    the same one m2b stamps into `render_profile.font_pack_hash`."""
    return RendererOptions(strict=True, emit_manifest=True)


def _policy_failure(row: dict[str, Any], font_policy: Any) -> str | None:
    """Typed code when `row` was not rendered under `font_policy`.

    The inequality itself is `targets.masks.projection_is_stale` — the
    derivation-side predicate — rather than a second copy of it here.
    Two copies of one comparison drift; this module only decides what an
    unprovable pack means (UNKNOWN, never "fine").
    """
    profile = row.get("render_profile")
    recorded = profile.get("font_pack_hash") if isinstance(profile, dict) else None
    if not recorded:
        return POLICY_UNKNOWN_CODE
    if projection_is_stale(profile, font_policy):
        return POLICY_MISMATCH_CODE
    return None


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


def _needs_mask(row: dict[str, Any], *, force: bool = False) -> bool:
    """Whether `row` should be (re-)derived.

    `force` is the DEC-042 §1 / DEC-043 §5 re-derivation path: on
    `m2a-v10` every row already carries `targets.mask_projection`
    inline, so the default (`force=False`) rule below would never touch
    a single one of them, and the fixed FONT_SUBSTITUTED semantics would
    never reach the store. `force` says "derive regardless of what the
    row already carries" — it is a caller-named override, not a default.
    """
    targets = row.get("targets")
    if not isinstance(targets, dict) or not targets.get("target_bundle"):
        return False
    if force:
        return True
    return not targets.get("mask_projection")


def _backfill_one(args: tuple[Path, str, dict[str, Any]]) -> dict[str, Any]:
    """Worker: derive + store one projection. Returns a ledger row, or a
    ``{"error": code}`` marker the parent counts and does not append."""
    store_dir, bundle_id, envelope = args
    try:
        stored_ast = json.loads(read_object(store_dir, envelope["ast"]))
    except StoreError as exc:
        return {"document_bundle_id": bundle_id, "error": exc.code}
    except (KeyError, TypeError, json.JSONDecodeError):
        return {"document_bundle_id": bundle_id, "error": "LOAD_RECORD_MALFORMED"}
    try:
        canonical = canonicalize(stored_ast)
    except CanonicalizationError:
        return {"document_bundle_id": bundle_id, "error": "CANONICALIZATION_FAILED"}
    options = _backfill_options()
    try:
        projection = build_mask_projection(
            canonical, options.font_policy, strict=options.strict
        )
        issues = validate_mask_projection(projection)
    except (KeyError, TypeError, ValueError):
        return {"document_bundle_id": bundle_id, "error": "MASK_PROJECTION_FAILED"}
    if issues:
        return {"document_bundle_id": bundle_id, "error": sorted({i.code for i in issues})[0]}
    ref = _store_ref(store_dir, to_canonical_json(projection), "application/json")
    return {
        "document_bundle_id": bundle_id,
        "mask_projection": ref,
        "masked_font_runs": projection["counts"]["affected_runs"],
        "ast_hash": projection["ast_hash"],
    }


def _iter_results(
    tasks: list[tuple[Path, str, dict[str, Any]]], worker_count: int
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


def backfill_masks(
    store_dir: Path,
    *,
    limit: int | None = None,
    workers: int | None = None,
    allow_mixed_profile: bool = False,
    force: bool = False,
    progress_every: int = 200,
) -> dict[str, Any]:
    """Backfill mask projections for rendered rows that lack one.

    A row is only masked against the policy it was RENDERED under: each
    `renders.jsonl` row carries `render_profile.font_pack_hash`, and a
    row whose hash differs from this backfill's policy (or that records
    none at all) fails with a typed code instead of getting a mask that
    disagrees with its own pixels. `allow_mixed_profile` overrides that
    explicitly, mirroring m2b's `--allow-mixed-profile`.

    `force` re-derives regardless of what the row or this tool's own
    ledger already carry (DEC-042 §1 / DEC-043 §5): every row is a
    candidate even if `targets.mask_projection` is already set inline
    (`_needs_mask`), and a bundle already present in `mask-projections.
    jsonl` is derived again rather than skipped as `done` — the ledger
    is last-wins, so the new line simply supersedes the old one. `force`
    is an explicit override with no default of its own; omitting it
    reproduces exactly today's resume behaviour."""
    store_dir = Path(store_dir)
    index_path = store_dir / "index.jsonl"
    renders_path = store_dir / "renders.jsonl"
    if not index_path.is_file() or not renders_path.is_file():
        raise M2bError(f"m2b-masks: {store_dir} has no index.jsonl + renders.jsonl")

    ledger_path = store_dir / LEDGER_NAME
    done = set(_last_wins_rows(ledger_path))
    envelopes = _index_envelopes(index_path)
    rows = _last_wins_rows(renders_path)
    candidates = [
        bundle_id for bundle_id, row in rows.items() if _needs_mask(row, force=force)
    ]
    skipped_already_done = 0 if force else sum(1 for b in candidates if b in done)
    missing_envelope = [
        b for b in candidates if (force or b not in done) and b not in envelopes
    ]
    pending = [b for b in candidates if (force or b not in done) and b in envelopes]
    if limit is not None:
        pending = pending[:limit]

    font_policy = _backfill_options().font_policy
    expected_pack_hash = compute_font_pack_hash(font_policy)
    failure_codes: dict[str, int] = {}
    overridden_codes: dict[str, int] = {}
    if not allow_mixed_profile:
        eligible = []
        for bundle_id in pending:
            code = _policy_failure(rows[bundle_id], font_policy)
            if code is None:
                eligible.append(bundle_id)
            else:
                failure_codes[code] = failure_codes.get(code, 0) + 1
        pending = eligible
    else:
        # The override still records WHAT it overrode: without this the
        # summary cannot say which masks were derived against a policy
        # their pixels were not produced under.
        for bundle_id in pending:
            code = _policy_failure(rows[bundle_id], font_policy)
            if code is not None:
                overridden_codes[code] = overridden_codes.get(code, 0) + 1
        if overridden_codes:
            print(
                f"[m2b-masks] WARNING: --allow-mixed-profile derived masks for "
                f"{sum(overridden_codes.values())} row(s) whose render profile does "
                f"not match this policy: {overridden_codes}",
                file=sys.stderr,
                flush=True,
            )
    tasks = [(store_dir, bundle_id, envelopes[bundle_id]) for bundle_id in pending]

    lock_path = ledger_path.with_name(LEDGER_NAME + ".lock")
    _acquire_lock(lock_path)
    started = time.monotonic()
    written = 0
    masked_runs_total = 0
    documents_with_masked_runs = 0
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
                    masked_runs_total += result["masked_font_runs"]
                    documents_with_masked_runs += 1 if result["masked_font_runs"] else 0
                if progress_every and position % progress_every == 0:
                    print(
                        f"[m2b-masks] {position}/{len(tasks)} "
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
        "font_pack_hash": expected_pack_hash,
        "candidates": len(candidates),
        "processed": len(tasks),
        "written": written,
        "failed": sum(failure_codes.values()),
        "skipped_already_done": skipped_already_done,
        "missing_index_envelope": len(missing_envelope),
        "failure_codes": dict(sorted(failure_codes.items(), key=lambda kv: -kv[1])),
        "allow_mixed_profile": allow_mixed_profile,
        "force": force,
        # Rows whose render profile did NOT match this policy but were
        # masked anyway because of the override — empty unless
        # `allow_mixed_profile` is set.
        "overridden_codes": dict(sorted(overridden_codes.items(), key=lambda kv: -kv[1])),
        "masked_font_runs_total": masked_runs_total,
        "documents_with_masked_runs": documents_with_masked_runs,
        "seconds_total": round(elapsed, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.dataset.migration.m2b_masks", description=__doc__
    )
    parser.add_argument("store_dir", type=Path, help="M2a/M2b store root")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--allow-mixed-profile",
        action="store_true",
        help="mask rows whose render_profile font pack differs from (or is missing "
        "for) the backfill policy — the mask may disagree with their pixels",
    )
    parser.add_argument(
        "--force-rederive-mask",
        action="store_true",
        help="re-derive masks for every eligible row, including ones that already "
        "carry a mask_projection (inline or in this tool's own ledger) — for the "
        "DEC-042 §1 semantics fix; NOT the default",
    )
    args = parser.parse_args()
    summary = backfill_masks(
        args.store_dir,
        limit=args.limit,
        workers=args.workers,
        allow_mixed_profile=args.allow_mixed_profile,
        force=args.force_rederive_mask,
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
