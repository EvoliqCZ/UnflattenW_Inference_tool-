"""M2b-augment — the CPU side of DEC-040 image augmentation (T5a).

For every TRAIN-split page (`document_bundle_id`, `page_index`) that the
deterministic selection policy (`augment.selection.is_selected`) picks
(~20% of train pages — DEC-041 point 2): read the stored page PNG,
resolve one family's ordered distortion recipe
(`augment.selection.resolve_recipe`), apply it
(`augment.ops.apply_recipe`), and store the resulting PNG content-
addressed alongside a `augmented-views.jsonl` ledger row that names the
`RenderView` identity (`identity.render_view_id`) and the exact
transform matrix a build step needs to move region-target boxes by the
same warp (`augment.geometry.transform_bbox_ppm`).

Train-only (RFC003 split-before-variants, §46): the candidate page set
is restricted to the release's TRAIN members as `samples._split_membership`
reads them — `splits/train.jsonl` plus the `partial.jsonl` rows whose
`split_name` is `train` (`_load_train_bundle_ids`); a store may back several releases, so
which one is authoritative is the caller's `--release-dir`, never
inferred.

Same conventions as `m2b_masks.py`: `renders.jsonl` is only ever read;
results land in a NEW, append-only ledger; resume is last-wins on
`(document_bundle_id, page_index)`; a derivation failure appends nothing
and is retried on the next run; a single pid lockfile guards the
appender; `--workers` fan the CPU-only derivation out across a process
pool.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from unflatten.dataset.augment import ops, selection
from unflatten.dataset.augment import policy as policy_mod
from unflatten.dataset.gates.store import StoreError, read_object
from unflatten.dataset.identity import domain_hash, render_view_id
from unflatten.dataset.migration.m2b import M2bError, _acquire_lock, _store_ref
from unflatten.dataset.samples import _split_membership

AUGMENT_VERSION = "0.1.0"
LEDGER_NAME = "augmented-views.jsonl"

#: The distortion pipeline itself raised — a typed op failure, never a
#: crash for the whole run (same principle as `m2b`'s per-document
#: isolation).
OP_FAILED_CODE = "AUGMENT_OP_FAILED"

_INPUT_SET_DOMAIN = "rfc003.augmentation-input-set/v1"
_RENDER_PROFILE_DOMAIN = "rfc003.augmentation-render-profile/v1"
_OUTPUT_SET_DOMAIN = "rfc003.augmentation-output-set/v1"


def _last_wins_rows(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                rows[row[key]] = row
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
    return rows


def _last_wins_ledger(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                rows[(row["document_bundle_id"], row["page_index"])] = row
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
    return rows


def _load_train_bundle_ids(release_dir: Path) -> frozenset[str]:
    """Every `document_bundle_id` the release assigns to TRAIN — the
    RFC003 split-before-variants boundary this module enforces."""
    path = Path(release_dir) / "splits" / "train.jsonl"
    if not path.is_file():
        raise M2bError(f"m2b-augment: {path} does not exist — no train split to read")
    # The SAME membership the cache's cohort uses, not a second reading of the
    # split files: `partial.jsonl` carries partially eligible documents of
    # every split with their real split in `split_name`. Reading only
    # train.jsonl skipped 5 025 of release-v14-r5's 14 702 train documents.
    # Val/test partials stay out -- the split is decided before variants.
    return frozenset(
        bundle_id
        for bundle_id, (split, _lineage) in _split_membership(Path(release_dir)).items()
        if split == "train"
    )


def _candidate_pages(
    store_dir: Path,
    train_ids: frozenset[str],
    policy: dict[str, Any],
    policy_hash: str,
) -> list[tuple[str, int, dict[str, Any]]]:
    """`(document_bundle_id, page_index, page_ref)` for every rendered
    train page the selection policy picks."""
    rows = _last_wins_rows(store_dir / "renders.jsonl", "document_bundle_id")
    candidates: list[tuple[str, int, dict[str, Any]]] = []
    for bundle_id, row in sorted(rows.items()):
        if bundle_id not in train_ids or row.get("render_status") != "rendered":
            continue
        pages = (row.get("artifacts") or {}).get("pages") or []
        for page_index, page_ref in enumerate(pages):
            if selection.is_selected(bundle_id, page_index, policy_hash, policy):
                candidates.append((bundle_id, page_index, page_ref))
    return candidates


def _render_profile_hash(policy: dict[str, Any], policy_hash: str) -> str:
    return domain_hash(
        _RENDER_PROFILE_DOMAIN,
        {
            "tool": "m2b_augment",
            "augment_version": AUGMENT_VERSION,
            "policy_id": policy["policy_id"],
            "policy_schema_version": policy["policy_schema_version"],
            "policy_hash": policy_hash,
        },
    )


def _augment_one(
    args: tuple[Path, str, int, dict[str, Any], dict[str, Any], str],
) -> dict[str, Any]:
    """Worker: derive + store one augmented view. Returns a ledger row,
    or a ``{"document_bundle_id", "page_index", "error": code}`` marker
    the parent counts and does not append."""
    store_dir, bundle_id, page_index, page_ref, policy, policy_hash = args
    try:
        source_bytes = read_object(store_dir, page_ref)
    except StoreError as exc:
        return {"document_bundle_id": bundle_id, "page_index": page_index, "error": exc.code}

    family = selection.select_family(bundle_id, page_index, policy_hash)
    resolved_ops, seeds = selection.resolve_recipe(
        bundle_id, page_index, family, policy, policy_hash
    )
    try:
        image = Image.open(io.BytesIO(source_bytes))
        width_px, height_px = image.size
        out_bytes, matrix = ops.apply_recipe(source_bytes, resolved_ops, seeds)
    except ops.OpError:
        return {"document_bundle_id": bundle_id, "page_index": page_index, "error": OP_FAILED_CODE}

    transform_chain = selection.transform_chain_hash(
        bundle_id, page_index, family, resolved_ops, policy_hash
    )
    input_set_hash = domain_hash(_INPUT_SET_DOMAIN, {"page_sha256": page_ref["sha256"]})
    render_profile_hash = _render_profile_hash(policy, policy_hash)
    out_ref = _store_ref(store_dir, out_bytes, "image/png")
    output_set_hash = domain_hash(_OUTPUT_SET_DOMAIN, {"output_sha256": out_ref["sha256"]})

    view_id = render_view_id(
        document_bundle_id=bundle_id,
        candidate_id=None,
        hypothesis_hash=None,
        view_kind=family,
        input_artifact_set_hash=input_set_hash,
        render_profile_hash=render_profile_hash,
        transform_chain_hash=transform_chain,
        output_artifact_set_hash=output_set_hash,
    )
    return {
        "augment_version": AUGMENT_VERSION,
        "document_bundle_id": bundle_id,
        "page_index": page_index,
        "view_kind": family,
        "render_view_id": view_id,
        "transform_chain_hash": transform_chain,
        "policy_id": policy["policy_id"],
        "policy_hash": policy_hash,
        "image": out_ref,
        "source_page_sha256": page_ref["sha256"],
        "width_px": width_px,
        "height_px": height_px,
        "ops": resolved_ops,
        "transform_matrix": [list(row) for row in matrix],
    }


def _iter_results(
    tasks: list[tuple[Path, str, int, dict[str, Any], dict[str, Any], str]], worker_count: int
) -> Iterator[dict[str, Any]]:
    if worker_count <= 1:
        for task in tasks:
            yield _augment_one(task)
        return
    pool = ProcessPoolExecutor(max_workers=worker_count)
    try:
        yield from pool.map(_augment_one, tasks, chunksize=8)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def augment_views(
    store_dir: Path,
    release_dir: Path,
    *,
    policy_path: Path | None = None,
    limit: int | None = None,
    workers: int | None = None,
    force: bool = False,
    progress_every: int = 200,
) -> dict[str, Any]:
    """Derive augmented train-page views for the store, per `--release-dir`'s
    train split. `force` re-derives every eligible page even if the
    ledger already carries a row for it (appended fresh, last-wins)."""
    store_dir = Path(store_dir)
    index_path = store_dir / "index.jsonl"
    renders_path = store_dir / "renders.jsonl"
    if not index_path.is_file() or not renders_path.is_file():
        raise M2bError(f"m2b-augment: {store_dir} has no index.jsonl + renders.jsonl")

    policy = policy_mod.load_policy(policy_path)
    policy_hash = policy_mod.policy_hash(policy)
    train_ids = _load_train_bundle_ids(Path(release_dir))

    ledger_path = store_dir / LEDGER_NAME
    done = set(_last_wins_ledger(ledger_path))
    candidates = _candidate_pages(store_dir, train_ids, policy, policy_hash)
    skipped_already_done = 0 if force else sum(
        1 for c in candidates if (c[0], c[1]) in done
    )
    pending = candidates if force else [c for c in candidates if (c[0], c[1]) not in done]
    if limit is not None:
        pending = pending[:limit]

    tasks = [
        (store_dir, bundle_id, page_index, page_ref, policy, policy_hash)
        for bundle_id, page_index, page_ref in pending
    ]

    lock_path = ledger_path.with_name(LEDGER_NAME + ".lock")
    _acquire_lock(lock_path)
    started = time.monotonic()
    written = 0
    failure_codes: dict[str, int] = {}
    family_counts: dict[str, int] = {}
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
                    family_counts[result["view_kind"]] = (
                        family_counts.get(result["view_kind"], 0) + 1
                    )
                if progress_every and position % progress_every == 0:
                    print(
                        f"[m2b-augment] {position}/{len(tasks)} "
                        f"(written={written}, failed={sum(failure_codes.values())})",
                        flush=True,
                    )
    finally:
        lock_path.unlink(missing_ok=True)

    elapsed = time.monotonic() - started
    return {
        "augment_version": AUGMENT_VERSION,
        "store": str(store_dir),
        "ledger": str(ledger_path),
        "policy_id": policy["policy_id"],
        "policy_hash": policy_hash,
        "train_bundles": len(train_ids),
        "candidates": len(candidates),
        "processed": len(tasks),
        "written": written,
        "failed": sum(failure_codes.values()),
        "skipped_already_done": skipped_already_done,
        "failure_codes": dict(sorted(failure_codes.items(), key=lambda kv: -kv[1])),
        "family_counts": family_counts,
        "force": force,
        "seconds_total": round(elapsed, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.dataset.migration.m2b_augment", description=__doc__
    )
    parser.add_argument("store_dir", type=Path, help="M2a/M2b store root")
    parser.add_argument(
        "release_dir", type=Path, help="release directory carrying splits/train.jsonl"
    )
    parser.add_argument("--policy", type=Path, default=None, help="augmentation policy JSON")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-derive every eligible page even if the ledger already carries one for it",
    )
    args = parser.parse_args()
    summary = augment_views(
        args.store_dir,
        args.release_dir,
        policy_path=args.policy,
        limit=args.limit,
        workers=args.workers,
        force=args.force,
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
