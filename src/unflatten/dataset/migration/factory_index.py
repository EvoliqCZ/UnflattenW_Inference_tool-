"""Index synthetic factory bundles into an M2a store (append-only repair).

First-of-kind gap (found 2026-09-21, release-v14-synth S0): the M2c
synthetic path admits factory bundles into a release without them ever
entering the M2a store's ``index.jsonl``. S0 §9.2 rows 7/9 then fail
(``documents_without_a_store_record``), and downstream stages that join
on the store (M2b renders/masks, exact-text cache schema gate) skip
those bundles.

Re-importing the factory DOCX through M2a is NOT the repair: M2a
derives ``lineage_root_id``/``label_revision_hash``/``document_bundle_id``
from the import, so a re-import would mint DIFFERENT bundle ids and
break the release identity that already pins these bundles. This tool
carries the factory envelope VERBATIM (same bundle id, same ast_hash,
same lineage_root_id as the release split rows) and only adds the
store-ledger fields a factory record never had.

Row shape (new values are explicit and documented, never backfilled
to look like an import):
- ``envelope``: factory envelope verbatim (identity preserved).
- ``source_sha256``: hex sha256 of the factory-generated DOCX
  (``run_manifest.documents[].artifacts.docx``) — the generated DOCX
  is the source; verified present before append.
- ``import``: ``status: "generated"`` + generator facts (NOT an import
  summary; must never be read as one).
- ``lineage``: factory ``lineage_root_id`` verbatim (must equal the
  release split row — verified where a split dir is given);
  ``original_filename`` is null (factory docs have no filename),
  ``source`` is ``"synthetic"``.
- ``rights``: ABSENT. The gates engine auto-passes
  ``source_origin == "synthetic_ast"`` records with the recorded
  synthetic basis — byte-identical gate semantics to the M2c synthetic
  path that already admitted these bundles. Writing a novel explicit
  block would change gate details versus the release's existing rows.

Fail-closed per bundle (a refusal appends nothing, ever):
1. factory envelope ``source_origin`` must be ``"synthetic_ast"``
   (scope guard — real docs go through M2a).
2. bundle must NOT already be in the store index (never duplicate).
3. ``envelope.ast`` + ``envelope.resources_manifest`` + docx blobs must
   exist under the store ``objects/`` with matching ``size_bytes``.
4. AST identity: ``ast_hash(canonicalize(stored_ast))`` must equal
   ``envelope.ast_hash`` — a canonicalizer drift refuses loudly
   instead of indexing a different tree under the same id.
5. where ``--split-dir`` is given, factory ``lineage_root_id`` must
   equal the split row's (S0 row 9 compares exactly this).

``index.jsonl`` is append-only here (existing rows are never touched;
the release never pins its hash). A ``factory_index_report.json``
sidecar records what was appended, by which tool version, from which
factory manifest hash. ``--dry-run`` verifies everything and writes
nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

TOOL_VERSION = "0.1.0"
FACTORY_ORIGIN = "synthetic_ast"


class FactoryIndexError(RuntimeError):
    """A refusal that must not degrade into a partial append."""


def _blob(store_objects: Path, ref: str) -> Path | None:
    name = (ref or "").split("/")[-1]
    if not name:
        return None
    path = store_objects / name
    return path if path.is_file() else None


def _check_blob(store_objects: Path, ref_dict: Any, what: str) -> bytes:
    if not isinstance(ref_dict, dict):
        raise FactoryIndexError(f"{what}: not a ref object")
    path = _blob(store_objects, ref_dict.get("ref", ""))
    if path is None:
        raise FactoryIndexError(f"{what}: blob missing in store objects")
    data = path.read_bytes()
    expect_size = ref_dict.get("size_bytes")
    if expect_size is not None and len(data) != expect_size:
        raise FactoryIndexError(
            f"{what}: size {len(data)} != pinned {expect_size}"
        )
    expect_sha = (ref_dict.get("sha256") or "").split(":")[-1]
    if expect_sha and hashlib.sha256(data).hexdigest() != expect_sha:
        raise FactoryIndexError(f"{what}: sha256 mismatch")
    return data


def _verify_ast_identity(store_objects: Path, envelope: dict[str, Any]) -> None:
    from unflatten.ast.canonicalizer import canonicalize
    from unflatten.ast.hashing import ast_hash

    ast_bytes = _check_blob(store_objects, envelope.get("ast"), "envelope.ast")
    try:
        stored = json.loads(ast_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactoryIndexError(f"envelope.ast: not JSON: {exc}") from exc
    try:
        recomputed = ast_hash(canonicalize(stored))
    except Exception as exc:  # noqa: BLE001 - any canonicalizer failure refuses
        raise FactoryIndexError(f"AST canonicalize/hash refused: {exc}") from exc
    pinned = envelope.get("ast_hash")
    if recomputed != pinned:
        raise FactoryIndexError(
            f"ast_hash recomputed {recomputed!r} != envelope {pinned!r}"
        )


def build_store_row(
    *,
    envelope: dict[str, Any],
    docx_sha256: str,
    generator_version: str,
    recipe_id: str,
    root_instance_index: int,
) -> dict[str, Any]:
    """Assemble one store index row; envelope carried verbatim."""
    if envelope.get("source_origin") != FACTORY_ORIGIN:
        raise FactoryIndexError(
            f"source_origin {envelope.get('source_origin')!r} != {FACTORY_ORIGIN!r} "
            "-- real documents go through M2a, never this tool"
        )
    return {
        "envelope": envelope,
        "source_sha256": docx_sha256,
        "import": {
            "status": "generated",
            "generator": "unflatten.dataset.synthetic.factory",
            "generator_version": generator_version,
            "recipe_id": recipe_id,
            "root_instance_index": root_instance_index,
        },
        "lineage": {
            "lineage_root_id": envelope.get("lineage_root_id"),
            "original_filename": None,
            "source": "synthetic",
        },
    }


def _split_lineage(split_dir: Path | None, bundle_id: str) -> str | None:
    if split_dir is None:
        return None
    for name in ("train.jsonl", "val_iid.jsonl", "test.jsonl", "partial.jsonl"):
        path = split_dir / name
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if bundle_id in line:
                return json.loads(line).get("lineage_root_id")
    return None


def run_factory_index(
    *,
    factory_dir: Path,
    store_dir: Path,
    bundle_ids: set[str] | None = None,
    split_dir: Path | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    factory_dir, store_dir = Path(factory_dir), Path(store_dir)
    store_objects = store_dir / "objects"
    index_path = store_dir / "index.jsonl"
    if not index_path.is_file():
        raise FactoryIndexError(f"no index.jsonl under {store_dir}")

    manifest = json.loads((factory_dir / "run_manifest.json").read_text())
    manifest_docs = {
        d["document_bundle_id"]: d for d in manifest.get("documents", [])
    }
    factory_records: dict[str, dict[str, Any]] = {}
    for line in (factory_dir / "index.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        bid = (record.get("envelope") or {}).get("document_bundle_id")
        if bid:
            factory_records[bid] = record

    indexed: set[str] = set()
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            bid = (json.loads(line).get("envelope") or {}).get(
                "document_bundle_id"
            )
        except (json.JSONDecodeError, AttributeError):
            continue
        if bid:
            indexed.add(bid)

    appended: list[str] = []
    refused: dict[str, str] = {}
    skipped_done = 0
    rows: list[dict[str, Any]] = []
    for bundle_id in sorted(factory_records):
        if bundle_ids is not None and bundle_id not in bundle_ids:
            continue
        if bundle_id in indexed:
            skipped_done += 1
            continue
        if limit is not None and len(rows) >= limit:
            break
        record = factory_records[bundle_id]
        envelope = record.get("envelope") or {}
        try:
            if envelope.get("document_bundle_id") != bundle_id:
                raise FactoryIndexError("envelope id mismatch")
            if envelope.get("source_origin") != FACTORY_ORIGIN:
                raise FactoryIndexError(
                    f"source_origin {envelope.get('source_origin')!r} != "
                    f"{FACTORY_ORIGIN!r} -- real documents go through M2a, "
                    "never this tool"
                )
            manifest_doc = manifest_docs.get(bundle_id)
            if manifest_doc is None:
                raise FactoryIndexError("no run_manifest documents entry")
            docx_ref = (manifest_doc.get("artifacts") or {}).get("docx") or {}
            docx_sha = (docx_ref.get("sha256") or "").split(":")[-1]
            if not docx_sha:
                raise FactoryIndexError("no docx sha256 in manifest entry")
            _check_blob(store_objects, docx_ref, "factory docx")
            _verify_ast_identity(store_objects, envelope)
            _check_blob(
                store_objects,
                envelope.get("resources_manifest"),
                "envelope.resources_manifest",
            )
            if split_dir is not None:
                split_lin = _split_lineage(split_dir, bundle_id)
                if split_lin is not None and (
                    split_lin != envelope.get("lineage_root_id")
                ):
                    raise FactoryIndexError(
                        f"lineage {envelope.get('lineage_root_id')!r} != "
                        f"split {split_lin!r}"
                    )
            rows.append(
                build_store_row(
                    envelope=envelope,
                    docx_sha256=docx_sha,
                    generator_version=manifest.get("generator_version", ""),
                    recipe_id=manifest_doc.get("recipe_id", ""),
                    root_instance_index=manifest_doc.get(
                        "root_instance_index", -1
                    ),
                )
            )
            appended.append(bundle_id)
        except FactoryIndexError as exc:
            refused[bundle_id] = str(exc)

    report = {
        "tool": "unflatten.dataset.migration.factory_index",
        "tool_version": TOOL_VERSION,
        "factory_dir": str(factory_dir),
        "store_dir": str(store_dir),
        "appended": len(rows),
        "appended_bundle_ids": appended,
        "refused": refused,
        "skipped_already_indexed": skipped_done,
        "dry_run": dry_run,
    }
    if dry_run or not rows:
        return report

    lock_path = store_dir / "factory_index.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FactoryIndexError(
            f"lock exists: {lock_path} -- another indexer holds it"
        ) from exc
    try:
        with os.fdopen(fd, "w") as lock_f:
            lock_f.write(f"{os.getpid()}\n")
        from unflatten.ast.canonical_json import to_canonical_json

        with open(index_path, "a") as index_f:
            for row in rows:
                index_f.write(to_canonical_json(row).decode("utf-8") + "\n")
    finally:
        try:
            os.remove(lock_path)
        except OSError:
            pass
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.dataset.migration.factory_index",
        description="Index factory bundles into an M2a store (append-only).",
    )
    parser.add_argument("factory_dir", type=Path)
    parser.add_argument("store_dir", type=Path)
    parser.add_argument(
        "--bundle-ids",
        type=Path,
        default=None,
        help="file with one document_bundle_id per line (default: all missing)",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="release splits dir for the lineage cross-check",
    )
    parser.add_argument(
        "--out-report", type=Path, default=None, help="write the report JSON here"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="verify everything, append nothing"
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    bundle_ids = None
    if args.bundle_ids is not None:
        bundle_ids = {
            line.strip()
            for line in args.bundle_ids.read_text().splitlines()
            if line.strip()
        }
    report = run_factory_index(
        factory_dir=args.factory_dir,
        store_dir=args.store_dir,
        bundle_ids=bundle_ids,
        split_dir=args.split_dir,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    data = json.dumps(report, indent=1, sort_keys=True)
    if args.out_report is not None:
        args.out_report.write_text(data + "\n")
    else:
        print(data)
    if report["refused"] and not args.dry_run:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
