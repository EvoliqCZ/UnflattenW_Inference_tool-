"""Restore resource blobs missing from an M2a object store.

Written after an ad-hoc garbage collection deleted 6,219 resource objects
from 17,199 bundles. The mistake was in the reachability walk: it followed
`fragment_index` transitively but treated `resources_manifest` as a leaf,
so every embedded image a document referenced *through* its manifest was
classified unreachable. The ASTs and manifests survived, which is what
makes repair possible at all — the manifest still records each blob's
path, size and SHA-256.

Repair is only possible because import is deterministic and the store is
content-addressed: re-importing the source DOCX reproduces byte-identical
resources, which land back at exactly the hashes the manifest names. The
tool verifies that rather than assuming it — a blob whose hash does not
match is a corruption, not a repair, and is refused.

The lesson worth keeping: a store walk that decides what to delete is not
a shell one-liner. If a garbage collector is ever wanted here, it belongs
next to this module, with the manifest indirection under test.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


class RepairError(RuntimeError):
    """The store cannot be repaired as given."""


def _read_object(store_dir: Path, ref: str) -> bytes | None:
    path = store_dir / "objects" / ref.split("/")[-1]
    return path.read_bytes() if path.is_file() else None


def find_missing_resources(store_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Bundles whose manifest names resource blobs that are not present.

    A bundle whose *manifest* is itself missing is skipped: that is a
    different failure, and guessing its resource list would invent data.
    """
    store_dir = Path(store_dir)
    index = store_dir / "index.jsonl"
    if not index.is_file():
        raise RepairError(f"no index.jsonl under {store_dir}")

    missing: dict[str, dict[str, Any]] = {}
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        envelope = record["envelope"]
        manifest_bytes = _read_object(
            store_dir, envelope["resources_manifest"]["ref"]
        )
        if manifest_bytes is None:
            continue
        gaps = [
            entry
            for entry in json.loads(manifest_bytes)
            if not (store_dir / "objects" / entry["ref"].split("/")[-1]).is_file()
        ]
        if gaps:
            missing[envelope["document_bundle_id"]] = {
                "source_sha256": record["source_sha256"],
                "missing": gaps,
            }
    return missing


def _reimport_resources(source_bytes: bytes) -> dict[str, bytes]:
    """Resources the importer extracts from `source_bytes`."""
    from unflatten.dataset.import_ import ImportOptions, import_docx

    result = import_docx(source_bytes, options=ImportOptions(allow_floating_drop=True))
    return dict(result.get_resources())


def _store_blob(store_dir: Path, blob: bytes) -> str:
    digest = hashlib.sha256(blob).hexdigest()
    path = store_dir / "objects" / digest
    tmp = path.with_name(f"{digest}.tmp.{os.getpid()}")
    tmp.write_bytes(blob)
    os.replace(tmp, path)
    return digest


def repair_store(
    store_dir: Path | str, raw_dir: Path | str, *, progress_every: int = 200
) -> dict[str, Any]:
    """Re-import sources to restore missing resource blobs."""
    store_dir, raw_dir = Path(store_dir), Path(raw_dir)
    missing = find_missing_resources(store_dir)
    restored = failed = 0
    codes: Counter[str] = Counter()

    for done, (_bundle_id, info) in enumerate(sorted(missing.items()), start=1):
        source = raw_dir / f"{info['source_sha256']}.docx"
        if not source.is_file():
            failed += len(info["missing"])
            codes["SOURCE_MISSING"] += len(info["missing"])
            continue
        try:
            resources = _reimport_resources(source.read_bytes())
        except Exception:  # noqa: BLE001 - any import failure is one typed code
            failed += len(info["missing"])
            codes["REIMPORT_FAILED"] += len(info["missing"])
            continue
        for entry in info["missing"]:
            blob = resources.get(entry["path"])
            if blob is None:
                failed += 1
                codes["RESOURCE_NOT_REPRODUCED"] += 1
                continue
            # The manifest is the authority on what the bytes must be.
            if hashlib.sha256(blob).hexdigest() != entry["sha256"].split(":")[-1]:
                failed += 1
                codes["RESOURCE_HASH_MISMATCH"] += 1
                continue
            _store_blob(store_dir, blob)
            restored += 1
        if progress_every and done % progress_every == 0:
            print(f"[repair] {done}/{len(missing)} bundles", flush=True)

    return {
        "documents": len(missing),
        "restored": restored,
        "failed": failed,
        "failure_codes": dict(codes),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m unflatten.dataset.migration.store_repair", description=__doc__
    )
    parser.add_argument("store_dir", type=Path)
    parser.add_argument("raw_dir", type=Path, help="source <sha256>.docx directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="report the gaps, restore nothing"
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        missing = find_missing_resources(args.store_dir)
        blobs = sum(len(v["missing"]) for v in missing.values())
        print(json.dumps({"documents": len(missing), "missing_blobs": blobs}, indent=1))
        return 0
    print(json.dumps(repair_store(args.store_dir, args.raw_dir), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
