"""Derive RFC003 §29 mask projections for factory bundles (no re-render).

Companion to :mod:`unflatten.dataset.migration.factory_index`: factory
bundles never went through M2b (no LibreOffice on the factory path is
needed — the mask is a pure function of the canonical AST and the
pinned FontPolicy, same derivation M2b applies in
``_build_mask_projection``). This tool computes that projection for
store-indexed factory bundles and stages the resulting lane for the
release rebuild.

Policy binding (fail-closed): the projection is only meaningful
against the policy the pixels were produced under. The tool reads the
factory manifest's ``render_profile.font_pack_hash`` and REFUSES the
whole run when it differs from the FontPolicy used for the
derivation — the exact ``MASK_POLICY_MISMATCH`` semantics of
``m2b_masks``, never a silent cross-policy mask.

Per bundle (a failure is a typed code in the report, never a crash):
1. read the store index row (must exist — run factory_index first)
   and load ``envelope.ast`` from the store objects;
2. ``ast_hash(canonicalize(stored))`` must equal ``envelope.ast_hash``;
3. ``build_mask_projection(canonical, font_policy, strict=True)`` —
   same strictness the factory compiled with;
4. ``validate_mask_projection`` must report no issues;
5. store the canonical JSON blob under the store ``objects/``
   (content-addressed; new hashes only, nothing rewritten).

``--augment-factory`` copies a factory output dir and adds the
computed ``mask_projection`` lane to each covered
``run_manifest.documents[].artifacts`` entry, plus a
``mask_derivation_report.json`` recording tool version, font pack
hash and per-bundle refs/codes. The source factory dir is never
modified. M2c then picks the lane up through its unchanged synthetic
path (``artifacts_by_bundle`` from run_manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

TOOL_VERSION = "0.1.0"
FACTORY_ORIGIN = "synthetic_ast"


class FactoryMaskError(RuntimeError):
    """A refusal that must not degrade into an unbound mask."""


def _font_policy() -> Any:
    from unflatten.renderer.api import RendererOptions

    return RendererOptions(strict=True, emit_manifest=True).font_policy


def _pack_hash(font_policy: Any) -> str:
    from unflatten.renderer.manifest import compute_font_pack_hash

    return compute_font_pack_hash(font_policy)


def derive_bundle_mask(
    *, ast_bytes: bytes, font_policy: Any
) -> dict[str, Any]:
    """Pure derivation + validation; raises :class:`FactoryMaskError`."""
    from unflatten.ast.canonicalizer import canonicalize
    from unflatten.ast.hashing import ast_hash
    from unflatten.dataset.targets.masks import (
        build_mask_projection,
        validate_mask_projection,
    )

    try:
        stored = json.loads(ast_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactoryMaskError(f"AST not JSON: {exc}") from exc
    try:
        canonical = canonicalize(stored)
    except Exception as exc:  # noqa: BLE001 - any canonicalizer failure refuses
        raise FactoryMaskError(f"canonicalize refused: {exc}") from exc
    try:
        projection = build_mask_projection(
            canonical, font_policy, strict=True
        )
        issues = validate_mask_projection(projection)
    except (KeyError, TypeError, ValueError) as exc:
        raise FactoryMaskError(f"mask derivation failed: {exc}") from exc
    if issues:
        codes = sorted({i.code for i in issues})
        raise FactoryMaskError(f"mask validation issues: {codes}")
    return {
        "canonical_ast_hash": ast_hash(canonical),
        "projection": projection,
    }


def _store_blob(store_objects: Path, data: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(data).hexdigest()
    path = store_objects / digest
    if not path.is_file():
        tmp = path.with_name(f"{digest}.tmp.{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return {
        "mime_type": "application/json",
        "ref": f"objects/{digest}",
        "sha256": f"sha256:{digest}",
        "size_bytes": len(data),
    }


def _read_store_blob(store_objects: Path, ref: str) -> bytes:
    name = (ref or "").split("/")[-1]
    path = store_objects / name
    if not name or not path.is_file():
        raise FactoryMaskError(f"store blob missing: {ref!r}")
    return path.read_bytes()


def run_factory_masks(
    *,
    store_dir: Path,
    bundle_ids: set[str],
    factory_manifest_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    from unflatten.ast.canonical_json import to_canonical_json

    store_dir = Path(store_dir)
    store_objects = store_dir / "objects"
    font_policy = _font_policy()
    pack_hash = _pack_hash(font_policy)

    if factory_manifest_path is not None:
        manifest = json.loads(Path(factory_manifest_path).read_text())
        recorded = (manifest.get("render_profile") or {}).get("font_pack_hash")
        if recorded is not None and recorded != pack_hash:
            raise FactoryMaskError(
                f"factory pixels rendered under {recorded!r} != current "
                f"policy pack {pack_hash!r} -- refusing cross-policy masks"
            )

    index: dict[str, dict[str, Any]] = {}
    for line in (store_dir / "index.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        bid = (record.get("envelope") or {}).get("document_bundle_id")
        if bid:
            index[bid] = record

    masks: dict[str, dict[str, Any]] = {}
    codes: dict[str, str] = {}
    for bundle_id in sorted(bundle_ids):
        record = index.get(bundle_id)
        if record is None:
            codes[bundle_id] = "NO_STORE_RECORD"
            continue
        envelope = record.get("envelope") or {}
        if envelope.get("source_origin") != FACTORY_ORIGIN:
            codes[bundle_id] = "NOT_FACTORY_ORIGIN"
            continue
        try:
            ast_bytes = _read_store_blob(
                store_objects, (envelope.get("ast") or {}).get("ref", "")
            )
            derived = derive_bundle_mask(
                ast_bytes=ast_bytes, font_policy=font_policy
            )
            if derived["canonical_ast_hash"] != envelope.get("ast_hash"):
                raise FactoryMaskError(
                    f"recomputed {derived['canonical_ast_hash']!r} != "
                    f"envelope {envelope.get('ast_hash')!r}"
                )
            data = to_canonical_json(derived["projection"])
            ref = (
                _store_blob(store_objects, data) if not dry_run else {
                    "mime_type": "application/json",
                    "ref": (
                        "objects/"
                        + hashlib.sha256(data).hexdigest()
                    ),
                    "sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
                    "size_bytes": len(data),
                }
            )
            masks[bundle_id] = ref
        except FactoryMaskError as exc:
            codes[bundle_id] = str(exc)

    return {
        "tool": "unflatten.dataset.migration.factory_masks",
        "tool_version": TOOL_VERSION,
        "font_pack_hash": pack_hash,
        "masks": masks,
        "failures": codes,
        "dry_run": dry_run,
    }


def augment_factory(
    *,
    factory_dir: Path,
    out_dir: Path,
    masks: dict[str, dict[str, Any]],
    derivation: dict[str, Any],
) -> dict[str, Any]:
    """Copy a factory dir; add the mask lane to covered manifest entries."""
    factory_dir, out_dir = Path(factory_dir), Path(out_dir)
    if out_dir.exists():
        raise FactoryMaskError(f"out_dir exists: {out_dir} -- refusing overwrite")
    shutil.copytree(factory_dir, out_dir)
    manifest_path = out_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    covered = 0
    for doc in manifest.get("documents", []):
        bid = doc.get("document_bundle_id")
        if bid in masks:
            artifacts = doc.setdefault("artifacts", {})
            if "mask_projection" in artifacts:
                raise FactoryMaskError(
                    f"{bid}: run_manifest already has a mask lane"
                )
            artifacts["mask_projection"] = masks[bid]
            covered += 1
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    (out_dir / "mask_derivation_report.json").write_text(
        json.dumps(
            {
                **derivation,
                "masks": masks,
                "augmented_documents": covered,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    return {"augmented_dir": str(out_dir), "augmented_documents": covered}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.dataset.migration.factory_masks",
        description="Derive mask projections for factory bundles (no re-render).",
    )
    parser.add_argument("store_dir", type=Path)
    parser.add_argument(
        "bundle_ids", type=Path, help="file with one document_bundle_id per line"
    )
    parser.add_argument(
        "--factory-manifest",
        type=Path,
        default=None,
        help="factory run_manifest.json (policy cross-check)",
    )
    parser.add_argument(
        "--augment-factory",
        type=Path,
        default=None,
        nargs=2,
        metavar=("FACTORY_DIR", "OUT_DIR"),
        help="copy factory dir and add the mask lane to covered entries",
    )
    parser.add_argument("--out-report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    bids = {
        line.strip()
        for line in args.bundle_ids.read_text().splitlines()
        if line.strip()
    }
    report = run_factory_masks(
        store_dir=args.store_dir,
        bundle_ids=bids,
        factory_manifest_path=args.factory_manifest,
        dry_run=args.dry_run,
    )
    if args.augment_factory is not None and not args.dry_run:
        if report["failures"]:
            print(
                json.dumps(
                    {
                        "refused_augment": (
                            "failures present -- augment nothing"
                        ),
                        "failures": report["failures"],
                    },
                    indent=1,
                )
            )
            return 2
        info = augment_factory(
            factory_dir=args.augment_factory[0],
            out_dir=args.augment_factory[1],
            masks=report["masks"],
            derivation={
                "tool": report["tool"],
                "tool_version": report["tool_version"],
                "font_pack_hash": report["font_pack_hash"],
            },
        )
        report["augment"] = info
    data = json.dumps(report, indent=1, sort_keys=True)
    if args.out_report is not None:
        args.out_report.write_text(data + "\n")
    else:
        print(data)
    if report["failures"] and not args.dry_run:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
