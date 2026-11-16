"""M2a — bulk import of the glm-docx registry into DocumentBundles.

Registry-driven (``documents.jsonl``): each record's raw DOCX (blob
store ``raw/<sha256>.docx``) is re-hashed, imported via
``unflatten.dataset.import_``, canonicalized, and stored content-
addressed with a bundle envelope, a lineage record and a rights record
(basis: DEC-001 option A — ``training_allowed`` for all sources, with
``source`` preserved in provenance so a later narrowing is a filter,
not a rebuild). Failures land in ``quarantine.jsonl`` with the full
typed issue list — never silently skipped (RFC003 §45).

Outputs under ``out_dir``: ``objects/<sha256>`` blobs, ``index.jsonl``
(one accepted record per line: envelope + lineage + rights + import
summary), ``quarantine.jsonl``, ``run-summary.json``.

Parallel and resumable: work is pure per-document; an existing
``index.jsonl``/``quarantine.jsonl`` pair is scanned on start and
already-processed source hashes are skipped.

One AST identity per bundle: the stored bytes, ``envelope.ast_hash``
and every downstream derivation (M2b observations/targets/fragments/
masks) are all the CANONICAL tree. ``import_docx`` canonicalizes only
as an output guard and returns the importer-id tree, so ``_process_one``
canonicalizes explicitly before storing. ``MIGRATION_VERSION`` 0.4.0
marked that store revision (m2a-v4) — it enters ``label_revision_hash``,
so every ``document_bundle_id`` differs from a 0.3.0 store. The bump from
0.3.0 records that table/cell borders and shading now resolve through the
style chain (including ``insideH``/``insideV``), ``nil`` borders no longer
import as visible ``single`` ink, ``text_frame`` nodes are produced, charts
and EMF/WMF are rasterized, and three new issue codes exist — labels that
change even where ``label_grade`` stays the same, which is exactly what
``label_revision_hash`` exists to surface.

The bump from 0.5.0 to 0.6.0 (DEC-058, the v16 re-import) records the
labels that changed since m2a-v5..v11: `w:tcW type="pct"` cell widths now
resolve against the table reference (RFC005-ERRATA-008 E2 / T2,
`IMPORT_CELL_WIDTH_PCT_RESOLVED`), and empty text frames are no longer
emitted (DEC-055 I2, `IMPORT_EMPTY_TEXT_FRAME_DROPPED`, switched on by this
very bump through `drop_empty_text_frames`). M2b's v16 aligner policy
rides the same bump (`m2b.default_aligner_policy`), so neither fix can reach
a store without moving every `document_bundle_id`.

The bump from 0.4.0 to 0.5.0 records two corrections that landed after
m2a-v4 was built, both of which change stored labels: chart/EMF
rasterization now embeds ``word/theme/theme1.xml`` in the minimal package
built around the chart part (``PACKAGE_BUILDER_VERSION`` 1 -> 2), because
without the theme LibreOffice resolved every ``a:schemeClr`` to the same
default and two of three chart series in a stacked bar rendered identical
black — the stored raster misrepresented the data, not merely omitted
ink; and rotated text frames are now dropped at import with
``IMPORT_TEXT_BOX_CONTENT_DROPPED`` (``_text_frame_is_rotated``, covering
DrawingML ``a:xfrm/@rot`` and VML ``rotation:``), since the ratified
``text_frame`` anchor (RFC001-ERRATA-004 E1) reuses ``floating_image``'s
anchor verbatim and that contract carries no rotation field, so a
rotated watermark could previously only have been emitted unrotated and
overlapping. The per-document rasterization budget also moves, 5 -> 50
(``_RASTERIZATION_BUDGET`` in ``dataset/import_/importer.py``), because
the old cap dropped 757 real graphics across 47 tail documents (up to
164 in one file) that the new cap recovers all but 208 of, at a
measured cost of 549 extra conversions across the whole corpus — so
those 47 documents' resources change too. A surgical re-import of only the affected documents would
leave a store with two importer versions both declaring
``migration_version`` 0.4.0, and ``label_revision_hash`` exists precisely
so two revisions cannot be indistinguishable — hence the bump covers the
whole store, not just the affected bundles.

``--profile`` selects the AST profile the import targets (DEC-003). It
defaults to ``word-core-v0``, so an existing invocation keeps producing
exactly what it produced before; the D-2b regeneration passes ``word-visual-v1.1``
EXPLICITLY. The flag is deliberately not implicit in the version bump:
the command that built a store should say which profile it asked for,
rather than leaving it to be inferred from the code of the day.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.ast.canonicalizer import canonicalize
from unflatten.ast.hashing import ast_file_hash, ast_hash
from unflatten.dataset import identity
from unflatten.dataset.import_ import ImportOptions, import_docx
from unflatten.dataset.import_.rasterize import RasterizerUnavailableError, rasterizer_pin

RIGHTS_EVIDENCE = "DEC-001-option-A"

#: The corpus privacy review is closed: DEC-002, by the owner, for an
#: internal project on private data that never leaves this machine. The store
#: therefore records ``approved`` directly. It used to hardcode ``pending``
#: and be patched afterwards by a script over ``index.jsonl`` — which the next
#: regeneration silently overwrote, leaving every record blocked at the
#: training admission boundary. A decision that lives only in patched data is
#: not recorded; it is waiting to be lost.
PRIVACY_REVIEW = "approved"
PRIVACY_REVIEW_EVIDENCE = "DEC-002"
# Resumed source numbering counters and instance start overrides change
# canonical labels. New imports must carry a distinct revision; existing
# stores remain immutable and are never restamped with this version.
MIGRATION_VERSION = "0.7.0"

#: Store revisions imported BEFORE DEC-055 I2 (empty text frames dropped at
#: import). I2 changes stored labels, so it may only reach a store under a
#: new `MIGRATION_VERSION` (hence a new `label_revision_hash`): binding it to
#: the bump means the fix can neither be forgotten at the next re-import nor
#: ride an unchanged revision. The bump itself is the owner's re-import step
#: (RFC005-ERRATA-008 §3, `test_migration_version_not_yet_bumped_for_errata_008_t2`).
_PRE_I2_MIGRATION_VERSIONS = frozenset({"0.1.0", "0.2.0", "0.3.0", "0.4.0", "0.5.0"})


def drop_empty_text_frames() -> bool:
    """`ImportOptions.drop_empty_text_frames` for this store revision."""
    return MIGRATION_VERSION not in _PRE_I2_MIGRATION_VERSIONS


class M2aError(RuntimeError):
    """A run-level refusal that must not degrade into per-document loss."""


def _store(out_dir: Path, data: bytes) -> tuple[str, str]:
    hex_digest = hashlib.sha256(data).hexdigest()
    path = out_dir / "objects" / hex_digest
    tmp = path.with_name(f"{hex_digest}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hex_digest, f"objects/{hex_digest}"


PROFILES = ("word-core-v0", "word-visual-v1", "word-visual-v1.1")


def _process_one(args: tuple[str, str, dict[str, Any], str]) -> dict[str, Any]:
    """Import a single registry record; pure function for pool workers.

    Returns a dict with either ``accepted`` (envelope pieces + bytes to
    store are recomputed in the parent to keep IPC small — here we
    return the canonical AST JSON + resources) or ``quarantined``.
    """
    raw_dir, registry_sha, registry_record, profile = args
    source_path = Path(raw_dir) / f"{registry_sha}.docx"
    result: dict[str, Any] = {
        "sha256": registry_sha,
        "source": registry_record.get("source"),
        "original_filename": registry_record.get("original_filename"),
    }
    try:
        data = source_path.read_bytes()
    except OSError as exc:
        result["quarantine"] = {"reason": "SOURCE_MISSING", "detail": str(exc)}
        return result
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != registry_sha:
        result["quarantine"] = {
            "reason": "SOURCE_HASH_MISMATCH",
            "detail": f"registry {registry_sha[:12]}… vs file {actual_sha[:12]}…",
        }
        return result

    imported = import_docx(
        data,
        options=ImportOptions(
            allow_floating_drop=True,
            profile=profile,
            drop_empty_text_frames=drop_empty_text_frames(),
        ),
    )
    issue_codes: dict[str, int] = {}
    for issue in imported.report:
        issue_codes[issue.code] = issue_codes.get(issue.code, 0) + 1
    result["import_status"] = imported.status
    result["issue_codes"] = issue_codes
    if imported.status == "failed" or imported.ast is None:
        error_codes = sorted(
            {i.code for i in imported.report if i.severity in ("error", "fatal")}
        )
        result["quarantine"] = {"reason": "IMPORT_FAILED", "codes": error_codes}
        return result

    # ONE identity per bundle: `import_docx` runs canonicalize() only as
    # an output guard and DISCARDS its result, so what it returns is the
    # importer-id tree. Canonicalizing here makes the tree that is
    # stored, the tree the envelope hashes, and the tree M2b derives
    # observations/targets/fragments/masks from the same tree.
    canonical = canonicalize(imported.ast)
    canonical_bytes = to_canonical_json(canonical)
    result["canonical_ast"] = canonical_bytes.decode("utf-8")
    result["ast_hash"] = ast_hash(canonical)
    result["ast_file_hash"] = ast_file_hash(canonical_bytes)
    result["resources"] = {
        path: data.hex() for path, data in imported.get_resources().items()
    }
    result["resource_mime_types"] = {
        r["path"]: r["mime_type"]
        for r in canonical.get("resources", [])
        if isinstance(r, dict) and "path" in r and "mime_type" in r
    }
    result["resource_provenance"] = imported.get_resource_provenance()
    return result


#: Fields the resource-manifest identity is computed over: WHAT the
#: resource is, never WHERE its bytes currently sit. Including the
#: object `ref` would make moving identical bytes a different document;
#: dropping `mime_type` would make a MIME change invisible. Same
#: projection the synthetic branch hashes (`synthetic/factory.py`).
MANIFEST_SEMANTIC_KEYS: tuple[str, ...] = (
    "path", "sha256", "size_bytes", "mime_type",
)


def semantic_manifest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The identity-bearing projection of stored manifest records."""
    return sorted(
        ({key: rec[key] for key in MANIFEST_SEMANTIC_KEYS} for rec in records),
        key=lambda rec: rec["path"],
    )


def _envelope(res: dict[str, Any], ast_ref: dict[str, Any],
               manifest_ref: dict[str, Any], manifest_records: list[dict[str, Any]],
               approximate: bool, raster_pin: dict[str, Any]) -> dict[str, Any]:
    lineage_root_id = identity.lineage_root_id_real(
        source_set_hash=f"sha256:{res['sha256']}"
    )
    resource_manifest_hash = identity.domain_hash(
        "rfc003.resource-manifest/v1", semantic_manifest(manifest_records)
    )
    label_revision_hash = identity.domain_hash(
        "rfc003.label-revision/v1",
        {
            "label_method": "docx_importer",
            "label_grade": "imported_normalized" if approximate else "imported_exact",
            "migration_version": MIGRATION_VERSION,
            "source_sha256": f"sha256:{res['sha256']}",
            "raster_pin_hash": raster_pin["raster_pin_hash"],
        },
    )
    bundle_id = identity.document_bundle_id(
        lineage_root_id=lineage_root_id,
        ast_hash=res["ast_hash"],
        ast_file_hash=res["ast_file_hash"],
        resource_manifest_hash=resource_manifest_hash,
        label_revision_hash=label_revision_hash,
    )
    return {
        "document_bundle_schema_version": "0.1.0",
        "document_bundle_id": bundle_id,
        "lineage_root_id": lineage_root_id,
        "ast_hash": res["ast_hash"],
        "ast_file_hash": res["ast_file_hash"],
        "resource_manifest_hash": resource_manifest_hash,
        "label_revision_hash": label_revision_hash,
        "source_origin": "native_docx",
        "label_method": "docx_importer",
        "label_grade": "imported_normalized" if approximate else "imported_exact",
        "processing_state": "canonicalized",
        "verification_tier": "machine_verified",
        "disposition": "active",
        "revision_id": "rev_000001",
        "supersedes_revision_id": None,
        "ast": ast_ref,
        "resources_manifest": manifest_ref,
    }


def run_m2a(registry_path: Path, raw_dir: Path, out_dir: Path,
            *, workers: int = 4, limit: int | None = None,
            retry_quarantined: bool = False,
            profile: str = "word-core-v0") -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(
            f"unknown AST profile {profile!r}; expected one of {', '.join(PROFILES)}"
        )
    try:
        raster_pin = rasterizer_pin()
    except RasterizerUnavailableError as exc:
        raise M2aError(f"graphic rasterizer unavailable: {exc}") from exc
    summary_path = out_dir / "run-summary.json"
    if summary_path.exists():
        try:
            recorded_pin = json.loads(summary_path.read_text()).get("rasterizer_pin")
        except (OSError, json.JSONDecodeError) as exc:
            raise M2aError(f"cannot read existing rasterizer pin: {exc}") from exc
        if recorded_pin is not None and recorded_pin != raster_pin:
            raise M2aError(
                "rasterizer pin mismatch: refusing to mix identity-bearing raster versions"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "objects").mkdir(exist_ok=True)
    index_path = out_dir / "index.jsonl"
    quarantine_path = out_dir / "quarantine.jsonl"

    done: set[str] = set()
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["source_sha256"])
            except (KeyError, json.JSONDecodeError):
                continue
    if quarantine_path.exists():
        if retry_quarantined:
            # Re-attempt everything previously quarantined (e.g. after an
            # importer widening); the old list is preserved as an audit
            # trail and a fresh quarantine.jsonl accumulates the residue.
            backup = quarantine_path.with_suffix(
                f".jsonl.retry-{len(list(out_dir.glob('quarantine.jsonl.retry-*')))}"
            )
            os.replace(quarantine_path, backup)
        else:
            for line in quarantine_path.read_text().splitlines():
                try:
                    done.add(json.loads(line)["source_sha256"])
                except (KeyError, json.JSONDecodeError):
                    continue

    records: list[tuple[str, str, dict[str, Any], str]] = []
    for line in registry_path.read_text().splitlines():
        rec = json.loads(line)
        sha = rec["sha256"]
        if sha in done or rec.get("extension") != "docx":
            continue
        records.append((str(raw_dir), sha, rec, profile))
    if limit is not None:
        records = records[:limit]

    accepted = quarantined = 0
    status_counts: dict[str, int] = {}
    with open(index_path, "a") as index_f, open(quarantine_path, "a") as quar_f, \
            ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_process_one, records, chunksize=8):
            if "quarantine" in res:
                quarantined += 1
                quar_f.write(json.dumps({
                    "source_sha256": res["sha256"],
                    "source": res["source"],
                    "original_filename": res["original_filename"],
                    "reason": res["quarantine"]["reason"],
                    "detail": res["quarantine"].get("detail"),
                    "codes": res["quarantine"].get("codes"),
                    "issue_codes": res.get("issue_codes"),
                    "rights": {
                        "training_allowed": True,
                        "evidence": RIGHTS_EVIDENCE,
                        "source": res["source"],
                    },
                }, sort_keys=True) + "\n")
                continue

            canonical_bytes = res["canonical_ast"].encode("utf-8")
            _hex, ast_obj_ref = _store(out_dir, canonical_bytes)
            ast_ref = {
                "ref": ast_obj_ref,
                "sha256": f"sha256:{_hex}",
                "size_bytes": len(canonical_bytes),
                "mime_type": "application/json",
            }
            manifest_records = []
            mime_types = res.get("resource_mime_types") or {}
            provenance = res.get("resource_provenance") or {}
            for rpath in sorted(res["resources"]):
                blob = bytes.fromhex(res["resources"][rpath])
                bhex, bref = _store(out_dir, blob)
                manifest_record = {
                    "path": rpath,
                    "sha256": f"sha256:{bhex}",
                    "size_bytes": len(blob),
                    "mime_type": mime_types.get(rpath, "application/octet-stream"),
                    "ref": bref,
                }
                if rpath in provenance:
                    manifest_record["derived_from"] = provenance[rpath]
                manifest_records.append(manifest_record)
            manifest_bytes = json.dumps(
                manifest_records, sort_keys=True, separators=(",", ":")
            ).encode()
            mhex, mref = _store(out_dir, manifest_bytes)
            manifest_ref = {
                "ref": mref,
                "sha256": f"sha256:{mhex}",
                "size_bytes": len(manifest_bytes),
                "mime_type": "application/json",
            }
            approximate = res["import_status"] == "success_with_warnings"
            envelope = _envelope(
                res, ast_ref, manifest_ref, manifest_records, approximate, raster_pin
            )
            accepted += 1
            status_counts[res["import_status"]] = (
                status_counts.get(res["import_status"], 0) + 1
            )
            index_f.write(json.dumps({
                "source_sha256": res["sha256"],
                "envelope": envelope,
                "lineage": {
                    "lineage_root_id": envelope["lineage_root_id"],
                    "source": res["source"],
                    "original_filename": res["original_filename"],
                },
                "rights": {
                    "training_allowed": True,
                    "derivatives_allowed": True,
                    # Historical DEC-001 value, kept by design: DEC-011
                    # (2026-09-06) supersedes it at the decision level, but
                    # the store is not recomputed and this field feeds no
                    # identity. Distribution is derived at C6 (the ancestry
                    # manifest) from the typed distribution_decision
                    # reference to DEC-011, not from this record.
                    "redistribution_allowed": False,
                    "evidence": RIGHTS_EVIDENCE,
                    "source": res["source"],
                    "privacy_class": "internal_non_sensitive"
                    if res["source"] == "internal" else "public_non_personal",
                    "privacy_review": PRIVACY_REVIEW,
                    "privacy_review_evidence": PRIVACY_REVIEW_EVIDENCE,
                },
                "import": {
                    "status": res["import_status"],
                    "issue_codes": res["issue_codes"],
                },
            }, sort_keys=True) + "\n")

    summary = {
        "migration_version": MIGRATION_VERSION,
        # Provenance: the store records which profile the run ASKED for,
        # so it never has to be inferred from the importer's default.
        "profile": profile,
        "registry": str(registry_path),
        "processed": len(records),
        "accepted": accepted,
        "quarantined": quarantined,
        "skipped_already_done": len(done),
        "status_counts": status_counts,
        "rights_evidence": RIGHTS_EVIDENCE,
        "privacy_review": PRIVACY_REVIEW,
        "privacy_review_evidence": PRIVACY_REVIEW_EVIDENCE,
        "rasterizer_pin": raster_pin,
    }
    (out_dir / "run-summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-quarantined", action="store_true")
    parser.add_argument(
        "--profile", choices=PROFILES, default="word-core-v0",
        help="AST profile to import into (DEC-003/RFC001-ERRATA-004). The D-2b "
             "regeneration uses word-visual-v1.1; the default reproduces a "
             "word-core-v0 store.",
    )
    args = parser.parse_args()
    summary = run_m2a(args.registry, args.raw_dir, args.out_dir,
                      workers=args.workers, limit=args.limit,
                      retry_quarantined=args.retry_quarantined,
                      profile=args.profile)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
