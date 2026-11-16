"""Light release assembly (RFC003 §40.6/§45/§46.2/§52, V0 subset).

``assemble_release`` writes, into a fresh ``out_dir``:

- ``splits/<name>.jsonl`` — one JCS member per line, authority IDs plus
  an explicit ``eligibility`` from the gate report (``authority_kind``,
  ``authority_id``, ``lineage_root_id``, ``eligibility``; no refs, no
  hashes, no back-references — §13.1/§40.6). ``train.jsonl`` contains
  ONLY ``train_ready`` members ("only TRAIN_READY trains by default",
  §34); train-assigned members with any other eligibility land in the
  top-level partial manifest ``splits/partial.jsonl`` (each line adds
  ``split_name``), reflected in ``counts.partial_documents`` and the
  manifest's ``partial_manifest`` entry. ``partial_ready`` members NEVER land
  in an eval-side manifest either — ALL ``partial_ready`` members route
  to the partial manifest regardless of their assigned split (each line
  keeps its ``split_name``). Eval-side manifests keep their remaining
  members inline, labeled by ``eligibility``;
- ``artifacts/index.jsonl`` — per document, the content-addressed refs
  of everything a training sample reads (page images, target bundle,
  observation sidecar, mask projection, fragment index), hashed into the
  manifest projection (``artifact_index``). Training-data selection
  resolves through THIS, never through the store's mutable append-only
  ``renders.jsonl``: appending a render row or rewriting that ledger
  must not change what an already assembled release delivers;
- ``quality/gate-report.jsonl`` — the per-record gate results INCLUDING
  their per-task eligibility, hashed into the manifest projection
  (``gate_report``) and pinned by a raw ``quality/gate-report.sha256``
  sidecar; what may train is part of the release identity, not
  provenance written next to it;
- ``quality/quarantine.jsonl`` — gate-quarantined records and
  leakage-quarantined lineages with §45 reason codes;
  ``lineage_root_id`` is looked up from the ``index.jsonl`` envelopes
  (ALL records, so quarantined documents keep their real lineage);
- ``coverage/report.json`` — coverage light: node-type counts,
  property-presence counts and per-recipe counts per split, integers
  only;
- ``policies/split-policy.json`` and
  ``policies/leakage-fingerprint-policy.json`` — pinned policy
  projections plus their hashes;
- ``release-manifest.json`` — dataset version, component-lock light,
  split manifest hashes and counts, closed by ``release_root_hash``.

``release_root_hash`` projection (documented whitelist): the hash is
``H("rfc003.release-root/v1", projection)`` where the projection is the
manifest object with EXACTLY the release_root_hash field removed —
i.e. every other manifest field (schema version, dataset name/version,
component_lock, policies, split_manifests, partial_manifest,
artifact_index, gate_report, quarantine_index, coverage_report,
held_out_template_family_ids, counts) is whitelisted
into the identity, and nothing else exists in the file. The crypto
graph stays acyclic: split manifests carry only authority IDs, and no
written artifact references the release root (§13.1/§52).

Immutability (§46.2): the assembler refuses to write into an existing
``out_dir`` — corrections build a new release directory with new
hashes, never an in-place edit. All inputs are loaded and validated
(including ast_version homogeneity across records) BEFORE ``out_dir``
is created, so a failing input never leaves a partial directory.

Split manifest hash: ``H("rfc003.split-manifest/v1", {...})`` over
schema version, split name, both policy hashes, the template-holdout
ids (empty list for non-OOD branches, never null) and the sorted
members. [SIMPLIFIED-VS-RFC] V0 has no separate template-holdout
policy artifact — `template_holdout_policy_hash` reuses the split
policy hash, whose projection pins the reservation rule.

CLI: ``python -m unflatten.dataset.split.release <factory_dir>
<gate_report> <out_dir>`` — loads gate-passing records, fingerprints
them from the verified object store, allocates the split and assembles
the release, printing a JSON summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.dataset.gates.result import GATE_RESULT_SCHEMA_VERSION
from unflatten.dataset.gates.store import read_object
from unflatten.dataset.identity import domain_hash
from unflatten.dataset.split.allocator import (
    DEFAULT_SPLIT_POLICY,
    TEMPLATE_OOD_SPLIT,
    SplitAssignment,
    SplitPolicy,
    allocate,
)
from unflatten.dataset.split.fingerprints import (
    DEFAULT_LEAKAGE_POLICY,
    FINGERPRINT_SCHEMA_VERSION,
    LeakageFingerprintPolicy,
    build_fingerprint,
)
from unflatten.dataset.targets.properties import emit_property_labels, normalize_pointer

RELEASE_SCHEMA_VERSION = "0.3.0"
RELEASE_ROOT_DOMAIN = "rfc003.release-root/v1"
SPLIT_MANIFEST_DOMAIN = "rfc003.split-manifest/v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "0.1.0"
DATASET_NAME = "unflatten-word-core-v0"
DATASET_VERSION = "0.2.0"

#: Artifact lanes a release PINS per document: everything a training
#: sample reads. They are copied into the release as content-addressed
#: refs and hashed into `release_root_hash`, so a later mutation of the
#: store's append-only `renders.jsonl` cannot change what an already
#: assembled release resolves to. Provenance lanes (docx, pdf, package
#: manifest) are deliberately not pinned — nothing trains on them.
PINNED_ARTIFACT_LANES: tuple[str, ...] = (
    "pages",
    "target_bundle",
    "observation_sidecar",
    "mask_projection",
    "fragment_index",
)

#: All release split branches, in fixed report order.
RELEASE_SPLITS: tuple[str, ...] = ("train", "val_iid", TEMPLATE_OOD_SPLIT, "test")

_JSON = "application/json"
_NDJSON = "application/x-ndjson"


class ReleaseError(Exception):
    """Release assembly refused (immutability or inconsistent inputs)."""


def load_eligible_records(
    factory_out_dir: Path, gate_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Allocator records for every gate result with ``disposition
    active``, with fingerprints built from hash-verified stored objects.
    """
    factory_out_dir = Path(factory_out_dir)
    envelopes = {
        record["envelope"]["document_bundle_id"]: record["envelope"]
        for record in _read_jsonl(factory_out_dir / "index.jsonl")
    }
    run_manifest = json.loads((factory_out_dir / "run_manifest.json").read_bytes())
    recipe_by_bundle = {
        doc["document_bundle_id"]: doc["recipe_id"] for doc in run_manifest["documents"]
    }
    missing_recipe = set(envelopes) - set(recipe_by_bundle)
    if missing_recipe:
        raise ReleaseError(
            f"run_manifest.json lacks documents for bundle(s) {sorted(missing_recipe)[:3]!r}"
        )

    # Completeness (re-verify should_fix): every index.jsonl bundle must
    # have exactly one gate result, and dispositions form a closed set —
    # a record silently missing from every manifest would violate the
    # never-silent principle.
    seen_bundles = [r["document_bundle_id"] for r in gate_results]
    if len(seen_bundles) != len(set(seen_bundles)):
        raise ReleaseError("duplicate gate results for a document_bundle_id")
    missing = set(envelopes) - set(seen_bundles)
    if missing:
        raise ReleaseError(
            f"index.jsonl records without a gate result: {sorted(missing)[:3]!r}"
            f" ({len(missing)} total)"
        )
    for result in gate_results:
        if result["disposition"] not in ("active", "quarantine"):
            raise ReleaseError(
                f"unknown disposition {result['disposition']!r} for "
                f"{result['document_bundle_id']!r}"
            )

    records: list[dict[str, Any]] = []
    for result in gate_results:
        if result["disposition"] != "active":
            continue
        bundle_id = result["document_bundle_id"]
        envelope = envelopes.get(bundle_id)
        if envelope is None:
            raise ReleaseError(f"gate result {bundle_id!r} has no index.jsonl record")
        canonical_ast = json.loads(read_object(factory_out_dir, envelope["ast"]))
        manifest_records = json.loads(
            read_object(factory_out_dir, envelope["resources_manifest"])
        )
        records.append(
            {
                "document_bundle_id": bundle_id,
                "lineage_root_id": envelope["lineage_root_id"],
                "recipe_id": recipe_by_bundle[bundle_id],
                "eligibility": result["eligibility"],
                "fingerprint": build_fingerprint(
                    canonical_ast, [r["sha256"] for r in manifest_records]
                ),
                "canonical_ast": canonical_ast,
            }
        )
    records.sort(key=lambda r: r["document_bundle_id"])
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


_SERIALIZATION_CHUNK_ROWS = 256


def _serialize_rows(rows: list[dict[str, Any]]) -> bytes:
    """Serialize one ordered chunk using the same JCS encoder as serial assembly."""
    return b"".join(to_canonical_json(row) + b"\n" for row in rows)


def _serialize_jsonl(
    rows: list[dict[str, Any]], executor: ProcessPoolExecutor | None, workers: int
) -> bytes:
    if executor is None or len(rows) < 2 * _SERIALIZATION_CHUNK_ROWS:
        return _serialize_rows(rows)
    chunks = (
        rows[i : i + _SERIALIZATION_CHUNK_ROWS]
        for i in range(0, len(rows), _SERIALIZATION_CHUNK_ROWS)
    )
    # Bound in-flight IPC and consume in submission order, regardless of completion
    # order. Only the parent writes files and their content-addressed references.
    pending = deque(
        executor.submit(_serialize_rows, chunk) for chunk in islice(chunks, 2 * workers)
    )
    parts: list[bytes] = []
    while pending:
        parts.append(pending.popleft().result())
        chunk = next(chunks, None)
        if chunk is not None:
            pending.append(executor.submit(_serialize_rows, chunk))
    return b"".join(parts)


def _write(path: Path, data: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
        "size_bytes": len(data),
    }


def _ref(root: Path, path: Path, data: bytes, mime_type: str) -> dict[str, Any]:
    return {
        "ref": path.relative_to(root).as_posix(),
        **_write(path, data),
        "mime_type": mime_type,
    }


#: Story context a node lives in — the §43 REPORT axis (DEC-003 [OPEN] J).
#: Keyed off the root the node hangs from and the story it is inside;
#: `body` is everything else, including table cells (which the criterion
#: has always folded into `paragraph`).
STORY_CONTEXTS: tuple[str, ...] = (
    "body",
    "header",
    "footer",
    "footnote",
    "endnote",
)

_STORY_CONTEXT_OF_TYPE: dict[str, str] = {
    "header": "header",
    "footer": "footer",
    "footnote": "footnote",
    "endnote": "endnote",
}


def _walk_counts(value: Any, context: str, counts: dict[str, int],
                 by_context: dict[str, dict[str, int]]) -> None:
    if isinstance(value, dict):
        node_type = value.get("type")
        if isinstance(node_type, str) and isinstance(value.get("id"), str):
            counts[node_type] = counts.get(node_type, 0) + 1
            context = _STORY_CONTEXT_OF_TYPE.get(node_type, context)
            bucket = by_context.setdefault(context, {})
            bucket[node_type] = bucket.get(node_type, 0) + 1
        for item in value.values():
            _walk_counts(item, context, counts, by_context)
    elif isinstance(value, list):
        for item in value:
            _walk_counts(item, context, counts, by_context)


def _node_type_counts(
    canonical_ast: dict[str, Any],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """`(counts by node type, counts by story context and node type)`.

    DEC-003 [OPEN] J, reading (b): the §43 criterion stays keyed on the
    NODE TYPE alone — a header paragraph is a paragraph, exactly as a
    cell paragraph has always been — and the story context is carried
    alongside as a report axis. Counting a header paragraph as a
    separate node type would contradict both the cell precedent and the
    profile alphabet (`events_vocab.PROFILE_NODE_TYPES` has no
    `header_paragraph`); dropping the axis would leave "1000 paragraphs"
    satisfiable with zero header paragraphs and no way to see it.

    The note stories hang off the DOCUMENT root, so they are walked
    from the roots, not from `sections` — walking `sections` alone
    counted every footnote body as absent.
    """
    counts: dict[str, int] = {}
    by_context: dict[str, dict[str, int]] = {}
    _walk_counts(canonical_ast.get("sections", []), "body", counts, by_context)
    for key in ("footnotes", "endnotes"):
        _walk_counts(canonical_ast.get(key) or [], "body", counts, by_context)
    return counts, by_context


def _property_presence(canonical_ast: dict[str, Any]) -> dict[str, int]:
    presence: dict[str, int] = {}
    for label in emit_property_labels(canonical_ast).values():
        for field in label["fields"]:
            key = label["node_type"] + normalize_pointer(field["node_property_pointer"])
            presence[key] = presence.get(key, 0) + 1
    return presence


def compute_coverage(canonical_ast: dict[str, Any]) -> dict[str, Any]:
    """Coverage counts for one canonical tree, in the record `coverage` shape."""
    node_types, by_context = _node_type_counts(canonical_ast)
    return {
        "node_types": node_types,
        "node_types_by_story": by_context,
        "property_presence": _property_presence(canonical_ast),
    }


def record_coverage(record: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Per-document coverage counts for `record`.

    A record may carry a precomputed ``coverage`` block instead of its
    ``canonical_ast``: the counts are a few hundred bytes while the tree
    is ~1 MB, and a release over 100k documents cannot hold every tree in
    memory at once. Records that still carry the tree are counted here."""
    coverage = record.get("coverage")
    if coverage is not None:
        return {
            "node_types": coverage.get("node_types", {}),
            # a record written before the story axis existed carries no
            # breakdown; an empty one reports "not measured here" rather
            # than claiming every node was body text.
            "node_types_by_story": coverage.get("node_types_by_story", {}),
            "property_presence": coverage.get("property_presence", {}),
        }
    return compute_coverage(record["canonical_ast"])


def record_ast_version(record: dict[str, Any]) -> str:
    """`record`'s AST version, from the record or its canonical tree."""
    version = record.get("ast_version")
    if version is not None:
        return version
    return record["canonical_ast"].get("ast_version", "0.1.0")


def _coverage_report(
    records: list[dict[str, Any]], assignment: SplitAssignment
) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for name in RELEASE_SPLITS:
        split_records = [
            r
            for r in records
            if assignment.assignments.get(r["lineage_root_id"]) == name
        ]
        node_types: dict[str, int] = {}
        by_story: dict[str, dict[str, int]] = {}
        presence: dict[str, int] = {}
        by_recipe: dict[str, int] = {}
        for record in split_records:
            if record["recipe_id"] is not None:  # recipe-less = imported real doc
                by_recipe[record["recipe_id"]] = by_recipe.get(record["recipe_id"], 0) + 1
            counts = record_coverage(record)
            for key, count in counts["node_types"].items():
                node_types[key] = node_types.get(key, 0) + count
            for context, per_type in counts["node_types_by_story"].items():
                bucket = by_story.setdefault(context, {})
                for key, count in per_type.items():
                    bucket[key] = bucket.get(key, 0) + count
            for key, count in counts["property_presence"].items():
                presence[key] = presence.get(key, 0) + count
        by_split[name] = {
            "documents": len(split_records),
            "lineages": len({r["lineage_root_id"] for r in split_records}),
            "by_recipe": by_recipe,
            "node_types": node_types,
            "node_types_by_story": by_story,
            "property_presence": presence,
        }
    return {"coverage_report_schema_version": "0.2.0", "by_split": by_split}


def _split_manifest_hash(
    split_name: str,
    members: list[dict[str, Any]],
    assignment: SplitAssignment,
) -> str:
    return domain_hash(
        SPLIT_MANIFEST_DOMAIN,
        {
            "split_manifest_schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "split_name": split_name,
            "split_policy_hash": assignment.split_policy_hash,
            "leakage_fingerprint_policy_hash": assignment.leakage_fingerprint_policy_hash,
            "template_holdout_policy_hash": assignment.split_policy_hash,
            "held_out_template_family_ids": (
                sorted(assignment.held_out_template_family_ids)
                if split_name == TEMPLATE_OOD_SPLIT
                else []
            ),
            "members": members,
        },
    )


def assemble_release(
    factory_out_dir: Path,
    gate_results: list[dict[str, Any]],
    assignment: SplitAssignment,
    out_dir: Path,
    *,
    split_policy: SplitPolicy = DEFAULT_SPLIT_POLICY,
    fingerprint_policy: LeakageFingerprintPolicy = DEFAULT_LEAKAGE_POLICY,
    records: list[dict[str, Any]] | None = None,
    lineage_by_bundle: dict[str, str] | None = None,
    component_lock_source: dict[str, str] | None = None,
    artifacts_by_bundle: dict[str, dict[str, Any]] | None = None,
    pinned_from_release_root_hash: str | None = None,
    supersedes: dict[str, Any] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Assemble the release into a FRESH `out_dir`; returns the manifest.

    By default all inputs load from ``factory_out_dir`` (factory shape).
    A combined-corpus caller (``migration.m2c``) may inject its own
    ``records`` (allocator records; ``recipe_id`` may be None for
    imported real documents), ``lineage_by_bundle`` lookup and
    ``component_lock_source`` (``dataset_namespace`` +
    ``generator_version``) and ``artifacts_by_bundle`` (bundle id ->
    artifact refs to pin) — the caller then owns the completeness checks
    that ``load_eligible_records`` performs for factory runs.

    ``workers`` controls independent JSONL serialization only (default serial).
    One bounded pool is reused across lanes; file bytes and identity are unchanged.

    ``pinned_from_release_root_hash`` and ``supersedes`` (§4.4) record a
    split-pinned migration from a prior release: both are always present
    in the manifest projection (``None`` when this release is not a
    migration), so the schema never grows an OPEN branch. They are NOT
    cross-checked against ``assignment.pinned_from_release_root_hash``
    here — the caller (``migration.m2c``) is the one that threads the
    same hash through ``allocate`` and ``assemble_release``.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    if split_policy.policy_hash() != assignment.split_policy_hash:
        raise ReleaseError("split_policy does not match the assignment's policy hash")
    if fingerprint_policy.policy_hash() != assignment.leakage_fingerprint_policy_hash:
        raise ReleaseError(
            "fingerprint_policy does not match the assignment's policy hash"
        )

    # --- load + validate ALL inputs before touching out_dir (§46.2) ------
    factory_out_dir = Path(factory_out_dir)
    if records is None:
        records = load_eligible_records(factory_out_dir, gate_results)
    missing_views = [
        r.get("document_bundle_id")
        for r in gate_results
        if "task_eligibility" not in r
    ]
    if missing_views:
        raise ReleaseError(
            f"gate result(s) {missing_views[:3]!r} carry no task_eligibility; "
            "per-task admission is part of the release identity (§33.4)"
        )
    ast_versions = sorted({record_ast_version(r) for r in records})
    if len(ast_versions) > 1:
        raise ValueError(
            "mixed ast_version across release records: "
            + " vs ".join(ast_versions)
        )
    if component_lock_source is None:
        run_manifest = json.loads(
            (factory_out_dir / "run_manifest.json").read_bytes()
        )
        component_lock_source = {
            "dataset_namespace": run_manifest["dataset_namespace"],
            "generator_version": run_manifest["generator_version"],
        }
    # lineage lookup over ALL index.jsonl envelopes — quarantined
    # documents keep their real lineage_root_id, never null
    if lineage_by_bundle is None:
        lineage_by_bundle = {
            rec["envelope"]["document_bundle_id"]: rec["envelope"]["lineage_root_id"]
            for rec in _read_jsonl(factory_out_dir / "index.jsonl")
        }
    # artifact refs to pin; a factory run records them in its run manifest
    if artifacts_by_bundle is None:
        run_documents = json.loads(
            (factory_out_dir / "run_manifest.json").read_bytes()
        )["documents"]
        artifacts_by_bundle = {
            doc["document_bundle_id"]: doc.get("artifacts") or {}
            for doc in run_documents
        }

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ReleaseError(
            f"release directory {out_dir} already exists — releases are "
            "immutable (§46.2); assemble into a fresh directory"
        )
    out_dir.mkdir(parents=True)

    # Serialize independent row chunks in one reusable pool; allocation, ordering,
    # coverage, hashing and all writes remain in the parent. Execution settings
    # deliberately do not enter any policy or release identity.
    chunk_count = (
        max(len(records), len(gate_results)) + _SERIALIZATION_CHUNK_ROWS - 1
    ) // _SERIALIZATION_CHUNK_ROWS
    pool_workers = min(workers, max(1, chunk_count))
    pool = (
        ProcessPoolExecutor(max_workers=pool_workers)
        if pool_workers > 1 else nullcontext(None)
    )
    with pool as executor:
        def jsonl(rows: list[dict[str, Any]]) -> bytes:
            return _serialize_jsonl(rows, executor, pool_workers)

        # --- split manifests -------------------------------------------------
        split_manifest_entries: list[dict[str, Any]] = []
        partial_members: list[dict[str, Any]] = []
        for name in RELEASE_SPLITS:
            assigned_members = sorted(
                (
                    {
                        "authority_kind": "document_bundle",
                        "authority_id": r["document_bundle_id"],
                        "lineage_root_id": r["lineage_root_id"],
                        "eligibility": r["eligibility"],
                    }
                    for r in records
                    if assignment.assignments.get(r["lineage_root_id"]) == name
                ),
                key=lambda m: (m["authority_kind"], m["authority_id"]),
            )
            if name == "train":
                # only TRAIN_READY trains by default (§34): everything else
                # goes to the top-level partial manifest, never train.jsonl
                members = [m for m in assigned_members if m["eligibility"] == "train_ready"]
                moved = [m for m in assigned_members if m["eligibility"] != "train_ready"]
            else:
                # partial_ready members never land in an eval-side manifest:
                # they route to the partial manifest, keeping split_name
                members = [
                    m for m in assigned_members if m["eligibility"] != "partial_ready"
                ]
                moved = [
                    m for m in assigned_members if m["eligibility"] == "partial_ready"
                ]
            partial_members.extend({**m, "split_name": name} for m in moved)
            data = jsonl(members)
            entry_ref = _ref(out_dir, out_dir / "splits" / f"{name}.jsonl", data, _NDJSON)
            split_manifest_entries.append(
                {
                    "split_name": name,
                    "split_manifest_hash": _split_manifest_hash(name, members, assignment),
                    **entry_ref,
                }
            )
        partial_members.sort(
            key=lambda m: (m["authority_kind"], m["authority_id"], m["split_name"])
        )
        partial_entry = {
            "split_name": "partial",
            "split_manifest_hash": _split_manifest_hash(
                "partial", partial_members, assignment
            ),
            **_ref(
                out_dir,
                out_dir / "splits" / "partial.jsonl",
                jsonl(partial_members),
                _NDJSON,
            ),
        }

        # --- pinned artifact index (inside the release identity) -------------
        artifact_rows = [
            {
                "document_bundle_id": record["document_bundle_id"],
                "artifacts": {
                    lane: (artifacts_by_bundle.get(record["document_bundle_id"]) or {})[
                        lane
                    ]
                    for lane in PINNED_ARTIFACT_LANES
                    if (artifacts_by_bundle.get(record["document_bundle_id"]) or {}).get(
                        lane
                    )
                    is not None
                },
            }
            for record in sorted(records, key=lambda r: r["document_bundle_id"])
        ]
        artifact_index_ref = _ref(
            out_dir,
            out_dir / "artifacts" / "index.jsonl",
            jsonl(artifact_rows),
            _NDJSON,
        )

        # --- gate report (inside the release identity) -----------------------
        # What may train is decided by the gate report and its per-task
        # eligibility. A release whose identity does not cover that decision
        # can have it swapped without changing release_root_hash, so the
        # report is written HERE, hashed into the manifest projection, and
        # pinned by a raw sha256 sidecar the sample loader verifies.
        gate_report_bytes = jsonl(gate_results)
        gate_report_ref = _ref(
            out_dir, out_dir / "quality" / "gate-report.jsonl", gate_report_bytes, _NDJSON
        )
        (out_dir / "quality" / "gate-report.sha256").write_text(
            gate_report_ref["sha256"] + "\n"
        )

        # --- quarantine index ------------------------------------------------
        quarantine_lines: list[dict[str, Any]] = []
        for result in gate_results:
            if result["disposition"] == "quarantine":
                quarantine_lines.append(
                    {
                        "document_bundle_id": result["document_bundle_id"],
                        "lineage_root_id": lineage_by_bundle.get(
                            result["document_bundle_id"]
                        ),
                        "reason_codes": list(result["reason_codes"]),
                        "source": "gates",
                    }
                )
        for lineage, reason in sorted(assignment.quarantined_lineages.items()):
            bundles = sorted(
                r["document_bundle_id"]
                for r in records
                if r["lineage_root_id"] == lineage
            )
            # SPLIT_COLLISION here is a pin conflict (§4.2 test-wins rule);
            # every other quarantine reason at this stage is a leakage-graph
            # decision (ambiguous-pair or late-collision quarantine).
            source = "pin" if reason == "SPLIT_COLLISION" else "leakage"
            for bundle_id in bundles or [None]:
                quarantine_lines.append(
                    {
                        "document_bundle_id": bundle_id,
                        "lineage_root_id": lineage,
                        "reason_codes": [reason],
                        "source": source,
                    }
                )
        quarantine_lines.sort(key=lambda q: to_canonical_json(q))
        quarantine_ref = _ref(
            out_dir,
            out_dir / "quality" / "quarantine.jsonl",
            jsonl(quarantine_lines),
            _NDJSON,
        )

        # --- coverage light --------------------------------------------------
        coverage = _coverage_report(records, assignment)
        coverage_ref = _ref(
            out_dir, out_dir / "coverage" / "report.json", to_canonical_json(coverage), _JSON
        )

        # --- pinned policies -------------------------------------------------
        policy_refs = {
            "split_policy": _ref(
                out_dir,
                out_dir / "policies" / "split-policy.json",
                to_canonical_json(
                    {**split_policy.to_dict(), "policy_hash": split_policy.policy_hash()}
                ),
                _JSON,
            ),
            "leakage_fingerprint_policy": _ref(
                out_dir,
                out_dir / "policies" / "leakage-fingerprint-policy.json",
                to_canonical_json(
                    {
                        **fingerprint_policy.to_dict(),
                        "policy_hash": fingerprint_policy.policy_hash(),
                    }
                ),
                _JSON,
            ),
        }

        # --- counts + manifest ----------------------------------------------
        assigned = {
            name: [
                r
                for r in records
                if assignment.assignments.get(r["lineage_root_id"]) == name
            ]
            for name in RELEASE_SPLITS
        }
        by_recipe: dict[str, int] = {}
        for record in records:
            if record["recipe_id"] is not None:  # recipe-less = imported real doc
                by_recipe[record["recipe_id"]] = by_recipe.get(record["recipe_id"], 0) + 1
        counts = {
            "documents": len(records),
            "lineages": len({r["lineage_root_id"] for r in records}),
            "by_split_documents": {n: len(assigned[n]) for n in RELEASE_SPLITS},
            "by_split_lineages": {
                n: len({r["lineage_root_id"] for r in assigned[n]}) for n in RELEASE_SPLITS
            },
            "by_recipe": by_recipe,
            "partial_documents": len(partial_members),
            "quarantined_documents": sum(
                1 for r in gate_results if r["disposition"] == "quarantine"
            ),
            "quarantined_lineages": len(assignment.quarantined_lineages),
        }
        ast_version = ast_versions[0] if records else "0.1.0"
        projection = {
            "release_schema_version": RELEASE_SCHEMA_VERSION,
            "dataset_name": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "component_lock": {
                "ast_version": ast_version,
                "dataset_namespace": component_lock_source["dataset_namespace"],
                "generator_version": component_lock_source["generator_version"],
                "gate_result_schema_version": GATE_RESULT_SCHEMA_VERSION,
                "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
                "split_policy_hash": assignment.split_policy_hash,
                "leakage_fingerprint_policy_hash": assignment.leakage_fingerprint_policy_hash,
                "pinned_from_release_root_hash": pinned_from_release_root_hash,
            },
            "policies": policy_refs,
            "split_manifests": split_manifest_entries,
            "partial_manifest": partial_entry,
            "artifact_index": artifact_index_ref,
            "gate_report": gate_report_ref,
            "quarantine_index": quarantine_ref,
            "coverage_report": coverage_ref,
            "held_out_template_family_ids": sorted(
                assignment.held_out_template_family_ids
            ),
            "counts": counts,
            "supersedes": supersedes,
        }
        manifest = {
            **projection,
            "release_root_hash": domain_hash(RELEASE_ROOT_DOMAIN, projection),
        }
        (out_dir / "release-manifest.json").write_bytes(to_canonical_json(manifest))
        return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.dataset.split.release",
        description="RFC003 D5 split allocation + light release assembly.",
    )
    parser.add_argument("factory_dir", type=Path, help="factory output directory")
    parser.add_argument("gate_report", type=Path, help="gate-report.jsonl path")
    parser.add_argument("out_dir", type=Path, help="fresh release directory")
    args = parser.parse_args(argv)

    gate_results = _read_jsonl(args.gate_report)
    records = load_eligible_records(args.factory_dir, gate_results)
    assignment = allocate(records)
    manifest = assemble_release(args.factory_dir, gate_results, assignment, args.out_dir)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "release_root_hash": manifest["release_root_hash"],
                "documents": manifest["counts"]["documents"],
                "lineages": manifest["counts"]["lineages"],
                "by_split_lineages": manifest["counts"]["by_split_lineages"],
                "quarantined_lineages": manifest["counts"]["quarantined_lineages"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
