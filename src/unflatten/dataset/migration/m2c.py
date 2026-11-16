"""M2c — dev release candidate over the M2a store (+ optional synthetic
factory run).

Glue over existing D5 machinery, no forks:

1. **Gates** — each ``index.jsonl`` bundle is joined with its
   ``renders.jsonl`` row (M2b enrichment; absent row = ``render_status
   "skipped"`` -> the ``partial_ready`` downgrade path) and adapted into
   the record shape ``gates.engine.evaluate_document`` expects. The
   engine reads the record's explicit ``rights`` block (basis
   DEC-001-option-A); ``privacy_review: pending`` passes with a recorded
   PRIVACY_REVIEW_PENDING warning — dev/research inventory only. The
   training path refuses those records at the admission boundary
   (``dataset.samples`` ``require_privacy_review``), RFC005 §9.2/§9.3.
2. **Split** — ``split.fingerprints`` + ``split.allocator`` over the
   gate-passing lineages of the combined population. Real documents are
   recipe-less (``recipe_id`` None): they participate through
   fingerprint/resource leakage edges only and never form a template
   family; synthetic records keep their factory ``recipe_id``.
3. **Release** — ``split.release.assemble_release`` with injected
   combined records, lineage lookup and component-lock source. The
   assembler writes the per-record gate report (including its per-task
   eligibility) under ``out_dir/quality/gate-report.jsonl`` and hashes
   its ref into ``release_root_hash``; the raw sha256 sidecar next to it
   is what the sample loader verifies.

The M2a store is only ever read; every write lands under ``out_dir``.

Gate-stage durability (§4.5, DEC-023): a checkpoint dir (``--gate-checkpoint``,
CLI default ``<out_dir>.gate/``) makes the gate stage resumable across a
crash — ``gate-results.jsonl``/``active-records.jsonl`` are appended per
document with ``flush()``, and ``identity.json`` pins the store bytes, the
gate-engine source closure and the run configuration, so a resume against
anything else refuses (``ReleaseError``) instead of splicing two gate-report
identities together.

CLI: ``python -m unflatten.dataset.migration.m2c <m2a_store> <out_dir>
[--synthetic <factory_dir>] [--limit N] [--gate-checkpoint <dir>]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.ast.canonicalizer import canonicalize
from unflatten.dataset import samples
from unflatten.dataset.gates.engine import evaluate_document
from unflatten.dataset.gates.result import GATE_RESULT_SCHEMA_VERSION, validate_gate_result
from unflatten.dataset.gates.store import StoreError, read_object
from unflatten.dataset.gates.task_eligibility import TASK_KINDS, task_eligibility
from unflatten.dataset.migration.m2b import derivation_hash
from unflatten.dataset.script_profile import SCRIPT_RANGES, dominant_script
from unflatten.dataset.split.allocator import allocate
from unflatten.dataset.split.fingerprints import build_fingerprint
from unflatten.dataset.split.release import (
    ReleaseError,
    assemble_release,
    compute_coverage,
    load_eligible_records,
)
from unflatten.dataset.targets.masks import MASK_PROJECTION_SCHEMA_VERSION
from unflatten.import_closure import closure

M2C_VERSION = "0.2.0"
#: component_lock source for a real-only release (no factory run).
REAL_DATASET_NAMESPACE = "unflatten-m2a-v0"
REAL_GENERATOR_VERSION = f"m2c-{M2C_VERSION}"

ELIGIBILITIES = ("train_ready", "eval_ready", "partial_ready", "ineligible")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file, tolerating one torn TRAILING line (a crashed
    appender) with a warning print; a torn line anywhere else is real
    corruption and still raises."""
    lines = path.read_bytes().splitlines()
    rows: list[dict[str, Any]] = []
    for k, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if k == len(lines) - 1:
                print(
                    f"[m2c] {path}: skipping torn trailing line {k + 1}",
                    flush=True,
                )
                continue
            raise
    return rows


def load_bundle_exclusions(path: Path) -> dict[str, str]:
    """Load a pinned JSONL exclusion set with one named reason per bundle."""
    exclusions: dict[str, str] = {}
    for line_number, row in enumerate(_read_jsonl(Path(path)), start=1):
        if set(row) != {"document_bundle_id", "reason"}:
            raise ReleaseError(
                f"{path}:{line_number}: exclusion row must contain exactly "
                "document_bundle_id and reason"
            )
        bundle_id = row["document_bundle_id"]
        reason = row["reason"]
        if not isinstance(bundle_id, str) or not bundle_id:
            raise ReleaseError(f"{path}:{line_number}: invalid document_bundle_id")
        if not isinstance(reason, str) or not reason:
            raise ReleaseError(f"{path}:{line_number}: invalid exclusion reason")
        if bundle_id in exclusions:
            raise ReleaseError(f"{path}:{line_number}: duplicate exclusion {bundle_id!r}")
        exclusions[bundle_id] = reason
    if not exclusions:
        raise ReleaseError(f"{path}: bundle exclusion file is empty")
    return exclusions


# ------------------------------------------------------------------ adapters --


def adapt_record(
    index_record: dict[str, Any], renders_record: dict[str, Any] | None
) -> dict[str, Any]:
    """Join one M2a index record with its M2b renders row into the gate
    engine's expected record shape. No renders row means the bundle was
    never enriched: ``render_status "skipped"`` (downgrade path), never
    a fabricated failure."""
    record: dict[str, Any] = {
        "envelope": index_record["envelope"],
        "rights": index_record.get("rights"),
        "render_status": "skipped",
        "artifacts": {},
        "observation": None,
        "observation_skip_reason": None,
        "targets": None,
    }
    if renders_record is not None:
        record["render_status"] = renders_record["render_status"]
        record["artifacts"] = renders_record.get("artifacts") or {}
        record["observation"] = renders_record.get("observation")
        record["observation_skip_reason"] = renders_record.get(
            "observation_skip_reason"
        )
        record["targets"] = renders_record.get("targets")
    return record


def join_store_records(
    store_dir: Path,
    limit: int | None = None,
    *,
    refuse_stale_target_ledger: bool = True,
    excluded_bundles: Collection[str] = (),
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """``(index_record, renders_record | None)`` pairs in index order.

    ``refuse_stale_target_ledger=False`` is for `m2b --align-only` only: an
    aligner change moves `derivation_hash`, so the ledger is stale until
    `m2b_targets` re-derives it from the sidecars align-only is about to
    write, and align-only re-derives every bundle and refuses drift itself.

    ``excluded_bundles``: bundles the release leaves out; their ledger rows are
    neither judged by the stale guard nor overlaid (a bundle excluded for an
    align timeout is never re-derived, 2026-09-24)."""
    store_dir = Path(store_dir)
    renders: dict[str, dict[str, Any]] = {}
    renders_path = store_dir / "renders.jsonl"
    if renders_path.exists():
        # last-wins per bundle id — the same resume semantic M2b uses on
        # its append-only ledger (a re-attempted skipped row supersedes)
        duplicates = 0
        for row in _read_jsonl(renders_path):
            if row["document_bundle_id"] in renders:
                duplicates += 1
            renders[row["document_bundle_id"]] = row
        if duplicates:
            print(
                f"[m2c] {renders_path}: {duplicates} duplicate bundle "
                "row(s) — last row wins",
                flush=True,
            )
    masks = _read_mask_ledger(store_dir)
    _refuse_stale_mask_ledger(store_dir, masks)
    targets_ledger = _read_target_ledger(store_dir)
    excluded = frozenset(excluded_bundles)
    targets_ledger = {b: row for b, row in targets_ledger.items() if b not in excluded}
    if refuse_stale_target_ledger:
        _refuse_stale_target_ledger(targets_ledger)
    joined: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    skipped_masks = 0
    skipped_targets = 0
    with open(store_dir / "index.jsonl") as f:
        for line in f:
            if limit is not None and len(joined) >= limit:
                break
            index_record = json.loads(line)
            bundle_id = index_record["envelope"]["document_bundle_id"]
            renders_record = renders.get(bundle_id)
            mask_row = masks.get(bundle_id)
            if mask_row is not None and renders_record is not None:
                if mask_row.get("ast_hash") == index_record["envelope"].get("ast_hash"):
                    renders_record = _with_mask(renders_record, mask_row)
                else:
                    skipped_masks += 1
            target_row = targets_ledger.get(bundle_id)
            if target_row is not None and renders_record is not None:
                if target_row.get("ast_hash") == index_record["envelope"].get("ast_hash"):
                    renders_record = _with_target(renders_record, target_row)
                else:
                    skipped_targets += 1
            joined.append((index_record, renders_record))
    if skipped_masks:
        print(
            f"[m2c] {skipped_masks} mask ledger row(s) skipped — derived "
            "from another AST than the bundle carries",
            flush=True,
        )
    if skipped_targets:
        print(
            f"[m2c] {skipped_targets} target ledger row(s) skipped — derived "
            "from another AST than the bundle carries",
            flush=True,
        )
    return joined


def _refuse_stale_mask_ledger(
    store_dir: Path, masks: dict[str, dict[str, Any]]
) -> None:
    """Refuse a mask ledger written under an older mask schema.

    DEC-043 §5 raised `MASK_PROJECTION_SCHEMA_VERSION` so that old and
    new masks are distinguishable — but nothing compared them, so a
    release could be assembled on stale masks and look healthy. It was:
    release-v14-r5 shipped 0.2.0 masks while its gate report said 0.3.0,
    and a 66,407-page cache was built on them.

    One object is enough: the ledger is written by a single backfill run
    under one schema, so its first row decides. Reading all 18,803 would
    cost minutes to learn the same fact.
    """
    if not masks:
        return
    first = next(iter(masks.values()))
    ref = first.get("mask_projection")
    if ref is None:
        return
    try:
        projection = json.loads(read_object(store_dir, ref))
    except (StoreError, ValueError):
        return  # unreadable is the gates' problem, per-bundle, not this one's
    found = projection.get("mask_projection_schema_version")
    if found != MASK_PROJECTION_SCHEMA_VERSION:
        raise ReleaseError(
            f"{store_dir / 'mask-projections.jsonl'}: masks are schema "
            f"{found!r}, this code writes {MASK_PROJECTION_SCHEMA_VERSION!r} "
            "— re-derive the masks (m2b_masks --force-rederive-mask) before "
            "assembling a release, or the release ships masks whose meaning "
            "does not match the code that reads them"
        )


def _read_mask_ledger(store_dir: Path) -> dict[str, dict[str, Any]]:
    """``document_bundle_id -> row`` from ``mask-projections.jsonl``.

    `m2b_masks` re-derives the font-substitution mask into this ledger
    rather than rewriting the append-only ``renders.jsonl``. Nothing read
    it, so a re-derivation never reached the release and a mask schema
    bump quarantined the whole corpus on masks that were already rebuilt.
    Last row wins, the same resume semantic the renders ledger uses."""
    ledger_path = store_dir / "mask-projections.jsonl"
    if not ledger_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(ledger_path):
        rows[row["document_bundle_id"]] = row
    return rows


def _with_mask(
    renders_record: dict[str, Any], mask_row: dict[str, Any]
) -> dict[str, Any]:
    """`renders_record` with the ledger's mask projection overlaid.

    An M2b row carries the same mask ref TWICE: under ``targets`` (what
    the gates validate) and under ``artifacts`` (what the release pins in
    `artifacts_by_bundle`, and therefore what the cache build reads).
    Both are replaced. Overlaying only ``targets`` is what produced
    release-v14-r5 on 2026-09-23: the gate saw the re-derived 0.3.0 masks
    and passed, while the release shipped the old 0.2.0 ones and the
    66,407-page cache built on it never saw the font-mask fix at all.

    Copied, not mutated: `renders.jsonl` is append-only and the caller
    may hold the same row elsewhere. A row with no ``targets`` at all is
    left alone — there is nothing for a mask to belong to."""
    targets = renders_record.get("targets")
    if targets is None:
        return renders_record
    updated = dict(renders_record)
    updated["targets"] = {
        **targets,
        "mask_projection": mask_row["mask_projection"],
        "masked_font_runs": mask_row["masked_font_runs"],
    }
    artifacts = renders_record.get("artifacts")
    if artifacts is not None and "mask_projection" in artifacts:
        updated["artifacts"] = {
            **artifacts,
            "mask_projection": mask_row["mask_projection"],
        }
    return updated


def _read_target_ledger(store_dir: Path) -> dict[str, dict[str, Any]]:
    """``document_bundle_id -> row`` from ``target-bundles.jsonl``.

    `m2b_targets` re-derives the target bundle + fragments into this
    ledger rather than rewriting the append-only ``renders.jsonl``
    (DEC-043 §1 hyperlink flattening). Nothing read it would mean the
    re-derivation never reached a release, the same gap the mask ledger
    had before `_read_mask_ledger`. Last row wins, the same resume
    semantic every other ledger in this module uses."""
    ledger_path = store_dir / "target-bundles.jsonl"
    if not ledger_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(ledger_path):
        rows[row["document_bundle_id"]] = row
    return rows


def _with_target(
    renders_record: dict[str, Any], target_row: dict[str, Any]
) -> dict[str, Any]:
    """`renders_record` with the ledger's target bundle + fragment index
    overlaid.

    Same lesson as `_with_mask` (23. 9., release-v14-r5): an M2b row
    carries `target_bundle`/`fragment_index` TWICE, under ``targets``
    (what the gates validate) and ``artifacts`` (what the release PINS
    in `artifacts_by_bundle`, and therefore what the cache build reads).
    Both copies are replaced together so they can never diverge.

    Copied, not mutated: `renders.jsonl` is append-only and the caller
    may hold the same row elsewhere. A row with no ``targets`` at all is
    left alone — there is nothing for a re-derived bundle to belong to."""
    targets = renders_record.get("targets")
    if targets is None:
        return renders_record
    updated = dict(renders_record)
    updated["targets"] = {
        **targets,
        "target_bundle": target_row["target_bundle"],
        "fragment_index": target_row["fragment_index"],
        "fragment_count": target_row["fragment_count"],
        "window_policy": target_row["window_policy"],
    }
    artifacts = renders_record.get("artifacts")
    if artifacts is not None:
        overlay = {}
        if "target_bundle" in artifacts:
            overlay["target_bundle"] = target_row["target_bundle"]
        if "fragment_index" in artifacts:
            overlay["fragment_index"] = target_row["fragment_index"]
        if overlay:
            updated["artifacts"] = {**artifacts, **overlay}
    return updated


def _refuse_stale_target_ledger(targets_ledger: dict[str, dict[str, Any]]) -> None:
    """Refuse a target-bundle ledger written by derivation code other than
    today's (DEC-043 §1 rebuild).

    The mask ledger pins a schema version field carried on every
    projection; a target bundle carries no such field, so the honest
    staleness signal here is `m2b.derivation_hash()` — the SAME
    source-digest closure over `_DERIVATION_MODULES` that m2b itself uses
    on resume to decide whether a stored render was derived by current
    code (see `m2b.py`'s own staleness check). `m2b_targets` stamps its
    own `derivation_hash()` onto every ledger row it writes; comparing the
    first row against today's value is enough (one ledger, one writer,
    one run), exactly as `_refuse_stale_mask_ledger` reads only one
    object rather than all of them."""
    if not targets_ledger:
        return
    # EVERY effective row, not the first: a ledger mixing derivation hashes
    # (a partial `--force-rederive --limit`) passed whenever its first row
    # happened to be current (review 24. 9.). The rows are already in memory,
    # so checking them all costs nothing.
    current = derivation_hash()
    stale = {
        row.get("derivation_hash")
        for row in targets_ledger.values()
        if row.get("derivation_hash") != current
    }
    if stale:
        count = sum(
            1 for row in targets_ledger.values() if row.get("derivation_hash") != current
        )
        raise ReleaseError(
            f"target-bundles.jsonl: {count} of {len(targets_ledger)} bundle(s) carry "
            f"derivation_hash {sorted(map(str, stale))!r}, this code derives "
            f"{current!r} — re-derive them (m2b_targets --force-rederive) before "
            "assembling a release, or the release ships target bundles the "
            "current code no longer agrees with"
        )


def load_prior_assignment(release_dir: Path) -> dict[str, str]:
    """``lineage_root_id -> split_name`` from EVERY split manifest of a
    prior release (§4.3), read exclusively through the pinned sha256 refs
    in ``release-manifest.json`` — never a directory glob, so a manifest
    entry deleted or a byte tampered with is caught by the manifest's own
    ``release_root_hash`` or ref-hash check, not silently skipped."""
    release_dir = Path(release_dir)
    manifest = samples._release_manifest(release_dir)
    prior: dict[str, str] = {}

    def add(lineage_root_id: str, split_name: str) -> None:
        if lineage_root_id in prior and prior[lineage_root_id] != split_name:
            raise ReleaseError(
                f"prior release {release_dir}: lineage {lineage_root_id!r} "
                f"appears in more than one split manifest "
                f"({prior[lineage_root_id]!r} and {split_name!r})"
            )
        prior[lineage_root_id] = split_name

    for entry in manifest.get("split_manifests") or []:
        split_name = entry["split_name"]
        rows = samples._pinned_rows_entry(
            release_dir, entry, key_label=f"split_manifests[{split_name}]"
        )
        for row in rows:
            add(row["lineage_root_id"], split_name)
    partial_entry = manifest.get("partial_manifest")
    if partial_entry is not None:
        rows = samples._pinned_rows_entry(
            release_dir, partial_entry, key_label="partial_manifest"
        )
        for row in rows:
            add(row["lineage_root_id"], row["split_name"])
    return prior


def _pinned_file(path: Path) -> dict[str, Any]:
    data = Path(path).read_bytes()
    return {
        "path": str(path),
        "sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
        "size_bytes": len(data),
    }


def build_migration_record(summary_path: Path, m2a_store: Path) -> dict[str, Any]:
    """Build the `supersedes.migration` record (§4.4/§2.6) from
    `<store>/align-only-summary.json` — NOT an opaque record the operator
    types: the align-only summary is pinned by sha256, and the derivation
    hash / source store / m2b version are read from its own last run,
    never retyped by hand where they could drift from what actually ran.
    A summary with no `runs`, or whose last run is missing a required
    field, is a caller error worth failing loudly on.

    A store re-imported under a new `MIGRATION_VERSION` and fully rendered
    (DEC-058 v16: every `document_bundle_id` moves, so `--align-only` cannot
    carry it) has no align-only summary. Its record is built from the saved
    M2b summary (`render_profile` present) by `_full_render_record`.

    A `combined_store` reads `<store>/composition-manifest.json`: component
    migration records, source/output pins, exact unions and profile exceptions
    are verified before returning the pinned composition provenance.
    """
    summary_path = Path(summary_path)
    try:
        summary = json.loads(summary_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(
            f"--migration {summary_path}: cannot read align-only summary: {exc}"
        ) from exc
    if isinstance(summary, dict) and summary.get("kind") == "combined_store":
        from unflatten.dataset.migration.composition import build_composition_record

        return build_composition_record(summary_path, Path(m2a_store))
    if isinstance(summary, dict) and "render_profile" in summary:
        return _full_render_record(summary_path, summary, Path(m2a_store))
    runs = summary.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ReleaseError(
            f"--migration {summary_path}: summary has no 'runs' entries"
        )
    last_run = runs[-1]
    required = ("source_store", "derivation_hash", "m2b_version")
    missing = [key for key in required if key not in last_run]
    if missing:
        raise ReleaseError(
            f"--migration {summary_path}: last run is missing field(s) {missing}"
        )
    summary_bytes = summary_path.read_bytes()
    return {
        "kind": "align_only",
        "source_store": last_run["source_store"],
        "store": Path(m2a_store).name,
        "aligner_derivation_hash": last_run["derivation_hash"],
        "m2b_version": last_run["m2b_version"],
        "align_only_summary": {
            "path": str(summary_path),
            "sha256": f"sha256:{hashlib.sha256(summary_bytes).hexdigest()}",
            "size_bytes": len(summary_bytes),
        },
    }


def _full_render_record(
    summary_path: Path, summary: dict[str, Any], m2a_store: Path
) -> dict[str, Any]:
    """`kind: "full_render"`: the M2b summary (`python -m ...m2b` stdout,
    saved as `<store>/m2b-render-summary.json`) and the M2a
    `<store>/run-summary.json` are both pinned by sha256; the store name,
    import revision and render profile are read from them, never retyped.
    Refused: a summary of another store, one that is not a fresh render
    (`alignment_source` other than `fresh_render`, no derivation hash) and
    a store without its import summary."""
    profile = summary.get("render_profile") or {}
    if profile.get("alignment_source") != "fresh_render" or not profile.get("derivation_hash"):
        raise ReleaseError(
            f"--migration {summary_path}: not a fresh render summary "
            f"(render_profile {profile!r})"
        )
    rendered_store = Path(str(summary.get("store") or ""))
    if rendered_store.name != m2a_store.name:
        raise ReleaseError(
            f"--migration {summary_path}: summary of store {str(rendered_store)!r}, "
            f"release built from {m2a_store.name!r}"
        )
    import_path = m2a_store / "run-summary.json"
    try:
        imported = json.loads(import_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"--migration: cannot read {import_path}: {exc}") from exc
    return {
        "kind": "full_render",
        "store": m2a_store.name,
        "m2a_migration_version": imported.get("migration_version"),
        "import_profile": imported.get("profile"),
        "m2b_version": summary.get("migration_version"),
        "aligner_derivation_hash": profile["derivation_hash"],
        "render_profile": profile,
        "render_summary": _pinned_file(summary_path),
        "import_summary": _pinned_file(import_path),
    }


def _checked_result(record: dict[str, Any], store_dir: Path) -> dict[str, Any]:
    result = evaluate_document(record, store_dir)
    serialized = result.to_dict()
    schema_issues = validate_gate_result(serialized)
    if schema_issues:  # engine bug, never data to quarantine
        raise RuntimeError(
            "engine produced a schema-invalid gate result: "
            + "; ".join(f"{i.code}@{i.path}" for i in schema_issues)
        )
    return serialized


def _evaluate_one(
    args: tuple[Path, dict[str, Any], dict[str, Any] | None],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Gate one joined pair; module-level so it is picklable for the pool."""
    store_dir, index_record, renders_record = args
    serialized = _checked_result(adapt_record(index_record, renders_record), store_dir)
    if serialized["disposition"] != "active":
        # quarantined: a failed hard gate makes every task view ineligible,
        # so the node histogram is not needed to project it.
        serialized["task_eligibility"] = task_eligibility(serialized, {})
        return serialized, None
    envelope = index_record["envelope"]
    canonical = canonicalize(json.loads(read_object(store_dir, envelope["ast"])))
    manifest_records = json.loads(read_object(store_dir, envelope["resources_manifest"]))
    coverage = compute_coverage(canonical)
    # §33.4: eligibility is per TaskView. Attached AFTER schema validation
    # of the gate result — it is a projection of that result, not evidence.
    serialized["task_eligibility"] = task_eligibility(serialized, coverage["node_types"])
    # The canonical tree is NOT carried out of the worker: at ~1 MB per
    # document a 100k release would need tens of GB of resident memory.
    # Everything the release needs from it — the fingerprint, the coverage
    # counts, the AST version — is derived here and travels as a few
    # hundred bytes instead.
    return serialized, {
        "document_bundle_id": envelope["document_bundle_id"],
        "lineage_root_id": envelope["lineage_root_id"],
        "recipe_id": None,
        "eligibility": serialized["eligibility"],
        "fingerprint": build_fingerprint(
            canonical, [r["sha256"] for r in manifest_records]
        ),
        "ast_version": canonical.get("ast_version", "0.1.0"),
        "coverage": coverage,
    }


def evaluate_real_store(
    store_dir: Path,
    limit: int | None = None,
    *,
    joined: list[tuple[dict[str, Any], dict[str, Any] | None]] | None = None,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Gate the joined M2a+renders population.

    Returns ``(gate_results, active_records)`` where active_records are
    allocator records (recipe-less: ``recipe_id`` None) with
    fingerprints built from the hash-verified stored objects.

    `workers` > 1 spreads the per-document gate work over a process pool.
    Gating is pure per document, so results stay byte-identical to the
    serial path — `imap` preserves index order, which the allocator and
    the gate report both depend on."""
    store_dir = Path(store_dir)
    pairs = join_store_records(store_dir, limit) if joined is None else joined
    tasks = [(store_dir, index_record, renders) for index_record, renders in pairs]

    gate_results: list[dict[str, Any]] = []
    active_records: list[dict[str, Any]] = []

    def collect(outcomes: Iterable[tuple[dict[str, Any], dict[str, Any] | None]]) -> None:
        for done, (serialized, active) in enumerate(outcomes, start=1):
            gate_results.append(serialized)
            if active is not None:
                active_records.append(active)
            if done % 500 == 0:
                print(f"[m2c] gated {done}/{len(tasks)}", flush=True)

    if workers > 1:
        with multiprocessing.Pool(processes=workers) as pool:
            collect(pool.imap(_evaluate_one, tasks, chunksize=8))
    else:
        collect(_evaluate_one(task) for task in tasks)
    return gate_results, active_records


# ------------------------------------------------------- gate durability --
# §4.5 (DEC-023): a gate stage that survives a crash mid-run without
# splicing two different gate-engine code states into one report.

_GATE_ENGINE_SEED = Path("src/unflatten/dataset/gates/engine.py")
_GATE_CLOSURE_PREFIX = "unflatten."


def gate_code_digest(repo_root: Path | None = None) -> str:
    """sha256 over the FULL ``unflatten.`` import closure reachable from
    ``gates.engine`` (§4.5) — deliberately broad: the engine reaches into
    ``observations.sidecar``, ``observations.aligner``, ``targets.*``,
    ``fragments.*``, ``ast.*`` and ``renderer.*``, and a change anywhere
    in that closure can change gate verdicts. A resume whose stored
    digest disagrees must refuse rather than assume the code did not
    matter."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]
    repo_root = Path(repo_root)
    sources = closure([repo_root / _GATE_ENGINE_SEED], prefix=_GATE_CLOSURE_PREFIX)
    payload = "\x00".join(
        f"{path}\x00{content}" for path, content in sorted(sources.items())
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _file_sha256(path: Path) -> str:
    data = path.read_bytes() if path.exists() else b""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _gate_checkpoint_identity(
    m2a_store: Path,
    *,
    limit: int | None,
    exclude_scripts: frozenset[str] | set[str] | None,
    bundle_exclusions: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "index_sha256": _file_sha256(m2a_store / "index.jsonl"),
        "renders_sha256": _file_sha256(m2a_store / "renders.jsonl"),
        # The mask ledger is the third store input to a gated record
        # (see `_read_mask_ledger`). Unpinned, a resume after a mask
        # re-derivation would reuse results computed from the masks the
        # re-derivation replaced.
        "mask_projections_sha256": _file_sha256(m2a_store / "mask-projections.jsonl"),
        # The target-bundle ledger is a FOURTH store input to a gated
        # record (see `_read_target_ledger`) — unpinned, a resume after a
        # DEC-043 §1 re-derivation would reuse gate results computed from
        # the target bundles the re-derivation replaced.
        "target_bundles_sha256": _file_sha256(m2a_store / "target-bundles.jsonl"),
        "gate_result_schema_version": GATE_RESULT_SCHEMA_VERSION,
        "m2c_version": M2C_VERSION,
        "exclude_scripts": sorted(exclude_scripts) if exclude_scripts else [],
        "bundle_exclusions": [
            {"document_bundle_id": bundle_id, "reason": reason}
            for bundle_id, reason in sorted((bundle_exclusions or {}).items())
        ],
        "limit": limit,
        "gate_code_digest": gate_code_digest(),
    }


def evaluate_real_store_checkpointed(
    store_dir: Path,
    checkpoint_dir: Path,
    *,
    limit: int | None = None,
    joined: list[tuple[dict[str, Any], dict[str, Any] | None]] | None = None,
    workers: int = 1,
    exclude_scripts: frozenset[str] | set[str] | None = None,
    bundle_exclusions: dict[str, str] | None = None,
    progress_every: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Durable, resumable gate stage (§4.5).

    ``checkpoint_dir/gate-results.jsonl`` and ``active-records.jsonl``
    are appended per document with ``flush()``. ``identity.json`` pins
    the store bytes, the run configuration and ``gate_code_digest``; a
    resume whose identity disagrees refuses (``ReleaseError``) instead
    of assembling a report out of two different runs. Only documents
    missing from the checkpoint are (re)gated; the return preserves
    ``index.jsonl`` order regardless of how much of it was already done.
    """
    store_dir = Path(store_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    identity_path = checkpoint_dir / "identity.json"
    results_path = checkpoint_dir / "gate-results.jsonl"
    active_path = checkpoint_dir / "active-records.jsonl"

    identity = _gate_checkpoint_identity(
        store_dir,
        limit=limit,
        exclude_scripts=exclude_scripts,
        bundle_exclusions=bundle_exclusions,
    )
    if identity_path.is_file():
        prior_identity = json.loads(identity_path.read_bytes())
        if prior_identity != identity:
            raise ReleaseError(
                f"gate checkpoint {checkpoint_dir}: identity mismatch on resume "
                "— refusing to splice a gate report across two identities "
                f"(checkpointed {prior_identity}, current {identity})"
            )
    else:
        identity_path.write_bytes(to_canonical_json(identity))

    if joined is None:
        joined = join_store_records(store_dir, limit)

    done_results: dict[str, dict[str, Any]] = {}
    if results_path.is_file():
        for row in _read_jsonl(results_path):
            done_results[row["document_bundle_id"]] = row
    done_active: dict[str, dict[str, Any]] = {}
    if active_path.is_file():
        for row in _read_jsonl(active_path):
            done_active[row["document_bundle_id"]] = row

    pending = [
        (index_record, renders_record)
        for index_record, renders_record in joined
        if index_record["envelope"]["document_bundle_id"] not in done_results
    ]
    tasks = [
        (store_dir, index_record, renders_record)
        for index_record, renders_record in pending
    ]

    total = len(joined)
    processed = len(done_results)
    start = time.monotonic()

    with open(results_path, "ab") as results_f, open(active_path, "ab") as active_f:

        def persist(serialized: dict[str, Any], active: dict[str, Any] | None) -> None:
            nonlocal processed
            results_f.write(to_canonical_json(serialized) + b"\n")
            results_f.flush()
            done_results[serialized["document_bundle_id"]] = serialized
            if active is not None:
                active_f.write(to_canonical_json(active) + b"\n")
                active_f.flush()
                done_active[serialized["document_bundle_id"]] = active
            processed += 1
            if processed % progress_every == 0:
                elapsed = time.monotonic() - start
                rate = processed / elapsed if elapsed else 0.0
                eta_s = (total - processed) / rate if rate else float("inf")
                print(
                    f"[m2c] gated {processed}/{total} ({rate:.1f}/s, "
                    f"ETA {eta_s:.0f}s)",
                    flush=True,
                )

        if workers > 1 and tasks:
            with multiprocessing.Pool(processes=workers) as pool:
                for serialized, active in pool.imap(_evaluate_one, tasks, chunksize=8):
                    persist(serialized, active)
        else:
            for task in tasks:
                persist(*_evaluate_one(task))

    gate_results = [
        done_results[index_record["envelope"]["document_bundle_id"]]
        for index_record, _ in joined
    ]
    active_records = [
        done_active[bundle_id]
        for bundle_id in (
            index_record["envelope"]["document_bundle_id"] for index_record, _ in joined
        )
        if bundle_id in done_active
    ]
    return gate_results, active_records


# -------------------------------------------------------------------- driver --


def build_release_candidate(
    m2a_store: Path,
    synthetic_dir: Path | None,
    out_dir: Path,
    *,
    limit: int | None = None,
    workers: int = 1,
    exclude_scripts: frozenset[str] | set[str] | None = None,
    bundle_exclusions: dict[str, str] | None = None,
    pin_splits_dir: Path | None = None,
    supersedes_reason: str | None = None,
    migration: dict[str, Any] | None = None,
    gate_checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Gate + split + assemble a dev release candidate; returns a summary.

    `exclude_scripts` drops documents whose dominant writing script is in
    the set BEFORE gating, so they never enter the release population.
    The store keeps them — this is a release admission policy, not a
    deletion — and the excluded ids are written to
    `out_dir/quality/script-exclusions.jsonl` as provenance.

    `bundle_exclusions` is the equivalent explicit, named admission policy
    for a pinned set of bundle ids. Every requested id must exist in this
    build's joined population; a stale or mistyped exclusion list is refused
    rather than silently excluding fewer documents than declared.

    `pin_splits_dir` (§4, spec-zarovnanie-korpusu) pins the new release's
    split allocation to a prior release directory (`load_prior_assignment`);
    `supersedes_reason` and `migration` (the verified migration record,
    §4.4 — the CLI builds it from the align-only/render summary or
    `composition-manifest.json` via `build_migration_record`) must come
    with it — a caller pinning splits without recording WHY and how is
    exactly the silent migration RFC003 §46.2 forbids.

    `gate_checkpoint_dir` (§4.5) makes the gate stage resumable across a
    crash (`evaluate_real_store_checkpointed`); `None` (the default for
    this function — the CLI always sets one) keeps the original in-memory
    gate stage, unchanged."""
    if pin_splits_dir is not None and (supersedes_reason is None or migration is None):
        raise ReleaseError(
            "pin_splits_dir requires both supersedes_reason and migration "
            "— a pinned migration must record why and how (§46.2)"
        )
    m2a_store = Path(m2a_store)
    out_dir = Path(out_dir)

    joined = join_store_records(m2a_store, limit, excluded_bundles=set(bundle_exclusions or {}))
    bundle_exclusions = dict(bundle_exclusions or {})
    joined_ids = {
        index_record["envelope"]["document_bundle_id"]
        for index_record, _ in joined
    }
    missing_exclusions = sorted(set(bundle_exclusions) - joined_ids)
    if missing_exclusions:
        raise ReleaseError(
            "bundle exclusion list names ids outside this release population: "
            f"{missing_exclusions[:5]!r}"
        )
    bundle_exclusion_rows = [
        {"document_bundle_id": bundle_id, "reason": reason}
        for bundle_id, reason in sorted(bundle_exclusions.items())
    ]
    if bundle_exclusions:
        joined = [
            (index_record, renders_record)
            for index_record, renders_record in joined
            if index_record["envelope"]["document_bundle_id"] not in bundle_exclusions
        ]
        print(
            f"[m2c] bundle filter: excluded {len(bundle_exclusions)} "
            f"of {len(joined) + len(bundle_exclusions)} document(s)",
            flush=True,
        )
    exclusions: list[dict[str, str]] = []
    if exclude_scripts:
        kept = []
        for index_record, renders_record in joined:
            envelope = index_record["envelope"]
            ast = json.loads(read_object(m2a_store, envelope["ast"]))
            script = dominant_script(ast)
            if script in exclude_scripts:
                exclusions.append(
                    {
                        "document_bundle_id": envelope["document_bundle_id"],
                        "dominant_script": script,
                    }
                )
            else:
                kept.append((index_record, renders_record))
        print(
            f"[m2c] script filter: excluded {len(exclusions)} of {len(joined)} "
            f"document(s) ({', '.join(sorted(exclude_scripts))})",
            flush=True,
        )
        joined = kept
    if gate_checkpoint_dir is not None:
        real_results, real_records = evaluate_real_store_checkpointed(
            m2a_store,
            gate_checkpoint_dir,
            limit=limit,
            joined=joined,
            workers=workers,
            exclude_scripts=exclude_scripts,
            bundle_exclusions=bundle_exclusions,
        )
    else:
        real_results, real_records = evaluate_real_store(
            m2a_store, limit, joined=joined, workers=workers
        )
    lineage_by_bundle = {
        index_record["envelope"]["document_bundle_id"]: index_record["envelope"][
            "lineage_root_id"
        ]
        for index_record, _ in joined
    }
    component_lock_source = {
        "dataset_namespace": REAL_DATASET_NAMESPACE,
        "generator_version": REAL_GENERATOR_VERSION,
    }
    # the refs the release PINS: resolved once, here, from the ledger row
    # this run actually gated — a later append to renders.jsonl cannot
    # move an assembled release
    artifacts_by_bundle = {
        index_record["envelope"]["document_bundle_id"]: (
            (renders_record or {}).get("artifacts") or {}
        )
        for index_record, renders_record in joined
    }

    syn_results: list[dict[str, Any]] = []
    syn_records: list[dict[str, Any]] = []
    if synthetic_dir is not None:
        synthetic_dir = Path(synthetic_dir)
        for factory_record in _read_jsonl(synthetic_dir / "index.jsonl"):
            syn_result = _checked_result(factory_record, synthetic_dir)
            # Factory records are geometry-complete by construction, but
            # asserting that instead of computing it would be exactly the
            # silent assumption the gates exist to prevent.
            syn_ast = json.loads(
                read_object(synthetic_dir, factory_record["envelope"]["ast"])
            )
            syn_result["task_eligibility"] = task_eligibility(
                syn_result, compute_coverage(syn_ast)["node_types"]
            )
            syn_results.append(syn_result)
            lineage_by_bundle[
                factory_record["envelope"]["document_bundle_id"]
            ] = factory_record["envelope"]["lineage_root_id"]
        for doc in json.loads(
            (synthetic_dir / "run_manifest.json").read_bytes()
        )["documents"]:
            artifacts_by_bundle[doc["document_bundle_id"]] = doc.get("artifacts") or {}
        syn_records = load_eligible_records(synthetic_dir, syn_results)
        run_manifest = json.loads((synthetic_dir / "run_manifest.json").read_bytes())
        component_lock_source = {
            "dataset_namespace": run_manifest["dataset_namespace"],
            "generator_version": run_manifest["generator_version"],
        }

    gate_results = real_results + syn_results
    bundle_ids = [r["document_bundle_id"] for r in gate_results]
    if len(bundle_ids) != len(set(bundle_ids)):
        raise ReleaseError(
            "duplicate document_bundle_id across the combined population"
        )

    records = sorted(
        real_records + syn_records, key=lambda r: r["document_bundle_id"]
    )
    pinned_from_release_root_hash: str | None = None
    supersedes: dict[str, Any] | None = None
    prior_assignment: dict[str, str] | None = None
    if pin_splits_dir is not None:
        pin_splits_dir = Path(pin_splits_dir)
        prior_manifest = samples._release_manifest(pin_splits_dir)
        pinned_from_release_root_hash = prior_manifest["release_root_hash"]
        prior_assignment = load_prior_assignment(pin_splits_dir)
        supersedes = {
            "release_root_hash": pinned_from_release_root_hash,
            "release_dir_name": pin_splits_dir.name,
            "reason": supersedes_reason,
            "migration": migration,
        }
    split_started = time.monotonic()
    last_split_percent = -1

    def split_progress(done: int, total: int) -> None:
        nonlocal last_split_percent
        percent = done * 100 // total if total else 100
        if percent == last_split_percent and done != total:
            return
        last_split_percent = percent
        elapsed = time.monotonic() - split_started
        rate = done / elapsed if elapsed else 0.0
        eta = (total - done) / rate if rate else float("inf")
        print(
            f"[m2c] split pairs {done}/{total} ({percent}%, "
            f"workers={workers}, ETA {eta:.0f}s)",
            flush=True,
        )

    print(f"[m2c] split: {len(records)} records, workers={workers}", flush=True)
    assignment = allocate(
        records,
        prior_assignment=prior_assignment,
        pinned_from_release_root_hash=pinned_from_release_root_hash,
        workers=workers,
        progress=split_progress,
    )
    print(f"[m2c] assembly: workers={workers}", flush=True)
    manifest = assemble_release(
        synthetic_dir if synthetic_dir is not None else m2a_store,
        gate_results,
        assignment,
        out_dir,
        records=records,
        lineage_by_bundle=lineage_by_bundle,
        component_lock_source=component_lock_source,
        artifacts_by_bundle=artifacts_by_bundle,
        pinned_from_release_root_hash=pinned_from_release_root_hash,
        supersedes=supersedes,
        workers=workers,
    )

    # The gate report (with its per-task eligibility) is written BY
    # assemble_release and hashed into release_root_hash — what may
    # train is part of the release identity, not provenance dropped next
    # to it afterwards.
    gate_report_sha256 = manifest["gate_report"]["sha256"]
    if exclude_scripts:
        (out_dir / "quality" / "script-exclusions.jsonl").write_bytes(
            b"".join(to_canonical_json(r) + b"\n" for r in exclusions)
        )
    if bundle_exclusions:
        (out_dir / "quality" / "bundle-exclusions.jsonl").write_bytes(
            b"".join(to_canonical_json(r) + b"\n" for r in bundle_exclusion_rows)
        )

    eligibility_counts = {name: 0 for name in ELIGIBILITIES}
    task_counts = {kind: dict.fromkeys(ELIGIBILITIES, 0) for kind in TASK_KINDS}
    # documents held back because NOTHING on their pages was observed —
    # the release-side count of the pages the feature cache would skip.
    # The two numbers should track each other; a wide divergence is a
    # finding, not a rounding difference.
    no_observed_placement = 0
    for result in gate_results:
        eligibility_counts[result["eligibility"]] += 1
        views = result.get("task_eligibility") or {}
        for kind, view in views.items():
            task_counts[kind][view["eligibility"]] += 1
        if "NO_OBSERVED_PLACEMENT" in (
            views.get("structure_v0", {}).get("reason_codes") or []
        ):
            no_observed_placement += 1
    return {
        "m2c_version": M2C_VERSION,
        "out_dir": str(out_dir),
        "documents": {"real": len(real_results), "synthetic": len(syn_results)},
        "eligibility": eligibility_counts,
        "quarantined": sum(
            1 for r in gate_results if r["disposition"] == "quarantine"
        ),
        "quarantined_lineages": len(assignment.quarantined_lineages),
        "by_split_documents": manifest["counts"]["by_split_documents"],
        "partial_documents": manifest["counts"]["partial_documents"],
        "release_root_hash": manifest["release_root_hash"],
        "gate_report_sha256": gate_report_sha256,
        "script_excluded": len(exclusions),
        "bundle_excluded": len(bundle_exclusions),
        "no_observed_placement": no_observed_placement,
        "task_eligibility": task_counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.dataset.migration.m2c", description=__doc__
    )
    parser.add_argument("m2a_store", type=Path, help="M2a store root (read-only)")
    parser.add_argument("out_dir", type=Path, help="fresh release directory")
    parser.add_argument(
        "--synthetic", type=Path, default=None, help="synthetic factory output dir"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "processes for document gates, leakage comparisons and release "
            "serialization (default 1 = serial)"
        ),
    )
    parser.add_argument(
        "--exclude-scripts",
        default=None,
        help=(
            "comma-separated dominant scripts to keep out of the release "
            f"(e.g. arabic,hebrew); known: {','.join(sorted(SCRIPT_RANGES))}"
        ),
    )
    parser.add_argument(
        "--exclude-bundles",
        type=Path,
        default=None,
        metavar="JSONL",
        help="pinned JSONL rows with document_bundle_id and a named reason; "
        "every id must exist in the release population",
    )
    parser.add_argument(
        "--pin-splits",
        type=Path,
        default=None,
        metavar="RELEASE_DIR",
        help="pin the split allocation to a prior release directory (§4)",
    )
    parser.add_argument(
        "--supersedes-reason",
        default=None,
        help="required with --pin-splits: why this release supersedes it",
    )
    parser.add_argument(
        "--migration",
        type=Path,
        default=None,
        metavar="SUMMARY_JSON",
        help=(
            "required with --pin-splits: path to <store>/align-only-summary.json "
            "(§2.6), or for a re-imported, fully rendered store the saved M2b "
            "summary <store>/m2b-render-summary.json (kind full_render, pins it "
            "and <store>/run-summary.json), or <store>/composition-manifest.json "
            "(kind combined_store, verifies component/source/output provenance); "
            "m2c builds the migration record "
            "itself and pins the summary file by sha256"
        ),
    )
    parser.add_argument(
        "--gate-checkpoint",
        type=Path,
        default=None,
        metavar="DIR",
        help="gate-stage checkpoint dir (§4.5); default <out_dir>.gate/",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    exclude_scripts = None
    if args.exclude_scripts:
        exclude_scripts = {s.strip() for s in args.exclude_scripts.split(",") if s.strip()}
        unknown = exclude_scripts - set(SCRIPT_RANGES)
        if unknown:
            parser.error(f"--exclude-scripts: unknown script(s) {sorted(unknown)}")
    bundle_exclusions = (
        load_bundle_exclusions(args.exclude_bundles)
        if args.exclude_bundles is not None
        else None
    )
    if args.pin_splits is not None:
        # §OPEN-2 resolved 2026-09-21 (release-v14-synth-r3): the allocator
        # already handles this combination -- pass 1 inherits prior pins,
        # pass 2 routes newcomer (synthetic) lineages into the largest
        # deficit (allocator.py, covered by
        # test_newcomer_goes_to_largest_deficit_and_others_keep_split).
        # The empirical proof is the post-build split diff against the
        # pinned release (r3: zero moves on shared lineages); a diff any
        # owner cannot reproduce from these flags re-opens this.
        if args.supersedes_reason is None:
            parser.error("--pin-splits requires --supersedes-reason")
        if args.migration is None:
            parser.error("--pin-splits requires --migration")
    migration = None
    if args.migration is not None:
        migration = build_migration_record(args.migration, args.m2a_store)
    gate_checkpoint_dir = args.gate_checkpoint
    if gate_checkpoint_dir is None:
        gate_checkpoint_dir = args.out_dir.parent / (args.out_dir.name + ".gate")
    summary = build_release_candidate(
        args.m2a_store,
        args.synthetic,
        args.out_dir,
        limit=args.limit,
        workers=args.workers,
        exclude_scripts=exclude_scripts,
        bundle_exclusions=bundle_exclusions,
        pin_splits_dir=args.pin_splits,
        supersedes_reason=args.supersedes_reason,
        migration=migration,
        gate_checkpoint_dir=gate_checkpoint_dir,
    )
    print(json.dumps(summary, indent=1, sort_keys=True))
    admitted = sum((summary.get("by_split_documents") or {}).values())
    admitted += summary.get("partial_documents") or 0
    if admitted == 0:
        # v15, 2026-09-23: this returned 0 on a release whose every split
        # was empty, so the chain built bindings and started a cache
        # against a corpus that did not exist. An empty release is a
        # failed release, not a small one.
        print(
            "[m2c] REFUSED: no document was admitted to any split — "
            "an empty release cannot be built on",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
