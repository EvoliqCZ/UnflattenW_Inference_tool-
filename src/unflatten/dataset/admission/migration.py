"""RFC005 §9.4 — the migration report for the existing corpus.

§9.4 is blunt about what this is for: *until the report exists, the count
is only an inventory count, not a training count*. So the report ends on
exactly that arithmetic — `inventory_count` (documents the store holds)
against `training_count` (documents admission actually admits) — and
`training_count` is `None`, with a stated reason, whenever admission did
not conclude. A blocked admission does not get to publish a training
count; that is how an inventory number becomes a training number by
accident.

All eleven §9.4 lines are computed from the store index, the release and
its gate report. Line 11 (legacy checkpoint exposure) comes from
`ancestry.py`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from unflatten.dataset.admission.cohort import Cohort
from unflatten.dataset.admission.evidence import ReleaseEvidence
from unflatten.dataset.gates.engine import GATE_IDS
from unflatten.dataset.gates.task_eligibility import TASK_KINDS
from unflatten.dataset.samples import _privacy_pending

#: §9.4 lines, in the order the RFC lists them.
MIGRATION_LINES: tuple[str, ...] = (
    "source_inventory",
    "rights_privacy_decisions",
    "lineage_reconstruction",
    "ast_profile_compatibility",
    "renderer_observation_gates",
    "exact_near_deduplication",
    "split_assignment",
    "taskview_eligibility",
    "coverage_vector",
    "quarantine_distribution",
    "legacy_checkpoint_exposure_overlap",
)

_RENDER_GATES: tuple[str, ...] = (
    "compile",
    "build_determinism",
    "text_conservation",
    "no_silent_loss",
    "approximations",
    "observation",
    "targets",
)
_AST_GATES: tuple[str, ...] = ("ast_schema", "ast_semantics", "profile", "canonicalization")


def _envelope(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("envelope") or {}


def _gate_histogram(
    ev: ReleaseEvidence, gate_ids: tuple[str, ...]
) -> dict[str, dict[str, int]]:
    histogram: dict[str, Counter] = {gate: Counter() for gate in gate_ids}
    for row in ev.gate_rows:
        for check in row.get("gate_report", []):
            gate = check.get("gate_id")
            if gate in histogram:
                histogram[gate][check.get("status")] += 1
    return {gate: dict(counter) for gate, counter in histogram.items()}


def _source_inventory(store_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "documents": len(store_index),
        "by_source_origin": dict(
            Counter(_envelope(r).get("source_origin") for r in store_index.values())
        ),
        "by_verification_tier": dict(
            Counter(_envelope(r).get("verification_tier") for r in store_index.values())
        ),
        "by_label_grade": dict(
            Counter(_envelope(r).get("label_grade") for r in store_index.values())
        ),
        "by_import_status": dict(
            Counter((r.get("import") or {}).get("status") for r in store_index.values())
        ),
    }


def _rights_privacy(
    ev: ReleaseEvidence, store_index: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Release-recorded decisions, plus any divergence in the live store.

    The release is the training authority (§9.2): a store that has since
    recorded a decision does not retroactively admit an already assembled
    release. The divergence is therefore reported as the reason a NEW
    release is needed, never applied to this one.
    """
    release_pending = sum(1 for row in ev.gate_rows if _privacy_pending(row))
    diverged = 0
    store_states: Counter = Counter()
    for row in ev.gate_rows:
        record = store_index.get(row["document_bundle_id"])
        if record is None:
            continue
        # Synthetic-origin records carry no rights block by design
        # (the gates auto-pass them on source_origin); an absent
        # decision is a measured state, not a missing key — None would
        # crash json.dumps(sort_keys=True) on mixed None/str keys.
        state = (record.get("rights") or {}).get("privacy_review")
        store_states[state if state is not None else "unrecorded"] += 1
        if _privacy_pending(row) and state in ("approved", "not_required"):
            diverged += 1
    return {
        "authority": "release gate report",
        "documents": len(ev.gate_rows),
        "release_privacy_review_incomplete": release_pending,
        "release_training_allowed": sum(
            1
            for row in ev.gate_rows
            for check in row.get("gate_report", [])
            if check.get("gate_id") == "rights"
            and (check.get("details") or {}).get("training_allowed") is True
        ),
        "store_privacy_review_states": dict(store_states),
        "store_divergence": {
            "documents": diverged,
            "meaning": (
                "the store records a closed privacy review for documents whose "
                "release gate report still records it as pending; only a NEW "
                "release can carry that decision (RFC003 §46.2 immutability)"
            ),
        },
    }


def _lineage(
    ev: ReleaseEvidence, store_index: dict[str, dict[str, Any]], conditions: dict[int, dict]
) -> dict[str, Any]:
    lineages = {_envelope(r).get("lineage_root_id") for r in store_index.values()}
    return {
        "store_lineages": len(lineages - {None}),
        "documents_without_lineage": sum(
            1 for r in store_index.values() if not _envelope(r).get("lineage_root_id")
        ),
        "release_members": len(ev.membership_rows),
        "ancestry_condition": conditions.get(9, {}).get("measured"),
    }


def _dedup(ev: ReleaseEvidence, store_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hashes = Counter(
        _envelope(r).get("ast_hash") for r in store_index.values() if _envelope(r).get("ast_hash")
    )
    return {
        "distinct_ast_hashes": len(hashes),
        "exact_duplicate_groups_in_store": sum(1 for n in hashes.values() if n > 1),
        "near_duplicate_lineages_quarantined": sum(
            1
            for row in ev.quarantine
            if "NEAR_DUPLICATE_COLLISION" in (row.get("reason_codes") or [])
        ),
        "not_recomputed": ["near_duplicate_fingerprint_graph"],
    }


def _split_assignment(ev: ReleaseEvidence, cohort: Cohort) -> dict[str, Any]:
    by_split = Counter(split for _b, split, _l, _f in ev.membership_rows)
    return {
        "by_split": dict(by_split),
        "by_manifest_file": dict(Counter(f for _b, _s, _l, f in ev.membership_rows)),
        "train_candidates": len(cohort.candidates),
        "train_admitted": len(cohort.admitted),
    }


def _taskview_eligibility(ev: ReleaseEvidence) -> dict[str, Any]:
    per_kind: dict[str, Counter] = {kind: Counter() for kind in TASK_KINDS}
    for row in ev.gate_rows:
        for kind, view in (row.get("task_eligibility") or {}).items():
            per_kind.setdefault(kind, Counter())[(view or {}).get("eligibility")] += 1
    return {kind: dict(counter) for kind, counter in per_kind.items()}


def _coverage_vector(coverage: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_types_below_minima": [
            row["node_type"] for row in coverage["node_type_minima"] if not row["meets_minima"]
        ],
        "failing_criteria": coverage["failing_criteria"],
        "permitted_run_classes": coverage["permitted_run_classes"],
    }


def _quarantine(ev: ReleaseEvidence) -> dict[str, Any]:
    reasons: Counter = Counter()
    sources: Counter = Counter()
    for row in ev.quarantine:
        sources[row.get("source")] += 1
        for code in row.get("reason_codes") or []:
            reasons[code] += 1
    return {
        "entries": len(ev.quarantine),
        "by_reason_code": dict(reasons),
        "by_source": dict(sources),
    }


def migration_report(
    ev: ReleaseEvidence,
    store_index: dict[str, dict[str, Any]],
    cohort: Cohort,
    conditions: list[dict[str, Any]],
    coverage: dict[str, Any],
    ancestry: dict[str, Any],
    dataset_outcome: str,
    blocking_reasons: list[str],
) -> dict[str, Any]:
    """The §9.4 migration report, with its inventory-vs-training arithmetic."""
    by_row = {row["row"]: row for row in conditions}
    lines = {
        "source_inventory": _source_inventory(store_index),
        "rights_privacy_decisions": _rights_privacy(ev, store_index),
        "lineage_reconstruction": _lineage(ev, store_index, by_row),
        "ast_profile_compatibility": {
            "component_lock": ev.manifest.get("component_lock"),
            "gates": _gate_histogram(ev, _AST_GATES),
        },
        "renderer_observation_gates": {"gates": _gate_histogram(ev, _RENDER_GATES)},
        "exact_near_deduplication": _dedup(ev, store_index),
        "split_assignment": _split_assignment(ev, cohort),
        "taskview_eligibility": _taskview_eligibility(ev),
        "coverage_vector": _coverage_vector(coverage),
        "quarantine_distribution": _quarantine(ev),
        "legacy_checkpoint_exposure_overlap": {
            "legacy_ancestry": ancestry["value"],
            "exposed_documents": 0 if ancestry["value"] is False else None,
            "evidence": ancestry["evidence"],
        },
    }
    admitted = dataset_outcome == "promoted"
    return {
        "gates_reported": list(GATE_IDS),
        "lines": {name: lines[name] for name in MIGRATION_LINES},
        "inventory_count": len(store_index),
        "training_count": len(cohort.admitted) if admitted else None,
        "training_count_refused_because": None if admitted else blocking_reasons,
    }
