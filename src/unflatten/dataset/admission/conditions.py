"""RFC005 §9.2 — the nine conditions that BLOCK quality training.

Seven of the nine are already decided by the gate engine, per document,
and recorded in the release's gate report. This module does not re-run a
gate: it aggregates what the release already says, over the cohort that
would actually carry gradient, and reports the count.

Two need care.

**Row 1** asks about a TaskView. No TaskView object is materialized in
this pipeline (`task_view_id` is null — `dataset/fragments/identity.py`),
so `training_allowed` can only be read at DocumentBundle granularity and
widened to that document's task views. The widening is recorded on the
row as an explicit approximation; it is never applied silently.

**Row 6** is measured by reading target bundles (`supervision.py`), and
is `unmeasured` — not `pass` — when that scan did not run.

The admitted cohort is deliberately narrow: a document counts as admitted
only if it is in the train split, qualifies for at least one task view,
its rights and privacy decisions are complete and no hard gate failed.
Everything else is reported as a count, not admitted by omission.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from unflatten.dataset.admission.cohort import (
    Cohort,
    gate_details,
    hard_gates_clean,
    rights_ok,
    statuses,
    train_ready_kinds,
)
from unflatten.dataset.admission.evidence import ReleaseEvidence
from unflatten.dataset.gates.engine import AST_VERSION_PREFIX, SUPPORTED_PROFILES
from unflatten.dataset.samples import _privacy_pending
from unflatten.dataset.split.release import PINNED_ARTIFACT_LANES

#: The nine §9.2 conditions, in RFC order. Each is stated as the thing
#: that BLOCKS: a row `pass`es when the blocking condition is absent.
CONDITIONS: tuple[tuple[int, str], ...] = (
    (1, "a used TaskView does not have training_allowed = true"),
    (2, "a rights or privacy decision is incomplete"),
    (3, "the dataset release is not immutable"),
    (4, "a cross-split lineage/leakage collision exists"),
    (5, "a TRAIN_READY TaskView did not pass 100 % of its applicable hard gates"),
    (6, "full_ast_exact_v0 carries required unknown or imputed_for_render supervision"),
    (7, "frozen pixels/targets cannot be deterministically re-verified"),
    (8, "the selected profile carries an unsupported visible V1/V2 feature"),
    (9, "input ancestry or fragment ownership is not unambiguous"),
)

_TASKVIEW_APPROXIMATION = {
    "granularity": "document_bundle",
    "reason": (
        "no TaskView object is materialized in this pipeline — task_view_id is "
        "null (unflatten/dataset/fragments/identity.py), so per-task rights "
        "cannot be read from a TaskView"
    ),
    "widening": (
        "the DocumentBundle's rights decision is attributed to every task view "
        "of that document"
    ),
}


def _row(
    number: int,
    status: str,
    measured: dict[str, Any],
    evidence: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "row": number,
        "condition": dict(CONDITIONS)[number],
        "status": status,
        "measured": measured,
        "evidence": evidence,
        "approximation": extra.get("approximation"),
        "not_recomputed": extra.get("not_recomputed", []),
    }


def _verdict(*counts: int) -> str:
    return "fail" if any(counts) else "pass"


def _row1(ev: ReleaseEvidence, cohort: Cohort) -> dict[str, Any]:
    rows = ev.gate_rows_by_bundle
    denied = sum(1 for b in cohort.candidates if not rights_ok(rows[b]))
    return _row(
        1,
        _verdict(denied),
        {"candidate_documents": len(cohort.candidates), "training_not_allowed": denied},
        "gate report `rights` gate details.training_allowed",
        approximation=_TASKVIEW_APPROXIMATION,
    )


def _row2(ev: ReleaseEvidence, cohort: Cohort) -> dict[str, Any]:
    rows = ev.gate_rows_by_bundle
    pending = sum(1 for b in cohort.candidates if _privacy_pending(rows[b]))
    rights_gate = sum(
        1 for b in cohort.candidates if statuses(rows[b]).get("rights") != "pass"
    )
    missing = sum(
        1 for b in cohort.candidates if not gate_details(rows[b], "rights")
    )
    return _row(
        2,
        _verdict(pending, rights_gate, missing),
        {
            "candidate_documents": len(cohort.candidates),
            "privacy_review_incomplete": pending,
            "rights_gate_not_pass": rights_gate,
            "rights_decision_missing": missing,
        },
        "gate report `privacy` gate (PRIVACY_REVIEW_PENDING) and `rights` gate",
    )


def _row3(immutability: dict[str, Any]) -> dict[str, Any]:
    failed = [c["name"] for c in immutability["checks"] if c["status"] == "fail"]
    return _row(
        3,
        "fail" if failed else "pass",
        {"checks": len(immutability["checks"]), "failed": failed},
        "release_root_hash recompute + per-artifact sha256/size (dataset.samples)",
    )


def _row4(ev: ReleaseEvidence, store_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_lineage: dict[str, set[str]] = {}
    by_bundle: dict[str, set[str]] = {}
    for bundle_id, split, lineage, _file in ev.membership_rows:
        by_lineage.setdefault(lineage, set()).add(split)
        by_bundle.setdefault(bundle_id, set()).add(split)
    quarantined = {
        row.get("lineage_root_id") for row in ev.quarantine if row.get("lineage_root_id")
    } | {
        row.get("document_bundle_id")
        for row in ev.quarantine
        if row.get("document_bundle_id")
    }
    leaked = sorted(
        {
            bundle_id
            for bundle_id, split, lineage, _file in ev.membership_rows
            if bundle_id in quarantined or lineage in quarantined
        }
    )
    by_ast_hash: dict[str, set[str]] = {}
    for bundle_id, (split, _lineage) in ev.membership.items():
        envelope = (store_index.get(bundle_id) or {}).get("envelope") or {}
        ast_hash = envelope.get("ast_hash")
        if ast_hash:
            by_ast_hash.setdefault(ast_hash, set()).add(split)
    duplicates = sum(1 for splits in by_ast_hash.values() if len(splits) > 1)
    multi_lineage = sum(1 for splits in by_lineage.values() if len(splits) > 1)
    multi_bundle = sum(1 for splits in by_bundle.values() if len(splits) > 1)
    return _row(
        4,
        _verdict(multi_lineage, multi_bundle, duplicates, len(leaked)),
        {
            "lineages": len(by_lineage),
            "lineages_in_multiple_splits": multi_lineage,
            "bundles_in_multiple_splits": multi_bundle,
            "exact_ast_duplicates_across_splits": duplicates,
            "quarantined_members_present_in_a_split": leaked[:20],
            "quarantined_identifiers": len(quarantined),
        },
        "split manifests + quarantine index + store ast_hash (§43.11/§43.12)",
        not_recomputed=["near_duplicate_fingerprint_graph"],
    )


def _row5(ev: ReleaseEvidence) -> dict[str, Any]:
    offenders = 0
    scope = 0
    for row in ev.gate_rows:
        if not train_ready_kinds(row):
            continue
        scope += 1
        if not hard_gates_clean(row):
            offenders += 1
    return _row(
        5,
        _verdict(offenders),
        {"train_ready_documents": scope, "train_ready_with_failed_gate": offenders},
        "gate report per-gate statuses, 100 % rule (RFC003 §34/§34.6)",
    )


def _row7(ev: ReleaseEvidence, cohort: Cohort, store_dir: Path, scan: dict) -> dict:
    rows = ev.gate_rows_by_bundle
    missing_ref = 0
    wrong_size = 0
    missing_lane = 0
    for bundle_id in cohort.candidates:
        artifacts = ev.artifacts.get(bundle_id) or {}
        for lane in PINNED_ARTIFACT_LANES:
            refs = artifacts.get(lane)
            if refs is None:
                missing_lane += 1
                continue
            for ref in refs if isinstance(refs, list) else [refs]:
                path = store_dir / ref.get("ref", "")
                if not path.is_file():
                    missing_ref += 1
                elif path.stat().st_size != ref.get("size_bytes"):
                    wrong_size += 1
    nondeterministic = sum(
        1
        for b in cohort.candidates
        if statuses(rows[b]).get("canonicalization") != "pass"
        or statuses(rows[b]).get("build_determinism") not in ("pass", "not_applicable")
    )
    hash_verified = ["target_bundle"] if scan["measured"]["bundles_scanned"] else []
    return _row(
        7,
        _verdict(missing_ref, wrong_size, missing_lane, nondeterministic),
        {
            "candidate_documents": len(cohort.candidates),
            "pinned_refs_missing": missing_ref,
            "pinned_refs_wrong_size": wrong_size,
            "pinned_lanes_absent": missing_lane,
            "determinism_gate_not_pass": nondeterministic,
            "lanes_hash_verified": hash_verified,
        },
        "pinned artifact index resolved against the content-addressed store "
        "+ `canonicalization`/`build_determinism` gates",
        not_recomputed=[
            lane for lane in PINNED_ARTIFACT_LANES if lane not in hash_verified
        ],
    )


def _row8(ev: ReleaseEvidence, cohort: Cohort) -> dict[str, Any]:
    rows = ev.gate_rows_by_bundle
    failed = sum(1 for b in cohort.candidates if statuses(rows[b]).get("profile") != "pass")
    lock = ev.manifest.get("component_lock") or {}
    ast_version = lock.get("ast_version", "")
    version_ok = str(ast_version).startswith(AST_VERSION_PREFIX)
    return _row(
        8,
        _verdict(failed, 0 if version_ok else 1),
        {
            "candidate_documents": len(cohort.candidates),
            "profile_gate_not_pass": failed,
            "profiles": sorted(SUPPORTED_PROFILES),
            "component_lock_ast_version": ast_version,
        },
        "gate report `profile` gate + release component_lock",
    )


def _row9(ev: ReleaseEvidence, cohort: Cohort, store_index: dict) -> dict[str, Any]:
    rows = ev.gate_rows_by_bundle
    membership = ev.membership
    no_record = 0
    lineage_mismatch = 0
    no_fragment_index = 0
    for bundle_id in cohort.candidates:
        record = store_index.get(bundle_id)
        if record is None:
            no_record += 1
            continue
        envelope = record.get("envelope") or {}
        if envelope.get("lineage_root_id") != membership[bundle_id][1]:
            lineage_mismatch += 1
        if (ev.artifacts.get(bundle_id) or {}).get("fragment_index") is None:
            no_fragment_index += 1
    targets_not_pass = sum(
        1 for b in cohort.candidates if statuses(rows[b]).get("targets") != "pass"
    )
    return _row(
        9,
        _verdict(no_record, lineage_mismatch, no_fragment_index, targets_not_pass),
        {
            "candidate_documents": len(cohort.candidates),
            "documents_without_a_store_record": no_record,
            "lineage_disagreement_release_vs_store": lineage_mismatch,
            "documents_without_a_pinned_fragment_index": no_fragment_index,
            "targets_gate_not_pass": targets_not_pass,
        },
        "store envelope lineage_root_id vs split manifest + pinned fragment index "
        "(RFC003-ERRATA-001 E1/E4)",
    )


def dataset_conditions(
    ev: ReleaseEvidence,
    store_index: dict[str, dict[str, Any]],
    cohort: Cohort,
    store_dir: Path,
    immutability: dict[str, Any],
    scan: dict[str, Any],
) -> list[dict[str, Any]]:
    """All nine §9.2 rows, with measured values."""
    return [
        _row1(ev, cohort),
        _row2(ev, cohort),
        _row3(immutability),
        _row4(ev, store_index),
        _row5(ev),
        _row(
            6,
            scan["status"],
            scan["measured"],
            "target-bundle scan for known_mask=false / origin=imputed_for_render "
            "(RFC003 §34.2)",
        ),
        _row7(ev, cohort, store_dir, scan),
        _row8(ev, cohort),
        _row9(ev, cohort, store_index),
    ]


def unmeasured_conditions(
    reason: str, immutability: dict[str, Any]
) -> list[dict[str, Any]]:
    """The nine rows when the release itself did not verify.

    Row 3 is still ANSWERED — "the release is not immutable" is exactly
    what was measured. The other eight are `unmeasured`, because a
    release whose identity does not recompute cannot be evidence for
    anything, and reporting them as clean would be the vacuous pass this
    report exists to prevent.
    """
    return [
        _row3(immutability)
        if number == 3
        else _row(number, "unmeasured", {"reason": reason}, "not measured")
        for number, _text in CONDITIONS
    ]
