"""The cohort the §9.2 conditions are measured over.

"Admitted" is deliberately narrow: a document counts only if it is in the
train split, qualifies for at least one task view, its rights and privacy
decisions are complete, and no hard gate failed. Nothing is admitted by
omission — every excluded document lands in a named counter instead.

The gate-report accessors live here too, because both the conditions and
the migration report read the same recorded evidence and must read it the
same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unflatten.dataset.admission.evidence import ReleaseEvidence
from unflatten.dataset.gates.task_eligibility import TASK_KINDS
from unflatten.dataset.samples import _privacy_pending

#: Gates that must have judged real evidence, not merely not failed
#: (`gates/task_eligibility.py` `PREREQUISITE_GATES`).
PREREQUISITE_GATES: tuple[str, ...] = ("integrity", "observation", "targets")


@dataclass(frozen=True)
class Cohort:
    """Who the §9.2 rows are measured over."""

    #: Train-split documents qualifying for >= 1 task view (the candidates).
    candidates: tuple[str, ...]
    #: Candidates whose rights/privacy/gates are all clean (the admitted).
    admitted: tuple[str, ...]
    #: CANDIDATES whose `full_ast_exact_v0` view is train_ready. Scoped to
    #: candidates, not to the admitted set: §9.2 row 6 asks what the
    #: full-AST targets contain, and that answer must not become vacuous
    #: because a different condition (privacy, say) emptied the admitted
    #: set first.
    full_ast: tuple[str, ...]
    by_task_kind: dict[str, int]
    excluded: dict[str, int]


def views(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return row.get("task_eligibility") or {}


def train_ready_kinds(row: dict[str, Any]) -> set[str]:
    return {
        kind
        for kind, view in views(row).items()
        if (view or {}).get("eligibility") == "train_ready"
    }


def statuses(row: dict[str, Any]) -> dict[str, str]:
    return {c.get("gate_id"): c.get("status") for c in row.get("gate_report", [])}


def gate_details(row: dict[str, Any], gate_id: str) -> dict[str, Any]:
    for check in row.get("gate_report", []):
        if check.get("gate_id") == gate_id:
            return check.get("details") or {}
    return {}


def hard_gates_clean(row: dict[str, Any]) -> bool:
    recorded = statuses(row)
    if any(status == "fail" for status in recorded.values()):
        return False
    return all(recorded.get(gate) == "pass" for gate in PREREQUISITE_GATES)


def rights_ok(row: dict[str, Any]) -> bool:
    return gate_details(row, "rights").get("training_allowed") is True


def build_cohort(evidence: ReleaseEvidence) -> Cohort:
    """The train cohort and the subset that admission actually admits."""
    membership = evidence.membership
    rows = evidence.gate_rows_by_bundle
    candidates: list[str] = []
    admitted: list[str] = []
    full_ast: list[str] = []
    by_task_kind = dict.fromkeys(TASK_KINDS, 0)
    excluded = {
        "rights_not_allowed": 0,
        "privacy_review_incomplete": 0,
        "failed_hard_gate": 0,
        "no_gate_row": 0,
    }
    for bundle_id, (split, _lineage) in sorted(membership.items()):
        if split != "train":
            continue
        row = rows.get(bundle_id)
        if row is None:
            excluded["no_gate_row"] += 1
            continue
        kinds = train_ready_kinds(row)
        if not kinds:
            continue
        candidates.append(bundle_id)
        if "full_ast_exact_v0" in kinds:
            full_ast.append(bundle_id)
        blocked = False
        if not rights_ok(row):
            excluded["rights_not_allowed"] += 1
            blocked = True
        if _privacy_pending(row):
            excluded["privacy_review_incomplete"] += 1
            blocked = True
        if not hard_gates_clean(row):
            excluded["failed_hard_gate"] += 1
            blocked = True
        if blocked:
            continue
        admitted.append(bundle_id)
        for kind in kinds:
            by_task_kind[kind] = by_task_kind.get(kind, 0) + 1
    return Cohort(
        candidates=tuple(candidates),
        admitted=tuple(admitted),
        full_ast=tuple(full_ast),
        by_task_kind=by_task_kind,
        excluded=excluded,
    )


