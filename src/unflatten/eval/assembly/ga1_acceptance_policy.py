"""RFC005 §12.2 G-A1 acceptance policy (DEC-049 point 7, item 9.1, frozen
2026-09-27).

The floors gate three of the seven rule classes M1 measured
(`docs/plans/spec-s4-assembler-m1-results.md` §5/§8): `section`, `table`,
`list`. The other four (`paragraph`, `row`, `list_item`, `run`) are
reported, never gated -- DEC-049 does not freeze a floor for them (§8:
their B0 point is the "never join" base rate, a non-inferiority floor
only, T6 open). The class names and the `accuracy`/`n` fields this reads
are exactly `harness.compute_summary`'s
`policies[policy_name].classes[class]` shape; this module never
recomputes a metric, only gates one someone else already measured.

Like `unflatten.model.s4_selection_policy`, this policy carries no
embedded self-hash -- `ga1_acceptance_policy_hash` computes a
`domain_hash` over the whole validated document, recorded in the verdict
so a later reader can tell which frozen policy produced it without
re-deriving it.

Floors and margin are `_micro` integers (parts per 1e6), the same
convention `s4_selection_policy.py`'s `floor_micro` uses: `domain_hash`
serializes through the RFC004 §3.1 CBOR profile
(`rfc8949-core-det-nfc-int-v0`), which has no float type.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unflatten.model.hashing import domain_hash

GA1_ACCEPTANCE_POLICY_SCHEMA_VERSION = "1.0.0"

#: DEC-049 point 7, item 9.1: the only three rule classes with a frozen
#: floor. Matches `harness.compute_summary`'s per-policy `classes` keys.
GATED_CLASSES = ("section", "table", "list")

_REQUIRED_KEYS = frozenset(
    {
        "ga1_acceptance_policy_schema_version",
        "policy_name",
        "floors_micro",
        "margin_micro",
        "minimum_documents",
        "minimum_decisions_per_class",
        "is_placeholder",
    }
)

#: DEC-049 point 7, item 9.1 (frozen, not a placeholder): B0's M1
#: measurement on train was 0.991 / 0.913 / 0.847 (§5.1); these floors sit
#: 1-1.7 points below it for the train -> val gap (§8/§9.1).
FROZEN_GA1_ACCEPTANCE_POLICY: dict[str, Any] = {
    "ga1_acceptance_policy_schema_version": GA1_ACCEPTANCE_POLICY_SCHEMA_VERSION,
    "policy_name": "B0",
    "floors_micro": {"section": 980_000, "table": 900_000, "list": 830_000},
    "margin_micro": 0,
    "minimum_documents": 200,
    "minimum_decisions_per_class": 30,
    "is_placeholder": False,
}

DEFAULT_POLICY_PATH = Path(__file__).parent / "ga1_acceptance_policy.v1.json"

_GA1_POLICY_DOMAIN = "rfc005.ga1-acceptance-policy/v1"


class Ga1AcceptancePolicyInvalid(ValueError):
    """A G-A1 acceptance policy document is malformed."""


def validate_ga1_acceptance_policy(doc: Mapping[str, Any]) -> None:
    missing = _REQUIRED_KEYS - set(doc)
    if missing:
        raise Ga1AcceptancePolicyInvalid(
            f"g-a1 acceptance policy missing key(s): {sorted(missing)}"
        )
    if not isinstance(doc["policy_name"], str) or not doc["policy_name"]:
        raise Ga1AcceptancePolicyInvalid(
            "g-a1 acceptance policy policy_name must be a non-empty string"
        )
    floors = doc["floors_micro"]
    if not isinstance(floors, dict) or set(floors) != set(GATED_CLASSES):
        raise Ga1AcceptancePolicyInvalid(
            f"g-a1 acceptance policy floors_micro must carry exactly {sorted(GATED_CLASSES)}, "
            f"got {sorted(floors) if isinstance(floors, dict) else floors!r}"
        )
    for cls, floor in floors.items():
        if isinstance(floor, bool) or not isinstance(floor, int) or not 0 <= floor <= 1_000_000:
            raise Ga1AcceptancePolicyInvalid(
                f"g-a1 acceptance policy floors_micro[{cls!r}] must be an integer in "
                f"0..1000000, got {floor!r}"
            )
    margin = doc["margin_micro"]
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise Ga1AcceptancePolicyInvalid(
            f"g-a1 acceptance policy margin_micro must be an integer >= 0, got {margin!r}"
        )
    for key in ("minimum_documents", "minimum_decisions_per_class"):
        value = doc[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise Ga1AcceptancePolicyInvalid(
                f"g-a1 acceptance policy {key} must be an integer >= 1, got {value!r}"
            )
    if not isinstance(doc["is_placeholder"], bool):
        raise Ga1AcceptancePolicyInvalid(
            "g-a1 acceptance policy is_placeholder must be a bool"
        )


def load_ga1_acceptance_policy(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Ga1AcceptancePolicyInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise Ga1AcceptancePolicyInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_ga1_acceptance_policy(payload)
    return payload


def ga1_acceptance_policy_hash(policy: Mapping[str, Any]) -> str:
    """`domain_hash` over the whole validated policy document."""
    validate_ga1_acceptance_policy(policy)
    return domain_hash(_GA1_POLICY_DOMAIN, dict(policy))


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def evaluate_acceptance(summary: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """PASS / FAIL / `insufficient_evidence` per gated class, plus the hard
    zeros (invariant violations, property conflicts), from one
    `harness.compute_summary` output for `policy['policy_name']`.

    A class is `insufficient_evidence` when the run has fewer than
    `minimum_documents` done documents overall, or fewer than
    `minimum_decisions_per_class` scored decisions of that class (DEC-049
    point 7, item 9.1) -- checked before the floor comparison, so a class
    never PASSes or FAILs on too little evidence. The overall verdict is
    FAIL if any hard zero is violated or any gated class FAILs, else
    `insufficient_evidence` if any gated class lacks evidence, else PASS.
    """
    validate_ga1_acceptance_policy(policy)
    policy_outcome = (summary.get("policies") or {}).get(policy["policy_name"]) or {}
    documents_done = (summary.get("documents") or {}).get("done", 0)
    all_classes = policy_outcome.get("classes") or {}
    classes: dict[str, Any] = {}
    class_statuses: set[str] = set()
    for cls, floor_micro in policy["floors_micro"].items():
        cell = all_classes.get(cls)
        n = cell["n"] if cell else 0
        accuracy = cell["accuracy"] if cell else None
        below_minimum = (
            documents_done < policy["minimum_documents"]
            or n < policy["minimum_decisions_per_class"]
        )
        if below_minimum:
            status = "insufficient_evidence"
        else:
            effective_floor = (floor_micro - policy["margin_micro"]) / 1_000_000
            status = "PASS" if accuracy is not None and accuracy >= effective_floor else "FAIL"
        class_statuses.add(status)
        classes[cls] = {
            "status": status, "accuracy": accuracy, "floor_micro": floor_micro, "n": n,
        }
    invariant_violations = policy_outcome.get("invariant_violations", 0)
    conflicts = policy_outcome.get("conflicts", 0)
    hard_zero_ok = invariant_violations == 0 and conflicts == 0
    if not hard_zero_ok or "FAIL" in class_statuses:
        overall = "FAIL"
    elif "insufficient_evidence" in class_statuses:
        overall = "insufficient_evidence"
    else:
        overall = "PASS"
    return {
        "policy_hash": ga1_acceptance_policy_hash(policy),
        "documents_done": documents_done,
        "classes": classes,
        "hard_zeros": {
            "invariant_violations": invariant_violations,
            "conflicts": conflicts,
            "status": "PASS" if hard_zero_ok else "FAIL",
        },
        "verdict": overall,
    }
