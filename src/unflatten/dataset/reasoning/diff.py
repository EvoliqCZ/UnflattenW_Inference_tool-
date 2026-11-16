"""D5 — `candidate-diff-policy` framework (RFC003-ERRATA-003 E3, §30.12).

Compares a model candidate (a Prediction IR / candidate typed target from a
sealed decode) against the gold typed target and emits typed defects. This
module is the FRAMEWORK the errata ratifies: a versioned policy plug-in
protocol over a pluggable defect register (OPEN-003-E3-1), mask
withdrawal against the GOLD field only (E3 rule 2), deterministic
ordering, and the structural guarantee of E3 rule 1 — the candidate's own
value never leaves this module as anything but a comparison operand, it
is not a field on :class:`DefectTarget`. The defect-type register itself
is supplied by the caller's policy plug-in; none is chosen here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from unflatten.dataset.identity import domain_hash

#: Identity domain for a diff policy's own hash.
DIFF_POLICY_HASH_DOMAIN = "rfc003.candidate-diff-policy/v1"

#: Identity domain for the deterministic hash of an emitted defect list.
DEFECT_TARGETS_HASH_DOMAIN = "rfc003.reasoning-defect-targets/v1"


@dataclass(frozen=True)
class RawDefect:
    """One candidate defect, as a `DiffPolicy` emits it.

    `node_id`/`node_property_pointer` locate the GOLD field this defect is
    about; `claim_known`/`claim_observable` are the policy's own upper
    bound (default: no additional restriction) — `compare` ANDs them with
    the gold field's own masks, so a policy can never flag a defect on a
    property gold does not supervise (E3 rule 2).
    """

    node_id: str
    node_property_pointer: str
    defect_type: str
    operation: str
    claim_known: bool = True
    claim_observable: bool = True


@dataclass(frozen=True)
class DefectTarget:
    """One emitted §30.12 typed defect. Carries no candidate value — only
    the locator, the defect type and the required operation (E3 rule 1:
    the candidate is an input, never a target)."""

    node_id: str
    node_property_pointer: str
    defect_type: str
    operation: str
    known_mask: bool
    observable_mask: bool
    supervision_mask: bool


class DiffPolicy(Protocol):
    """A versioned plug-in supplying the OPEN-003-E3-1 defect register."""

    policy_id: str
    policy_version: str

    def diff(
        self, gold_typed: dict[str, Any], candidate_typed: dict[str, Any]
    ) -> list[RawDefect]:
        """Return every candidate defect BEFORE mask withdrawal.

        `candidate_typed` is read only as a comparison operand (E3 rule 1);
        nothing from it may appear in the returned `RawDefect`s beyond the
        locator/defect_type/operation the policy derives from the
        comparison.
        """
        ...


def diff_policy_hash(policy: DiffPolicy) -> str:
    """`H(DIFF_POLICY_HASH_DOMAIN, {policy_id, policy_version})`."""
    return domain_hash(
        DIFF_POLICY_HASH_DOMAIN,
        {"policy_id": policy.policy_id, "policy_version": policy.policy_version},
    )


def _gold_field_masks(
    gold_typed: dict[str, Any], node_id: str, pointer: str
) -> tuple[bool, bool]:
    """`(known_mask, observable_mask)` of the gold field, or `(False,
    False)` when it does not exist — a defect on a field gold never
    labeled can never be supervised (E3 rule 2)."""
    entry = gold_typed.get(node_id)
    if entry is None:
        return False, False
    for field in entry["fields"]:
        if field["node_property_pointer"] == pointer:
            return bool(field["known_mask"]), bool(field.get("observable_mask", False))
    return False, False


def compare(
    policy: DiffPolicy,
    *,
    gold_typed: dict[str, Any],
    candidate_typed: dict[str, Any],
) -> list[DefectTarget]:
    """Run `policy` and return defects with masks tightened and a
    deterministic order (by `(node_id, node_property_pointer,
    defect_type)`, RFC003 §12.1 discipline)."""
    targets: list[DefectTarget] = []
    for raw in policy.diff(gold_typed, candidate_typed):
        gold_known, gold_observable = _gold_field_masks(
            gold_typed, raw.node_id, raw.node_property_pointer
        )
        known = raw.claim_known and gold_known
        observable = raw.claim_observable and gold_observable
        targets.append(
            DefectTarget(
                node_id=raw.node_id,
                node_property_pointer=raw.node_property_pointer,
                defect_type=raw.defect_type,
                operation=raw.operation,
                known_mask=known,
                observable_mask=observable,
                supervision_mask=known and observable,
            )
        )
    targets.sort(key=lambda t: (t.node_id, t.node_property_pointer, t.defect_type))
    return targets


def defect_targets_hash(targets: list[DefectTarget]) -> str:
    """Deterministic hash of an emitted defect list (order-sensitive: the
    caller must pass `compare`'s own ordered output)."""
    payload = [
        {
            "node_id": t.node_id,
            "node_property_pointer": t.node_property_pointer,
            "defect_type": t.defect_type,
            "operation": t.operation,
            "known_mask": t.known_mask,
            "observable_mask": t.observable_mask,
            "supervision_mask": t.supervision_mask,
        }
        for t in targets
    ]
    return domain_hash(DEFECT_TARGETS_HASH_DOMAIN, payload)
