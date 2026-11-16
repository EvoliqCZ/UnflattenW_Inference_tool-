"""D4 — `reasoning-derivation-policy` framework (RFC003-ERRATA-003 E2, §30.11).

Deterministic derivation of auxiliary reasoning targets from a gold
Canonical AST for `task_kind == "reasoning_constructive_v0"`. This module
is the FRAMEWORK the errata ratifies: a versioned policy plug-in protocol,
mask-tightening enforcement (E2 rule 2), and the two structural refusals
(E2 rule 3 continuation family, DEC-033 single-page input). The tag
register itself (OPEN-003-E3-2 — which structural values a family emits)
is supplied by the caller's policy plug-in; none is chosen here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from unflatten.dataset.identity import domain_hash

#: RFC003-ERRATA-003 E2 rule 3: forbidden while DEC-033 (no cross-page
#: continuation) stands.
FORBIDDEN_REASONING_FAMILY = "continuation"

#: Identity domain for a derivation policy's own hash (RFC003-ERRATA-003 E2
#: rule 6: `reasoning_derivation_policy_hash` is part of `task_view_id`
#: identity — see `unflatten.dataset.identity.task_view_id`).
DERIVATION_POLICY_HASH_DOMAIN = "rfc003.reasoning-derivation-policy/v1"


class ReasoningRefusedError(ValueError):
    """A structural E2 refusal: continuation family or multi-page input."""


@dataclass(frozen=True)
class RawAuxTag:
    """One candidate aux tag, as a `DerivationPolicy` emits it.

    `source_node_id`/`source_pointer` name the single gold typed-target
    field this tag is derived from — `derive_aux_targets` looks up that
    field's own masks and ANDs them into the result, so a policy can never
    grant a tag more supervision than its source already carries (E2
    rule 2). `claim_known`/`claim_observable` are the policy's own upper
    bound on the tag (default: no additional restriction); the final masks
    are always `claim AND source`, never `claim` alone.
    """

    node_id: str
    tag_name: str
    value: Any
    source_node_id: str
    source_pointer: str
    claim_known: bool = True
    claim_observable: bool = True


@dataclass(frozen=True)
class AuxTarget:
    """One derived §30.11 auxiliary reasoning target, masks already
    tightened against its source field (E2 rule 2)."""

    node_id: str
    tag_name: str
    value: Any
    known_mask: bool
    observable_mask: bool
    supervision_mask: bool


class DerivationPolicy(Protocol):
    """A versioned plug-in supplying the OPEN-003-E3-2 tag register."""

    policy_id: str
    policy_version: str

    def derive(self, canonical_ast: dict[str, Any]) -> list[RawAuxTag]:
        """Return every candidate tag BEFORE mask tightening.

        Structural values only (E2 rule 1: no natural-language text) — the
        framework does not and cannot check that; it is the policy's
        contract.
        """
        ...


def derivation_policy_hash(policy: DerivationPolicy) -> str:
    """`H(DERIVATION_POLICY_HASH_DOMAIN, {policy_id, policy_version})`."""
    return domain_hash(
        DERIVATION_POLICY_HASH_DOMAIN,
        {"policy_id": policy.policy_id, "policy_version": policy.policy_version},
    )


def _require_not_continuation(reasoning_spec_id: str) -> None:
    parts = reasoning_spec_id.split(".")
    family = parts[1] if len(parts) >= 3 else None
    if family == FORBIDDEN_REASONING_FAMILY:
        raise ReasoningRefusedError(
            f"reasoning_spec_id family {FORBIDDEN_REASONING_FAMILY!r} is forbidden "
            "while DEC-033 stands (RFC003-ERRATA-003 E2 rule 3)"
        )


def _require_single_page(page_indices: list[int]) -> None:
    if len(page_indices) != 1:
        raise ReasoningRefusedError(
            "reasoning_constructive_v0 derivation requires exactly one page "
            f"(DEC-033 standalone-page scope), got {len(page_indices)} pages"
        )


def _source_field_masks(
    typed_target: dict[str, Any], node_id: str, pointer: str
) -> tuple[bool, bool]:
    """`(known_mask, observable_mask)` of the named field, or `(False, False)`
    when it does not exist — a missing source can never grant supervision
    (E2 rule 2: "a missing or unobservable tag is never zero or negative",
    i.e. it withdraws rather than defaults to observable)."""
    entry = typed_target.get(node_id)
    if entry is None:
        return False, False
    for field in entry["fields"]:
        if field["node_property_pointer"] == pointer:
            return bool(field["known_mask"]), bool(field.get("observable_mask", False))
    return False, False


def derive_aux_targets(
    policy: DerivationPolicy,
    *,
    canonical_ast: dict[str, Any],
    typed_target: dict[str, Any],
    reasoning_spec_id: str,
    page_indices: list[int],
) -> list[AuxTarget]:
    """Run `policy` and tighten every tag's masks against its source field.

    Raises :class:`ReasoningRefusedError` for the continuation family or a
    multi-page selection before the policy ever runs (E2 rules 2 input
    scope, and rule 3).
    """
    _require_not_continuation(reasoning_spec_id)
    _require_single_page(page_indices)

    targets: list[AuxTarget] = []
    for raw in policy.derive(canonical_ast):
        source_known, source_observable = _source_field_masks(
            typed_target, raw.source_node_id, raw.source_pointer
        )
        known = raw.claim_known and source_known
        observable = raw.claim_observable and source_observable
        targets.append(
            AuxTarget(
                node_id=raw.node_id,
                tag_name=raw.tag_name,
                value=raw.value,
                known_mask=known,
                observable_mask=observable,
                supervision_mask=known and observable,
            )
        )
    return targets
