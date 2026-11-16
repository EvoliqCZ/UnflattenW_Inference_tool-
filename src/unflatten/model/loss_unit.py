"""M1 LossUnit construction and fail-closed verification (design §4.6, RFC005 §29.6).

Split out of `training_materializer.py` because the split follows a real
seam where the *kind* of data changes: verified `ScalarPolicyRange`s in,
`LossUnit`s (with fail-closed authority verification) out. The LossUnit
shape and its verification are self-contained --
`training_materializer.py` only calls `build_loss_unit` and
`assert_no_duplicate_or_unowned_locators`, so the two halves are
separately testable.

Owns acceptance criterion 8 (design §16): "every output contribution has
exactly one LossUnit/family; zero duplicate or unowned locators." The
`owner_scope` type keeps the full ratified four-value RFC005 enum;
`assert_no_duplicate_or_unowned_locators` is where M1 narrows to the two
values it actually produces (`global_target`, `fragment_local`) at
runtime, per design §4.6's disposition -- narrowing the type itself would
be an implicit RFC005 amendment.

This module authorizes no gradient update: it declares only dataclasses
and pure functions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from unflatten.model.boundary_projector import OwnerScope
from unflatten.model.hashing import domain_hash

_LOSS_UNIT_DOMAIN = "rfc005.loss-unit/v1"

LOSS_UNIT_SCHEMA_VERSION = "1.0.0"

#: Reason code for a structural TEXT_BEGIN/TEXT_END wrapper pair whose text
#: event has no owning grant at all (bug D1, 2026-09-07): a span with
#: non-empty text but zero `char_ranges` across every grant emits no L_text
#: atoms, so the wrapper pair was the only signal and, unwithdrawn, taught
#: the model that a real (often whitespace) run has empty text.
TEXT_UNOWNED = "text_unowned"

#: Reason code for a structural TEXT_BEGIN/TEXT_END pair a page *presents*
#: but does not structurally own -- the continuation of a run that began on
#: an earlier page (design §12.6, P1). Appended next to `not_supervised`,
#: which alone conflates "unresolved placement" with "this page presents
#: text it does not own"; the second code is what makes that withdrawal
#: countable in a report instead of invisible inside a bucket. It changes
#: no verdict, no mask and no schema. Nothing emits it yet -- this entry
#: only makes a row carrying it *readable*, ahead of the producer (P5),
#: because a reader without it refuses every such row.
TEXT_CONTINUATION_PRESENTATION = "text_continuation_presentation"

#: Every `reason_codes` value any M1 code path may emit into
#: `EffectiveLossMask`/`ScalarPolicyRange` -- a closed registry kept on the
#: Python side only. `loss_unit.schema.json` deliberately leaves
#: `reason_codes` entries as free-form strings (its byte hash is an
#: owner binding, D2 2026-09-07: extending the schema enum was rejected so
#: this registry would not require touching that file).
KNOWN_REASON_CODES: tuple[str, ...] = (
    "not_known",
    "not_observable",
    "not_supervised",
    "not_owned",
    "not_stage_enabled",
    TEXT_UNOWNED,
    TEXT_CONTINUATION_PRESENTATION,
)

#: The closed LossUnit envelope this module emits. `loss_unit_schema_hash`
#: reads these exact bytes, so the schema file itself is the artifact and
#: the hash cannot drift from it.
LOSS_UNIT_SCHEMA_PATH = Path(__file__).parent / "schemas" / "loss_unit.schema.json"

type LossFamily = Literal["L_structure", "L_text"]

#: The two owner_scope values M1 may ever emit (design §4.6 disposition:
#: "M1 validates that only global_target and fragment_local occur"). The
#: full four-value enum stays in the type; narrowing happens only here.
_M1_OWNER_SCOPES: frozenset[str] = frozenset({"global_target", "fragment_local"})


class LossUnitAuthorityMismatch(ValueError):
    """A LossUnit's authority/locator/ownership-proof fields disagree.

    M1-internal (§9.3); mirrors the proposed `LOSS_UNIT_AUTHORITY_MISMATCH`
    registry record (§9.2). Raised fail-closed before batching, per RFC005
    §29.6 (`docs/rfc/RFC005-training-strategy-v0.1-2.md:2004-2019`).
    """


class LossUnitBranchInvalid(ValueError):
    """A LossUnit's kind/locator branch, or output-locator uniqueness, is invalid.

    M1-internal (§9.3); mirrors the proposed `LOSS_UNIT_BRANCH_INVALID`
    registry record (§9.2). Covers duplicate output locators, unowned
    (schema-inconsistent) locators, and any `owner_scope` M1 does not
    itself produce.
    """


@dataclass(frozen=True, slots=True)
class TextRangeLocator:
    text_span_id: str
    scalar_start: int
    scalar_end: int


@dataclass(frozen=True, slots=True)
class EventRangeLocator:
    event_start: int
    event_end: int


type Locator = TextRangeLocator | EventRangeLocator


@dataclass(frozen=True, slots=True)
class Rfc003OwnershipProof:
    document_bundle_id: str
    target_bundle_hash: str
    fragment_id: str | None
    fragment_schema_version: str | None
    fragment_sequence_index: int | None
    node_id: str
    structure_owner: bool | None
    typed_target_owner: bool | None
    text_range_owner: bool | None
    page_evidence: str | None
    owned_char_ranges: tuple[tuple[int, int], ...]
    ownership_projection_hash: str


@dataclass(frozen=True, slots=True)
class EffectiveLossMask:
    mask_projection_hash: str
    known_mask: bool
    observable_mask: bool
    supervision_mask: bool
    ownership_enabled: bool
    stage_enabled: bool
    effective: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LossUnit:
    loss_unit_id: str
    loss_unit_kind: Literal["rfc003_supervised"]
    output_locator: str
    attachment: Literal["decoder_step"]
    loss_family: LossFamily
    stage_enabled: bool
    owner_scope: OwnerScope
    owner_fragment_id: str | None
    locator: Locator
    authority_target_hash: str
    projection_hash: str
    mask_projection_hash: str
    ownership_proof: Rfc003OwnershipProof
    effective_mask: EffectiveLossMask
    composite_vocabulary_hash: str
    composite_grammar_hash: str
    capability_profile_hash: str
    head_registry_hash: str | None


def verify_loss_unit_authority(unit: LossUnit) -> None:
    """Fail-closed authority/ownership verification (design §4.6).

    Checked: `ownership_proof.target_bundle_hash == authority_target_hash`;
    `ownership_proof.fragment_id == owner_fragment_id`, non-null exactly
    when `owner_scope == "fragment_local"`; for a `TextRangeLocator`, the
    locator range lies inside `ownership_proof.owned_char_ranges`.
    """
    proof = unit.ownership_proof
    if proof.target_bundle_hash != unit.authority_target_hash:
        raise LossUnitAuthorityMismatch(
            f"ownership_proof.target_bundle_hash {proof.target_bundle_hash!r} != "
            f"authority_target_hash {unit.authority_target_hash!r} "
            f"(loss_unit_id={unit.loss_unit_id!r})"
        )
    if proof.fragment_id != unit.owner_fragment_id:
        raise LossUnitAuthorityMismatch(
            f"ownership_proof.fragment_id {proof.fragment_id!r} != "
            f"owner_fragment_id {unit.owner_fragment_id!r} (loss_unit_id={unit.loss_unit_id!r})"
        )
    fragment_local = unit.owner_scope == "fragment_local"
    if fragment_local and proof.fragment_id is None:
        raise LossUnitAuthorityMismatch(
            f"owner_scope 'fragment_local' requires a non-null fragment_id "
            f"(loss_unit_id={unit.loss_unit_id!r})"
        )
    if not fragment_local and proof.fragment_id is not None:
        raise LossUnitAuthorityMismatch(
            f"owner_scope {unit.owner_scope!r} requires a null fragment_id "
            f"(loss_unit_id={unit.loss_unit_id!r})"
        )
    if isinstance(unit.locator, TextRangeLocator):
        inside = any(
            lo <= unit.locator.scalar_start and unit.locator.scalar_end <= hi
            for lo, hi in proof.owned_char_ranges
        )
        if not inside:
            raise LossUnitAuthorityMismatch(
                f"locator [{unit.locator.scalar_start}, {unit.locator.scalar_end}) "
                f"is not inside owned_char_ranges {proof.owned_char_ranges!r} "
                f"(loss_unit_id={unit.loss_unit_id!r})"
            )


def assert_no_duplicate_or_unowned_locators(loss_units: Sequence[LossUnit]) -> None:
    """Criterion 8: exactly one LossUnit/family per contribution, zero
    duplicate or unowned locators.

    "Unowned" here means schema/authority-inconsistent -- an `owner_scope`
    M1 does not itself produce, or a unit that fails
    `verify_loss_unit_authority`. Both are checked before the duplicate
    scan so a caller sees the more specific failure first.
    """
    for unit in loss_units:
        if unit.owner_scope not in _M1_OWNER_SCOPES:
            raise LossUnitBranchInvalid(
                f"owner_scope {unit.owner_scope!r} is not produced by M1 "
                f"(loss_unit_id={unit.loss_unit_id!r})"
            )
        verify_loss_unit_authority(unit)
    seen: set[tuple[str, str, str]] = set()
    for unit in loss_units:
        key = (unit.output_locator, unit.loss_family, unit.loss_unit_kind)
        if key in seen:
            raise LossUnitBranchInvalid(
                f"duplicate output_locator {unit.output_locator!r} for family "
                f"{unit.loss_family!r}"
            )
        seen.add(key)


def loss_unit_schema_hash() -> str:
    """`loss_unit_schema_hash`: the raw SHA-256 of `loss_unit.schema.json`'s
    own bytes, as `sha256:<64hex>`.

    A *raw artifact* hash, not a `domain_hash` semantic one, and
    deliberately so: what `ExactTextCacheIdentity.loss_unit_schema_hash`
    and `s1_run_manifest.loss_unit_schema_hash` pin is the schema
    document a run validated its LossUnits against, which is a file, not
    a projection. The Component Lock draws the same distinction and says
    why -- "the raw-vs-semantic distinction is load-bearing and the two
    must never be assumed equal"
    (`schemas/component_lock.schema.json`). Uncached, because a cached
    value would keep answering for a file that has since changed on disk.
    """
    return f"sha256:{hashlib.sha256(LOSS_UNIT_SCHEMA_PATH.read_bytes()).hexdigest()}"


def compute_loss_unit_id(payload_without_id: dict[str, Any]) -> str:
    """`loss_unit_id` for a payload with its own `loss_unit_id` field
    already removed. Public so a reader verifying an on-disk `LossUnit`
    (`exact_text_cache.py`) can recompute the id the same way this module
    does, without duplicating `_LOSS_UNIT_DOMAIN` -- a second copy of that
    domain string would silently stop matching this one if it were ever
    bumped to `/v2` here."""
    return domain_hash(_LOSS_UNIT_DOMAIN, payload_without_id)


def _loss_unit_id(payload_without_id: dict[str, Any]) -> str:
    return compute_loss_unit_id(payload_without_id)


def loss_unit_payload(unit: LossUnit) -> dict[str, Any]:
    """The closed JSON payload for `unit`, validatable against
    `loss_unit.schema.json`. Public so cache writers and tests can
    serialize a `LossUnit` without duplicating the field mapping."""
    return {
        "loss_unit_schema_version": LOSS_UNIT_SCHEMA_VERSION,
        "loss_unit_id": unit.loss_unit_id,
        "loss_unit_kind": unit.loss_unit_kind,
        "output_locator": unit.output_locator,
        "attachment": unit.attachment,
        "loss_family": unit.loss_family,
        "stage_enabled": unit.stage_enabled,
        "owner_scope": unit.owner_scope,
        "owner_fragment_id": unit.owner_fragment_id,
        "locator": _locator_payload(unit.locator),
        "authority_target_hash": unit.authority_target_hash,
        "projection_hash": unit.projection_hash,
        "mask_projection_hash": unit.mask_projection_hash,
        "ownership_proof": _rfc003_ownership_proof_payload(unit.ownership_proof),
        "effective_mask": _effective_mask_payload(unit.effective_mask),
        "composite_vocabulary_hash": unit.composite_vocabulary_hash,
        "composite_grammar_hash": unit.composite_grammar_hash,
        "capability_profile_hash": unit.capability_profile_hash,
        "head_registry_hash": unit.head_registry_hash,
    }


def _rfc003_ownership_proof_payload(proof: Rfc003OwnershipProof) -> dict[str, Any]:
    return {
        "document_bundle_id": proof.document_bundle_id,
        "target_bundle_hash": proof.target_bundle_hash,
        "fragment_id": proof.fragment_id,
        "fragment_schema_version": proof.fragment_schema_version,
        "fragment_sequence_index": proof.fragment_sequence_index,
        "node_id": proof.node_id,
        "structure_owner": proof.structure_owner,
        "typed_target_owner": proof.typed_target_owner,
        "text_range_owner": proof.text_range_owner,
        "page_evidence": proof.page_evidence,
        "owned_char_ranges": [list(r) for r in proof.owned_char_ranges],
        "ownership_projection_hash": proof.ownership_projection_hash,
    }


def _effective_mask_payload(mask: EffectiveLossMask) -> dict[str, Any]:
    return {
        "mask_projection_hash": mask.mask_projection_hash,
        "known_mask": mask.known_mask,
        "observable_mask": mask.observable_mask,
        "supervision_mask": mask.supervision_mask,
        "ownership_enabled": mask.ownership_enabled,
        "stage_enabled": mask.stage_enabled,
        "effective": mask.effective,
        "reason_codes": list(mask.reason_codes),
    }


def _locator_payload(locator: Locator) -> dict[str, Any]:
    if isinstance(locator, TextRangeLocator):
        return {
            "text_span_id": locator.text_span_id,
            "scalar_start": locator.scalar_start,
            "scalar_end": locator.scalar_end,
        }
    return {"event_start": locator.event_start, "event_end": locator.event_end}


def build_loss_unit(
    *,
    output_locator: str,
    loss_family: LossFamily,
    stage_enabled: bool,
    owner_scope: OwnerScope,
    owner_fragment_id: str | None,
    locator: Locator,
    authority_target_hash: str,
    projection_hash: str,
    mask_projection_hash: str,
    ownership_proof: Rfc003OwnershipProof,
    effective_mask: EffectiveLossMask,
    composite_vocabulary_hash: str,
    composite_grammar_hash: str,
    capability_profile_hash: str,
    head_registry_hash: str | None,
) -> LossUnit:
    """Build one closed, verified LossUnit (`loss_unit_kind` is always
    `rfc003_supervised` in M1 -- §4.6). `head_registry_hash` is the bound
    registry hash supplied by the materializer off the cache identity
    (non-null since cache 2.0.0, DEC-008 addendum A / ERRATUM E1)."""
    payload_without_id: dict[str, Any] = {
        "loss_unit_schema_version": LOSS_UNIT_SCHEMA_VERSION,
        "loss_unit_kind": "rfc003_supervised",
        "output_locator": output_locator,
        "attachment": "decoder_step",
        "loss_family": loss_family,
        "stage_enabled": stage_enabled,
        "owner_scope": owner_scope,
        "owner_fragment_id": owner_fragment_id,
        "locator": _locator_payload(locator),
        "authority_target_hash": authority_target_hash,
        "projection_hash": projection_hash,
        "mask_projection_hash": mask_projection_hash,
        "ownership_proof": _rfc003_ownership_proof_payload(ownership_proof),
        "effective_mask": _effective_mask_payload(effective_mask),
        "composite_vocabulary_hash": composite_vocabulary_hash,
        "composite_grammar_hash": composite_grammar_hash,
        "capability_profile_hash": capability_profile_hash,
        "head_registry_hash": head_registry_hash,
    }
    unit = LossUnit(
        loss_unit_id=_loss_unit_id(payload_without_id),
        loss_unit_kind="rfc003_supervised",
        output_locator=output_locator,
        attachment="decoder_step",
        loss_family=loss_family,
        stage_enabled=stage_enabled,
        owner_scope=owner_scope,
        owner_fragment_id=owner_fragment_id,
        locator=locator,
        authority_target_hash=authority_target_hash,
        projection_hash=projection_hash,
        mask_projection_hash=mask_projection_hash,
        ownership_proof=ownership_proof,
        effective_mask=effective_mask,
        composite_vocabulary_hash=composite_vocabulary_hash,
        composite_grammar_hash=composite_grammar_hash,
        capability_profile_hash=capability_profile_hash,
        head_registry_hash=head_registry_hash,
    )
    verify_loss_unit_authority(unit)
    return unit
