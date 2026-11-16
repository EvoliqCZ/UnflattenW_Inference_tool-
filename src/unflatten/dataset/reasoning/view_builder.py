"""Dormant `reasoning_corrective_v0` TaskView builder (RFC003-ERRATA-003 E3,
DEC-047 dormant code). `docs/plans/reasoning-data-readiness-2026-09-28.md`
§2 row d: "zostavenie TaskView + Input Ancestry pre korektívny príklad ...
neexistuje". This module is that step, and only that step.

Train-only by construction (E3 rule 3): a `split_name != "train"` is
refused before anything is built, never silently coerced. It computes no
release membership itself -- `split_name` is still the caller's fact
(§40.6 lives outside the envelope, same divide `envelope_validator.
validate_reasoning_corrective_rules` already draws) -- and it emits
nothing to disk or a release: the caller decides whether to write the
returned in-memory record anywhere, and DEC-047 requires it not build a
release from this at all today.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from unflatten.dataset import identity
from unflatten.dataset.envelope_validator import (
    validate_input_ancestry,
    validate_reasoning_corrective_rules_from_ancestry,
    validate_task_view,
)
from unflatten.dataset.reasoning.candidate_producer import SealedCandidateArtefact

#: The only split this builder will ever materialize a view for (E3 rule 3).
TRAIN_SPLIT = "train"


class ReasoningViewRefusedError(ValueError):
    """A structural E3 refusal: non-train split, or an envelope/rule the
    schema/identity/rule validators do not accept."""


@dataclass(frozen=True, slots=True)
class ReasoningCorrectiveView:
    """A dormant, in-memory `reasoning_corrective_v0` record: not written
    to a release, a cache or a training loop by this module."""

    task_view: dict[str, Any]
    input_ancestry: dict[str, Any]
    run_class: str


def _model_candidate_binding(
    *,
    ordinal: int,
    source_page_index: int,
    candidate: SealedCandidateArtefact,
    input_artifact: Mapping[str, Any],
    source_artifact: Mapping[str, Any],
    transform_chain_hash: str,
    lineage_edge_id: str,
) -> dict[str, Any]:
    provenance = candidate.provenance
    return {
        "input_ordinal": ordinal,
        "input_artifact": dict(input_artifact),
        "source_kind": "model_candidate",
        "source_page_index": source_page_index,
        "source_input_ordinal": None,
        "source_fragment_id": None,
        "source_artifact": dict(source_artifact),
        "derivation_kind": "transform_chain",
        "transform_chain_hash": transform_chain_hash,
        "lineage_edge_id": lineage_edge_id,
        "candidate_origin": "frozen_model",
        "producer_checkpoint_sha256": provenance.checkpoint_sha256,
        "producer_component_lock_hash": provenance.checkpoint_component_lock_hash,
        "producer_decode_policy_hash": provenance.decode_policy_hash,
        "producer_exposure": provenance.producer_exposure,
        "corruption_policy_hash": None,
    }


def build_reasoning_corrective_view(
    *,
    task_view_template: Mapping[str, Any],
    document_bundle_id: str,
    render_page_binding: Mapping[str, Any],
    parent_task_view_id: str | None,
    candidate: SealedCandidateArtefact,
    candidate_input_artifact: Mapping[str, Any],
    candidate_source_artifact: Mapping[str, Any],
    transform_chain_hash: str,
    lineage_edge_id: str,
    split_name: str,
    producer_ancestry: Mapping[str, Any] | None,
    release_name: str,
    val_test_ids: Iterable[str],
) -> ReasoningCorrectiveView:
    """Build a `reasoning_corrective_v0` TaskView + Input Ancestry pair from
    a sealed candidate and a template envelope (the render-view-specific
    fields: `input_selection`, `target_projection`, `masks`, `eligibility` --
    everything a `reasoning_corrective_v0` view inherits unchanged from its
    parent authoritative-target task_kind, RFC003-ERRATA-003 E1).

    Refuses (`ReasoningViewRefusedError`) for any split but `"train"`
    before doing anything else, and again if the assembled envelope fails
    schema validation or any E3 build-time rule (exposure computed from
    `producer_ancestry`, never trusted as a bool -- see
    `unflatten.dataset.reasoning.exposure`).
    """
    if split_name != TRAIN_SPLIT:
        raise ReasoningViewRefusedError(
            f"reasoning_corrective_v0 views may only be built for split "
            f"{TRAIN_SPLIT!r} (RFC003-ERRATA-003 E3 rule 3), got {split_name!r}"
        )

    render_view_id = task_view_template["render_view_id"]
    binding = _model_candidate_binding(
        ordinal=render_page_binding["input_ordinal"] + 1,
        source_page_index=render_page_binding["source_page_index"],
        candidate=candidate,
        input_artifact=candidate_input_artifact,
        source_artifact=candidate_source_artifact,
        transform_chain_hash=transform_chain_hash,
        lineage_edge_id=lineage_edge_id,
    )
    bindings = [dict(render_page_binding), binding]
    ancestry_hash = identity.input_ancestry_hash(
        document_bundle_id=document_bundle_id,
        candidate_id=None,
        render_view_id=render_view_id,
        parent_task_view_id=parent_task_view_id,
        bindings=bindings,
    )
    ancestry = {
        "input_ancestry_schema_version": "0.2.0",
        "input_ancestry_hash": ancestry_hash,
        "authority_ref": {"document_bundle_id": document_bundle_id, "candidate_id": None},
        "render_view_id": render_view_id,
        "parent_task_view_id": parent_task_view_id,
        "bindings": bindings,
    }

    task_view = dict(task_view_template)
    task_view["task_kind"] = "reasoning_corrective_v0"
    # `reasoning_corrective_v0` carries no reasoning_* fields (E3): a 0.2.0
    # envelope requires them present-but-null; a 0.1.0 envelope forbids the
    # keys existing at all (schema `additionalProperties`/`properties: false`
    # for the else branch) -- so only inject them under 0.2.0.
    if task_view.get("task_view_schema_version") == "0.2.0":
        task_view["reasoning_spec_id"] = None
        task_view["reasoning_derivation_policy_hash"] = None
        task_view["reasoning_aux_target_ref"] = None
    task_view["task_view_id"] = identity.task_view_id(
        render_view_id=render_view_id,
        task_kind="reasoning_corrective_v0",
        input_selection_hash=task_view["input_selection"]["input_selection_hash"],
        projection_hash=task_view["target_projection"]["projection_hash"],
        mask_policy_hash=task_view["masks"]["mask_policy_hash"],
        mask_projection_hash=task_view["masks"]["mask_projection_hash"],
    )

    schema_issues = validate_task_view(task_view) + validate_input_ancestry(ancestry)
    if schema_issues:
        raise ReasoningViewRefusedError(
            f"assembled envelope failed schema/identity validation: {schema_issues}"
        )

    rule_issues = validate_reasoning_corrective_rules_from_ancestry(
        task_view,
        ancestry,
        split_name=split_name,
        producer_ancestry=producer_ancestry,
        release_name=release_name,
        val_test_ids=val_test_ids,
    )
    if rule_issues:
        raise ReasoningViewRefusedError(f"E3 build-time rule violation: {rule_issues}")

    return ReasoningCorrectiveView(
        task_view=task_view, input_ancestry=ancestry, run_class=candidate.provenance.run_class
    )
