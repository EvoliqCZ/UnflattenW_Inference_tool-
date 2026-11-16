"""RFC005 §20.4-§20.5 S6 paired-exposure runner scaffolding (DEC-045/DEC-047).

RFC005 §20.5: "Dense control a sparse challenger používajú rovnakú
frozen `StageDataMatrix` a rovnaký precomputed ordered exposure plan až na
batch packing ... Odchýlka v exposure, orderi alebo supervised-unit
budgete ruší paired G2 claim." This module builds exactly ONE
`ordered_exposure_plan` (by composing `unflatten.model.s2_sampler.
build_ordered_exposure_plan` -- the same frozen-plan builder `s4_plan.py`
already uses -- never by re-deriving the sampler) and wraps it, unchanged,
into two small "arm" envelopes that differ ONLY in `arm_id` and
`architecture_manifest_ref`. Composition, not edits: nothing in
`s4_plan.py` or `s2_sampler.py` is modified, and the shared, closed
`ordered_exposure_plan.schema.json` gains no new field -- the arm identity
lives in a NEW, separate, versioned envelope this module builds and
hashes under its own domain (`rfc005.paired-exposure-arm/v1`), wrapping a
plan built to the existing schema without touching it.

RFC005 §20.4's G2 acceptance conjunction (all dense hard gates pass AND
no critical-slice regression AND router stability AND material benefit AND
dropless H200 fit) is expressed here as one `general_stage_gate_policy`
document (W3-11) with every threshold a named placeholder -- `is_placeholder:
true` on every metric -- so `evaluate_policy` can only ever return
`insufficient_evidence` for this draft until an owner freezes real
thresholds (DEC-047 condition 4: "prahy ... sú polia politik, nie
konštanty").

DEC-047 dormant-code boundary: this module runs no gradient, moves no
hashed identity, and is not read by the S4 v15b path or `s4_plan.py`'s own
CLI. `S6R`/`G2` still require their own G2/G3 ratification (§20.4) before
any of this scaffolding may gate a real promotion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from unflatten.model.general_stage_gate_policy import (
    GENERAL_STAGE_GATE_POLICY_SCHEMA_VERSION,
    LineagePairedSample,
    PolicyVerdict,
    evaluate_policy,
    general_stage_gate_policy_hash,
)
from unflatten.model.hashing import domain_hash
from unflatten.model.s2_sampler import build_ordered_exposure_plan

#: Design registry row `rfc005.paired-exposure-arm/v1`.
PAIRED_EXPOSURE_ARM_DOMAIN = "rfc005.paired-exposure-arm/v1"
PAIRED_EXPOSURE_ARM_SCHEMA_VERSION = "1.0.0"

ARM_DENSE_CONTROL = "dense_control"
ARM_SPARSE_CHALLENGER = "sparse_challenger"

#: RFC005 §20.4's four-way conjunction, named as separate metrics of one
#: G2 general_stage_gate_policy draft. Every floor is a placeholder
#: (is_placeholder=True): the owner has frozen no number yet.
G2_GATE_METRIC_IDS: tuple[str, ...] = (
    "no_critical_slice_regression",
    "router_stability",
    "material_benefit",
    "dropless_h200_fit",
)


class S6PairedExposureRefused(ValueError):
    """A paired exposure request cannot be honoured truthfully -- a
    refusal, never a silent repair."""


def _build_arm(
    shared_plan: Mapping[str, Any], *, arm_id: str, architecture_manifest_ref: str
) -> dict[str, Any]:
    arm: dict[str, Any] = {
        "paired_exposure_arm_schema_version": PAIRED_EXPOSURE_ARM_SCHEMA_VERSION,
        "arm_id": arm_id,
        "architecture_manifest_ref": architecture_manifest_ref,
        "exposure_plan": dict(shared_plan),
    }
    arm["paired_exposure_arm_hash"] = domain_hash(PAIRED_EXPOSURE_ARM_DOMAIN, arm)
    return arm


def build_paired_exposure_plans(
    *,
    matrix: Mapping[str, Any],
    lineages: Sequence[Mapping[str, Any]],
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
    run_seed: int,
    dense_architecture_manifest_ref: str,
    sparse_architecture_manifest_ref: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One frozen `StageDataMatrix` + ordered exposure plan (built once,
    via `build_ordered_exposure_plan`, exactly as `s4_plan.py` already
    does), wrapped into `(dense_arm, sparse_arm)` -- byte-identical except
    `arm_id`, `architecture_manifest_ref` and the hash each carries over
    its own arm_id/ref (§20.5: "až na batch packing", which lives
    outside this scaffolding, not in the plan itself)."""
    if dense_architecture_manifest_ref == sparse_architecture_manifest_ref:
        raise S6PairedExposureRefused(
            "dense and sparse architecture_manifest_ref are identical "
            f"({dense_architecture_manifest_ref!r}) -- a paired G2 comparison needs "
            "two distinct architecture manifests, not two labels for the same one"
        )
    shared_plan = build_ordered_exposure_plan(
        matrix=matrix,
        lineages=lineages,
        sampler_policy=sampler_policy,
        exposure_policy=exposure_policy,
        run_seed=run_seed,
    )
    dense_arm = _build_arm(
        shared_plan,
        arm_id=ARM_DENSE_CONTROL,
        architecture_manifest_ref=dense_architecture_manifest_ref,
    )
    sparse_arm = _build_arm(
        shared_plan,
        arm_id=ARM_SPARSE_CHALLENGER,
        architecture_manifest_ref=sparse_architecture_manifest_ref,
    )
    return dense_arm, sparse_arm


def build_g2_gate_policy_draft() -> dict[str, Any]:
    """RFC005 §20.4's acceptance conjunction as one `general_stage_gate_
    policy` (W3-11) draft, stage `S6`, gate `G2_moe_acceptance`. Every
    metric is `is_placeholder: true` with a floor of 0 -- a value with no
    meaning of its own, present only so the document validates; the
    verdict this draft can produce is `insufficient_evidence` on every
    metric until an owner names a real floor, margin, alpha and minimum
    lineage/denominator count for each (DEC-047 condition 4)."""
    metrics = [
        {
            "metric_id": metric_id,
            "denominator": "attempted",
            "absolute_quality_floor_micro": 0,
            "relative_margin_micro": 0,
            "confidence_bound_method": "paired_lineage_bootstrap",
            "alpha_micro": 50_000,
            "minimum_independent_lineages": 1,
            "minimum_attempted_denominator": 1,
            "critical_slices": [],
            "is_placeholder": True,
            "missing_evidence_action": "insufficient_evidence",
        }
        for metric_id in G2_GATE_METRIC_IDS
    ]
    return {
        "general_stage_gate_policy_schema_version": GENERAL_STAGE_GATE_POLICY_SCHEMA_VERSION,
        "policy_version": "0.1.0",
        "stage": "S6",
        "gate_name": "G2_moe_acceptance",
        "missing_evidence_action": "insufficient_evidence",
        "metrics": metrics,
    }


@dataclass(frozen=True)
class ArmMetricSample:
    """One lineage's own value for one metric, from one arm's evaluation
    run -- the shape a decode/score pipeline over that arm's checkpoint is
    expected to produce (RFC005 §12.2: production-equivalent free-running
    output, never a teacher-forced diagnostic)."""

    lineage_root_id: str
    value_micro: int


@dataclass(frozen=True)
class ArmEvaluationResult:
    """One arm's (dense control or sparse challenger) measured metric
    values, keyed by the SAME `metric_id`s the G2 gate policy names.
    `metric_values[metric_id]` may omit a lineage the arm's own decode did
    not attempt -- pairing (`_paired_samples_for_metric`) only uses
    lineages both arms attempted."""

    arm_id: str
    metric_values: Mapping[str, Sequence[ArmMetricSample]]


@dataclass(frozen=True)
class RouterTelemetry:
    """Placeholder shape for the sparse arm's router telemetry (RFC005
    §20.3: shared expert always active, dropless production path, router
    collapse/dead experts are stage failures). Populated by the sparse MoE
    module (a separate, later agent's work per the task); this dataclass
    only fixes the field names a `build_g2_report` caller must supply."""

    dead_expert_count: int
    router_entropy_micro: int
    token_drop_count: int
    total_tokens: int


def _paired_samples_for_metric(
    dense_samples: Sequence[ArmMetricSample], sparse_samples: Sequence[ArmMetricSample]
) -> list[LineagePairedSample]:
    dense_by_lineage = {s.lineage_root_id: s.value_micro for s in dense_samples}
    return [
        LineagePairedSample(
            lineage_root_id=s.lineage_root_id,
            candidate_value_micro=s.value_micro,
            reference_value_micro=dense_by_lineage[s.lineage_root_id],
        )
        for s in sparse_samples
        if s.lineage_root_id in dense_by_lineage
    ]


def build_g2_report(
    *,
    dense_result: ArmEvaluationResult,
    sparse_result: ArmEvaluationResult,
    router_telemetry: RouterTelemetry,
    policy: Mapping[str, Any],
    num_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """The sparse challenger evaluated against the dense control, paired
    by lineage, under `policy` (a `general_stage_gate_policy` document --
    ordinarily `build_g2_gate_policy_draft()`'s draft, or an owner-frozen
    successor). `router_telemetry` is carried through verbatim: this
    module makes no router-stability determination of its own beyond what
    `policy`'s own `router_stability` metric (if ever unfrozen) can
    express over samples -- it does not invent a second, ad hoc rule."""
    samples_by_metric = {
        metric_id: _paired_samples_for_metric(
            dense_result.metric_values.get(metric_id, ()),
            sparse_result.metric_values.get(metric_id, ()),
        )
        for metric_id in {
            *dense_result.metric_values,
            *sparse_result.metric_values,
        }
    }
    policy_verdict: PolicyVerdict = evaluate_policy(
        policy, samples_by_metric, num_resamples=num_resamples, seed=seed
    )
    return {
        "policy_hash": general_stage_gate_policy_hash(dict(policy)),
        "dense_arm_id": dense_result.arm_id,
        "sparse_arm_id": sparse_result.arm_id,
        "router_telemetry": {
            "dead_expert_count": router_telemetry.dead_expert_count,
            "router_entropy_micro": router_telemetry.router_entropy_micro,
            "token_drop_count": router_telemetry.token_drop_count,
            "total_tokens": router_telemetry.total_tokens,
        },
        "verdict": policy_verdict.verdict,
        "metric_verdicts": [m.as_dict() for m in policy_verdict.metric_verdicts],
    }
