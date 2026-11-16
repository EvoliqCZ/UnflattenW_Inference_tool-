"""RFC005 §12.2 generalized StageGatePolicy for S6/S7/S8 (DEC-047 dormant code).

DEC-047 (`docs/decisions/DEC-047-spiaci-kod-v-predstihu.md`) names a
"zovšeobecnená StageGatePolicy" among the pieces of dormant code the
owner allowed ahead of S6/S7/S8 ratification. This module is that piece:
a NEW, separately versioned policy registry -- `general_stage_gate_policy_
schema_version`, domain `rfc005.general-stage-gate-policy/v1` -- that does
not edit, import as a base class, or share a hash domain with
`stage_gate_report.py` (S1/S2 contract policy), `s2_stage_gate_policy.py`
(S2 quality policy) or `s4_selection_policy.py` (S4 checkpoint selection).
Each of those three keeps its own schema, its own hash, and every existing
frozen policy JSON under `policies/` byte-for-byte unchanged (DEC-047
condition 3); `tests/model/test_general_stage_gate_policy.py::
TestExistingPolicyHashesUnchanged` pins all three literally.

RFC005 §12.2 in one sentence: every expression of "minimum", "better",
"non-inferior", "material benefit" or "passes" from S2 onward must resolve
against a pre-frozen policy carrying a metric id and denominator, a
mandatory absolute floor, an optional non-inferiority margin that can
never compensate a failed floor, a confidence-bound method and alpha, a
minimum independent-lineage count, a minimum attempted/accepted
denominator, critical slices, and a missing/insufficient-evidence action
-- and an underpowered or zero-coverage cohort is `insufficient_evidence`,
never a pass. This module generalizes that shape to carry an arbitrary
list of metrics (S6/S7/S8 do not share S2's single `exact_match_rate`
metric), and its evaluator resolves a verdict of exactly `pass`, `fail` or
`insufficient_evidence` -- `evaluate_metric_gate`/`evaluate_policy` never
return a fourth value.

The confidence-bound method is `paired_lineage_bootstrap`: this module
imports `lineage_bootstrap_margin_micro` from
`unflatten.model.select_checkpoint` (RFC005 §39.2's lineage-level
bootstrap) rather than re-deriving it -- the task's own instruction is
"reuse math ... only by importing; do not change select_checkpoint or
s4_selection_policy". Nothing here is imported BY select_checkpoint or
s4_selection_policy, so the reuse is one-directional.

No gradient, no default-on switch, no hashed identity moved (DEC-047
conditions 1/3): this module is pure policy-shape validation plus a CPU
arithmetic evaluator over caller-supplied samples.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import jsonschema.protocols
import jsonschema.validators

from unflatten.model.hashing import domain_hash
from unflatten.model.select_checkpoint import lineage_bootstrap_margin_micro

_SCHEMA_DIR = Path(__file__).parent / "schemas"

#: Design registry row `rfc005.general-stage-gate-policy/v1`. Owned here:
#: no other module computes a generalized stage gate policy's identity hash.
_POLICY_DOMAIN = "rfc005.general-stage-gate-policy/v1"

GENERAL_STAGE_GATE_POLICY_SCHEMA_VERSION = "1.0.0"

Verdict = Literal["pass", "fail", "insufficient_evidence"]


class GeneralStageGatePolicyInvalid(ValueError):
    """A generalized stage gate policy is malformed: the schema refuses
    it, or a semantic check the schema cannot express (distinct metric
    ids, distinct slice ids per metric) does not hold."""


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / "general_stage_gate_policy.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _validate_schema(doc: dict[str, Any]) -> None:
    errors = sorted(_validator().iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        pointer = "/" + "/".join(str(p) for p in first.absolute_path)
        raise GeneralStageGatePolicyInvalid(
            f"general stage gate policy invalid at {pointer or '/'}: {first.message}"
        )


def validate_general_stage_gate_policy(doc: dict[str, Any]) -> None:
    _validate_schema(doc)
    metric_ids = [m["metric_id"] for m in doc["metrics"]]
    if len(set(metric_ids)) != len(metric_ids):
        raise GeneralStageGatePolicyInvalid(
            f"general stage gate policy repeats a metric_id: {metric_ids} -- "
            "one policy entry per metric"
        )
    for metric in doc["metrics"]:
        slice_ids = [s["slice_id"] for s in metric["critical_slices"]]
        if len(set(slice_ids)) != len(slice_ids):
            raise GeneralStageGatePolicyInvalid(
                f"general stage gate policy metric {metric['metric_id']!r} repeats a "
                f"critical slice_id: {slice_ids}"
            )


def general_stage_gate_policy_hash(policy: dict[str, Any]) -> str:
    """`domain_hash` over the whole validated document -- no embedded
    self-hash to project away, same convention as `s2_stage_gate_policy_
    hash`/`s4_selection_policy_hash`."""
    validate_general_stage_gate_policy(policy)
    return domain_hash(_POLICY_DOMAIN, policy)


def load_general_stage_gate_policy(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GeneralStageGatePolicyInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GeneralStageGatePolicyInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_general_stage_gate_policy(payload)
    return payload


@dataclass(frozen=True)
class LineagePairedSample:
    """One lineage's contribution to one metric under paired G2-style
    exposure (RFC005 §20.5: dense control and sparse challenger share the
    same frozen `StageDataMatrix` and exposure plan, so a lineage's
    candidate and reference values are directly paired). `reference_
    value_micro` is `None` when the metric has no reference arm (a bare
    absolute-floor metric, not a non-inferiority one) -- the margin check
    is then skipped, never guessed as zero."""

    lineage_root_id: str
    candidate_value_micro: int
    reference_value_micro: int | None = None


@dataclass(frozen=True)
class MetricGateVerdict:
    metric_id: str
    verdict: Verdict
    reason: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "detail": self.detail,
        }


def _insufficient(metric_id: str, reason: str, **detail: Any) -> MetricGateVerdict:
    return MetricGateVerdict(
        metric_id=metric_id, verdict="insufficient_evidence", reason=reason, detail=detail
    )


def evaluate_metric_gate(
    metric: Mapping[str, Any],
    samples: Sequence[LineagePairedSample],
    *,
    num_resamples: int,
    seed: int,
) -> MetricGateVerdict:
    """RFC005 §12.2's floor+margin+confidence-interval+minimum-lineage
    rule, generalized to one metric of an arbitrary S6/S7/S8 policy.

    Returns `insufficient_evidence` -- never `pass` -- while the metric's
    floor is still an unratified `is_placeholder`, while the cohort has
    fewer than `minimum_independent_lineages` distinct lineages or fewer
    than `minimum_attempted_denominator` samples, or (for a metric with a
    reference arm) while some samples carry a reference value and others
    do not. Otherwise `pass` iff BOTH the absolute floor and the
    non-inferiority margin hold on the lineage-bootstrap lower confidence
    bound (relative improvement may never compensate a failed floor,
    §12.2); `fail` otherwise.
    """
    metric_id = metric["metric_id"]
    if metric["is_placeholder"]:
        return _insufficient(metric_id, "policy_placeholder_not_frozen")

    lineages = {s.lineage_root_id for s in samples}
    if len(lineages) < metric["minimum_independent_lineages"]:
        return _insufficient(
            metric_id,
            "minimum_independent_lineages_not_met",
            observed_independent_lineages=len(lineages),
            minimum_independent_lineages=metric["minimum_independent_lineages"],
        )
    if len(samples) < metric["minimum_attempted_denominator"]:
        return _insufficient(
            metric_id,
            "minimum_attempted_denominator_not_met",
            observed_attempted_denominator=len(samples),
            minimum_attempted_denominator=metric["minimum_attempted_denominator"],
        )

    reference_present = [s.reference_value_micro is not None for s in samples]
    has_reference_arm = any(reference_present)
    if has_reference_arm and not all(reference_present):
        return _insufficient(metric_id, "partial_reference_arm_coverage")

    candidate_values = [float(s.candidate_value_micro) for s in samples]
    candidate_point_micro = sum(candidate_values) / len(candidate_values)
    candidate_margin_micro = lineage_bootstrap_margin_micro(
        candidate_values, num_resamples=num_resamples, alpha_micro=metric["alpha_micro"], seed=seed
    )
    candidate_lower_bound_micro = candidate_point_micro - candidate_margin_micro
    floor_satisfied = candidate_lower_bound_micro >= metric["absolute_quality_floor_micro"]

    margin_satisfied = True
    diff_lower_bound_micro: float | None = None
    if has_reference_arm:
        diffs = [
            float(s.candidate_value_micro) - float(s.reference_value_micro)  # type: ignore[arg-type]
            for s in samples
        ]
        diff_margin_micro = lineage_bootstrap_margin_micro(
            diffs, num_resamples=num_resamples, alpha_micro=metric["alpha_micro"], seed=seed
        )
        diff_point_micro = sum(diffs) / len(diffs)
        diff_lower_bound_micro = diff_point_micro - diff_margin_micro
        margin_satisfied = diff_lower_bound_micro >= -metric["relative_margin_micro"]

    passed = floor_satisfied and margin_satisfied
    return MetricGateVerdict(
        metric_id=metric_id,
        verdict="pass" if passed else "fail",
        reason="ok" if passed else "floor_or_margin_not_satisfied",
        detail={
            "candidate_point_micro": candidate_point_micro,
            "candidate_lower_bound_micro": candidate_lower_bound_micro,
            "absolute_quality_floor_micro": metric["absolute_quality_floor_micro"],
            "absolute_quality_floor_satisfied": floor_satisfied,
            "relative_margin_micro": metric["relative_margin_micro"],
            "diff_lower_bound_micro": diff_lower_bound_micro,
            "relative_margin_satisfied": margin_satisfied,
        },
    )


def evaluate_critical_slice_gate(
    metric_id: str,
    slice_policy: Mapping[str, Any],
    samples: Sequence[LineagePairedSample],
    *,
    num_resamples: int,
    seed: int,
) -> MetricGateVerdict:
    """A critical slice is the same floor rule as `evaluate_metric_gate`
    but keyed to `slice_policy["minimum_fidelity_micro"]`/`slice_policy
    ["minimum_independent_lineages"]` instead of the metric's own floor --
    RFC005 §12.2's "no critical-slice regression beyond predeclared
    margin" (§20.4) reduced to an absolute floor, the same reduction
    `s2_stage_gate_policy.schema.json` already uses for S2's own critical
    slices."""
    slice_id = slice_policy["slice_id"]
    metric_composite_id = f"{metric_id}/{slice_id}"
    lineages = {s.lineage_root_id for s in samples}
    if len(lineages) < slice_policy["minimum_independent_lineages"]:
        return _insufficient(
            metric_composite_id,
            "minimum_independent_lineages_not_met",
            observed_independent_lineages=len(lineages),
            minimum_independent_lineages=slice_policy["minimum_independent_lineages"],
        )
    if not samples:
        return _insufficient(metric_composite_id, "no_samples")

    values = [float(s.candidate_value_micro) for s in samples]
    point_micro = sum(values) / len(values)
    margin_micro = lineage_bootstrap_margin_micro(
        values, num_resamples=num_resamples, alpha_micro=1000, seed=seed
    )
    lower_bound_micro = point_micro - margin_micro
    passed = lower_bound_micro >= slice_policy["minimum_fidelity_micro"]
    return MetricGateVerdict(
        metric_id=metric_composite_id,
        verdict="pass" if passed else "fail",
        reason="ok" if passed else "minimum_fidelity_not_satisfied",
        detail={
            "point_micro": point_micro,
            "lower_bound_micro": lower_bound_micro,
            "minimum_fidelity_micro": slice_policy["minimum_fidelity_micro"],
        },
    )


@dataclass(frozen=True)
class PolicyVerdict:
    verdict: Verdict
    metric_verdicts: tuple[MetricGateVerdict, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "metric_verdicts": [m.as_dict() for m in self.metric_verdicts],
        }


def evaluate_policy(
    policy: Mapping[str, Any],
    samples_by_metric: Mapping[str, Sequence[LineagePairedSample]],
    *,
    num_resamples: int,
    seed: int,
) -> PolicyVerdict:
    """Every metric AND every one of its critical slices, combined:
    `insufficient_evidence` if any component is, else `fail` if any
    component failed, else `pass`. An underpowered cohort or a placeholder
    floor can never be outvoted into a pass by other metrics passing."""
    validate_general_stage_gate_policy(dict(policy))
    verdicts: list[MetricGateVerdict] = []
    for metric in policy["metrics"]:
        metric_id = metric["metric_id"]
        samples = samples_by_metric.get(metric_id, ())
        verdicts.append(
            evaluate_metric_gate(metric, samples, num_resamples=num_resamples, seed=seed)
        )
        for slice_policy in metric["critical_slices"]:
            verdicts.append(
                evaluate_critical_slice_gate(
                    metric_id, slice_policy, samples, num_resamples=num_resamples, seed=seed
                )
            )

    if any(v.verdict == "insufficient_evidence" for v in verdicts):
        overall: Verdict = "insufficient_evidence"
    elif any(v.verdict == "fail" for v in verdicts):
        overall = "fail"
    else:
        overall = "pass"
    return PolicyVerdict(verdict=overall, metric_verdicts=tuple(verdicts))
