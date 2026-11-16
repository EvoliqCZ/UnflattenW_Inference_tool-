"""S4.5 promotion-gate report (RFC005-ERRATA-010 E7.3; dormant).

The report is a pure function of its inputs: :func:`verify_rl_stage_gate_report`
rebuilds it from the same real run manifest, artefacts, ledger and arms plus
the measurements stored in the report, and compares. The RFC006 conditions and
the reward-hacking signal are owner attestations with evidence refs; only the
decision is derived.

Decision order (the first that applies):

1. ``blocked``: no SFT-continuation arm, a reward-hacking signal, or one of the
   RFC006-ERRATA-001 E1(b)1-7 conditions not held;
2. ``insufficient_evidence``: any gate / correlation / zero-advantage result is
   insufficient (placeholder threshold, missing evidence, under-powered);
3. ``insufficient_evidence`` / ``policy_not_ratified``: the policy is not
   RATIFIED -- its thresholds are not frozen, so no result is a verdict on
   them and none may be reported as ``rejected`` (E7.4);
4. ``rejected``: any result failed (ERRATA-010 E7.6);
5. otherwise all passed, and ``promoted`` only for a ``promotion`` run under a
   RATIFIED policy with a bound ratification record; else
   ``insufficient_evidence`` with the reason named.

Hard failures raise instead: a forbidden source/lineage in the exposure
(``RlExposureForbidden`` from ``ledger.verify()``), a manifest or arms that do
not verify, a reward-vs-quality correlation measured on anything but a
lineage-disjoint free-running development cohort.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from unflatten.model.hashing import domain_hash
from unflatten.model.rl.exposure import RlExposureLedger
from unflatten.model.rl.gate_policy import POLICY_HASH_FIELD, decide, evaluate_results
from unflatten.model.rl.ids import STAGE_GATE_REPORT_DOMAIN, STAGE_ID
from unflatten.model.rl.paired_plan import verify_paired_arms
from unflatten.model.rl.policies import FORBIDDEN_SOURCES, MANIFEST_HASH_FIELD, _schema_check
from unflatten.model.rl.run_manifest import RunArtefacts, policy_status, verify_rl_run_manifest

SCHEMA_VERSION: Final = "1.0.0"
REPORT_HASH_FIELD: Final = "stage_gate_report_hash"
_SCHEMA: Final = "rl_stage_gate_report"
COHORT_OK: Final = "dev_validation_free_running"
#: RFC006-ERRATA-001 E1(b) 1-7, in order.
CONDITION_IDS: Final = (
    "commit_authority_unchanged",
    "renderer_evaluator_no_gradient",
    "no_refiner_weights_learned",
    "reward_lexicographic",
    "rollouts_host_side_only",
    "train_data_only",
    "no_refiner_component_lock_dependency",
)
REFUSED_COHORTS: Final = ("shaping_rollouts", "teacher_forced_dev", *FORBIDDEN_SOURCES)


class RlGateReportInvalid(ValueError):
    """The report cannot be built or does not verify."""


class RlRewardCohortRefused(RlGateReportInvalid):
    """Reward-vs-quality correlation measured on an inadmissible cohort."""


@dataclass(frozen=True, slots=True)
class GateMeasurements:
    rfc006_conditions: Mapping[str, Mapping[str, Any]]
    zero_advantage_rate_micro: int
    reward_hacking_signal: Mapping[str, Any]
    reward_quality: Mapping[str, Any]
    gate_evidence: Mapping[str, Mapping[str, Any]]


def _check_conditions(conditions: Mapping[str, Mapping[str, Any]]) -> None:
    if set(conditions) != set(CONDITION_IDS):
        raise RlGateReportInvalid(
            f"rfc006_conditions must be exactly {CONDITION_IDS}; "
            f"missing {sorted(set(CONDITION_IDS).difference(conditions))}, "
            f"unexpected {sorted(set(conditions).difference(CONDITION_IDS))}"
        )
    for name, c in conditions.items():
        if not isinstance(c.get("holds"), bool) or not c.get("evidence_ref"):
            raise RlGateReportInvalid(
                f"condition {name!r} needs a boolean 'holds' and an evidence_ref"
            )


def _check_cohort(rq: Mapping[str, Any], arts: RunArtefacts, ledger: RlExposureLedger) -> None:
    cohort = rq.get("cohort")
    if cohort in REFUSED_COHORTS or cohort != COHORT_OK:
        raise RlRewardCohortRefused(
            f"reward-quality correlation measured on cohort {cohort!r}; only lineage-disjoint "
            f"{COHORT_OK!r} is admissible (ERRATA-010 E7.3d)"
        )
    if type(rq.get("correlation_micro")) is not int or not rq.get("evidence_ref"):
        raise RlGateReportInvalid(
            "reward_quality needs an int correlation_micro and an evidence_ref"
        )
    ids = list(rq.get("lineage_ids", ()))
    if not ids or len(set(ids)) != len(ids):
        raise RlRewardCohortRefused("cohort lineage_ids must be a non-empty list of unique ids")
    dm = arts.dataset_manifest
    shaped = set(ids) & (ledger.lineage_ids() | set(dm["lineage_ids"]))
    if shaped:
        raise RlRewardCohortRefused(
            f"correlation cohort overlaps the lineages the reward shaped: {sorted(shaped)}"
        )
    barred = set(ids) & set(dm["exclude_lineage_ids"])
    if barred:
        raise RlRewardCohortRefused(
            f"correlation cohort uses calibration/dev/control lineages: {sorted(barred)}"
        )


_EVIDENCE_KEYS: Final = {
    "zero_count": ("count",),
    "non_regression": ("candidate_lower_bound_micro", "diff_lower_bound_micro"),
    "benefit": ("diff_lower_bound_micro",),
    "not_worse": ("diff_lower_bound_micro",),
}


def _check_evidence(gate_id: str, kind: str, ev: Mapping[str, Any]) -> None:
    if ev.get("kind") != kind:
        raise RlGateReportInvalid(
            f"{gate_id}: evidence kind {ev.get('kind')!r} != policy kind {kind!r}"
        )
    keys = ("independent_lineages", "attempted_denominator", *_EVIDENCE_KEYS[kind])
    bad = [k for k in keys if type(ev.get(k)) is not int]
    if bad or not ev.get("evidence_ref"):
        raise RlGateReportInvalid(
            f"{gate_id}: evidence needs integer {bad or list(keys)} and an evidence_ref"
        )


def _check_ledger(
    manifest: Mapping[str, Any], arts: RunArtefacts, ledger: RlExposureLedger
) -> None:
    ledger.verify()
    pairs = (
        ("run_id", ledger.run_id, manifest["run_id"]),
        (
            "dataset_manifest_hash",
            ledger.dataset_manifest_hash,
            arts.dataset_manifest[MANIFEST_HASH_FIELD],
        ),
        ("reward_policy_hash", ledger.reward_policy_hash, manifest["reward_policy_hash"]),
    )
    for name, got, want in pairs:
        if got != want:
            raise RlGateReportInvalid(f"ledger {name} {got!r} != run manifest {want!r}")


def build_rl_stage_gate_report(
    *,
    manifest: Mapping[str, Any],
    artefacts: RunArtefacts,
    ledger: RlExposureLedger,
    rl_arm: Mapping[str, Any],
    sft_arm: Mapping[str, Any] | None,
    candidate_checkpoint_id: str,
    measurements: GateMeasurements,
) -> dict[str, Any]:
    verify_rl_run_manifest(manifest, artefacts)
    _check_ledger(manifest, artefacts, ledger)
    if sft_arm is not None:
        verify_paired_arms(rl_arm, sft_arm)
    m = measurements
    _check_conditions(m.rfc006_conditions)
    _check_cohort(m.reward_quality, artefacts, ledger)
    if not candidate_checkpoint_id:
        raise RlGateReportInvalid("candidate_checkpoint_id must be a non-empty string")
    gates = list(artefacts.stage_policy["gates"])
    by_id = {g["gate_id"]: g for g in gates}
    for gate_id, ev in m.gate_evidence.items():
        if gate_id not in by_id:
            raise RlGateReportInvalid(f"evidence for unknown gate {gate_id!r}")
        _check_evidence(gate_id, by_id[gate_id]["kind"], ev)
    results = evaluate_results(
        artefacts.stage_policy,
        m.gate_evidence,
        m.reward_quality,
        m.zero_advantage_rate_micro,
    )
    blockers = []
    if sft_arm is None:
        blockers.append("missing_sft_continuation_arm")
    if m.reward_hacking_signal.get("detected") is True:
        blockers.append("reward_hacking_signal")
    blockers += [
        f"condition_not_held:{n}" for n in CONDITION_IDS if not m.rfc006_conditions[n]["holds"]
    ]
    status = policy_status(artefacts)
    decision, reason = decide(
        results,
        blockers,
        status=status,
        run_class=manifest["run_class"],
        ratified=manifest["ratification_record_hash"] is not None,
    )
    ledger_doc = ledger.to_document()
    body = {
        "schema_version": SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "run_class": manifest["run_class"],
        "run_id": manifest["run_id"],
        "run_manifest_hash": manifest["run_manifest_hash"],
        "stage_policy_hash": artefacts.stage_policy[POLICY_HASH_FIELD],
        "policy_status": status,
        "ratification_record_hash": manifest["ratification_record_hash"],
        "dataset_manifest_hash": artefacts.dataset_manifest[MANIFEST_HASH_FIELD],
        "exposure_ledger_hash": ledger_doc["exposure_ledger_hash"],
        "exposure_record_count": ledger_doc["record_count"],
        "rl_arm_hash": rl_arm["paired_arm_hash"],
        "sft_arm_hash": None if sft_arm is None else sft_arm["paired_arm_hash"],
        "parent_checkpoint_id": artefacts.policy_parent.checkpoint_id,
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "rfc006_conditions": {n: dict(m.rfc006_conditions[n]) for n in CONDITION_IDS},
        "zero_advantage_rate_micro": m.zero_advantage_rate_micro,
        "reward_hacking_signal": dict(m.reward_hacking_signal),
        "reward_quality": {
            **dict(m.reward_quality),
            "lineage_ids": sorted(m.reward_quality["lineage_ids"]),
        },
        "gate_evidence": {k: dict(v) for k, v in sorted(m.gate_evidence.items())},
        "results": results,
        "blockers": blockers,
        "decision": decision,
        "decision_reason": reason,
    }
    doc = {**body, REPORT_HASH_FIELD: domain_hash(STAGE_GATE_REPORT_DOMAIN, body)}
    _schema_check(_SCHEMA, doc)
    return doc


def verify_rl_stage_gate_report(
    report: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    artefacts: RunArtefacts,
    ledger: RlExposureLedger,
    rl_arm: Mapping[str, Any],
    sft_arm: Mapping[str, Any] | None,
) -> None:
    try:
        _schema_check(_SCHEMA, report)
    except ValueError as exc:
        raise RlGateReportInvalid(f"report schema: {exc}") from exc
    body = {k: v for k, v in report.items() if k != REPORT_HASH_FIELD}
    if domain_hash(STAGE_GATE_REPORT_DOMAIN, body) != report[REPORT_HASH_FIELD]:
        raise RlGateReportInvalid("stage_gate_report_hash does not verify against the payload")
    rebuilt = build_rl_stage_gate_report(
        manifest=manifest,
        artefacts=artefacts,
        ledger=ledger,
        rl_arm=rl_arm,
        sft_arm=sft_arm,
        candidate_checkpoint_id=report["candidate_checkpoint_id"],
        measurements=GateMeasurements(
            rfc006_conditions=report["rfc006_conditions"],
            zero_advantage_rate_micro=report["zero_advantage_rate_micro"],
            reward_hacking_signal=report["reward_hacking_signal"],
            reward_quality=report["reward_quality"],
            gate_evidence=report["gate_evidence"],
        ),
    )
    if rebuilt != dict(report):
        differing = sorted(k for k in rebuilt if rebuilt[k] != report.get(k))
        raise RlGateReportInvalid(
            f"report differs from the one re-derived from its real inputs at {differing}"
        )
