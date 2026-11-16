"""S4.5 promotion-gate policy and per-gate evaluation (RFC005-ERRATA-010 E7.3/E7.4).

The general stage gate policy (``general_stage_gate_policy``) is deliberately
NOT reused: its schema pins ``stage`` to S6/S7/S8 (owner decision: do not widen
it), its evaluator returns bootstrap floats in ``detail`` (this package's CBOR
profile has none), and it has no "measurable benefit" direction. The semantics
that matter are the same and are kept: a placeholder threshold, a missing
evidence record or an under-powered cohort is ``insufficient_evidence``, never
pass/fail. Confidence bounds are integer-micro lower bounds produced by the
evaluation harness (lineage-level bootstrap, run outside this package) and
carried in the evidence record with a reference.

Gate groups follow E7.3: ``a`` zero invalid-success / silent loss, ``b``
non-regression (exact text, Unicode/diacritics, structure, tables, fragments
and critical slices), ``c`` benefit against S4 AND the compute-matched SFT
continuation. Reward-vs-quality correlation (d) and reproducibility (e) live
in the report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from unflatten.model.rl.ids import STAGE_ID, STAGE_POLICY_DOMAIN
from unflatten.model.rl.policies import STATUSES, RlPolicyInvalid, _seal, _verify

POLICY_SCHEMA: Final = "rl_stage_gate_policy"
POLICY_HASH_FIELD: Final = "stage_policy_hash"
SCHEMA_VERSION: Final = "1.0.0"
SLICE_PREFIX: Final = "non_regression_slice:"

#: gate_id -> (group, kind); the closed core of E7.3. Critical slices are added.
REQUIRED_GATES: Final = {
    "invalid_success_zero": ("a", "zero_count"),
    "silent_loss_zero": ("a", "zero_count"),
    "non_regression_exact_text": ("b", "non_regression"),
    "non_regression_unicode_diacritics": ("b", "non_regression"),
    "non_regression_structure": ("b", "non_regression"),
    "non_regression_tables": ("b", "non_regression"),
    "non_regression_fragments": ("b", "non_regression"),
    "benefit_vs_s4": ("c", "benefit"),
    "benefit_vs_sft_continuation": ("c", "benefit"),
    # D2 mitigation: the update path does not cover loop-guard substitutions, so
    # dependence on the guard must not grow. Evidence is the share (micro) of
    # candidates with >= 1 intervention: ``diff_lower_bound_micro`` is the lower
    # bound of (S4 parent share - candidate share); passes when >= -margin_micro.
    "guard_intervention": ("b", "not_worse"),
}
#: Threshold keys that apply to each gate kind; the others must be null.
APPLICABLE: Final = {
    "zero_count": ("min_independent_lineages", "min_attempted_denominator"),
    "non_regression": (
        "floor_micro",
        "margin_micro",
        "min_independent_lineages",
        "min_attempted_denominator",
    ),
    "benefit": ("margin_micro", "min_independent_lineages", "min_attempted_denominator"),
    "not_worse": ("margin_micro", "min_independent_lineages", "min_attempted_denominator"),
}
_ALL_KEYS: Final = (
    "floor_micro",
    "margin_micro",
    "min_independent_lineages",
    "min_attempted_denominator",
)


class RlGatePolicyInvalid(RlPolicyInvalid):
    """The S4.5 gate policy is malformed, incomplete or inconsistent with its status."""


def gate_entry_placeholder(entry: Mapping[str, Any]) -> bool:
    return any(entry[k] is None for k in APPLICABLE[entry["kind"]])


def _entry(gate_id: str, group: str, kind: str, given: Mapping[str, int]) -> dict[str, Any]:
    stray = sorted(set(given) - set(APPLICABLE[kind]))
    if stray:
        raise RlGatePolicyInvalid(f"{gate_id}: threshold(s) {stray} do not apply to kind {kind!r}")
    entry: dict[str, Any] = {"gate_id": gate_id, "group": group, "kind": kind}
    for key in _ALL_KEYS:
        entry[key] = given.get(key)
    return entry


def build_rl_stage_gate_policy(
    *,
    critical_slice_ids: Sequence[str],
    thresholds: Mapping[str, Mapping[str, int]] | None = None,
    min_correlation_micro: int | None = None,
    min_correlation_lineages: int | None = None,
    max_zero_advantage_rate_micro: int | None = None,
    status: str = "PROPOSED",
) -> dict[str, Any]:
    if status not in STATUSES:
        raise RlGatePolicyInvalid(f"status {status!r} not in {STATUSES}")
    slices = list(critical_slice_ids)
    if not slices or len(set(slices)) != len(slices) or any(not s for s in slices):
        raise RlGatePolicyInvalid("critical_slice_ids must be a non-empty list of unique ids")
    spec = dict(REQUIRED_GATES)
    spec.update({f"{SLICE_PREFIX}{s}": ("b", "non_regression") for s in slices})
    given = dict(thresholds or {})
    unknown = sorted(set(given) - set(spec))
    if unknown:
        raise RlGatePolicyInvalid(f"thresholds for unknown gate id(s): {unknown}")
    gates = [_entry(g, grp, kind, given.get(g, {})) for g, (grp, kind) in spec.items()]
    doc = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "stage_id": STAGE_ID,
        "gates": gates,
        "reward_quality": {
            "min_correlation_micro": min_correlation_micro,
            "min_independent_lineages": min_correlation_lineages,
        },
        "shaping": {"max_zero_advantage_rate_micro": max_zero_advantage_rate_micro},
    }
    _check_semantics({**doc, POLICY_HASH_FIELD: "sha256:" + "0" * 64})
    return _seal(POLICY_SCHEMA, STAGE_POLICY_DOMAIN, POLICY_HASH_FIELD, doc)


def _check_semantics(doc: Mapping[str, Any]) -> None:
    ids = [g["gate_id"] for g in doc["gates"]]
    if len(set(ids)) != len(ids):
        raise RlGatePolicyInvalid(f"duplicate gate ids: {ids}")
    missing = sorted(set(REQUIRED_GATES) - set(ids))
    if missing:
        raise RlGatePolicyInvalid(f"required gate(s) missing from the policy: {missing}")
    if not any(i.startswith(SLICE_PREFIX) for i in ids):
        raise RlGatePolicyInvalid("no critical-slice non-regression gate in the policy")
    for g in doc["gates"]:
        if (
            g["gate_id"] in REQUIRED_GATES
            and (g["group"], g["kind"]) != REQUIRED_GATES[g["gate_id"]]
        ):
            raise RlGatePolicyInvalid(f"{g['gate_id']}: group/kind altered from E7.3")
        stray = [k for k in _ALL_KEYS if k not in APPLICABLE[g["kind"]] and g[k] is not None]
        if stray:
            raise RlGatePolicyInvalid(f"{g['gate_id']}: threshold(s) {stray} do not apply")
        if g["kind"] == "benefit" and g["margin_micro"] == 0:
            raise RlGatePolicyInvalid(f"{g['gate_id']}: a benefit margin of 0 is not a benefit")
    if doc["status"] == "RATIFIED" and policy_placeholders(doc):
        raise RlGatePolicyInvalid(
            f"RATIFIED policy still carries placeholders: {policy_placeholders(doc)}"
        )


def policy_placeholders(doc: Mapping[str, Any]) -> list[str]:
    out = [g["gate_id"] for g in doc["gates"] if gate_entry_placeholder(g)]
    if any(v is None for v in doc["reward_quality"].values()):
        out.append("reward_quality")
    if doc["shaping"]["max_zero_advantage_rate_micro"] is None:
        out.append("shaping.max_zero_advantage_rate")
    return out


def verify_rl_stage_gate_policy(doc: Mapping[str, Any]) -> None:
    _verify(POLICY_SCHEMA, STAGE_POLICY_DOMAIN, POLICY_HASH_FIELD, doc)
    _check_semantics(doc)


def evaluate_gate(entry: Mapping[str, Any], evidence: Mapping[str, Any] | None) -> tuple[str, str]:
    """``(verdict, reason)`` with verdict in pass | fail | insufficient_evidence."""
    if gate_entry_placeholder(entry):
        return "insufficient_evidence", "policy_placeholder_not_frozen"
    if evidence is None:
        return "insufficient_evidence", "missing_evidence"
    if evidence["independent_lineages"] < entry["min_independent_lineages"]:
        return "insufficient_evidence", "minimum_independent_lineages_not_met"
    if evidence["attempted_denominator"] < entry["min_attempted_denominator"]:
        return "insufficient_evidence", "minimum_attempted_denominator_not_met"
    kind = entry["kind"]
    if kind == "zero_count":
        ok = evidence["count"] == 0
        return ("pass", "ok") if ok else ("fail", "nonzero_count")
    if kind == "non_regression":
        ok = (
            evidence["candidate_lower_bound_micro"] >= entry["floor_micro"]
            and evidence["diff_lower_bound_micro"] >= -entry["margin_micro"]
        )
        return ("pass", "ok") if ok else ("fail", "floor_or_margin_not_satisfied")
    if kind == "not_worse":
        ok = evidence["diff_lower_bound_micro"] >= -entry["margin_micro"]
        return ("pass", "ok") if ok else ("fail", "worse_than_parent")
    ok = evidence["diff_lower_bound_micro"] >= entry["margin_micro"]
    return ("pass", "ok") if ok else ("fail", "benefit_not_demonstrated")


def evaluate_results(
    policy: Mapping[str, Any],
    gate_evidence: Mapping[str, Mapping[str, Any]],
    reward_quality: Mapping[str, Any],
    zero_advantage_rate_micro: int,
) -> dict[str, Any]:
    """Per-gate, reward-quality and zero-advantage results under ``policy``."""
    gates = policy["gates"]
    results = [
        {
            "gate_id": g["gate_id"],
            **dict(zip(("verdict", "reason"), evaluate_gate(g, gate_evidence.get(g["gate_id"])))),
        }
        for g in gates
    ]
    rq, rqp = reward_quality, policy["reward_quality"]
    if rqp["min_correlation_micro"] is None or rqp["min_independent_lineages"] is None:
        rq_res = ("insufficient_evidence", "policy_placeholder_not_frozen")
    elif len(rq["lineage_ids"]) < rqp["min_independent_lineages"]:
        rq_res = ("insufficient_evidence", "minimum_independent_lineages_not_met")
    elif rq["correlation_micro"] < rqp["min_correlation_micro"]:
        rq_res = ("fail", "correlation_below_minimum")
    else:
        rq_res = ("pass", "ok")
    cap = policy["shaping"]["max_zero_advantage_rate_micro"]
    if cap is None:
        za_res = ("insufficient_evidence", "policy_placeholder_not_frozen")
    elif zero_advantage_rate_micro > cap:
        za_res = ("fail", "zero_advantage_rate_above_maximum")
    else:
        za_res = ("pass", "ok")
    return {
        "gates": results,
        "reward_quality": dict(zip(("verdict", "reason"), rq_res)),
        "zero_advantage": dict(zip(("verdict", "reason"), za_res)),
    }


def decide(
    results: Mapping[str, Any], blockers: list[str], *, status: str, run_class: str, ratified: bool
) -> tuple[str, str]:
    """``(decision, decision_reason)``; order documented in ``gate_report``."""
    if blockers:
        return "blocked", "blocked"
    verdicts = [g["verdict"] for g in results["gates"]]
    verdicts += [results["reward_quality"]["verdict"], results["zero_advantage"]["verdict"]]
    if "insufficient_evidence" in verdicts:
        return "insufficient_evidence", "placeholder_or_missing_evidence"
    if status != "RATIFIED":  # thresholds of a PROPOSED policy are not frozen: no verdict on them
        return "insufficient_evidence", "policy_not_ratified"
    if "fail" in verdicts:
        return "rejected", "gate_failed"
    if run_class != "promotion":
        return "insufficient_evidence", "discovery_run_not_promotable"
    if not ratified:
        return "insufficient_evidence", "ratification_record_missing"
    return "promoted", "all_gates_passed"
