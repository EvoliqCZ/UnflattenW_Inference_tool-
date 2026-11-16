"""RFC005 §23.3/§53 #51 Product Audit Plan — mechanism only.

§23.3: "Audit-plan hash pokrýva required test branch manifests, frozen
`candidate_mode`/`promotion_scope`, product claim/deployment scope,
ScoringMask, metric a critical-slice registries, absolute floors a
non-inferiority margins, CI/bootstrap/multiplicity policy, human-
evaluation protocol a sealed evaluator runtime/code." This module
provides the closed field set and its hash, and nothing else — no
default thresholds, no S9 wiring, no gate. Per the task instruction,
this is mechanism only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from unflatten.model.hashing import domain_hash

_DOMAIN = "unflatten.rfc005.product_audit_plan.v0"

#: RFC005 §22.6 — the only two closed `candidate_mode` values.
CANDIDATE_MODES = ("auto_accept", "review_only")


@dataclass(frozen=True)
class ProductAuditPlan:
    """The closed field set §23.3 requires the audit-plan hash to cover."""

    required_test_branch_manifest_hashes: tuple[str, ...]
    candidate_mode: str
    promotion_scope: str
    product_claim_scope: str
    deployment_scope: str
    scoring_mask_policy_hash: str
    metric_registry_hash: str
    critical_slice_registry_hash: str
    absolute_floors_hash: str
    non_inferiority_margins_hash: str
    ci_bootstrap_multiplicity_policy_hash: str
    human_evaluation_protocol_hash: str
    sealed_evaluator_runtime_hash: str

    def __post_init__(self) -> None:
        if not self.required_test_branch_manifest_hashes:
            raise ValueError("required_test_branch_manifest_hashes must be non-empty")
        if self.candidate_mode not in CANDIDATE_MODES:
            raise ValueError(
                f"candidate_mode must be one of {CANDIDATE_MODES}, got {self.candidate_mode!r}"
            )

    def compute_hash(self) -> str:
        """The audit-plan hash §23.3 requires access records to bind to."""
        payload = asdict(self)
        payload["required_test_branch_manifest_hashes"] = list(
            payload["required_test_branch_manifest_hashes"]
        )
        return domain_hash(_DOMAIN, payload)
