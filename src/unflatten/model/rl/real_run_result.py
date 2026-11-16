"""Result and per-update bookkeeping of the S4.5 real run (PROPOSED, dormant)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from unflatten.model.rl.exposure import RlExposureLedger
from unflatten.model.rl.real_run_ports import PageRecord
from unflatten.model.rl.update import RlStepResult


@dataclass(frozen=True)
class StepTelemetry:
    """Per-update summary. Diagnostics only; none of it is a quality claim."""

    step: int
    loss: float
    policy_loss: float
    kl_mean: float
    grad_norm: float
    clipped: bool
    lr_scale_micro: int
    zero_advantage_rate_micro: int | None
    groups_total: int
    excluded_insufficient_evidence: int
    gate_failed_count: int
    rollouts: int
    scored: int
    guard_intervention_counts: tuple[int | None, ...]
    #: build outcome -> number of rollouts of this update, sorted by outcome.
    outcomes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class RealRunResult:
    run_id: str
    start_step: int
    steps_done: int
    updates_run: int
    stopped_by: str
    checkpoint_paths: tuple[Path, ...]
    ledger_record_count: int
    ledger_head_hash: str | None
    data_plan_hash: str
    selection_hash: str
    telemetry: tuple[StepTelemetry, ...]
    #: never True: a dormant run is not evidence of quality.
    quality_claim: bool = False


def step_telemetry(result: RlStepResult, records: list[PageRecord]) -> StepTelemetry:
    return StepTelemetry(
        step=result.step,
        loss=result.loss,
        policy_loss=result.policy_loss,
        kl_mean=result.kl_mean,
        grad_norm=result.grad_norm,
        clipped=result.clipped,
        lr_scale_micro=result.lr_scale_micro,
        zero_advantage_rate_micro=result.zero_advantage_rate_micro,
        groups_total=result.groups_total,
        excluded_insufficient_evidence=result.excluded_insufficient_evidence,
        gate_failed_count=result.gate_failed_count,
        rollouts=sum(r.rollout_count for r in records),
        scored=sum(1 for r in records for s in r.statuses if s == "scored"),
        guard_intervention_counts=tuple(c for r in records for c in r.guard_intervention_counts),
        outcomes=tuple(sorted(Counter(o for r in records for o in r.outcomes).items())),
    )


def append_ledger(
    ledger: RlExposureLedger, records: list[tuple[PageRecord, int]], step: int
) -> None:
    for rec, epoch in records:
        ledger.append(
            lineage_root_id=rec.lineage_root_id,
            source=rec.source,
            cell=rec.cell,
            epoch=epoch,
            view_ids=(rec.view_id,),
            rollout_count=rec.rollout_count,
            guard_intervention_counts=rec.guard_intervention_counts,
            rollout_seed_digest=rec.rollout_seed_digest,
            optimizer_step=step,
        )
