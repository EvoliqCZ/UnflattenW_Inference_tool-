"""Reward of one rollout candidate (design D1/D6; PROPOSED, dormant).

Chain adapter -> ``score_candidate``. ``(status, scalar_micro)`` are those of the
sealed evaluation, a missing value stays ``None``. A stream the builder could not
even report on (``BuilderReportInvalid``) is a named, counted FAILED candidate;
every other exception propagates as a defect. The LibreOffice probe is status
only and outside every hash (it runs only when the rollout policy says so).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.model.builder_report import BuilderReportInvalid
from unflatten.model.rl.chain_adapter import (
    AdapterBuild,
    candidate_inputs_from_trajectory,
    control_event_counts,
)
from unflatten.model.rl.real_run_ports import LoopContext, PageGold, PageVisual
from unflatten.model.rl.reward import EVALUATION_HASH_FIELD, score_candidate
from unflatten.model.rl.reward_inputs import CandidateInputs
from unflatten.model.rl.rollout import RolloutTrajectory

BUILDER_CRASH_OUTCOME = "BuilderReportInvalid"


@dataclass(frozen=True)
class Scored:
    status: str
    scalar_micro: int | None
    outcome: str
    probe_status: str | None
    components: Mapping[str, Any] | None = None
    gates: Sequence[Mapping[str, Any]] | None = None


def _builder_crash_inputs(
    ctx: LoopContext, trajectory: RolloutTrajectory, candidate_id: str
) -> CandidateInputs:
    """A FAILED candidate for a stream the builder could not even report on
    (``BuilderReportInvalid``, e.g. an unclamped integer head: ``/columns/count`` -2
    derives a negative container length). Named, counted in the telemetry under
    its outcome and never repaired; any other builder exception still propagates."""
    failed = AdapterBuild("failed", None, BUILDER_CRASH_OUTCOME)
    return CandidateInputs(
        candidate_id=candidate_id,
        chain_flags=dict(ctx.reward_policy["chain_policy_flags"]),
        build_candidate=lambda: failed,
        control_event_counts=control_event_counts(trajectory.token_ids, ctx.opened.vocabulary),
        rfc002_render="not_run",
        typed={},
        fragment_violations=(),
        continuity_micro=None,
    )


def score_page_candidate(
    ctx: LoopContext,
    trajectory: RolloutTrajectory,
    predictions: Sequence[Any],
    visual: PageVisual,
    gold: PageGold,
    candidate_id: str,
    regions: Sequence[Any] = (),
) -> Scored:
    """Status, scalar, build outcome and (status-only) render probe of one candidate."""
    try:
        cand = candidate_inputs_from_trajectory(
            trajectory,
            predictions,
            candidate_id=candidate_id,
            document_bundle_id=gold.document_bundle_id,
            chain_flags=ctx.reward_policy["chain_policy_flags"],
            gold=gold.gold,
            gold_node_ids=gold.gold_node_ids,
            gold_profile=gold.gold_profile,
            vocabulary=ctx.opened.vocabulary,
            tokenizer=ctx.opened.tokenizer,
            raster_source=visual.raster_source,
            render_dir=ctx.ports.render_dir,
            region_predictions=regions,
        )
    except BuilderReportInvalid:
        cand = _builder_crash_inputs(ctx, trajectory, candidate_id)
    evaluation = score_candidate(cand, gold.gold, ctx.reward_policy)
    probe_status = None
    if ctx.rollout_policy["render_probe"] == "status_only" and cand.rfc002_render == "rendered":
        assert ctx.ports.render_dir is not None and ctx.ports.render_probe is not None
        docx = Path(ctx.ports.render_dir) / f"{gold.document_bundle_id}.docx"
        if docx.exists():
            probe_status = ctx.ports.render_probe(docx, evaluation[EVALUATION_HASH_FIELD], 0).status
    outcome = cand.build_candidate().page_outcome  # type: ignore[attr-defined]
    return Scored(
        evaluation["status"],
        evaluation["scalar_micro"],
        outcome,
        probe_status,
        evaluation.get("components"),
        evaluation.get("gates"),
    )
