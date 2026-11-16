"""RFC006 §26 Relative Fidelity Verifier — deterministic decision logic.

"Commit-authoritative improvement MUSÍ vždy nanovo vypočítať independent
frozen Render Comparison Engine" (§26.1): the hard gates below are
computed directly from the `ConservationReport` and an injected
`RenderEngine`, never from a learned critic's confidence or defect
count.

Scope: gates 1-9 (§26.3) are exactly the `ConservationReport` plus
render-readiness/profile agreement; gates 10-12 (no new
clipping/overflow, no new unsupported fallback, visited-state) are
checked against the injected `RenderObservation`. The full pixel metric
vector (§26.4/§26.5, Tier 0-4 with real comparison channels) is out of
scope here (see `render_engine` module docstring) — `Policy.metric_key`
names whichever single lower-is-better integer channel the caller wired
up as the targeted-improvement metric, and `protected_metric_keys`
names the channels treated as protected regressions (§26.6 "no protected
metric regression beyond budget").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from unflatten.ast.hashing import ast_hash
from unflatten.refinement.patch_applier import ConservationReport
from unflatten.refinement.patch_language import ASTPatchSet
from unflatten.refinement.render_engine import RenderEngine, RenderObservation

Decision = Literal["accept_candidate", "rollback_candidate", "abstain"]


@dataclass(frozen=True)
class VerifierPolicy:
    """Frozen decision policy (RFC006 §26.4 "Metric Registry", simplified
    to the one targeted channel and the protected-channel budgets this
    package's `RenderObservation.metrics` actually carries)."""

    metric_key: str
    min_improvement: int = 1
    protected_metric_keys: tuple[str, ...] = ()
    max_protected_regression: int = 0


@dataclass(frozen=True)
class VerifierResult:
    decision: Decision
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _hard_gate_reasons(
    conservation: ConservationReport,
    candidate_ast: dict,
    baseline_obs: RenderObservation,
    candidate_obs: RenderObservation,
    visited_ast_hashes: frozenset[str],
) -> tuple[str, ...]:
    reasons: list[str] = []
    reasons.extend(conservation.violations)  # gates 4-9
    if not candidate_obs.render_ready:  # gate 2
        reasons.append("RENDER_NOT_READY")
    if baseline_obs.profile != candidate_obs.profile:  # gate 3/8
        reasons.append("RENDERER_PROFILE_CHANGED")
    if candidate_obs.clipping_overflow_count > baseline_obs.clipping_overflow_count:  # gate 10
        reasons.append("NEW_CLIPPING_OR_OVERFLOW")
    new_fallback = (
        candidate_obs.unsupported_fallback_count > baseline_obs.unsupported_fallback_count
    )
    if new_fallback:  # gate 11
        reasons.append("NEW_UNSUPPORTED_FALLBACK")
    if ast_hash(candidate_ast) in visited_ast_hashes:  # gate 12
        reasons.append("CANDIDATE_ALREADY_VISITED")
    return tuple(reasons)


def decide(
    *,
    candidate_ast: dict,
    conservation: ConservationReport,
    render_engine: RenderEngine,
    baseline_ast: dict,
    policy: VerifierPolicy,
    visited_ast_hashes: frozenset[str] = frozenset(),
) -> VerifierResult:
    """RFC006 §26.6 Decision, over one candidate.

    Renders both `baseline_ast` and `candidate_ast` in full (§26.8 "Local
    crop-only acceptance je zakázaná" — there is no crop-only code path
    here to begin with).
    """
    baseline_obs = render_engine.render(baseline_ast)
    candidate_obs = render_engine.render(candidate_ast)

    hard_gate_reasons = _hard_gate_reasons(
        conservation, candidate_ast, baseline_obs, candidate_obs, visited_ast_hashes
    )
    if hard_gate_reasons:
        return VerifierResult("rollback_candidate", hard_gate_reasons)

    improvement = baseline_obs.metrics[policy.metric_key] - candidate_obs.metrics[policy.metric_key]
    if improvement <= 0:
        return VerifierResult("rollback_candidate", ("TIE_OR_WORSE",))
    if improvement < policy.min_improvement:
        return VerifierResult("abstain", ("BELOW_IMPROVEMENT_MARGIN",))

    for key in policy.protected_metric_keys:
        regression = candidate_obs.metrics[key] - baseline_obs.metrics[key]
        if regression > policy.max_protected_regression:
            return VerifierResult("rollback_candidate", (f"PROTECTED_REGRESSION:{key}",))

    return VerifierResult("accept_candidate", ())


@dataclass(frozen=True)
class Candidate:
    """One offline/shadow candidate for `select_best` (§26.7 tie-break)."""

    ast: dict
    patch_set: ASTPatchSet
    conservation: ConservationReport


def select_best(
    candidates: list[Candidate],
    *,
    render_engine: RenderEngine,
    baseline_ast: dict,
    policy: VerifierPolicy,
    visited_ast_hashes: frozenset[str] = frozenset(),
) -> tuple[Candidate, VerifierResult] | None:
    """RFC006 §26.7 tie-break over multiple candidates.

    Order: larger target improvement -> smaller worst protected
    regression -> fewer patch operations -> lexicographically smaller
    patch hash. Returns `None` if no candidate cleared the absolute
    improvement gate (§26.7 "Najlepší kandidát sa však nesmie prijať, ak
    žiadny neprekročil absolute improvement gate").
    """
    baseline_obs = render_engine.render(baseline_ast)
    scored: list[tuple[tuple[int, int, int, str], Candidate, VerifierResult]] = []
    for candidate in candidates:
        result = decide(
            candidate_ast=candidate.ast,
            conservation=candidate.conservation,
            render_engine=render_engine,
            baseline_ast=baseline_ast,
            policy=policy,
            visited_ast_hashes=visited_ast_hashes,
        )
        if result.decision != "accept_candidate":
            continue
        candidate_obs = render_engine.render(candidate.ast)
        target_key = policy.metric_key
        improvement = baseline_obs.metrics[target_key] - candidate_obs.metrics[target_key]
        worst_regression = max(
            (
                candidate_obs.metrics[k] - baseline_obs.metrics[k]
                for k in policy.protected_metric_keys
            ),
            default=0,
        )
        key = (
            -improvement,
            worst_regression,
            len(candidate.patch_set.operations),
            candidate.patch_set.patch_set_hash,
        )
        scored.append((key, candidate, result))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    _, candidate, result = scored[0]
    return candidate, result
