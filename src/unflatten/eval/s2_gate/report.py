"""S2 exit-gate measurement -- the six criteria (design M5, governance M5).

DEC-009 makes tonight's S2 run discovery-class. A six-criterion conjunction
named "overall verdict" is shaped exactly like a Stage Gate Report, which a
discovery-class run may not produce (governance review M5). So this module
never emits a field named `verdict` or `pass`, and never fuses the six
criteria into one boolean. Instead it emits `criteria_state` -- each of the
six criteria mapped to one of `measured | blocked | insufficient_evidence`
-- plus a top-level `promotable` and `promotion_blockers`. Per memory
`literal-true-when-written`, a value that reads as a verdict outlives the
prose that qualified it, so `promotable` is never a literal: it is
computed by `_compute_promotable` from the criteria states and the blocker
list, and `test_report.py` proves the path on which it comes out `True` --
a state nothing can reach is a state that does not exist.

**Criteria 2 and 4 resolve `blocked` by construction, not by omission**
(governance M2), *unless a frozen S2 stage gate policy is supplied*:
§12.2 conditions every expression of "better" on a pre-frozen
`stage-gate-policy`, and today that policy does not exist, so both
criteria default to `blocked` (`no_frozen_stage_gate_policy` /
`no_predeclared_regression_margin`). Since DEC-012, a validated S2 gate
policy (loaded through `unflatten.model.s2_stage_gate_policy`, whose
`quality` branch is `applies`) can be passed to
`criterion_2_exact_text_vs_baseline` and `criterion_4_slices` as
`s2_gate_policy`; then each resolves `measured` or
`insufficient_evidence` from the policy's own thresholds, with the
determination (and the policy's hash) in `detail["policy_determination"]`
-- a policy that fails validation is refused, never defaulted. With no
policy, both criteria still report their numbers; neither reports a
comparison verdict.

**Criterion 6 resolves `blocked` unless a real tier map is supplied**
(design F1): there is no pyramid-tier label anywhere in the m2a-v6 store or
the `-c5`/eval caches. `criterion_6_pyramid_tiers` is reachable both ways
-- `test_report.py` supplies a fake tier map and proves the criterion
becomes `measured`, so `blocked` is proven not to be hardcoded (design §9
guard 4).

**`insufficient_evidence` is distinct from `blocked`** (governance,
"Cohort size" section): `blocked` is §12.2's word for a missing policy;
`insufficient_evidence` is §39.5's word for a cohort too small to support
an interval. There is no ratified minimum cohort size -- `MIN_LINEAGE_FOR_INTERVAL`
below is this module's own declared floor, not a ratified one, named and
versioned so a later reader can tell the two apart.

**Every reported value carries its denominator, lineage count included**
(§36.4/§12.2/§39.5): the criterion functions read `score.Denominator`
straight off the `score.CohortScore` they are given and never substitute a
smaller or larger one -- narrowing the cohort narrows what is reported,
visibly (design §9 guard 8, governance guard 8).

**The 437-vs-183 reconciliation is this module's job, not score.py's**
(score.py's own module docstring: "score.py's own eligible/attempted
columns are pass-through equal... that reconciliation is report.py's job
... over multiple cohorts including pages this module never sees").
`score.CohortScore.denominator` only reflects the rows `score.py` could
even see -- the ones with an exact-text cache row. The typed
pre-decode error classes (`capability_excluded`/`grammar_refused`/
`text_encoding_failed`/`over_length`, governance M6) never had an
exact-text row to begin with, so they never reach `score.score_cohort` at
all, and `score.Denominator` alone would silently understate the true
attempted denominator (437) with the narrower one `score.py` actually
scored (~183) -- precisely the "criterion-1 tautology" governance guard 8
warns against, one layer up. `reconcile_denominator` folds the cohort
manifest's full accounting back in so `criterion_1_structural_validity`
reports a rate over the TRUE attempted population, penalizing (not
hiding) every page that never reached a decode at all.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Literal

from unflatten.eval.s2_gate import sealed
from unflatten.eval.s2_gate.decode import CohortManifest
from unflatten.eval.s2_gate.score import CohortScore, Metric
from unflatten.model.s2_stage_gate_policy import s2_stage_gate_policy_hash

CriterionState = Literal["measured", "blocked", "insufficient_evidence"]

#: DEC-033 §12 / design §9: the S2 gate cohort's default slice (single-page
#: documents, comparable with S2 v12b). `s2_gate.cohort`'s `selection_rule`
#: carries no `"slice"` key for this default, so a manifest missing the key
#: entirely is this slice, same as one that names it explicitly.
DEFAULT_COHORT_SLICE = "single_page_docs"


def _cohort_slice(manifest: CohortManifest | None) -> str:
    """The report's own `"slice"` field (design §9): read from the cohort
    manifest's `selection_rule`, defaulting to `DEFAULT_COHORT_SLICE` when
    no manifest is supplied or its `selection_rule` carries no `"slice"`
    key -- never a literal, so a default-slice report and a manifest built
    before this slice existed report the same thing."""
    if manifest is None:
        return DEFAULT_COHORT_SLICE
    return manifest.selection_rule.get("slice", DEFAULT_COHORT_SLICE)

#: DEC-009 dodatok A's two mandatory `promotion_blockers` for any
#: `s1run_`-prefixed (discovery-class) run (governance M5).
DEC_009_PROMOTION_BLOCKERS: tuple[str, ...] = (
    "run_class_discovery",
    "run_id_not_rfc005_44_1",
)

#: This module's OWN declared floor on independent-lineage count needed to
#: report an interval-backed criterion as `measured` rather than
#: `insufficient_evidence` (§39.5). **Not ratified** -- there is no
#: ratified minimum cohort size (governance review, "Cohort size" section);
#: this is a floor on cohort strength this module refuses to pretend past,
#: never a quality threshold on the metric's value. Named and versioned so
#: a later reader cannot mistake it for a ratified §12.2 policy field.
MIN_LINEAGE_FOR_INTERVAL = 30

#: RFC005 §37.1: criterion 1 measures the grammar-valid hypothesis RATE
#: over the attempted denominator, not merely "replay terminated". Gate 7
#: was `blocked` at S1 on 7.0% `UnsupportedResumeInV0`
#: (`docs/plans/s1-gates-7-13-design.md:534`) -- stated up front per the
#: governance should_fix, so a reader is not surprised the branch fires
#: again here.
EXPECTED_BLOCKED_BRANCH_NOTE = (
    "gate 7 was blocked at S1 on 7.0% UnsupportedResumeInV0 "
    "(docs/plans/s1-gates-7-13-design.md:534); that branch is expected to "
    "recur here and is not itself a criterion-1 failure"
)

CHECKPOINT_RESUME_TEST_FILE = "tests/model/test_s1_train.py"
CHECKPOINT_RESUME_TEST_EXPRESSION = "checkpoint or resume"


@dataclass(frozen=True, slots=True)
class Criterion:
    """One of the six criteria's resolution. `state` is never `"pass"` or
    `"fail"` -- only `measured | blocked | insufficient_evidence`
    (governance M5). `detail` carries the numbers; it never carries a
    comparison verdict for a `blocked` criterion."""

    state: CriterionState
    reason: str | None
    detail: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state, "reason": self.reason, "detail": dict(self.detail)}


def _denominator_block(score: CohortScore) -> dict[str, Any]:
    return dataclasses.asdict(score.denominator)


def _metric_block(metric: Any) -> dict[str, Any]:
    return dataclasses.asdict(metric)


# ---------------------------------------------------------------------------
# The M6 reconciliation -- the TRUE (437-wide) denominator, one layer above
# score.py's own (module docstring above; score.py's docstring names this
# report.py's job).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconciledDenominator:
    """The honest attempted denominator (governance M6), folding the
    cohort manifest's full accounting back into `score.py`'s narrower one.

    `attempted_row_count`/`attempted_lineage_count` are the TRUE 437-wide
    counts (`sealed.CohortAccounting.attempted` plus the manifest's own
    lineage roots) -- never `score.Denominator.attempted_row_count`, which
    only counts rows that had an exact-text cache entry to begin with.
    `typed_upstream_error_count` is the pre-decode error classes that
    never reached `score.py` at all; `scoreable_row_count` is exactly
    `score.Denominator.attempted_row_count`, kept alongside so a reader can
    see both numbers and how they reconcile
    (`scoreable_row_count + typed_upstream_error_count == attempted_row_count`).
    """

    eligible_row_count: int
    eligible_lineage_count: int
    withdrawn_ratified_count: int
    attempted_row_count: int
    attempted_lineage_count: int
    scoreable_row_count: int
    scored_row_count: int
    typed_upstream_error_count: int
    score_layer_failed_count: int
    score_layer_missing_count: int

    def __post_init__(self) -> None:
        reconciled = self.scoreable_row_count + self.typed_upstream_error_count
        if reconciled != self.attempted_row_count:
            raise ValueError(
                "ReconciledDenominator does not reconcile: scoreable_row_count"
                f"({self.scoreable_row_count}) + typed_upstream_error_count"
                f"({self.typed_upstream_error_count}) = {reconciled} != "
                f"attempted_row_count({self.attempted_row_count})"
            )

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def reconcile_denominator(
    *, manifest: CohortManifest, cohort_accounting: sealed.CohortAccounting, score: CohortScore
) -> ReconciledDenominator:
    """Fold `manifest`'s full per-disposition accounting and
    `cohort_accounting` (the sealed artefact's own, decode.py-verified
    count) back into `score`'s narrower, exact-text-cache-bounded
    denominator (module docstring, "The 437-vs-183 reconciliation")."""
    non_withdrawn = [p for p in manifest.pages if p.disposition != sealed.WITHDRAWN_DISPOSITION]
    typed_upstream_error_count = (
        cohort_accounting.capability_excluded
        + cohort_accounting.grammar_refused
        + cohort_accounting.text_encoding_failed
        + cohort_accounting.over_length
        + sum(getattr(cohort_accounting, key) for key in sealed.ADDITIVE_DISPOSITIONS)
    )
    return ReconciledDenominator(
        eligible_row_count=cohort_accounting.eligible,
        eligible_lineage_count=len({p.lineage_root_id for p in manifest.pages}),
        withdrawn_ratified_count=cohort_accounting.withdrawn_text_coverage_floor,
        attempted_row_count=cohort_accounting.attempted,
        attempted_lineage_count=len({p.lineage_root_id for p in non_withdrawn}),
        scoreable_row_count=score.denominator.attempted_row_count,
        scored_row_count=score.denominator.scored_row_count,
        typed_upstream_error_count=typed_upstream_error_count,
        score_layer_failed_count=score.denominator.failed_count,
        score_layer_missing_count=score.denominator.missing_count,
    )


# ---------------------------------------------------------------------------
# Criterion 1 -- free-running structural validity, no grammar bypass
# ---------------------------------------------------------------------------


def _grammar_bypass_evidence(pages: Sequence[sealed.PageDecodeResult]) -> dict[str, Any]:
    """must_fix 5, round 2 (code review 2026-09-07): derive
    `no_grammar_bypass_evidence` from the sealed artefact's own per-page
    wiring rather than asserting a boolean (memory
    `literal-true-when-written`).

    `PageDecodeResult.decode_fn_identity` names the WRAPPER
    (`s1_runner_inputs.HypothesisCapture`) -- identical for a real
    `decode_free_running` run and for every test's injected fake, so it
    cannot distinguish them (`decode.py:_capture_wiring`'s own docstring).
    `decode_fn_inner_identity` names what the wrapper actually wraps;
    `decode_free_running` has no bypass mode at all -- no flag, no
    parameter that disables the grammar constraint anywhere in its module
    -- so naming IT is the whole proof, and a `True` here would just be the
    old literal moved one field over. Both are reported, never collapsed
    into one boolean, so a reader sees the wrapper and what it wraps.
    `pages_missing_inner_identity` counts attempted pages (a wrapper
    identity is present) whose inner identity is `None` -- surfaced as
    data, not swallowed, since an attempted page with no inner identity is
    exactly the case this evidence exists to catch."""
    wrapper_identities = sorted({p.decode_fn_identity for p in pages if p.decode_fn_identity})
    inner_identities = sorted(
        {p.decode_fn_inner_identity for p in pages if p.decode_fn_inner_identity}
    )
    pages_missing_inner_identity = sum(
        1 for p in pages if p.decode_fn_identity and not p.decode_fn_inner_identity
    )
    return {
        "decode_fn_wrapper_identities": wrapper_identities,
        "decode_fn_inner_identities": inner_identities,
        "pages_missing_inner_identity": pages_missing_inner_identity,
    }


def criterion_1_structural_validity(
    score: CohortScore,
    *,
    manifest: CohortManifest,
    cohort_accounting: sealed.CohortAccounting,
    cohort_pages: Sequence[sealed.PageDecodeResult] = (),
    min_lineage_for_interval: int = MIN_LINEAGE_FOR_INTERVAL,
) -> Criterion:
    """§37.1's grammar-valid hypothesis rate over the TRUE attempted
    denominator (governance M6/should_fix; `reconcile_denominator` above),
    replayed via the existing gate 7 (`score.replay_gate7`/
    `score.PageScore.replay_status`, M1: not redefined here). A page that
    never reached a decode at all (one of the typed pre-decode error
    classes) counts against the rate, not out of it -- exactly the guard 8
    tautology this must not commit. `insufficient_evidence` if the
    reconciled attempted lineage count is below `min_lineage_for_interval`
    (§39.5) -- otherwise `measured`.

    `cohort_pages` is the sealed artefact's own `PageDecodeResult` tuple
    (not `score.CohortScore.pages`, which is narrower and drops the wiring
    fields) -- passed here only to derive `no_grammar_bypass_evidence`."""
    reconciled = reconcile_denominator(
        manifest=manifest, cohort_accounting=cohort_accounting, score=score
    )
    if reconciled.attempted_lineage_count < min_lineage_for_interval:
        return Criterion(
            state="insufficient_evidence",
            reason="attempted_lineage_count_below_declared_floor",
            detail={
                "attempted_lineage_count": reconciled.attempted_lineage_count,
                "min_lineage_for_interval": min_lineage_for_interval,
                "denominator": reconciled.as_dict(),
            },
        )
    valid_count = sum(1 for p in score.pages if p.replay_status == "pass")
    attempted = reconciled.attempted_row_count
    rate = valid_count / attempted if attempted > 0 else None
    return Criterion(
        state="measured",
        reason=None,
        detail={
            "grammar_valid_hypothesis_rate": rate,
            "grammar_valid_count": valid_count,
            "denominator": reconciled.as_dict(),
            "no_grammar_bypass_evidence": _grammar_bypass_evidence(cohort_pages),
            "expected_branch_note": EXPECTED_BLOCKED_BRANCH_NOTE,
        },
    )


# ---------------------------------------------------------------------------
# Criterion 2 -- exact text vs frozen-init baseline (BLOCKED, §12.2)
# ---------------------------------------------------------------------------


class ArmComparisonRefused(ValueError):
    """Refused: the two arms report.py was handed are not a valid
    trained-vs-baseline pair (must_fix 6, code review 2026-09-07)."""


def refuse_unless_arms_are_comparable(
    *,
    trained_artefact: sealed.SealedCohortArtefact,
    baseline_artefact: sealed.SealedCohortArtefact,
    cohort_manifest_sha256: str,
) -> None:
    """The guard `arm` was put on `SealedCohortArtefact` FOR (memory
    `inert-guard-pattern`): refuses if the two artefacts carry the same
    `arm` -- an arm cannot be compared with itself -- or if either
    artefact's `cohort_manifest_sha256` disagrees with the other's or with
    the cohort manifest actually being reported against, which would mean
    at least one arm was decoded over a different cohort."""
    if trained_artefact.arm == baseline_artefact.arm:
        raise ArmComparisonRefused(
            f"trained and baseline artefacts both carry arm={trained_artefact.arm!r} -- "
            "an arm cannot be compared with itself"
        )
    if (
        trained_artefact.arm != sealed.DecodeArm.TRAINED
        or baseline_artefact.arm != sealed.DecodeArm.FROZEN_INIT_BASELINE
    ):
        raise ArmComparisonRefused(
            f"expected trained_artefact.arm={sealed.DecodeArm.TRAINED!r} and "
            f"baseline_artefact.arm={sealed.DecodeArm.FROZEN_INIT_BASELINE!r}, got "
            f"{trained_artefact.arm!r} and {baseline_artefact.arm!r} -- the two "
            "arms were swapped (should_fix, code review 2026-09-07)"
        )
    shas = {
        "trained": trained_artefact.cohort_manifest_sha256,
        "baseline": baseline_artefact.cohort_manifest_sha256,
        "manifest": cohort_manifest_sha256,
    }
    if len(set(shas.values())) != 1:
        raise ArmComparisonRefused(
            f"cohort_manifest_sha256 disagrees across trained/baseline/manifest: {shas} "
            "-- the two arms (or the manifest) would not be measuring the same cohort"
        )


#: Fixed two-sided 95% normal-approximation multiplier. No scipy dependency
#: is added for this (module docstring precedent in `distance.py`); alpha
#: is fixed at 0.05 and named, not configurable, since nothing here claims
#: a comparison verdict that would need a different one.
_Z_95 = 1.959963984540054


def paired_confidence_interval(differences: Sequence[float]) -> tuple[float, float] | None:
    """Two-sided ~95% normal-approximation CI on the mean of `differences`.
    `None` (with no crash) if fewer than 2 paired observations exist -- a
    caller must check for `None` rather than read it as a zero-width
    interval."""
    n = len(differences)
    if n < 2:
        return None
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    se = sqrt(variance / n)
    margin = _Z_95 * se
    return (mean - margin, mean + margin)


def wilson_score_interval(numerator: int, denominator: int) -> tuple[float, float] | None:
    """DEC-030 D4 (review should_fix 2): a per-arm 95% Wilson score
    interval on a single proportion `numerator / denominator` (e.g. one
    arm's own `approx_text_match_rate`) -- distinct from
    `paired_confidence_interval`, which bounds the paired difference
    between two arms, not either arm's own rate. Closed form, no scipy
    dependency (same precedent as `paired_confidence_interval`), reusing
    `_Z_95`. `None` when `denominator == 0` -- there is no proportion to
    bound."""
    if denominator == 0:
        return None
    p = numerator / denominator
    z2 = _Z_95 * _Z_95
    scale = 1 + z2 / denominator
    center = (p + z2 / (2 * denominator)) / scale
    margin = (
        _Z_95 * sqrt(p * (1 - p) / denominator + z2 / (4 * denominator * denominator))
    ) / scale
    return (center - margin, center + margin)


def _paired_exact_match_differences(trained: CohortScore, baseline: CohortScore) -> list[float]:
    baseline_by_id = {p.document_bundle_id: p for p in baseline.pages}
    diffs: list[float] = []
    for tp in trained.pages:
        bp = baseline_by_id.get(tp.document_bundle_id)
        if bp is None:
            continue
        diffs.append(float(tp.exact_match) - float(bp.exact_match))
    return diffs


def _paired_approx_match_differences(trained: CohortScore, baseline: CohortScore) -> list[float]:
    """DEC-030: same pairing as `_paired_exact_match_differences`, over
    `PageScore.approx_match` instead of `PageScore.exact_match`."""
    baseline_by_id = {p.document_bundle_id: p for p in baseline.pages}
    diffs: list[float] = []
    for tp in trained.pages:
        bp = baseline_by_id.get(tp.document_bundle_id)
        if bp is None:
            continue
        diffs.append(float(tp.approx_match) - float(bp.approx_match))
    return diffs


def _exact_match_rate_over_reconciled(
    score: CohortScore, reconciled: ReconciledDenominator
) -> dict[str, Any]:
    """The M6-honest exact-match rate (must_fix 4): numerator is still the
    matches `score.py` could see, but the denominator is the TRUE
    reconciled attempted count -- a page that never reached a decode at
    all counts against the rate, not out of it, exactly like criterion 1's
    `grammar_valid_hypothesis_rate`."""
    matched_count = sum(1 for p in score.pages if p.exact_match)
    attempted = reconciled.attempted_row_count
    value = matched_count / attempted if attempted > 0 else None
    return {"value": value, "matched_count": matched_count, "denominator": reconciled.as_dict()}


def _approx_text_match_rate_over_reconciled(
    score: CohortScore, reconciled: ReconciledDenominator
) -> dict[str, Any]:
    """DEC-030: the same M6-honest reconciliation as
    `_exact_match_rate_over_reconciled`, over `PageScore.approx_match`
    instead of `PageScore.exact_match` -- this is now the headline
    criterion-2 number, `_exact_match_rate_over_reconciled` the
    diagnostic. DEC-030 D4 (review should_fix 2): carries its own per-arm
    `wilson_95pct_ci` on the rate itself -- deliberately NOT added to
    `_exact_match_rate_over_reconciled`, which stays a diagnostic only."""
    matched_count = sum(1 for p in score.pages if p.approx_match)
    attempted = reconciled.attempted_row_count
    value = matched_count / attempted if attempted > 0 else None
    return {
        "value": value,
        "matched_count": matched_count,
        "denominator": reconciled.as_dict(),
        "wilson_95pct_ci": wilson_score_interval(matched_count, attempted),
    }


def _arm_detail_m6(score: CohortScore, reconciled: ReconciledDenominator) -> dict[str, Any]:
    """One arm's criterion-2 block (must_fix 4, DEC-030). `approx_text_match_rate`
    is the headline number, recomputed over the reconciled 437-wide
    denominator; `exact_match_rate_diagnostic` is the same reconciliation
    over the raw (DEC-030-demoted) exact match. `character_error_rate`/
    `word_error_rate` cannot be reconciled the same way: the typed
    pre-decode error classes never had reference text to begin with, so
    there is nothing to divide by past the 183 scoreable rows -- named
    `_diagnostic` per §36.4:2430 rather than silently kept under their
    un-suffixed, pre-review names."""
    return {
        "approx_text_match_rate": _approx_text_match_rate_over_reconciled(score, reconciled),
        "exact_match_rate_diagnostic": _exact_match_rate_over_reconciled(score, reconciled),
        "character_error_rate_diagnostic": _metric_block(score.character_error_rate),
        "word_error_rate_diagnostic": _metric_block(score.word_error_rate),
        "denominator": reconciled.as_dict(),
    }


def _criterion_2_policy_outcome(
    policy: Mapping[str, Any],
    *,
    rate: float | None,
    attempted_row_count: int,
    attempted_lineage_count: int,
    ci: tuple[float, float] | None,
) -> tuple[CriterionState, str | None, dict[str, Any]]:
    """DEC-012 determination for criterion 2 from a validated S2 gate
    policy: `measured` when the evidence supports the policy's own
    thresholds, `insufficient_evidence` (with the policy's own
    `missing_evidence_action`) when it does not. The margin is a
    non-inferiority band on the paired CI's lower bound -- it may NOT
    compensate a failed absolute floor: `satisfied` is floor AND margin,
    never either/or. An invalid policy refuses here
    (`S2StageGatePolicyInvalid`), it never falls back to a default."""
    policy_hash = s2_stage_gate_policy_hash(dict(policy))
    metric = policy["quality"]["metrics"]["exact_match_rate"]
    floor_micro = metric["absolute_quality_floor_micro"]
    margin_micro = metric["relative_margin_micro"]
    if rate is None:
        return (
            "insufficient_evidence",
            "exact_match_rate_unavailable",
            {
                "policy_hash": policy_hash,
                "action": "insufficient_evidence",
                "reason": "exact_match_rate_unavailable",
            },
        )
    if attempted_row_count < metric["minimum_attempted_denominator"]:
        return (
            "insufficient_evidence",
            "minimum_attempted_denominator_not_met",
            {
                "policy_hash": policy_hash,
                "action": "insufficient_evidence",
                "reason": "minimum_attempted_denominator_not_met",
                "attempted_row_count": attempted_row_count,
                "minimum_attempted_denominator": metric["minimum_attempted_denominator"],
            },
        )
    if attempted_lineage_count < metric["minimum_independent_lineages"]:
        return (
            "insufficient_evidence",
            "minimum_independent_lineage_count_not_met",
            {
                "policy_hash": policy_hash,
                "action": "insufficient_evidence",
                "reason": "minimum_independent_lineage_count_not_met",
                "attempted_lineage_count": attempted_lineage_count,
                "minimum_independent_lineages": metric["minimum_independent_lineages"],
            },
        )
    if ci is None:
        return (
            "insufficient_evidence",
            "paired_confidence_interval_unavailable",
            {
                "policy_hash": policy_hash,
                "action": "insufficient_evidence",
                "reason": "paired_confidence_interval_unavailable",
            },
        )
    floor_satisfied = rate >= floor_micro / 1e6
    margin_satisfied = ci[0] >= -(margin_micro / 1e6)
    return (
        "measured",
        None,
        {
            "policy_hash": policy_hash,
            "denominator": metric["denominator"],
            "exact_match_rate": rate,
            "attempted_row_count": attempted_row_count,
            "attempted_lineage_count": attempted_lineage_count,
            "absolute_quality_floor_micro": floor_micro,
            "relative_margin_micro": margin_micro,
            "confidence_bound_method": metric["confidence_bound_method"],
            "alpha_micro": metric["alpha_micro"],
            "minimum_independent_lineages": metric["minimum_independent_lineages"],
            "minimum_attempted_denominator": metric["minimum_attempted_denominator"],
            "paired_ci_lower_bound": ci[0],
            "absolute_quality_floor_satisfied": floor_satisfied,
            "relative_margin_satisfied": margin_satisfied,
            "satisfied": floor_satisfied and margin_satisfied,
        },
    )


def criterion_2_exact_text_vs_baseline(
    trained: CohortScore,
    baseline: CohortScore,
    *,
    manifest: CohortManifest,
    cohort_manifest_sha256: str,
    trained_artefact: sealed.SealedCohortArtefact,
    baseline_artefact: sealed.SealedCohortArtefact,
    s2_gate_policy: Mapping[str, Any] | None = None,
) -> Criterion:
    """`blocked`, reason `no_frozen_stage_gate_policy` (governance M2),
    when no frozen S2 stage gate policy is supplied: §12.2 conditions
    every expression of "better" on a pre-frozen `stage-gate-policy`, and
    one that does not exist is the default here. When a validated
    `s2_gate_policy` (DEC-012) IS supplied, the criterion resolves
    `measured`/`insufficient_evidence` from the policy's own thresholds
    (`_criterion_2_policy_outcome`), the determination -- including the
    policy's hash -- in `detail["policy_determination"]`. Either way both
    arms' paired approx-text-match (DEC-030, the headline)/exact-match
    (now a diagnostic)/CER/WER are reported with denominators and paired
    confidence intervals on both the approx-match and exact-match
    differences, labelled `single_seed_provisional` (§39.3); with no
    policy, no comparison verdict of any kind is made.

    Refuses (`ArmComparisonRefused`) via `refuse_unless_arms_are_comparable`
    before computing anything (must_fix 6): the two artefacts must carry
    different `arm`s and the same `cohort_manifest_sha256` as `manifest`,
    or this criterion is not actually comparing what it claims to. An
    invalid `s2_gate_policy` refuses (`S2StageGatePolicyInvalid`), never
    defaults."""
    refuse_unless_arms_are_comparable(
        trained_artefact=trained_artefact,
        baseline_artefact=baseline_artefact,
        cohort_manifest_sha256=cohort_manifest_sha256,
    )
    diffs = _paired_exact_match_differences(trained, baseline)
    ci = paired_confidence_interval(diffs)
    approx_diffs = _paired_approx_match_differences(trained, baseline)
    approx_ci = paired_confidence_interval(approx_diffs)
    trained_reconciled = reconcile_denominator(
        manifest=manifest, cohort_accounting=trained_artefact.accounting, score=trained
    )
    baseline_reconciled = reconcile_denominator(
        manifest=manifest, cohort_accounting=baseline_artefact.accounting, score=baseline
    )
    if s2_gate_policy is None:
        state: CriterionState = "blocked"
        reason: str | None = "no_frozen_stage_gate_policy"
        policy_determination: dict[str, Any] | None = None
    else:
        rate_block = _exact_match_rate_over_reconciled(trained, trained_reconciled)
        state, reason, policy_determination = _criterion_2_policy_outcome(
            s2_gate_policy,
            rate=rate_block["value"],
            attempted_row_count=trained_reconciled.attempted_row_count,
            attempted_lineage_count=trained_reconciled.attempted_lineage_count,
            ci=ci,
        )
    detail = {
        # must_fix 6: derived from `baseline_artefact` itself, never a
        # written literal -- a citation string cannot prove which
        # artefact was actually used as the baseline; these fields can.
        "baseline_citation": {
            "arm": baseline_artefact.arm,
            "cohort_manifest_sha256": baseline_artefact.cohort_manifest_sha256,
            "eval_cache_identity": dict(baseline_artefact.eval_cache_identity),
        },
        "trained": _arm_detail_m6(trained, trained_reconciled),
        "frozen_init_baseline": _arm_detail_m6(baseline, baseline_reconciled),
        "paired_exact_match_difference_95pct_ci": ci,
        "paired_n": len(diffs),
        # DEC-030: the paired difference on the new headline metric.
        "paired_approx_match_difference_95pct_ci": approx_ci,
        "single_seed_provisional": True,
    }
    if policy_determination is not None:
        detail["policy_determination"] = policy_determination
    return Criterion(state=state, reason=reason, detail=detail)


# ---------------------------------------------------------------------------
# Criterion 3 -- zero silent truncation, zero invalid success
# ---------------------------------------------------------------------------


def _loop_guard_evidence(pages: Sequence[sealed.PageDecodeResult]) -> dict[str, Any]:
    """DEC-031 task 2: `loop_guard_pages`/summed counters recovered from
    `PageDecodeResult.detail` -- the JSON string `decode.py`'s
    `_loop_guard_detail` writes whenever the decode ran with
    `DecodeConfig.loop_guard == "v1"`. `None`/unparsable `detail` (every
    page from a `loop_guard=None` decode, including every artefact sealed
    before this task) contributes nothing -- an old artefact reports all
    zeros here, never a crash. `loop_guard_pages` counts only pages whose
    counters actually fired at least once: a page the guard never touched
    is evidence the guard was inert there, not a guard event. A `"v2"`
    page (DEC-036) nests its v1 counters under `loop_guard_counters["v1"]`;
    its own evidence is `_loop_guard_v2_evidence`."""
    totals = {"ban_tab": 0, "ban_empty_paragraph": 0, "ban_empty_row": 0, "closeout": 0, "noop": 0}
    loop_guard_pages = 0
    payloads = []
    for page in pages:
        if not page.detail:
            continue
        try:
            payload = json.loads(page.detail)
        except json.JSONDecodeError:
            continue
        payloads.append((page, payload))
        counters = payload.get("loop_guard_counters")
        if not counters:
            continue
        if payload.get("loop_guard") == "v2":
            counters = counters.get("v1") or {}
        if any(counters.get(key, 0) for key in totals):
            loop_guard_pages += 1
        for key in totals:
            totals[key] += counters.get(key, 0)
    return {
        "loop_guard_pages": loop_guard_pages,
        "loop_guard_counters": totals,
        **_loop_guard_v2_evidence(payloads),
    }


_LOOP_GUARD_V2_RULES = (
    "text_run",
    "periodic",
    "close_recovery",
    "close_recovery_unavailable",
    "noop",
)
#: Width of one `first_intervention_step` histogram bucket, in decode steps.
_FIRST_STEP_BUCKET = 256


def _loop_guard_v2_evidence(payloads: Sequence[tuple[Any, Mapping[str, Any]]]) -> dict[str, Any]:
    """DEC-036 design §5. A page is *altered* when it carries at least one
    intervention that changed the decision -- a `noop` changed nothing and
    does not count. `terminated_after_intervention` is the terminated
    subset of altered pages. The histogram buckets each altered page's
    first altering step. One artefact must name exactly one caps hash;
    two are refused, never merged. Pages without v2 detail (every older
    artefact) contribute zeros."""
    rule_totals = dict.fromkeys(_LOOP_GUARD_V2_RULES, 0)
    altered = terminated_after = 0
    histogram: dict[str, int] = {}
    caps_values: set[str] = set()
    for page, payload in payloads:
        if payload.get("loop_guard") != "v2":
            continue
        counters = payload.get("loop_guard_counters") or {}
        for rule in _LOOP_GUARD_V2_RULES:
            rule_totals[rule] += counters.get(rule, 0)
        if payload.get("loop_guard_caps_sha256") is not None:
            caps_values.add(payload["loop_guard_caps_sha256"])
        interventions = payload.get("loop_guard_interventions") or ()
        steps = [i["step"] for i in interventions if i["rule"] != "noop"]
        if not steps:
            continue
        altered += 1
        terminated_after += page.outcome == "terminated"
        bucket = str(min(steps) // _FIRST_STEP_BUCKET * _FIRST_STEP_BUCKET)
        histogram[bucket] = histogram.get(bucket, 0) + 1
    if len(caps_values) > 1:
        raise ValueError(
            f"one sealed artefact names {len(caps_values)} loop guard v2 caps hashes "
            f"{sorted(caps_values)}; its pages were not decoded under one guard identity"
        )
    return {
        "loop_guard_v2_pages_altered": altered,
        "loop_guard_v2_terminated_after_intervention": terminated_after,
        "loop_guard_v2_rule_totals": rule_totals,
        "loop_guard_v2_first_intervention_step_histogram": dict(
            sorted(histogram.items(), key=lambda kv: int(kv[0]))
        ),
        "loop_guard_v2_caps_sha256_values": sorted(caps_values),
    }


def criterion_3_zero_silent_truncation_and_invalid_success(
    score: CohortScore,
    *,
    manifest: CohortManifest | None = None,
    cohort_accounting: sealed.CohortAccounting | None = None,
    cohort_pages: Sequence[sealed.PageDecodeResult] = (),
) -> Criterion:
    """Both are hard-zero counts (design §4), never redefined here (M1):
    silent truncation is `score.PageScore.outcome == "not_terminated"`
    (gate 13's own `"silent_truncation"` branch, decode.py); invalid
    success is `score.CohortScore.invalid_success_count`
    (`score.is_invalid_success`, gate 7 + text-match derived per M1). Both
    counts are properties only of pages that actually reached a decode
    (`score`'s own rows), so -- unlike criterion 1 -- they are not
    recomputed over the reconciled 437-wide population; the reconciled
    denominator is still attached for context (§36.4) when `manifest`/
    `cohort_accounting` are supplied. Always `measured` -- a count needs
    no interval and no policy.

    `cohort_pages` (DEC-031 task 2): the sealed artefact's own
    `PageDecodeResult` tuple (not `score.CohortScore.pages`, same reason
    criterion 1 takes it) -- passed here only to derive
    `_loop_guard_evidence`. Does not change verdict logic: the loop guard
    is diagnostic detail on this criterion, not a new pass/fail rule."""
    silent_truncation_count = sum(1 for p in score.pages if p.outcome == "not_terminated")
    invalid_success_count = score.invalid_success_count
    if manifest is not None and cohort_accounting is not None:
        denominator_block = reconcile_denominator(
            manifest=manifest, cohort_accounting=cohort_accounting, score=score
        ).as_dict()
    else:
        denominator_block = _denominator_block(score)
    return Criterion(
        state="measured",
        reason=None,
        detail={
            "silent_truncation_count": silent_truncation_count,
            "invalid_success_count": invalid_success_count,
            "zero_silent_truncation": silent_truncation_count == 0,
            "zero_invalid_success": invalid_success_count == 0,
            "denominator": denominator_block,
            **_loop_guard_evidence(cohort_pages),
        },
    )


# ---------------------------------------------------------------------------
# Criterion 4 -- control-event and diacritics slices (BLOCKED, §12.2)
# ---------------------------------------------------------------------------


def _criterion_4_policy_outcome(
    policy: Mapping[str, Any],
    *,
    slices: Mapping[str, Metric],
) -> tuple[CriterionState, str | None, dict[str, Any]]:
    """DEC-012 determination for criterion 4 from a validated S2 gate
    policy: every `critical_slices` entry is checked against the cohort's
    own slice fidelity. A listed slice with no value (`None` -- the
    empty-reference cohort) is `insufficient_evidence`, the policy's own
    `missing_evidence_action`; a measured slice below its
    `minimum_fidelity_micro` floor is reported per slice, and
    `all_slices_satisfied` is the conjunction, never a single fused
    boolean. An invalid policy refuses here, never defaults."""
    policy_hash = s2_stage_gate_policy_hash(dict(policy))
    per_slice: list[dict[str, Any]] = []
    unavailable: str | None = None
    for entry in policy["quality"]["critical_slices"]:
        slice_id = entry["slice_id"]
        value = slices[slice_id].value
        if value is None:
            unavailable = f"slice_fidelity_unavailable:{slice_id}"
            break
        minimum_micro = entry["minimum_fidelity_micro"]
        per_slice.append(
            {
                "slice_id": slice_id,
                "minimum_fidelity_micro": minimum_micro,
                "fidelity": value,
                "fidelity_satisfied": value >= minimum_micro / 1e6,
            }
        )
    if unavailable is not None:
        return (
            "insufficient_evidence",
            unavailable,
            {
                "policy_hash": policy_hash,
                "action": "insufficient_evidence",
                "reason": unavailable,
            },
        )
    return (
        "measured",
        None,
        {
            "policy_hash": policy_hash,
            "slices": per_slice,
            "all_slices_satisfied": all(s["fidelity_satisfied"] for s in per_slice),
        },
    )


def criterion_4_slices(
    score: CohortScore,
    *,
    manifest: CohortManifest,
    cohort_accounting: sealed.CohortAccounting,
    s2_gate_policy: Mapping[str, Any] | None = None,
) -> Criterion:
    """`blocked`, reason `no_predeclared_regression_margin` (governance
    M2), when no frozen S2 stage gate policy is supplied: §12.2 makes
    critical slices and their regression margins a frozen-policy field.
    When a validated `s2_gate_policy` (DEC-012) IS supplied, the
    criterion resolves `measured`/`insufficient_evidence` from the
    policy's own `critical_slices` floors
    (`_criterion_4_policy_outcome`), the determination -- including the
    policy's hash -- in `detail["policy_determination"]`.

    must_fix 4: the slice values themselves stay 183-scoped -- diacritic
    and control-event classification need the actual reference text, which
    the typed pre-decode error classes never had -- so they are named
    `_diagnostic` (§36.4:2430) and the attached `denominator` is the
    reconciled, TRUE 437-wide one, not `score.py`'s narrower pass-through.
    An invalid `s2_gate_policy` refuses (`S2StageGatePolicyInvalid`),
    never defaults."""
    reconciled = reconcile_denominator(
        manifest=manifest, cohort_accounting=cohort_accounting, score=score
    )
    if s2_gate_policy is None:
        state: CriterionState = "blocked"
        reason: str | None = "no_predeclared_regression_margin"
        policy_determination: dict[str, Any] | None = None
    else:
        state, reason, policy_determination = _criterion_4_policy_outcome(
            s2_gate_policy,
            slices={
                "diacritic_slice": score.diacritic_slice,
                "control_event_slice": score.control_event_slice,
            },
        )
    detail = {
        "diacritic_slice_diagnostic": _metric_block(score.diacritic_slice),
        "control_event_slice_diagnostic": _metric_block(score.control_event_slice),
        "denominator": reconciled.as_dict(),
    }
    if policy_determination is not None:
        detail["policy_determination"] = policy_determination
    return Criterion(state=state, reason=reason, detail=detail)


# ---------------------------------------------------------------------------
# Criterion 5 -- checkpoint resume test (existing proof, invoked and reported)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckpointResumeEvidence:
    test_file: str
    test_expression: str
    exit_code: int
    passed_count: int | None
    raw_summary: str


_PASSED_RE = re.compile(r"(\d+) passed")

#: `report.py` lives at `src/unflatten/eval/s2_gate/report.py`; the repo
#: root is four parents up. `run_checkpoint_resume_tests` must invoke
#: `uv run pytest <test_file>` with this as `cwd` -- `test_file` is a
#: path relative to the repo root, and the caller (an eval CLI) may run
#: from any working directory (e.g. an `/mnt/ai-disk/.../eval/s2-c5`
#: report directory). Without `cwd`, pytest cannot resolve `test_file`,
#: collects nothing, and exits 4 ("no tests ran") -- the exact failure
#: this module must not silently call `measured`.
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _default_pytest_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        argv, capture_output=True, text=True, check=False, cwd=_REPO_ROOT
    )


def run_checkpoint_resume_tests(
    *,
    test_file: str = CHECKPOINT_RESUME_TEST_FILE,
    test_expression: str = CHECKPOINT_RESUME_TEST_EXPRESSION,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _default_pytest_runner,
) -> CheckpointResumeEvidence:
    """Invoke the EXISTING checkpoint/resume proof
    (`tests/model/test_s1_train.py`, 26 green tests) rather than
    reimplementing it (design §4 row 5). `runner` is injectable so a test
    can supply a fake `CompletedProcess` without actually shelling out to
    pytest. The default runner sets `cwd=_REPO_ROOT` so `test_file`
    resolves regardless of the caller's working directory."""
    result = runner(["uv", "run", "pytest", test_file, "-k", test_expression, "-q"])
    summary_line = next(
        (line for line in reversed(result.stdout.splitlines()) if line.strip()), ""
    )
    match = _PASSED_RE.search(result.stdout)
    passed_count = int(match.group(1)) if match else None
    return CheckpointResumeEvidence(
        test_file=test_file,
        test_expression=test_expression,
        exit_code=result.returncode,
        passed_count=passed_count,
        raw_summary=summary_line,
    )


def criterion_5_checkpoint_resume(evidence: CheckpointResumeEvidence) -> Criterion:
    """`measured` only when the existing proof actually ran and passed:
    `exit_code == 0` and a parsed `passed_count > 0` (design §4 row 5,
    governance should_fix). A non-zero exit code, an unparseable summary
    (`passed_count=None`), or a zero `passed_count` (nothing collected --
    e.g. pytest usage-error exit code 4, "no tests ran") is `insufficient_
    evidence`, not `measured`: a run that collected and executed nothing
    is evidence of nothing, not evidence that happens to be reported
    honestly. Either way `detail` reports the raw numbers, never hiding
    them."""
    detail = {
        "test_file": evidence.test_file,
        "test_expression": evidence.test_expression,
        "exit_code": evidence.exit_code,
        "passed_count": evidence.passed_count,
        "raw_summary": evidence.raw_summary,
        "note": "existing proof invoked and reported, not reimplemented "
        "(design §4 row 5)",
    }
    if evidence.exit_code != 0 or not evidence.passed_count:
        return Criterion(
            state="insufficient_evidence",
            reason="checkpoint_resume_tests_did_not_run_or_pass",
            detail=detail,
        )
    return Criterion(state="measured", reason=None, detail=detail)


# ---------------------------------------------------------------------------
# Criterion 6 -- P1 and P2 reported separately (BLOCKED unless a real tier
# map exists)
# ---------------------------------------------------------------------------


def criterion_6_pyramid_tiers(pyramid_tier_by_bundle: Mapping[str, str] | None) -> Criterion:
    """`blocked`, reason `no_pyramid_tier_label`, when `pyramid_tier_by_bundle`
    is `None` or empty -- the real situation today (design F1): no tier
    label exists anywhere in the m2a-v6 store or the `-c5`/eval caches
    (`input_contract.py:42`, sole occurrence, in `FORBIDDEN_CHANNELS`).
    `measured` when a real map IS supplied, reporting the P1/P2 (and any
    other tier's) counts -- this branch exists and is exercised by
    `test_report.py` with a fake tier map so `blocked` is proven not to be
    hardcoded (design §9 guard 4)."""
    if not pyramid_tier_by_bundle:
        return Criterion(
            state="blocked",
            reason="no_pyramid_tier_label",
            detail={
                "citation": "input_contract.py:42 FORBIDDEN_CHANNELS (sole occurrence); "
                "no P1/P2 data in the m2a-v6 store or the -c5/eval caches (design F1)"
            },
        )
    tier_counts = Counter(pyramid_tier_by_bundle.values())
    return Criterion(
        state="measured",
        reason=None,
        detail={"tier_counts": dict(tier_counts), "total": sum(tier_counts.values())},
    )


# ---------------------------------------------------------------------------
# The report -- criteria_state, promotable, promotion_blockers
# ---------------------------------------------------------------------------

CRITERION_NAMES: tuple[str, ...] = (
    "structural_validity_no_grammar_bypass",
    "exact_text_vs_frozen_init_baseline",
    "zero_silent_truncation_and_invalid_success",
    "control_event_and_diacritics_slices",
    "checkpoint_resume",
    "p1_p2_reported_separately",
)


@dataclass(frozen=True, slots=True)
class S2ExitGateReport:
    """The report `report.py` emits. No field named `verdict` or `pass`
    (governance M5): `criteria` and `criteria_state` name the six
    criteria's own resolutions; `promotable` and `promotion_blockers` are
    DEC-009's discovery-class facts, constructed, never a literal."""

    criteria: Mapping[str, Criterion]
    criteria_state: Mapping[str, CriterionState]
    promotable: bool
    promotion_blockers: tuple[str, ...]
    slice: str = DEFAULT_COHORT_SLICE

    def as_dict(self) -> dict[str, Any]:
        return {
            "criteria": {name: c.as_dict() for name, c in self.criteria.items()},
            "criteria_state": dict(self.criteria_state),
            "promotable": self.promotable,
            "promotion_blockers": list(self.promotion_blockers),
            "slice": self.slice,
        }


def _compute_promotable(
    criteria_state: Mapping[str, CriterionState], promotion_blockers: Sequence[str]
) -> bool:
    """A function, never a literal (governance M5, memory
    `literal-true-when-written`): `True` only if every criterion is
    `measured` AND no promotion blocker remains. Tested on the path where
    it comes out `True` (`test_report.py`), so it cannot outlive its
    truth."""
    all_measured = all(state == "measured" for state in criteria_state.values())
    return all_measured and len(promotion_blockers) == 0


def build_report(
    *,
    criterion_1: Criterion,
    criterion_2: Criterion,
    criterion_3: Criterion,
    criterion_4: Criterion,
    criterion_5: Criterion,
    criterion_6: Criterion,
    promotion_blockers: Sequence[str] = DEC_009_PROMOTION_BLOCKERS,
    manifest: CohortManifest | None = None,
) -> S2ExitGateReport:
    """Assemble the six criteria into an `S2ExitGateReport`. `promotion_blockers`
    defaults to DEC-009's two mandatory blockers for a discovery-class run;
    a caller (a test proving `promotable` CAN be `True`) may pass `()`
    instead. `manifest` (DEC-033 §12, design §9) is optional and used only
    to derive the report's own `"slice"` field via `_cohort_slice` -- no
    caller that omits it changes behaviour, so this is additive to every
    existing caller."""
    criteria = {
        CRITERION_NAMES[0]: criterion_1,
        CRITERION_NAMES[1]: criterion_2,
        CRITERION_NAMES[2]: criterion_3,
        CRITERION_NAMES[3]: criterion_4,
        CRITERION_NAMES[4]: criterion_5,
        CRITERION_NAMES[5]: criterion_6,
    }
    criteria_state = {name: c.state for name, c in criteria.items()}
    promotable = _compute_promotable(criteria_state, promotion_blockers)
    return S2ExitGateReport(
        criteria=criteria,
        criteria_state=criteria_state,
        promotable=promotable,
        promotion_blockers=tuple(promotion_blockers),
        slice=_cohort_slice(manifest),
    )
