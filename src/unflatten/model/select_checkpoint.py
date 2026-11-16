"""`python -m unflatten.model.select_checkpoint` -- RFC005 §40 / DEC-041 A
post-hoc checkpoint selection.

Picks the best checkpoint of an S4 run from its checkpoint history by
FREE-RUNNING decoding over a pinned selection cohort, never by training
loss (RFC005 §12.2: "early stopping, selection a calibration používajú
production-equivalent `free_running` výstupy. Teacher-forced/oracle
metrics sú iba diagnostika."). This module never reads `s4_eval.jsonl`
(the trainer's teacher-forced diagnostic log) or any `train_loss`/
`val_loss` field -- `test_select_checkpoint.py::
test_module_source_never_references_training_loss_or_diagnostic_eval_log`
parses this file's own AST and refuses those names anywhere in it, the
same structural guarantee `eval/s2_gate/decode.py` gives for reference
text.

Candidate evaluation (the actual free-running decode + score) is
injected as a `CandidateEvaluator` callable -- this module owns only the
selection algorithm (RFC005 §40.1 lexicographic order, §40.2 Pareto
frontier, §40.3 `best_under_selection_policy_hash`, §40.4 trial budget
and cohort provenance), never a model, a GPU, or a decode implementation
of its own. `default_candidate_evaluator` is the real wiring (shells out
to the existing, already-proven `unflatten.eval.s2_gate.cli` decode/score
and `unflatten.eval.builder_chain.cli` subcommands -- module docstring:
"reuse, do not rebuild") and is exercised only by a fake-subprocess test;
every other test injects a small stub evaluator per the task's own
instruction.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import subprocess
import sys
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unflatten.model import checkpoint_ancestry as ca
from unflatten.model import events_vocab
from unflatten.model.s4_selection_policy import (
    has_placeholder_floor,
    load_s4_selection_policy,
    s4_selection_policy_hash,
    validate_s4_selection_policy,
)

SELECTION_REPORT_SCHEMA_VERSION = "1.0.0"

#: The three checkpoint-history shapes `s4_train.py` writes (module
#: docstring of `_save_checkpoint_with_history`/`_best_eval_path`).
_HISTORY_GLOB = "checkpoint-step-*.pt"
_BEST_EVAL_NAME = "checkpoint-best-eval.pt"
_LATEST_NAME = "checkpoint.pt"

#: `s4_train.load_or_select_eval_pages`'s pinned diagnostic-eval file --
#: read ONLY to check cohort disjointness (RFC005 §40.4), never for any
#: metric value.
_EVAL_PAGES_FILENAME = "s4_eval_pages.json"


class SelectionRefused(ValueError):
    """A selection run refused: the trial budget was exceeded, the
    selection cohort overlaps the run's own diagnostic eval pages, or a
    named `--candidates` path does not exist."""


# ---------------------------------------------------------------------------
# Candidate enumeration (DEC-041 A: history + best-eval + latest, dedup)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One checkpoint file under consideration. `candidate_id` is the
    file's own stem (e.g. `checkpoint-step-000010000`, `checkpoint-best-eval`,
    `checkpoint`) -- stable and human-readable, never the bare sha256."""

    candidate_id: str
    path: Path
    sha256: str


#: `promotion.RendererBranch.status` value meaning a Word file can be produced
#: (level 7). `rendered` is not a status the branch can take; matching it made
#: renderer_ready_rate 0 on every real report.
RENDER_READY_STATUS = "render_ready"

def enumerate_candidates(run_dir: Path, *, extra: Sequence[Path] = ()) -> list[Candidate]:
    """`extra` given: exactly those paths, in that order -- the owner's
    shortlist (24.9., RUNBOOK §11.5: the eval shortlists 2-3 checkpoints and
    only those are decoded). Otherwise every history copy (`checkpoint-step-
    *.pt`, sorted) + `checkpoint-best-eval.pt` + `checkpoint.pt` (latest).
    Deduplicated by sha256 -- `_save_checkpoint_with_history` writes the
    history/best-eval copies as byte-for-byte `shutil.copy2` duplicates of
    `checkpoint.pt` at the step they are taken, so several names can name
    identical content; the first name encountered wins the dedup, in the
    order above."""
    if extra:
        candidate_paths: list[Path] = list(extra)
    else:
        candidate_paths = list(sorted(run_dir.glob(_HISTORY_GLOB)))
        best_eval = run_dir / _BEST_EVAL_NAME
        if best_eval.exists():
            candidate_paths.append(best_eval)
        latest = run_dir / _LATEST_NAME
        if latest.exists():
            candidate_paths.append(latest)

    by_sha: dict[str, Candidate] = {}
    for path in candidate_paths:
        if not path.exists():
            raise SelectionRefused(f"candidate checkpoint {path} does not exist")
        sha = ca.file_sha256(path)
        if sha in by_sha:
            continue
        by_sha[sha] = Candidate(candidate_id=path.stem, path=path, sha256=sha)
    if not by_sha:
        raise SelectionRefused(
            f"{run_dir}: no checkpoint candidates found ({_HISTORY_GLOB}, "
            f"{_BEST_EVAL_NAME}, {_LATEST_NAME} all absent, and no --candidates supplied)"
        )
    return list(by_sha.values())


# ---------------------------------------------------------------------------
# §40.4: selection cohort must be disjoint from the run's diagnostic eval pages
# ---------------------------------------------------------------------------


def refuse_unless_cohort_disjoint_from_eval_pages(
    policy: Mapping[str, Any], run_dir: Path
) -> None:
    """RFC005 §40.4: the selection cohort must never be the run's own
    calibration/diagnostic split. `s4_eval_pages.json` (module docstring)
    is the one pinned split this run dir can name; if it exists, refuse
    any pinned selection page that also appears there. Silent when the
    file does not exist yet (a fresh run dir before its first diagnostic
    eval) -- nothing to check against, not a pass in disguise: the real
    provenance guarantee is that the cohort was drawn from `val_iid`
    outside this run's own splits, which is a data-pipeline property this
    function cannot see and does not claim to verify."""
    eval_pages_path = run_dir / _EVAL_PAGES_FILENAME
    if not eval_pages_path.exists():
        return
    payload = json.loads(eval_pages_path.read_text())
    eval_pages = {(bundle, page) for bundle, page in payload["pages"]}
    pinned = {(bundle, page) for bundle, page in policy["cohort"]["pinned_pages"]}
    overlap = eval_pages & pinned
    if overlap:
        raise SelectionRefused(
            f"selection cohort overlaps {run_dir / _EVAL_PAGES_FILENAME} on "
            f"{sorted(overlap)} -- the selection cohort must be disjoint from the "
            "run's own diagnostic eval pages (RFC005 §40.4)"
        )


# ---------------------------------------------------------------------------
# Candidate evaluation (injected)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """One candidate's measured evidence: `metrics_micro` maps every
    policy gate's `metric_id` to a rate in micro-units, or `None` when the
    evaluator could not measure it (`insufficient_evidence`, RFC005
    §12.2 -- never coerced to 0). `tie_break_value` is the raw (non-micro)
    performance number the policy's `tie_break.metric_id` names, or `None`.

    `per_lineage_metric_samples` (RFC005-ERRATA-005 E3): optional, `None`
    unless the evaluator populated it -- for a metric it does carry, maps
    `lineage_root_id` to that lineage's own micro-scale contribution to the
    metric. Never per-page: RFC005 §39.2 treats pages of one document as
    non-independent, so a bootstrap over per-page values would understate
    noise. Used only by `propose_reference_floors`; the ordinary selection
    algorithm (`rank_candidates`/`pareto_frontier`) never reads it."""

    candidate: Candidate
    metrics_micro: Mapping[str, int | None]
    tie_break_value: float | None
    per_lineage_metric_samples: Mapping[str, Mapping[str, float]] | None = None
    #: RFC005-ERRATA-005 / T2c: level 5's own denominator block (RFC005
    #: §12.2 "every reported value carries its denominator") --
    #: `aligned_pages`/`typed_unaligned_pages`/`compared_nodes`/
    #: `pointers_covered` from `TypedMacroAccuracyResult`. `None` unless the
    #: evaluator attempted the typed alignment at all (never a guessed
    #: all-zero denominator for a metric that was never measured).
    typed_head_macro_accuracy_coverage: Mapping[str, int] | None = None


CandidateEvaluator = Callable[[Candidate, Mapping[str, Any], Path], CandidateResult]


def _cached_result_path(out_dir: Path, candidate: Candidate, cache_tag: str = "") -> Path:
    """W0-2: `cache_tag` is the same chain-policy prefix `_chain_tag` builds
    (`""` for the plain chain) -- a `CandidateResult`'s `metrics_micro` can
    include values read from a chain report, and a chain report built
    under a DIFFERENT policy set (`headless_defaults`/`table_geometry`/...)
    is cached under its OWN `chain-<tag><sha>.json` name already; the
    metrics cache must be keyed the same way, or reusing `out_dir` with a
    different policy set silently hands back the previous policy set's
    stale `metrics-<sha>.json`."""
    return out_dir / f"metrics-{cache_tag}{candidate.sha256}.json"


def _read_cached_result(candidate: Candidate, path: Path) -> CandidateResult:
    payload = json.loads(path.read_text())
    samples = payload.get("per_lineage_metric_samples")
    coverage = payload.get("typed_head_macro_accuracy_coverage")
    return CandidateResult(
        candidate=candidate,
        metrics_micro=dict(payload["metrics_micro"]),
        tie_break_value=payload["tie_break_value"],
        per_lineage_metric_samples=(
            {metric: dict(v) for metric, v in samples.items()} if samples is not None else None
        ),
        typed_head_macro_accuracy_coverage=(dict(coverage) if coverage is not None else None),
    )


def _write_cached_result(result: CandidateResult, path: Path) -> None:
    samples = result.per_lineage_metric_samples
    coverage = result.typed_head_macro_accuracy_coverage
    path.write_text(
        json.dumps(
            {
                "candidate_id": result.candidate.candidate_id,
                "sha256": result.candidate.sha256,
                "metrics_micro": dict(result.metrics_micro),
                "tie_break_value": result.tie_break_value,
                "per_lineage_metric_samples": (
                    {metric: dict(v) for metric, v in samples.items()}
                    if samples is not None
                    else None
                ),
                "typed_head_macro_accuracy_coverage": (
                    dict(coverage) if coverage is not None else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _trial_budget_ledger_path(out_dir: Path, cache_tag: str) -> Path:
    return out_dir / f"trial-budget-spent-{cache_tag}.json"


def _read_trial_budget_spent(out_dir: Path, cache_tag: str) -> int:
    """W0-3(a): how many NEW trials this `out_dir`/`cache_tag` has already
    spent, across every past `evaluate_candidates` call -- `0` the first
    time. RFC005 §40.4's budget is cumulative over the search, not reset
    by re-invoking the CLI in a fresh process (finding 3, `docs/runs/
    v15b-layout-diagnostic-handoff-2026-09-26.md` §9)."""
    path = _trial_budget_ledger_path(out_dir, cache_tag)
    if not path.exists():
        return 0
    return json.loads(path.read_text())["new_trials_spent"]


def _write_trial_budget_spent(out_dir: Path, cache_tag: str, spent: int) -> None:
    _trial_budget_ledger_path(out_dir, cache_tag).write_text(
        json.dumps({"new_trials_spent": spent}, indent=2, sort_keys=True)
    )


def evaluate_candidates(
    candidates: Sequence[Candidate],
    policy: Mapping[str, Any],
    *,
    out_dir: Path,
    evaluator: CandidateEvaluator,
    cache_tag: str = "",
) -> list[CandidateResult]:
    """Evaluate every candidate, resuming from `out_dir`'s cached
    `metrics-<cache_tag><sha256>.json` files (module docstring: "resumable:
    skip a candidate whose sealed artefact exists and verifies" -- verified
    here by sha256-keyed filename, since the cache itself is this
    function's own write). Every NEW evaluation counts against
    `policy["trial_budget"]["max_candidates"]` (RFC005 §40.4); a resumed
    (cached) candidate costs nothing, since it is not a new selection
    trial. Exceeding the budget refuses before running the evaluator, not
    after (a partially-run trial still counts and cannot be silently
    discarded).

    W0-3(a): the spent count is CUMULATIVE across invocations, persisted
    in `out_dir` (`trial-budget-spent-<cache_tag>.json`), not reset to
    zero on every fresh call/process -- a caller cannot exceed the frozen
    budget merely by re-running the CLI against the same `out_dir`/
    `cache_tag` with a different candidate batch each time.

    `cache_tag` (W0-2) MUST be the same chain-policy tag `_chain_tag`
    would build for the evaluator's own `RealEvaluatorConfig` -- the
    caller's job, since this function takes an opaque `CandidateEvaluator`
    and cannot derive it. Reusing `out_dir` under a different policy set
    with the SAME (default `""`) tag is exactly the stale-reuse bug this
    parameter exists to prevent; `main()` always passes the real one."""
    budget = policy["trial_budget"]["max_candidates"]
    results: list[CandidateResult] = []
    new_trials = _read_trial_budget_spent(out_dir, cache_tag)
    for candidate in candidates:
        cache_path = _cached_result_path(out_dir, candidate, cache_tag)
        if cache_path.exists():
            results.append(_read_cached_result(candidate, cache_path))
            continue
        if new_trials >= budget:
            raise SelectionRefused(
                f"trial budget {budget} exceeded: {new_trials} new evaluation(s) already "
                f"spent under this out_dir/cache_tag -- the frozen trial_budget (RFC005 "
                "§40.4) counts every new trial cumulatively across invocations, never "
                "reset per process; rerun with a wider budget or an untouched cohort, "
                "never a post-hoc raise on the same cohort"
            )
        result = evaluator(candidate, policy, out_dir)
        _write_cached_result(result, cache_path)
        results.append(result)
        new_trials += 1
        _write_trial_budget_spent(out_dir, cache_tag, new_trials)
    return results


# ---------------------------------------------------------------------------
# RFC005 §40.1 lexicographic order + §40.2 Pareto frontier
# ---------------------------------------------------------------------------


def _gate_value_and_pass(gate: Mapping[str, Any], result: CandidateResult) -> tuple[bool, int]:
    """`(passes, sort_value)` for one gate against `result` -- a missing
    measurement (`None`) never compensates a floor: it sorts as the worst
    possible value and never counts as passing (§12.2: "Nulová alebo
    underpowered ... je insufficient_evidence, nie úspech")."""
    value = result.metrics_micro.get(gate["metric_id"])
    if value is None:
        return False, -1
    return value >= gate["floor_micro"], value


def _sorted_gates(policy: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return sorted(policy["gates"], key=lambda g: g["level"])


def _lexicographic_key(
    result: CandidateResult, gates: Sequence[Mapping[str, Any]], tie_break: Mapping[str, Any]
) -> tuple[Any, ...]:
    """A total order over candidates matching RFC005 §40.1: gate 1's pass/
    fail dominates every later gate, then gate 2's, and so on through level
    7; a tie across all seven resolves by §40.1 level 8's `tie_break`
    metric. Never a weighted sum of gate values -- that is exactly the
    "invented hidden weight" §40.2 forbids."""
    key: list[Any] = [_gate_value_and_pass(gate, result) for gate in gates]
    tb_value = result.tie_break_value
    if tie_break["direction"] == "min":
        # Higher key sorts first (reverse=True below); "min is better" is
        # expressed by negating so a smaller raw value produces a larger key.
        tb_key = -tb_value if tb_value is not None else float("-inf")
    else:
        tb_key = tb_value if tb_value is not None else float("-inf")
    key.append(tb_key)
    return tuple(key)


def rank_candidates(
    results: Sequence[CandidateResult], policy: Mapping[str, Any]
) -> list[CandidateResult]:
    """`results` ordered best-first under RFC005 §40.1's lexicographic
    order. A candidate that fails an earlier gate always loses to one that
    passes it, regardless of any later gate's value -- see
    `_lexicographic_key`."""
    gates = _sorted_gates(policy)
    tie_break = policy["tie_break"]
    return sorted(
        results, key=lambda r: _lexicographic_key(r, gates, tie_break), reverse=True
    )


def _tie_break_normalized(result: CandidateResult, tie_break: Mapping[str, Any]) -> float:
    value = result.tie_break_value
    if value is None:
        return float("-inf")
    return -value if tie_break["direction"] == "min" else value


def _dominates(
    a: CandidateResult,
    b: CandidateResult,
    *,
    gates: Sequence[Mapping[str, Any]],
    tie_break: Mapping[str, Any],
) -> bool:
    """`a` Pareto-dominates `b`: at least as good as `b` on every gate
    metric AND performance, strictly better on at least one (§40.2)."""
    a_values = [
        (m if (m := a.metrics_micro.get(g["metric_id"])) is not None else -1) for g in gates
    ]
    b_values = [
        (m if (m := b.metrics_micro.get(g["metric_id"])) is not None else -1) for g in gates
    ]
    a_tb = _tie_break_normalized(a, tie_break)
    b_tb = _tie_break_normalized(b, tie_break)
    at_least_as_good = all(av >= bv for av, bv in zip(a_values, b_values)) and a_tb >= b_tb
    strictly_better = any(av > bv for av, bv in zip(a_values, b_values)) or a_tb > b_tb
    return at_least_as_good and strictly_better


def pareto_frontier(
    results: Sequence[CandidateResult], policy: Mapping[str, Any]
) -> list[CandidateResult]:
    """The candidates no other candidate Pareto-dominates (§40.2). A
    single-element frontier is a genuine dominant winner; more than one
    element means quality/compute trade-offs with no dominant winner, and
    the caller must not invent a weight to break the tie -- it reports the
    frontier instead of a `best`."""
    gates = _sorted_gates(policy)
    tie_break = policy["tie_break"]
    return [
        r
        for r in results
        if not any(
            _dominates(other, r, gates=gates, tie_break=tie_break)
            for other in results
            if other is not r
        )
    ]


# ---------------------------------------------------------------------------
# selection_report.json
# ---------------------------------------------------------------------------


def _gate_outcome(gate: Mapping[str, Any], result: CandidateResult) -> dict[str, Any]:
    value = result.metrics_micro.get(gate["metric_id"])
    if value is None:
        return {
            "level": gate["level"],
            "metric_id": gate["metric_id"],
            "value_micro": None,
            "state": "insufficient_evidence",
        }
    return {
        "level": gate["level"],
        "metric_id": gate["metric_id"],
        "value_micro": value,
        "state": "pass" if value >= gate["floor_micro"] else "fail",
    }


def _passes_every_nonzero_floor(
    result: CandidateResult, gates: Sequence[Mapping[str, Any]]
) -> bool:
    for gate in gates:
        if gate["floor_micro"] == 0:
            continue
        value = result.metrics_micro.get(gate["metric_id"])
        if value is None or value < gate["floor_micro"]:
            return False
    return True


def build_selection_report(
    results: Sequence[CandidateResult],
    policy: Mapping[str, Any],
    *,
    report_only: bool = False,
) -> dict[str, Any]:
    """RFC005 §40.3: `best` is always `best_under_selection_policy_hash`,
    never `lowest_train_loss` or a hand-picked filename -- the report's
    `selection_policy_hash` field is present unconditionally (what any
    declared best is scoped under), and `best_under_selection_policy_hash`
    carries the winning candidate's id only when a best can honestly be
    declared:

    - never while any gate's floor is a DEC-035 §5 placeholder, unless
      `report_only` (then the report ranks and states no best is declared);
    - never when the Pareto frontier has more than one member (§40.2: no
      invented hidden weight breaks a genuine trade-off) -- the report
      carries the full frontier instead;
    - never when the winner is below a floor > 0 or leaves such a gate
      unmeasured (RFC005-ERRATA-005 E3, §39.5).
    """
    gates = _sorted_gates(policy)
    policy_hash = s4_selection_policy_hash(dict(policy))
    placeholder = has_placeholder_floor(policy)
    ranked = rank_candidates(results, policy)
    frontier = pareto_frontier(results, policy)
    frontier_ids = {r.candidate.candidate_id for r in frontier}

    best_candidate_id = None
    no_best_reason = None
    if placeholder and not report_only:
        no_best_reason = "policy_has_placeholder_floor"
    elif len(frontier) > 1:
        no_best_reason = "pareto_frontier_has_no_dominant_winner"
    elif report_only:
        no_best_reason = "report_only"
    elif not _passes_every_nonzero_floor(frontier[0], gates):
        # A floor is a minimum (RFC005-ERRATA-005 E3), not an ordering hint;
        # an unmeasured metric under a floor > 0 is insufficient evidence
        # (§39.5), never a pass. A floor of 0 never blocks.
        no_best_reason = "best_candidate_fails_a_floor"
    else:
        best_candidate_id = frontier[0].candidate.candidate_id

    candidates_doc = [
        {
            "candidate_id": r.candidate.candidate_id,
            "sha256": r.candidate.sha256,
            "gates": [_gate_outcome(g, r) for g in gates],
            "tie_break_value": r.tie_break_value,
            "on_pareto_frontier": r.candidate.candidate_id in frontier_ids,
            "typed_head_macro_accuracy_coverage": (
                dict(r.typed_head_macro_accuracy_coverage)
                if r.typed_head_macro_accuracy_coverage is not None
                else None
            ),
        }
        for r in ranked
    ]

    return {
        "selection_report_schema_version": SELECTION_REPORT_SCHEMA_VERSION,
        "selection_policy_hash": policy_hash,
        "report_only": report_only,
        "policy_has_placeholder_floor": placeholder,
        "trial_budget": dict(policy["trial_budget"]),
        "ranking_order": [r.candidate.candidate_id for r in ranked],
        "pareto_frontier": sorted(frontier_ids),
        "best_under_selection_policy_hash": best_candidate_id,
        "no_best_reason": no_best_reason,
        "candidates": candidates_doc,
    }


# ---------------------------------------------------------------------------
# RFC005-ERRATA-005 E3 (owner decision 2026-09-24): reference-checkpoint
# floors -- "not worse than v14-r4", measured on the SAME selection cohort
# by THIS tool, minus a noise margin from a lineage-level bootstrap.
# ---------------------------------------------------------------------------


def lineage_bootstrap_margin_micro(
    samples: Sequence[float], *, num_resamples: int, alpha_micro: int, seed: int
) -> int:
    """RFC005 §39.2's lineage-level (never per-page) bootstrap, applied
    here to a single arm -- the reference checkpoint's own per-lineage
    contributions to one metric, already in micro-units. Returns the
    one-sided noise margin: the reference's own point estimate minus the
    `alpha_micro/2`-percentile of the resampled means, i.e. how far below
    its point estimate this cohort's own sampling noise alone could
    plausibly put a re-measurement. A floor of `reference_value_micro -
    margin_micro` is "not worse than the reference, allowing for this
    cohort's own measurement noise" -- never a bare equality to one noisy
    draw. Clamped at 0: a margin cannot be negative."""
    if not samples:
        raise ValueError("lineage_bootstrap_margin_micro: samples is empty")
    rng = random.Random(seed)
    n = len(samples)
    point_estimate = sum(samples) / n
    resample_means = sorted(
        sum(samples[rng.randrange(n)] for _ in range(n)) / n for _ in range(num_resamples)
    )
    alpha = alpha_micro / 1_000_000
    lower_index = max(0, min(num_resamples - 1, int((alpha / 2) * num_resamples)))
    lower_percentile = resample_means[lower_index]
    return max(0, round(point_estimate - lower_percentile))


def _cohort_manifest_pages(cohort_manifest: Path) -> list[Mapping[str, Any]] | None:
    """`document_bundle_id`/`lineage_root_id`/`disposition` straight off
    the on-disk cohort manifest JSON, WITHOUT importing
    `eval.s2_gate.decode` (torch-only -- that module's own docstring:
    "phase A is GPU-only"; this reads the identical file a second,
    lighter way, only for lineage grouping). The manifest's own
    sha256/`policy_hash` are already verified by the `decode`/`score`
    subprocess calls that produced `sealed_path`/`score_path` -- this
    function trusts that verification rather than repeating it.

    `None` when the file does not exist or carries no `pages` list: a
    caller must treat a missing manifest as "no per-lineage evidence for
    this gate", never guess a lineage grouping."""
    if not cohort_manifest.exists():
        return None
    payload = json.loads(cohort_manifest.read_text())
    pages = payload.get("pages")
    if pages is None:
        return None
    return list(pages)


def _lineage_by_bundle(pages: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {p["document_bundle_id"]: p["lineage_root_id"] for p in pages}


#: Duplicated literals (not imported -- keeps this module's per-lineage
#: helpers free of `eval.s2_gate.sealed`'s torch-adjacent import chain);
#: MUST equal `sealed.WITHDRAWN_DISPOSITION`/`sealed.ATTEMPT_DISPOSITION`.
WITHDRAWN_DISPOSITION_LITERAL = "withdrawn_text_coverage_floor"
ATTEMPT_DISPOSITION_LITERAL = "attempt"


def _per_lineage_no_typed_failure_rate(pages: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Level 1's per-lineage samples: `sealed_doc["accounting"]`'s own
    `decoded / attempted` rate (module docstring of
    `default_candidate_evaluator`), split by `lineage_root_id` instead of
    pooled over the whole cohort. Mirrors `compute_cohort_accounting`'s
    own `attempted = eligible - withdrawn` / `decoded = count(disposition
    == "attempt")` exactly, at lineage granularity."""
    attempted: dict[str, int] = {}
    decoded: dict[str, int] = {}
    for page in pages:
        if page["disposition"] == WITHDRAWN_DISPOSITION_LITERAL:
            continue
        lineage = page["lineage_root_id"]
        attempted[lineage] = attempted.get(lineage, 0) + 1
        if page["disposition"] == ATTEMPT_DISPOSITION_LITERAL:
            decoded[lineage] = decoded.get(lineage, 0) + 1
    return {
        lineage: round(1_000_000 * decoded.get(lineage, 0) / count)
        for lineage, count in attempted.items()
        if count > 0
    }


def _per_lineage_row_rate(
    rows: Sequence[Mapping[str, Any]],
    *,
    lineage_of: Callable[[Mapping[str, Any]], str | None],
    value_key: str,
) -> dict[str, int]:
    """One lineage-grouped rate over any `rows` sequence carrying a truthy/
    falsy `value_key` -- shared by level 2 (`score_doc["pages"]`'s
    `exact_match`) and levels 3/7 (`chain_doc["rows"]`'s `strict_valid`/
    `renderer_ready`). A row whose `lineage_of` resolves to `None` (e.g. a
    chain row whose bundle id is absent from the cohort manifest) is
    skipped, never counted against an invented lineage."""
    numerator: dict[str, int] = {}
    denominator: dict[str, int] = {}
    for row in rows:
        lineage = lineage_of(row)
        if lineage is None:
            continue
        denominator[lineage] = denominator.get(lineage, 0) + 1
        if row[value_key]:
            numerator[lineage] = numerator.get(lineage, 0) + 1
    return {
        lineage: round(1_000_000 * numerator.get(lineage, 0) / count)
        for lineage, count in denominator.items()
        if count > 0
    }


def _per_lineage_critical_slice_min_fidelity(
    pages: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """Level 4's per-lineage samples: `min(diacritic_slice, control_event_
    slice)`, each slice's own ratio summed over ONLY that lineage's pages
    (mirrors `score._slice_fidelity_metric`'s sum-then-ratio shape,
    applied per lineage instead of pooled)."""
    diacritic_matched: dict[str, int] = {}
    diacritic_reference: dict[str, int] = {}
    control_matched: dict[str, int] = {}
    control_reference: dict[str, int] = {}
    for page in pages:
        lineage = page["lineage_root_id"]
        diacritic = page["diacritic_slice"]
        control = page["control_event_slice"]
        diacritic_matched[lineage] = diacritic_matched.get(lineage, 0) + diacritic["matched"]
        diacritic_reference[lineage] = (
            diacritic_reference.get(lineage, 0) + diacritic["reference_count"]
        )
        control_matched[lineage] = control_matched.get(lineage, 0) + control["matched"]
        control_reference[lineage] = control_reference.get(lineage, 0) + control["reference_count"]
    samples: dict[str, int] = {}
    for lineage, diacritic_ref in diacritic_reference.items():
        values: list[float] = []
        if diacritic_ref > 0:
            values.append(diacritic_matched[lineage] / diacritic_ref)
        control_ref = control_reference.get(lineage, 0)
        if control_ref > 0:
            values.append(control_matched.get(lineage, 0) / control_ref)
        if values:
            samples[lineage] = round(1_000_000 * min(values))
    return samples


# ---------------------------------------------------------------------------
# Fresh S4 (DEC-058) selection metrics -- DEC-055 V5 plus visible text omitted.
# Computed only when the policy names them (a v15b policy never does), so a
# v15b candidate result carries exactly the keys it always did.
# ---------------------------------------------------------------------------

#: Level 7 companion: fraction of attempted pages that did NOT loop. A page
#: looped when it hit the decode step cap or the loop guard intervened with
#: any rule but `noop` -- a loop the guard caught is still a loop.
LOOP_FREE_RATE = "loop_free_rate"
#: Level 2 companion: fraction of the page's visible glyphs the decoded text
#: keeps (1 - "visible text omitted"), micro over glyphs, word multiset.
VISIBLE_TEXT_RETAINED_RATE = "visible_text_retained_rate"
#: Tie-break (level 8): decode wall time per successfully reconstructed page
#: (terminated AND strict-valid in the builder chain), so a faster failure
#: never reads as a speed-up (DEC-055 V5, owner's proposal §5.2).
SECONDS_PER_SUCCESSFUL_PAGE = "seconds_per_successful_page"


def page_looped(page: Mapping[str, Any]) -> bool:
    if page.get("outcome") == "max_steps_exceeded":
        return True
    detail = page.get("detail")
    if not detail:
        return False
    payload = json.loads(detail)
    return any(
        intervention.get("rule") != "noop"
        for intervention in payload.get("loop_guard_interventions") or ()
    )


def loop_free_rate_micro(pages: Sequence[Mapping[str, Any]], *, attempted: int) -> int | None:
    """`1 - looped / attempted` over the sealed artefact's pages (micro)."""
    if not attempted:
        return None
    looped = sum(1 for page in pages if page_looped(page))
    return round(1_000_000 * (attempted - looped) / attempted)


def _glyphs(word: str) -> int:
    return sum(1 for ch in word if not ch.isspace())


def visible_text_omitted(page_text: str, hypothesis_text: str) -> tuple[int, int]:
    """`(omitted_glyphs, page_glyphs)`: the page's words (whitespace split)
    as a multiset minus the decoded words; the glyphs of what is left over
    were not produced. Order-free on purpose -- a reading-order difference
    (text frames, columns) is not an omission; extra decoded words do not
    offset missing ones."""
    from collections import Counter

    page_words = Counter(page_text.split())
    missing = page_words - Counter(hypothesis_text.split())
    return (
        sum(_glyphs(word) * count for word, count in missing.items()),
        sum(_glyphs(word) * count for word, count in page_words.items()),
    )


def visible_text_retained_rate_micro(
    score_pages: Sequence[Mapping[str, Any]],
    *,
    page_text: Mapping[tuple[str, int], str] | None = None,
) -> int | None:
    """Micro retained-glyph rate over every scored page (a failed or missing
    page has an empty hypothesis, so it omits everything). The page's text
    is `page_text[(bundle, page)]` when given (the store's text layer, V1c
    census logic) and the rebuilt gold `reference_text` otherwise."""
    omitted = total = 0
    for page in score_pages:
        reference = page["reference_text"]
        if page_text is not None:
            key = (page["document_bundle_id"], int(page["page_index"]))
            if key not in page_text:
                raise SelectionRefused(
                    f"{key}: the page has no text-layer row -- a layer must cover every "
                    "scored cohort page, never a silent fallback to the gold text"
                )
            reference = page_text[key]
        page_omitted, page_total = visible_text_omitted(reference, page["hypothesis_text"])
        omitted += page_omitted
        total += page_total
    if not total:
        return None
    return round(1_000_000 * (total - omitted) / total)


def seconds_per_successful_page(decode_seconds: float, *, successes: int) -> float | None:
    return decode_seconds / successes if successes else None


def load_page_text_layer(path: Path) -> dict[tuple[str, int], str]:
    """JSONL `{document_bundle_id, page_index, text}` (one line per cohort
    page) -> the `page_text` mapping `visible_text_retained_rate_micro` takes."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if "text" not in record:
                raise SelectionRefused(
                    f"{path}: {record['document_bundle_id']} page {record['page_index']} has no "
                    f"text (status {record.get('status')!r})"
                )
            rows[(record["document_bundle_id"], int(record["page_index"]))] = record["text"]
    return rows


# ---------------------------------------------------------------------------
# Level 5 (`typed_head_macro_accuracy_rate`): the pure macro-accuracy
# computation. RFC005 §12.2 forbids a teacher-forced source (`eval/
# s3_typed`) for selection; a FREE-RUNNING source needs each decoded page's
# predicted typed values joined to gold typed targets by an unambiguous
# node alignment. `eval/s2_gate/sealed.py` and `eval/builder_chain/
# sealed_adapter.py` only ever match a decoded page's OWN node ids across
# ITS OWN two lifecycle trackers, or join a decoded node to an assembled
# node the SAME page built -- neither compares the decoded tree's shape to
# a separate GOLD canonical-AST tree for the same bundle, and `eval/
# s3_typed` gets its alignment for free from teacher forcing (position IS
# node identity) -- a mechanism free-running cannot reuse.
#
# **Path A (below, plan T2c, owner decision 2026-09-24): structural-
# skeleton identity.** A page's decoded and gold structural skeletons
# (`page_structural_skeleton`) are compared as plain token-name sequences;
# an exact match means the i-th skeleton-bearing node on each side is the
# same node, with no tree-matching search at all -- a page whose decoded
# structure differs even slightly from gold (extra/missing/reordered node)
# is simply UNALIGNED, counted in `typed_unaligned_pages`, never partially
# matched. Path B (best-effort alignment when structures differ) remains
# unbuilt, per the plan's own scoping ("az v dalsom cykle, ak pokrytie A
# nestaci").
#
# `typed_macro_accuracy` is the metric's pure DEFINITION (exercised here
# only against hand-built `TypedPageAlignment` fixtures);
# `align_typed_page`/`typed_head_macro_accuracy_for_cohort` are path A's
# alignment, wired into `default_candidate_evaluator` below.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypedAlignedNodeSample:
    """One gold-aligned node's typed pointer values on one page.
    `pointer_values` maps a typed property pointer to `(predicted, gold)`
    -- already resolved by the caller's alignment, never looked up here."""

    pointer_values: Mapping[str, tuple[Any, Any]]


@dataclass(frozen=True, slots=True)
class TypedPageAlignment:
    """One page's contribution to the typed macro-accuracy metric. An
    empty `aligned_nodes` means the page contributed no unambiguously
    aligned gold node -- counted in `typed_unaligned_pages`, never
    silently dropped or coerced to a 0 (RFC005 §12.2)."""

    document_bundle_id: str
    lineage_root_id: str
    aligned_nodes: tuple[TypedAlignedNodeSample, ...]


@dataclass(frozen=True, slots=True)
class TypedMacroAccuracyResult:
    value: float | None
    pointers_covered: int
    aligned_node_count: int
    pages_contributing: int
    typed_unaligned_pages: int
    per_lineage_samples: Mapping[str, int]


def typed_macro_accuracy(pages: Sequence[TypedPageAlignment]) -> TypedMacroAccuracyResult:
    """RFC005 §12.2-compliant FREE-RUNNING typed metric: the macro average,
    over every typed pointer that appears on at least one aligned node,
    of that pointer's own exact-match accuracy across every aligned node
    carrying it. `value` is `None` when no pointer was ever covered
    (`pointers_covered == 0`) -- never a guessed 0.

    A page with no aligned nodes contributes nothing to any pointer and
    is counted only in `typed_unaligned_pages` (never in
    `pages_contributing`, never as a wrong answer for any pointer).

    `per_lineage_samples` (micro-scale, RFC005 §39.2): per lineage, the
    SAME macro-average-over-pointers definition, restricted to that
    lineage's own aligned nodes -- a lineage that contributed no aligned
    node at all carries no entry (nothing to average).

    Never reads a checkpoint, a cache, or teacher-forced targets --
    `pages` already carries every `(predicted, gold)` pair resolved by
    the caller's own free-running alignment."""
    correct_by_pointer: dict[str, int] = {}
    total_by_pointer: dict[str, int] = {}
    per_lineage_pointer_correct: dict[str, dict[str, int]] = {}
    per_lineage_pointer_total: dict[str, dict[str, int]] = {}
    aligned_node_count = 0
    pages_contributing = 0
    typed_unaligned_pages = 0

    for page in pages:
        if not page.aligned_nodes:
            typed_unaligned_pages += 1
            continue
        pages_contributing += 1
        lineage_total = per_lineage_pointer_total.setdefault(page.lineage_root_id, {})
        lineage_correct = per_lineage_pointer_correct.setdefault(page.lineage_root_id, {})
        for node in page.aligned_nodes:
            aligned_node_count += 1
            for pointer, (predicted, gold) in node.pointer_values.items():
                total_by_pointer[pointer] = total_by_pointer.get(pointer, 0) + 1
                lineage_total[pointer] = lineage_total.get(pointer, 0) + 1
                if predicted == gold:
                    correct_by_pointer[pointer] = correct_by_pointer.get(pointer, 0) + 1
                    lineage_correct[pointer] = lineage_correct.get(pointer, 0) + 1

    pointers_covered = len(total_by_pointer)
    value = (
        None
        if pointers_covered == 0
        else sum(
            correct_by_pointer.get(pointer, 0) / total
            for pointer, total in total_by_pointer.items()
        )
        / pointers_covered
    )

    per_lineage_samples: dict[str, int] = {}
    for lineage, totals in per_lineage_pointer_total.items():
        if not totals:
            continue
        correct = per_lineage_pointer_correct.get(lineage, {})
        lineage_value = sum(
            correct.get(pointer, 0) / total for pointer, total in totals.items()
        ) / len(totals)
        per_lineage_samples[lineage] = round(1_000_000 * lineage_value)

    return TypedMacroAccuracyResult(
        value=value,
        pointers_covered=pointers_covered,
        aligned_node_count=aligned_node_count,
        pages_contributing=pages_contributing,
        typed_unaligned_pages=typed_unaligned_pages,
        per_lineage_samples=per_lineage_samples,
    )


# ---------------------------------------------------------------------------
# Level 5 alignment ("path A", RFC005-ERRATA-005 / plan T2c, owner decision
# 2026-09-24): the structural-skeleton alignment `typed_macro_accuracy`'s own
# module docstring above says does not exist yet. A page is aligned iff its
# DECODED and GOLD structural skeletons (every node open/close/leaf/resume/
# suspend token, excluding TEXT_BEGIN/TEXT_END/END_DOCUMENT/FRAGMENT_BEGIN/
# END_FRAGMENT and every text/byte-fallback token) are byte-identical
# sequences -- then the i-th skeleton-bearing node on each side is the same
# node, RFC004 §18.2's node event alphabet requiring no separate proof that
# preorder identity survives free-running decode.
# ---------------------------------------------------------------------------

#: Reverse of `events_vocab.TOKEN_IDS` (`name -> local_id`): a packed
#: composite token id's structural token name. Built here rather than
#: imported from `free_running_decode._TOKEN_NAME_BY_ID` (the same
#: dict-comprehension) or `node_lifecycle`/`composite_vocab.unpack`, both of
#: which pull in torch (`node_lifecycle.py: import torch`, for its
#: `NodeState.anchor` tensor field) -- this module stays torch-free for its
#: own alignment arithmetic, on the same precedent `_cohort_manifest_pages`
#: sets above (a second, lighter read of the same fact, not a second
#: authority for it).
_STRUCTURAL_TOKEN_NAME_BY_ID: Mapping[int, str] = {
    local_id: name for name, local_id in events_vocab.TOKEN_IDS.items()
}

#: A packed composite token id < `len(events_vocab.TOKEN_IDS)` (71) is
#: ALWAYS a structural token, regardless of the Qwen native vocabulary size
#: it was packed against: `composite_vocab.packed_ranges` fixes
#: `structural_start = 0` unconditionally, so `StructuralRef(local_id)` packs
#: to exactly `local_id` (`composite_vocab.pack`) -- no `PackedRanges`/
#: `CompositeVocabulary` instance is needed to tell a structural id from a
#: Qwen/byte-fallback one. Both `ExactTextCacheRow.packed_token_ids` (the
#: gold side) and `sealed.PageDecodeResult.accepted_tokens` (the decoded
#: side) are packed under the same scheme (`training_materializer.py`/
#: `free_running_decode.py` both call `composite_vocab.pack`), so this one
#: threshold check applies unchanged to either sequence.
_STRUCTURAL_TOKEN_COUNT = len(events_vocab.TOKEN_IDS)

#: Token names that own a node slot in `node_lifecycle.NodeLifecycleTracker`
#: (`apply_structural_token`'s own routing: OPEN/LEAF/RESUME/SUSPEND/CLOSE
#: return a `NodeState`; TEXT_BEGIN/TEXT_END/END_DOCUMENT/FRAGMENT_BEGIN/
#: END_FRAGMENT return `None`) -- the skeleton is exactly this filtered
#: token-name sequence, never the raw token stream.
_SKELETON_TOKEN_NAMES: frozenset[str] = (
    frozenset(events_vocab.OPEN_TOKEN.values())
    | frozenset(events_vocab.CLOSE_TOKEN.values())
    | frozenset(events_vocab.LEAF_TOKEN.values())
    | frozenset(events_vocab.RESUME_TOKEN.values())
    | frozenset(events_vocab.SUSPEND_TOKEN.values())
)

#: The subset of `_SKELETON_TOKEN_NAMES` that consumes a new node ordinal
#: (`node_lifecycle.NodeLifecycleTracker.open`/`.leaf`/`.resume` each append
#: a new `NodeState`; `.close`/`.suspend` mutate/pop an existing one and
#: never append). Matches `NodeState`/`.states`'s own docstring: "acceptance
#: order (open/leaf order, not closure order)".
_ORDINAL_TOKEN_NAMES: frozenset[str] = (
    frozenset(events_vocab.OPEN_TOKEN.values())
    | frozenset(events_vocab.LEAF_TOKEN.values())
    | frozenset(events_vocab.RESUME_TOKEN.values())
)


def _structural_token_name(packed_token_id: int) -> str | None:
    """`packed_token_id`'s structural token name, or `None` for a text/
    byte-fallback token (see `_STRUCTURAL_TOKEN_COUNT`'s own docstring)."""
    if 0 <= packed_token_id < _STRUCTURAL_TOKEN_COUNT:
        return _STRUCTURAL_TOKEN_NAME_BY_ID[packed_token_id]
    return None


def page_structural_skeleton(packed_token_ids: Sequence[int]) -> tuple[str, ...]:
    """The ordered node open/close/leaf/resume/suspend token-name sequence
    in `packed_token_ids`, excluding every text/byte-fallback token and
    every non-node-owning structural control (TEXT_BEGIN/TEXT_END/
    END_DOCUMENT/FRAGMENT_BEGIN/END_FRAGMENT). Two pages ALIGN (path A) iff
    their skeletons compare equal as tuples -- text content never enters
    this comparison, so a page that decoded the right structure with wrong
    text still aligns."""
    return tuple(
        name
        for token_id in packed_token_ids
        if (name := _structural_token_name(token_id)) is not None
        and name in _SKELETON_TOKEN_NAMES
    )


def _gold_node_ordinal_by_anchor_token_index(packed_token_ids: Sequence[int]) -> dict[int, int]:
    """GOLD side: `anchor_token_index -> node ordinal`, replaying
    `packed_token_ids` the same way `NodeLifecycleTracker` assigns
    `local_node_id`s ("n0", "n1", ... acceptance order) without importing
    that (torch-adjacent) tracker -- every OPEN/LEAF/RESUME token consumes
    the next ordinal in token order, CLOSE/SUSPEND never do.

    Keyed by token position because `TypedPropertyTarget.anchor_token_index`
    IS a token position: `typed_property_targets._anchor_token_indices`
    (typed_property_targets.py:419-441) records, for the node's own OPEN/
    LEAF event, `token_index` from `enumerate(sequence.loss_units)` --
    `row_from_materialized`'s own docstring says `loss_units` holds "exactly
    one entry per unpadded token, in order", so that enumeration index IS
    the token's position in `packed_token_ids`. A RESUME token is never an
    anchor (`typed_property_targets.py:224`, "an anchor may never be a
    resume token") but still consumes an ordinal slot here, keeping ordinals
    aligned with the decoded side's `node_states` order (`page_from_dict`:
    `node_states` also carries a RESUME-created `continuation_proxy` slot,
    RFC003's fragment protocol, `node_lifecycle.py` module docstring)."""
    ordinal_by_token_index: dict[int, int] = {}
    ordinal = 0
    for token_index, token_id in enumerate(packed_token_ids):
        name = _structural_token_name(token_id)
        if name is not None and name in _ORDINAL_TOKEN_NAMES:
            ordinal_by_token_index[token_index] = ordinal
            ordinal += 1
    return ordinal_by_token_index


class _MissingTypedPrediction:
    """Sentinel: a supervised gold pointer with no decoded prediction on its
    aligned node. Never equal to any real value -- including `None`, itself
    a legitimate typed value (a `borders_is_null=True` container,
    `typed_decode.TypedPrediction`'s own docstring) -- so
    `typed_macro_accuracy`'s plain `==` counts it wrong without a separate
    branch for "missing"."""

    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return id(self)


_MISSING_TYPED_PREDICTION: Any = _MissingTypedPrediction()


def align_typed_page(
    *,
    document_bundle_id: str,
    lineage_root_id: str,
    gold_packed_token_ids: Sequence[int],
    gold_typed_targets: Sequence[Any],
    decoded_accepted_tokens: Sequence[int],
    decoded_node_states: Sequence[Mapping[str, Any]],
    decoded_typed_predictions: Sequence[Mapping[str, Any]],
) -> TypedPageAlignment:
    """RFC005-ERRATA-005 / T2c path A: align one page's FREE-RUNNING typed
    predictions to its gold typed targets.

    `gold_typed_targets` is `ExactTextCacheRow.typed_targets` (duck-typed
    here as `.supervision_mask`/`.anchor_token_index`/
    `.node_property_pointer`/`.value` -- real `TypedPropertyTarget`
    instances in production, so this function imports neither dataclass and
    stays free of `exact_text_cache.py`'s torch-heavy reader). Only targets
    with `supervision_mask=True` AND a non-`None` `.value` are used as
    labels -- `supervision_mask=False` is the unsupervised case, and a
    `known_mask=True, value=None` record is the documented `known_null`
    case (RFC003 §29.1) `s4_typed_loss.py`'s own per-key batches exclude the
    same way (module docstring above `TypedBatch`/`_encode_target`) rather
    than folding into a per-key tensor; this function excludes both from
    the label set for the same reason, per the task's own instruction
    ("never known_null/unsupervised/DEFERRED as labels"). A DEFERRED
    pointer that still carries a real value and a real head (colour/border
    pointers, `typed_decode.py`'s own carve-out) is NOT separately excluded
    here -- distinguishing it needs `typed_excluded.deferred`, a registry
    this function does not import; flagged in the task report as an open
    item, not silently routed around.

    `decoded_node_states`/`decoded_typed_predictions` are the sealed
    artefact's own JSON-loaded dicts (`sealed.page_to_dict`'s
    `node_states`/`typed_predictions` lists) -- dict access, not attribute
    access, because that is the shape `default_candidate_evaluator` actually
    has on hand after `json.loads`.

    Value comparison space (task instruction, resolved against the actual
    code rather than the task's suggested example): NOT
    `s4_typed_loss._encode_target` -- that function's enum branch returns a
    DOMAIN INDEX (`domain.index(value)`, `s4_typed_loss.py:281`), but
    `typed_decode._value`'s enum branch returns the DOMAIN STRING itself
    (`enum_domain(node_type, pointer)[argmax]`, `typed_decode.py:72`), and
    `TypedPropertyTarget.value` already stores that same domain string
    (`typed_property_targets.py:110`) -- using `_encode_target` here would
    compare an index against a string and mark every aligned enum pointer
    wrong. Boolean and integer/quantized-scalar pointers already match in
    raw units too: `IntegerPropertyHead.predict_value` reconstructs the
    RAW-unit integer (`typed_property_heads.py:438-449`, "no
    floating-point value emitted"), the same raw units
    `TypedPropertyTarget.value` stores and `IntegerPropertyHead.loss`'s own
    `residual_loss` term compares against UNSCALED
    (`typed_property_heads.py:460-463`). So every pointer family compares
    by plain `==` in the value space `typed_decode.decode_typed` already
    emits -- never a bin id, never an invented tolerance.

    A supervised gold pointer with no decoded prediction on its aligned
    node counts wrong (`_MISSING_TYPED_PREDICTION`). A decoded prediction
    with no gold target for its pointer is ignored -- no label to compare
    it against."""
    if page_structural_skeleton(gold_packed_token_ids) != page_structural_skeleton(
        decoded_accepted_tokens
    ):
        return TypedPageAlignment(
            document_bundle_id=document_bundle_id,
            lineage_root_id=lineage_root_id,
            aligned_nodes=(),
        )

    gold_ordinal_by_anchor = _gold_node_ordinal_by_anchor_token_index(gold_packed_token_ids)
    decoded_ordinal_by_local_node_id = {
        state["local_node_id"]: ordinal for ordinal, state in enumerate(decoded_node_states)
    }

    gold_by_ordinal: dict[int, dict[str, Any]] = {}
    for target in gold_typed_targets:
        if not target.supervision_mask or target.value is None:
            continue
        ordinal = gold_ordinal_by_anchor.get(target.anchor_token_index)
        if ordinal is None:
            continue
        gold_by_ordinal.setdefault(ordinal, {})[target.node_property_pointer] = target.value

    decoded_by_ordinal: dict[int, dict[str, Any]] = {}
    for prediction in decoded_typed_predictions:
        ordinal = decoded_ordinal_by_local_node_id.get(prediction["local_node_id"])
        if ordinal is None:
            continue
        decoded_by_ordinal.setdefault(ordinal, {})[prediction["pointer"]] = prediction["value"]

    aligned_nodes = tuple(
        TypedAlignedNodeSample(
            pointer_values={
                pointer: (
                    decoded_by_ordinal.get(ordinal, {}).get(pointer, _MISSING_TYPED_PREDICTION),
                    gold_value,
                )
                for pointer, gold_value in gold_pointers.items()
            }
        )
        for ordinal, gold_pointers in gold_by_ordinal.items()
    )
    return TypedPageAlignment(
        document_bundle_id=document_bundle_id,
        lineage_root_id=lineage_root_id,
        aligned_nodes=aligned_nodes,
    )


def typed_head_macro_accuracy_for_cohort(
    sealed_pages: Sequence[Mapping[str, Any]],
    *,
    manifest_pages: Sequence[Mapping[str, Any]] | None,
    rows_by_bundle: Mapping[str, Any],
) -> TypedMacroAccuracyResult | None:
    """Wire `align_typed_page` + `typed_macro_accuracy` over one sealed
    artefact's own pages. `None` (not attempted) when there is no cohort
    manifest to supply `lineage_root_id` grouping -- mirrors every other
    per-lineage helper's own `None`-when-ungroupable convention above
    (`_per_lineage_row_rate` et al. via `manifest_pages is None`).

    `rows_by_bundle` is the caller's already-read exact-text cache rows
    (`document_bundle_id -> ExactTextCacheRow`, duck-typed as
    `.packed_token_ids`/`.typed_targets`) -- this function never opens a
    cache itself, so it stays reachable from a plain unit test.

    A `"terminated"` page with no gold row (its bundle id is outside the
    val cache) or a non-`"terminated"` page (no `accepted_tokens`/
    `node_states` at all, `sealed.PageDecodeResult`'s own docstring)
    contributes no `TypedPageAlignment` -- never a guessed unaligned-page
    entry for a page this function cannot even see the tokens of."""
    if manifest_pages is None:
        return None
    lineage_by_bundle = _lineage_by_bundle(manifest_pages)
    alignments: list[TypedPageAlignment] = []
    for page in sealed_pages:
        if page.get("outcome") != "terminated":
            continue
        bundle_id = page["document_bundle_id"]
        row = rows_by_bundle.get(bundle_id)
        if row is None:
            continue
        alignments.append(
            align_typed_page(
                document_bundle_id=bundle_id,
                lineage_root_id=lineage_by_bundle.get(bundle_id, bundle_id),
                gold_packed_token_ids=row.packed_token_ids.tolist(),
                gold_typed_targets=row.typed_targets,
                decoded_accepted_tokens=page["accepted_tokens"],
                decoded_node_states=page["node_states"],
                decoded_typed_predictions=page["typed_predictions"],
            )
        )
    if not alignments:
        return None
    return typed_macro_accuracy(alignments)


def _read_exact_text_rows_by_bundle(cache_dir: Path, bundle_ids: Collection[str]) -> dict[str, Any]:
    """The exact-text rows of `bundle_ids`, keyed by bundle id -- mirrors
    `eval.s2_gate.cli._read_exact_text_rows` (same deferred torch/reader
    import, same `features` drop: level 5 never reads visual features,
    only `packed_token_ids`/`typed_targets`) rather than importing that CLI
    module, which this file avoids for the reasons `_cohort_manifest_pages`
    already documents."""
    import torch

    from unflatten.model.exact_text_cache import ExactTextCacheReader

    reader = ExactTextCacheReader(cache_dir, gradient_bearing=False, cache_shards=1)
    wanted = set(bundle_ids)
    positions = {
        bundle: position
        for position, (bundle, _page) in enumerate(reader.page_keys)
        if bundle in wanted
    }
    return {
        bundle: dataclasses.replace(reader[position], features=torch.empty(0))
        for bundle, position in sorted(positions.items(), key=lambda item: item[1])
    }


#: RFC005-ERRATA-005 E3, owner decision 2026-09-24: only these three gates
#: get a "not worse than v14-r4" REFERENCE floor. Free-running token
#: generation (`free_running_decode.py`) never reads a typed/table_plan/
#: node_visual_aligner head before termination -- only `typed_decode`/
#: `region_decode` do (module docstring ~1139-1148) -- and the reference
#: is always decoded with both off (`main`'s `--propose-reference-floors`
#: branch), so only the metrics free-running token generation alone can
#: produce are honestly measurable on it: level 1 (`no_typed_failure_rate`),
#: level 2 (`exact_match_rate`), level 4 (`critical_slice_min_fidelity`).
REFERENCE_FLOOR_METRIC_IDS: frozenset[str] = frozenset(
    {"no_typed_failure_rate", "exact_match_rate", "critical_slice_min_fidelity"}
)

#: Levels 3 (`strict_valid_rate`), 5 (`typed_head_macro_accuracy_rate`) and
#: 7 (`renderer_ready_rate`): the reference checkpoint's older model has no
#: trained typed/table_plan/node_visual_aligner heads (measured: 702 of 991
#: non-backbone `QwenDenseCandidateModel` keys missing from
#: `s4-discovery-v14-r4/checkpoint.pt`), so "not worse than v14-r4" cannot
#: honestly be computed for them even with typed/region decode on. These
#: gates get a FIXED floor the owner sets by hand -- this function only
#: marks `floor_source.kind == "fixed"`, never a value.
FIXED_FLOOR_METRIC_IDS: frozenset[str] = frozenset(
    {"strict_valid_rate", "typed_head_macro_accuracy_rate", "renderer_ready_rate"}
)

#: Level 6 (`p5_development_quality_rate`): no P5 human-verified cohort
#: exists at all (module docstring of `default_candidate_evaluator`) --
#: `propose_reference_floors` never touches this gate, reference or fixed.


class ReferenceFloorProposalRefused(ValueError):
    """`propose_reference_floors` was asked to derive a floor from
    evidence it does not have: no `reference_floor_bootstrap` block on the
    policy, the reference candidate's result carries no
    `per_lineage_metric_samples` at all, or one of
    `REFERENCE_FLOOR_METRIC_IDS` (levels 1/2/4, always reference-floored
    per owner decision 2026-09-24) has no usable per-lineage evidence."""


def propose_full_heads_reference_floors(
    base_policy: Mapping[str, Any],
    reference_result: CandidateResult,
    reference_metric_ids: Collection[str],
) -> dict[str, Any]:
    """Fresh S4: `propose_reference_floors` for a reference with trained heads
    (v15b step 224 000): `reference_metric_ids` get a reference floor, every
    other gate but level 6 is marked fixed. Still one arm, still proposals
    only (`is_placeholder` forced true)."""
    reference_ids = frozenset(reference_metric_ids)
    fixed_ids = frozenset(
        g["metric_id"]
        for g in base_policy["gates"]
        if g["metric_id"] not in reference_ids and g["metric_id"] != "p5_development_quality_rate"
    )
    return _propose_floors(base_policy, reference_result, reference_ids, fixed_ids)


def propose_reference_floors(
    base_policy: Mapping[str, Any], reference_result: CandidateResult
) -> dict[str, Any]:
    """RFC005-ERRATA-005 E3 (owner decision 2026-09-24): measure
    `reference_result` (the v14-r4 reference checkpoint, decoded WITHOUT
    typed/region decode -- see `REFERENCE_FLOOR_METRIC_IDS`'s own
    docstring) and return a NEW policy document whose gates are set as
    follows:

    - `REFERENCE_FLOOR_METRIC_IDS` (levels 1, 2, 4): `floor_source: kind
      "reference"`, `floor_micro = max(0, reference_value - margin)` --
      refused (`ReferenceFloorProposalRefused`) if the reference result
      carries no usable per-lineage evidence for one of them, since these
      three always get a reference floor, never a silently skipped one.
    - `FIXED_FLOOR_METRIC_IDS` (levels 3, 5, 7): `floor_source: kind
      "fixed"` only -- `floor_micro` is left for the owner to fill by hand.
    - level 6 (`p5_development_quality_rate`): left completely untouched.

    Every touched gate's `is_placeholder` is forced back to `True`
    regardless of its previous value: a freshly measured or newly marked
    floor is a PROPOSAL, and only the owner's explicit confirmation
    (editing the policy file) may set it to `False`; `select_checkpoint`
    never confirms its own proposal."""
    return _propose_floors(
        base_policy, reference_result, REFERENCE_FLOOR_METRIC_IDS, FIXED_FLOOR_METRIC_IDS
    )


def _propose_floors(
    base_policy: Mapping[str, Any],
    reference_result: CandidateResult,
    reference_ids: Collection[str],
    fixed_ids: Collection[str],
) -> dict[str, Any]:
    bootstrap = base_policy.get("reference_floor_bootstrap")
    if bootstrap is None:
        raise ReferenceFloorProposalRefused(
            "policy has no 'reference_floor_bootstrap' block -- cannot derive a "
            "reference floor's noise margin without one (RFC005-ERRATA-005 E3)"
        )
    if reference_result.per_lineage_metric_samples is None:
        raise ReferenceFloorProposalRefused(
            f"{reference_result.candidate.candidate_id}: reference candidate result carries "
            "no per_lineage_metric_samples -- cannot derive any reference floor's noise "
            "margin without them (RFC005 §39.2)"
        )
    policy = json.loads(json.dumps(base_policy))  # deep copy via round-trip, no external dep
    for gate in policy["gates"]:
        metric_id = gate["metric_id"]
        if metric_id in fixed_ids:
            gate["floor_source"] = {"kind": "fixed"}
            gate["is_placeholder"] = True
            continue
        if metric_id not in reference_ids:
            continue  # level 6: untouched, no measurement path exists
        samples_by_lineage = reference_result.per_lineage_metric_samples.get(metric_id)
        reference_value = reference_result.metrics_micro.get(metric_id)
        if not samples_by_lineage or reference_value is None:
            raise ReferenceFloorProposalRefused(
                f"{metric_id!r} is one of {sorted(reference_ids)} -- owner "
                "decision 2026-09-24 requires a reference floor for every one of them -- "
                "but the reference result carries no usable per-lineage evidence for it"
            )
        margin = lineage_bootstrap_margin_micro(
            list(samples_by_lineage.values()),
            num_resamples=bootstrap["num_resamples"],
            alpha_micro=bootstrap["alpha_micro"],
            seed=bootstrap["seed"],
        )
        gate["floor_micro"] = max(0, reference_value - margin)
        gate["floor_source"] = {
            "kind": "reference",
            "reference_checkpoint_sha256": reference_result.candidate.sha256,
            "reference_value_micro": reference_value,
            "margin_micro": margin,
        }
        gate["is_placeholder"] = True
    return policy


# ---------------------------------------------------------------------------
# The real evaluator: shells out to eval/s2_gate + eval/builder_chain
# (module docstring: "reuse, do not rebuild"). Not exercised on a GPU by
# the default test suite; see docs/plans/spec-v15-images-selection-
# promotion-plan.md T2 for the H200 runbook.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RealEvaluatorConfig:
    """Every fixed argument `default_candidate_evaluator`'s subprocess
    calls need beyond the per-candidate checkpoint path -- the same shape
    `eval/s2_gate/cli.py decode` and `eval/builder_chain/cli.py` already
    require. Constructing one for a real H200 run is the runbook's job,
    not this module's; nothing here guesses a default path."""

    cohort_manifest: Path
    feature_cache_dir: Path
    exact_text_cache_dir: Path
    run_manifest_bindings: Path
    component_lock: Path
    allowlist: Path
    tokenizer_artifact_set_hash: str
    expect_s1_run_manifest_hash: str
    page_raster_source: tuple[Path, Path]
    device: str = "cuda"
    #: RFC005-ERRATA-005 T2 reference decode mode: pass `eval/s2_gate
    #: decode`'s `--reference-partial-heads` for this candidate's decode
    #: (`propose_reference_floors`'s evaluator call only -- never set for
    #: an ordinary selection candidate). Refused together with the
    #: policy's own `typed_decode`/`region_decode` (untrained heads under
    #: this mode) at the point the decode args are built, mirroring
    #: `eval/s2_gate/cli.py`'s own refusal of the combination.
    reference_partial_heads: bool = False
    #: Owner 2026-09-25 ("cesta 1"): run the builder chain with its opt-in
    #: `headless-defaults.v0` policy (PROPOSED, RFC004-ERRATA-003). Its chain
    #: report is cached under its own name, never reused as the plain one.
    headless_defaults: bool = False
    #: RFC004-ERRATA-004 (RATIFIED 2026-09-26): the two follow-up chain policies. Each
    #: combination caches its chain report under its own name (see
    #: `_chain_tag`), so a reused output directory never hands back a report
    #: built under fewer policies. `table_geometry` needs `headless_defaults`.
    table_geometry: bool = False
    recovered_image_materialization: bool = False
    #: RFC004-ERRATA-004 E5 (RATIFIED 2026-09-26): story-kind dedup, needs headless_defaults too.
    story_dedup: bool = False
    #: RFC004-ERRATA-007 (PROPOSED, owner request 2026-09-27): deterministic run
    #: `/language` fill-in, needs headless_defaults too (one chain order).
    language_detect: bool = False
    #: RFC004-ERRATA-009 (PROPOSED, owner-approved 2026-09-27): page-number-fields.v0 and
    #: autolink.v0, same "own cache tag" contract; neither requires headless_defaults.
    page_number_fields: bool = False
    autolink: bool = False
    #: RFC004-ERRATA-010 (RATIFIED 2026-10-05, DEC-059): leader-fill.v0, own cache tag.
    leader_fill: bool = False
    #: RFC004-ERRATA-008 (PROPOSED, owner 2026-09-27): the two follow-up chain policies. Same
    #: contract as the RFC004-ERRATA-004 pair above -- each combination caches its chain report
    #: under its own name (`_chain_tag`). `runaway_row_repair` needs `headless_defaults`.
    runaway_row_repair: bool = False
    page_geometry_from_raster: bool = False
    #: Fresh S4 (rebuild gap G5): decode a page-image (7.0.0) cache through the
    #: frozen ViT (`eval/s2_gate decode --visual-input-policy/--page-image-store`);
    #: both or neither. `None` (every v15b selection): the feature path, unchanged.
    visual_input_policy: Path | None = None
    page_image_store: Path | None = None
    #: Optional JSONL of the cohort pages' visible text from the store's text
    #: layer (`scripts/selection_page_text_layer.py`); without it
    #: `visible_text_retained_rate` compares against the rebuilt gold text.
    page_text_layer: Path | None = None
    #: `eval/s2_gate decode --prefill-chunk-tokens` (W0-11): feed the prefix pass in
    #: chunks so a 300 DPI page decodes on a 12 GB card (owner 2026-10-05: v14-r4
    #: reference floors measured on the local RTX 4070). `None`: one prefix pass.
    prefill_chunk_tokens: int | None = None
    run_subprocess: Callable[..., subprocess.CompletedProcess[str]] = field(
        default=subprocess.run, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (self.visual_input_policy is None) != (self.page_image_store is None):
            raise ValueError("visual_input_policy and page_image_store go together")
        if self.table_geometry and not self.headless_defaults:
            raise ValueError(
                "table_geometry requires headless_defaults: table-geometry.v0 consumes "
                "the grid headless-defaults.v0 derives"
            )
        if self.story_dedup and not self.headless_defaults:
            raise ValueError("story_dedup requires headless_defaults: one chain order")
        if self.language_detect and not self.headless_defaults:
            raise ValueError("language_detect requires headless_defaults: one chain order")
        if self.runaway_row_repair and not self.headless_defaults:
            raise ValueError("runaway_row_repair requires headless_defaults: one chain order")


def _chain_policy_tag(
    *,
    headless_defaults: bool,
    story_dedup: bool,
    table_geometry: bool,
    recovered_image_materialization: bool,
    language_detect: bool,
    page_number_fields: bool = False,
    autolink: bool = False,
    runaway_row_repair: bool = False,
    page_geometry_from_raster: bool = False,
    leader_fill: bool = False,
) -> str:
    """The policy set a chain report was built under, in application order,
    as a file-name prefix (`""` for the plain chain). Takes plain booleans
    (not a `RealEvaluatorConfig`) so `main()` -- which never builds a
    `RealEvaluatorConfig` itself, only the args that feed one -- can derive
    the same W0-2 `evaluate_candidates(cache_tag=...)` value `_chain_tag`
    derives for the evaluator."""
    parts = [
        tag for flag, tag in (
            (headless_defaults, "headless-defaults.v0-"),
            (story_dedup, "story-kind-dedup.v0-"),
            (runaway_row_repair, "runaway-row-repair.v0-"),
            (page_geometry_from_raster, "page-geometry-from-raster.v0-"),
            (table_geometry, "table-geometry.v0-"),
            (recovered_image_materialization, "recovered-image-materialization.v0-"),
            (language_detect, "language-detect.v0-"),
            (page_number_fields, "page-number-fields.v0-"),
            (autolink, "autolink.v0-"),
            (leader_fill, "leader-fill.v0-"),
        ) if flag
    ]
    return "".join(parts)


def _chain_tag(config: RealEvaluatorConfig) -> str:
    """The policy set a chain report was built under, in application order,
    as a file-name prefix (`""` for the plain chain)."""
    return _chain_policy_tag(
        headless_defaults=config.headless_defaults,
        story_dedup=config.story_dedup,
        table_geometry=config.table_geometry,
        recovered_image_materialization=config.recovered_image_materialization,
        language_detect=config.language_detect,
        page_number_fields=config.page_number_fields,
        autolink=config.autolink,
        runaway_row_repair=config.runaway_row_repair,
        page_geometry_from_raster=config.page_geometry_from_raster,
        leader_fill=config.leader_fill,
    )


def candidate_checkpoint_id(candidate: Candidate) -> str:
    """The ancestry-derived `checkpoint_id` (`ckpt_...`) the sealed artefact carries,
    read from the candidate's sidecar and bound to this file by its sha256 -- the
    builder chain refuses a bare file hash as --checkpoint-id (24.9. prod)."""
    from unflatten.model import checkpoint_ancestry as ca

    path = ca.sidecar_path(candidate.path)
    try:
        record = json.loads(path.read_text())["checkpoint"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise SelectionRefused(f"{path}: no readable ancestry sidecar checkpoint record") from exc
    if record.get("checkpoint_sha256") != candidate.sha256:
        raise SelectionRefused(
            f"{path}: sidecar names checkpoint_sha256 {record.get('checkpoint_sha256')!r}, "
            f"not this candidate's {candidate.sha256!r}"
        )
    return record["checkpoint_id"]


class RealEvaluatorFailed(RuntimeError):
    """One of the three subprocess phases (decode / score / builder-chain
    report) exited non-zero."""


def _run(config: RealEvaluatorConfig, args: Sequence[str]) -> None:
    completed = config.run_subprocess(list(args), capture_output=True, text=True)
    if completed.returncode != 0:
        raise RealEvaluatorFailed(f"{' '.join(args)} failed: {completed.stderr}")


def default_candidate_evaluator(config: RealEvaluatorConfig) -> CandidateEvaluator:
    """Build the real, GPU-resident `CandidateEvaluator`: for one
    candidate, free-running decode over the policy's pinned cohort
    (`unflatten.eval.s2_gate.cli decode --arm trained`), score it
    (`... score`), and run the builder chain
    (`unflatten.eval.builder_chain.cli --sealed ...`) for the structural
    metrics -- the same three tools T2's own module docstring names,
    invoked exactly as the existing `s2-gate`/`builder-chain-report` CLIs
    already validate their own inputs, never reimplemented here.

    Metric mapping (documented once, here, not restated at every call
    site): `no_typed_failure_rate` from the sealed artefact's own
    `accounting` (fraction of non-withdrawn pages that reached `attempt`
    disposition, i.e. that hit no PRE-decode typed classification failure
    -- `capability_excluded`/`grammar_refused`/`text_encoding_failed`/
    `over_length`, RFC005 §36.1's "M6 denominator block"). W0-3(b),
    `docs/runs/v15b-layout-diagnostic-handoff-2026-09-26.md` §9 finding 1
    (CONFIRMED, `tests/eval/s2_gate/test_decode.py::
    test_accounting_decoded_reflects_pre_decode_disposition_not_runtime_termination`):
    this is a pre-decode disposition rate, NOT whether free-running decode
    itself actually reached `outcome == "terminated"` at runtime -- a page
    that hits `max_steps_exceeded`/`vocabulary_exhausted` after starting
    still counts here, because `accounting` is computed before `decode_fn`
    is ever called (`decode.compute_cohort_accounting`) and is never
    revised from the real per-page outcomes. Redefining this metric to
    reflect runtime termination is a live TODO, deliberately NOT done here
    -- it would change what the FROZEN v15b `s4_selection_policy`'s
    `no_typed_failure_rate: 100%` gate measures mid-selection.
    `exact_match_rate`, `critical_slice_min_fidelity` from `... score`'s
    `CohortScore`; `strict_valid_rate`, `renderer_ready_rate` from the
    builder-chain report's `strict_valid_count`/`renderer_status_histogram`
    over `total`. `typed_head_macro_accuracy_rate` (T2c path A) reads the
    val exact-text cache's gold rows for this candidate's own attempted
    pages and aligns them against the sealed artefact's typed predictions
    (`typed_head_macro_accuracy_for_cohort`/`align_typed_page`) -- `None`
    only when no page aligned at all (`insufficient_evidence`), never a
    guessed number. `p5_development_quality_rate` has no measurement path
    yet (no P5 human-verified cohort exists in this codebase today) --
    always reported as `None`. `tie_break_value` is this function's own
    wall-clock seconds for the decode subprocess, divided by the cohort's
    page count.
    """

    def _evaluate(
        candidate: Candidate, policy: Mapping[str, Any], out_dir: Path
    ) -> CandidateResult:
        sealed_path = out_dir / f"sealed-{candidate.sha256}.json"
        score_path = out_dir / f"score-{candidate.sha256}.json"
        chain_tag = _chain_tag(config)
        chain_path = out_dir / f"chain-{chain_tag}{candidate.sha256}.json"

        if config.reference_partial_heads and (
            policy["decode"]["typed_decode"] or policy["decode"]["region_decode"]
        ):
            raise SelectionRefused(
                "RealEvaluatorConfig.reference_partial_heads=True refuses to decode "
                "with typed_decode/region_decode on -- those heads are untrained on a "
                "partial-heads-loaded reference checkpoint (RFC005-ERRATA-005 T2)"
            )

        started = time.monotonic()
        if not sealed_path.exists():
            _run(
                config,
                [
                    sys.executable,
                    "-m",
                    "unflatten.eval.s2_gate.cli",
                    "decode",
                    "--cohort-manifest",
                    str(config.cohort_manifest),
                    "--feature-cache-dir",
                    str(config.feature_cache_dir),
                    "--exact-text-cache-dir",
                    str(config.exact_text_cache_dir),
                    "--run-manifest-bindings",
                    str(config.run_manifest_bindings),
                    "--component-lock",
                    str(config.component_lock),
                    "--allowlist",
                    str(config.allowlist),
                    "--tokenizer-artifact-set-hash",
                    config.tokenizer_artifact_set_hash,
                    "--arm",
                    "trained",
                    "--checkpoint",
                    str(candidate.path),
                    "--expect-s1-run-manifest-hash",
                    config.expect_s1_run_manifest_hash,
                    "--device",
                    config.device,
                    "--max-steps",
                    str(policy["decode"]["max_steps"]),
                    "--loop-guard",
                    policy["decode"]["loop_guard"],
                    *(["--typed-decode"] if policy["decode"]["typed_decode"] else []),
                    *(["--region-decode"] if policy["decode"]["region_decode"] else []),
                    *(["--reference-partial-heads"] if config.reference_partial_heads else []),
                    *(
                        [
                            "--visual-input-policy",
                            str(config.visual_input_policy),
                            "--page-image-store",
                            str(config.page_image_store),
                        ]
                        if config.visual_input_policy is not None
                        else []
                    ),
                    *(
                        ["--prefill-chunk-tokens", str(config.prefill_chunk_tokens)]
                        if config.prefill_chunk_tokens is not None
                        else []
                    ),
                    "--output",
                    str(sealed_path),
                ],
            )
            decoded_now = True
        else:
            decoded_now = False
        elapsed = time.monotonic() - started
        wanted = {gate["metric_id"] for gate in policy["gates"]} | {
            policy["tie_break"]["metric_id"]
        }
        timing_path = out_dir / f"timing-{candidate.sha256}.json"
        if SECONDS_PER_SUCCESSFUL_PAGE in wanted:
            # A resumed selection finds the sealed artefact already there and
            # would time nothing; the wall time of the decode that made it is kept.
            if decoded_now or not timing_path.exists():
                timing_path.write_text(json.dumps({"decode_seconds": elapsed}))
            decode_seconds = float(json.loads(timing_path.read_text())["decode_seconds"])

        if not score_path.exists():
            _run(
                config,
                [
                    sys.executable,
                    "-m",
                    "unflatten.eval.s2_gate.cli",
                    "score",
                    "--sealed-artefact",
                    str(sealed_path),
                    "--exact-text-cache-dir",
                    str(config.exact_text_cache_dir),
                    "--cohort-manifest",
                    str(config.cohort_manifest),
                    "--allowlist",
                    str(config.allowlist),
                    "--tokenizer-artifact-set-hash",
                    config.tokenizer_artifact_set_hash,
                    "--output",
                    str(score_path),
                ],
            )
        score_doc = json.loads(score_path.read_text())

        if not chain_path.exists():
            _run(
                config,
                [
                    sys.executable,
                    "-m",
                    # the package: `builder_chain/cli.py` has no __main__ block, so
                    # `-m ...builder_chain.cli` exits 0 and writes nothing (24.9. prod).
                    "unflatten.eval.builder_chain",
                    "--sealed",
                    str(sealed_path),
                    "--checkpoint-id",
                    candidate_checkpoint_id(candidate),
                    "--page-raster-source",
                    str(config.page_raster_source[0]),
                    str(config.page_raster_source[1]),
                    "--out",
                    str(chain_path),
                    "--render-dir",
                    str(chain_path.parent / f"docx-{chain_tag}{candidate.sha256}"),
                    *(["--headless-defaults"] if config.headless_defaults else []),
                    *(["--story-dedup"] if config.story_dedup else []),
                    *(["--runaway-row-repair"] if config.runaway_row_repair else []),
                    *(
                        ["--page-geometry-from-raster"]
                        if config.page_geometry_from_raster else []
                    ),
                    *(["--table-geometry"] if config.table_geometry else []),
                    *(
                        ["--recovered-image-materialization"]
                        if config.recovered_image_materialization else []
                    ),
                    *(["--language-detect"] if config.language_detect else []),
                    *(["--page-number-fields"] if config.page_number_fields else []),
                    *(["--autolink"] if config.autolink else []),
                    *(["--leader-fill"] if config.leader_fill else []),
                ],
            )
        chain_doc = json.loads(chain_path.read_text())

        sealed_doc = json.loads(sealed_path.read_text())
        accounting = sealed_doc["accounting"]
        attempted = accounting["attempted"]
        no_typed_failure_rate = (
            round(1_000_000 * accounting["decoded"] / attempted) if attempted else None
        )

        total = chain_doc["total"]
        strict_valid_rate = (
            round(1_000_000 * chain_doc["strict_valid_count"] / total) if total else None
        )
        rendered = sum(
            count
            for key, count in chain_doc["renderer_status_histogram"].items()
            if key.split("|", 1)[0] == RENDER_READY_STATUS
        )
        renderer_ready_rate = round(1_000_000 * rendered / total) if total else None

        exact_match_value = score_doc["exact_match_rate"]["value"]
        exact_match_rate = (
            round(1_000_000 * exact_match_value) if exact_match_value is not None else None
        )

        diacritic = score_doc["diacritic_slice"]["value"]
        control_event = score_doc["control_event_slice"]["value"]
        slice_values = [v for v in (diacritic, control_event) if v is not None]
        critical_slice_min_fidelity = round(1_000_000 * min(slice_values)) if slice_values else None

        manifest_pages = _cohort_manifest_pages(config.cohort_manifest)

        # T2c path A (RFC005-ERRATA-005, owner decision 2026-09-24): read
        # the val exact-text cache's gold rows for every sealed page this
        # candidate attempted, and align them (`typed_head_macro_
        # accuracy_for_cohort`). `p5_development_quality_rate` (level 6)
        # stays `None` unchanged -- no P5 human-verified cohort exists.
        sealed_pages = sealed_doc.get("pages", [])
        attempted_bundle_ids = [
            page["document_bundle_id"]
            for page in sealed_pages
            if page.get("outcome") == "terminated"
        ]
        rows_by_bundle = (
            _read_exact_text_rows_by_bundle(config.exact_text_cache_dir, attempted_bundle_ids)
            if attempted_bundle_ids
            else {}
        )
        typed_result = typed_head_macro_accuracy_for_cohort(
            sealed_pages, manifest_pages=manifest_pages, rows_by_bundle=rows_by_bundle
        )
        typed_head_macro_accuracy_rate = (
            round(1_000_000 * typed_result.value)
            if typed_result is not None and typed_result.value is not None
            else None
        )
        typed_head_macro_accuracy_coverage = (
            {
                "aligned_pages": typed_result.pages_contributing,
                "typed_unaligned_pages": typed_result.typed_unaligned_pages,
                "compared_nodes": typed_result.aligned_node_count,
                "pointers_covered": typed_result.pointers_covered,
            }
            if typed_result is not None
            else None
        )

        metrics_micro: dict[str, int | None] = {
            "no_typed_failure_rate": no_typed_failure_rate,
            "exact_match_rate": exact_match_rate,
            "strict_valid_rate": strict_valid_rate,
            "critical_slice_min_fidelity": critical_slice_min_fidelity,
            "typed_head_macro_accuracy_rate": typed_head_macro_accuracy_rate,
            "p5_development_quality_rate": None,
            "renderer_ready_rate": renderer_ready_rate,
        }
        tie_break_value = elapsed / attempted if attempted else None
        fresh_samples: dict[str, Mapping[str, int]] = {}
        if LOOP_FREE_RATE in wanted:
            score_lineage = {
                p["document_bundle_id"]: p["lineage_root_id"]
                for p in score_doc.get("pages") or ()
            }
            metrics_micro[LOOP_FREE_RATE] = loop_free_rate_micro(sealed_pages, attempted=attempted)
            fresh_samples[LOOP_FREE_RATE] = _per_lineage_row_rate(
                [{**p, "loop_free": not page_looped(p)} for p in sealed_pages],
                lineage_of=lambda p: score_lineage.get(p["document_bundle_id"]),
                value_key="loop_free",
            )
        if VISIBLE_TEXT_RETAINED_RATE in wanted and score_doc.get("pages") is not None:
            page_text = (
                load_page_text_layer(config.page_text_layer)
                if config.page_text_layer is not None
                else None
            )
            metrics_micro[VISIBLE_TEXT_RETAINED_RATE] = visible_text_retained_rate_micro(
                score_doc["pages"], page_text=page_text
            )
            by_lineage: dict[str, list[Mapping[str, Any]]] = {}
            for page in score_doc["pages"]:
                by_lineage.setdefault(page["lineage_root_id"], []).append(page)
            fresh_samples[VISIBLE_TEXT_RETAINED_RATE] = {
                lineage: value
                for lineage, pages in by_lineage.items()
                if (value := visible_text_retained_rate_micro(pages, page_text=page_text))
                is not None
            }
        if SECONDS_PER_SUCCESSFUL_PAGE in wanted:
            terminated = {
                p["document_bundle_id"] for p in sealed_pages if p.get("outcome") == "terminated"
            }
            chain_rows_now = chain_doc.get("rows")
            tie_break_value = (
                seconds_per_successful_page(
                    decode_seconds,
                    successes=sum(
                        1
                        for row in chain_rows_now
                        if row["strict_valid"] and row["item_id"] in terminated
                    ),
                )
                if chain_rows_now is not None
                else None
            )

        # RFC005-ERRATA-005 E3 / task point A: per-lineage samples for
        # every gate metric actually measured above (levels 1, 2, 3, 4, 5,
        # 7) -- `None` unless the underlying artefact carries the
        # per-page evidence a lineage grouping needs. `p5_development_
        # quality_rate` (level 6) is never populated here -- no P5 cohort
        # exists, task's own instruction that level 6 stays untouched.
        score_pages = score_doc.get("pages")
        chain_rows = chain_doc.get("rows")
        per_lineage_metric_samples: dict[str, Mapping[str, int]] = {}
        if manifest_pages is not None:
            per_lineage_metric_samples["no_typed_failure_rate"] = (
                _per_lineage_no_typed_failure_rate(manifest_pages)
            )
        if score_pages is not None:
            per_lineage_metric_samples["exact_match_rate"] = _per_lineage_row_rate(
                score_pages, lineage_of=lambda p: p["lineage_root_id"], value_key="exact_match"
            )
            per_lineage_metric_samples["critical_slice_min_fidelity"] = (
                _per_lineage_critical_slice_min_fidelity(score_pages)
            )
        if chain_rows is not None and manifest_pages is not None:
            lineage_by_bundle = _lineage_by_bundle(manifest_pages)
            per_lineage_metric_samples["strict_valid_rate"] = _per_lineage_row_rate(
                chain_rows,
                lineage_of=lambda r: lineage_by_bundle.get(r["item_id"]),
                value_key="strict_valid",
            )
            per_lineage_metric_samples["renderer_ready_rate"] = _per_lineage_row_rate(
                chain_rows,
                lineage_of=lambda r: lineage_by_bundle.get(r["item_id"]),
                value_key="renderer_ready",
            )
        if typed_result is not None:
            per_lineage_metric_samples["typed_head_macro_accuracy_rate"] = (
                typed_result.per_lineage_samples
            )
        per_lineage_metric_samples.update(fresh_samples)

        return CandidateResult(
            candidate=candidate,
            metrics_micro=metrics_micro,
            tie_break_value=tie_break_value,
            per_lineage_metric_samples=per_lineage_metric_samples or None,
            typed_head_macro_accuracy_coverage=typed_head_macro_accuracy_coverage,
        )

    return _evaluate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="select-checkpoint",
        description="RFC005 §40 / DEC-041 A post-hoc checkpoint selection",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="*", default=())
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rank and report every candidate but never declare a best (required "
        "while the policy carries a placeholder floor, RFC005 §12.2/DEC-035 §5)",
    )
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        default=None,
        help="RealEvaluatorConfig field; all of --cohort-manifest/--feature-cache-dir/"
        "--exact-text-cache-dir/--run-manifest-bindings/--component-lock/--allowlist/"
        "--tokenizer-artifact-set-hash/--expect-s1-run-manifest-hash/--page-raster-source "
        "are required together to run the real evaluator from this CLI directly",
    )
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--exact-text-cache-dir", type=Path, default=None)
    parser.add_argument("--run-manifest-bindings", type=Path, default=None)
    parser.add_argument("--component-lock", type=Path, default=None)
    parser.add_argument("--allowlist", type=Path, default=None)
    parser.add_argument("--tokenizer-artifact-set-hash", default=None)
    parser.add_argument("--expect-s1-run-manifest-hash", default=None)
    parser.add_argument(
        "--page-raster-source",
        type=Path,
        nargs=2,
        metavar=("RELEASE_DIR", "STORE_DIR"),
        default=None,
    )
    parser.add_argument(
        "--headless-defaults",
        action="store_true",
        help="run the builder chain with its opt-in headless-defaults.v0 policy (PROPOSED, "
        "RFC004-ERRATA-003): defaults for head-less properties and a derived table grid, "
        "recorded and counted in the chain report",
    )
    parser.add_argument(
        "--table-geometry",
        action="store_true",
        help="run the builder chain with the opt-in table-geometry.v0 policy (RATIFIED, "
        "RFC004-ERRATA-004 E2; requires --headless-defaults): one coherent table geometry "
        "derived from the grid, recorded and counted in the chain report",
    )
    parser.add_argument(
        "--story-dedup",
        action="store_true",
        help="run the builder chain with the opt-in story-kind-dedup.v0 policy (RATIFIED, "
        "RFC004-ERRATA-004 E5; requires --headless-defaults)",
    )
    parser.add_argument(
        "--recovered-image-materialization",
        action="store_true",
        help="run the builder chain with the opt-in recovered-image-materialization.v0 policy "
        "(RATIFIED, RFC004-ERRATA-004 E1): render-recovered images shown uncropped at their "
        "300 DPI pixel size, recorded and counted in the chain report",
    )
    parser.add_argument(
        "--language-detect",
        action="store_true",
        help="run the builder chain with the opt-in language-detect.v0 policy (PROPOSED, "
        "RFC004-ERRATA-007; requires --headless-defaults): deterministically fill a run's "
        "null /language, recorded and counted in the chain report",
    )
    parser.add_argument(
        "--page-number-fields",
        action="store_true",
        help="run the builder chain with the opt-in page-number-fields.v0 policy (PROPOSED, "
        "RFC004-ERRATA-009): a header/footer paragraph matching a closed page-number pattern "
        "gets a real PAGE/NUMPAGES field, recorded and counted",
    )
    parser.add_argument(
        "--autolink",
        action="store_true",
        help="run the builder chain with the opt-in autolink.v0 policy (PROPOSED, "
        "RFC004-ERRATA-009): a run's URL/e-mail span becomes a real hyperlink node, recorded "
        "and counted",
    )
    parser.add_argument(
        "--leader-fill",
        action="store_true",
        help="run the builder chain with the leader-fill.v0 policy (RATIFIED, "
        "RFC004-ERRATA-010, DEC-059): a canonical fill leader in a body paragraph becomes a "
        "tab to a stop with the matching leader, recorded and counted",
    )
    parser.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=None,
        help="pass to the decode subprocess (eval/s2_gate --prefill-chunk-tokens): feed the "
        "prefix pass in chunks of this many tokens so a 300 DPI page fits a 12 GB GPU",
    )
    parser.add_argument(
        "--runaway-row-repair",
        action="store_true",
        help="run the builder chain with the opt-in runaway-row-repair.v0 policy (PROPOSED, "
        "RFC004-ERRATA-008; requires --headless-defaults): a runaway row's trailing empty "
        "cells dropped to the table's modal count, before table-geometry",
    )
    parser.add_argument(
        "--page-geometry-from-raster",
        action="store_true",
        help="run the builder chain with the opt-in page-geometry-from-raster.v0 policy "
        "(PROPOSED, RFC004-ERRATA-008): section page size/orientation reconciled against the "
        "300 DPI raster before table-geometry",
    )
    parser.add_argument(
        "--propose-reference-floors",
        type=Path,
        default=None,
        metavar="REFERENCE_CHECKPOINT",
        help="RFC005-ERRATA-005 E3: instead of running selection, decode REFERENCE_CHECKPOINT "
        "(e.g. the v14-r4 checkpoint) on this policy's own cohort and write "
        "<out>/proposed_policy.json with every measurable gate's floor set to "
        "'not worse than this reference, minus a bootstrap noise margin' -- still "
        "is_placeholder=true until the owner confirms it",
    )
    parser.add_argument(
        "--reference-decode",
        choices=("partial_heads", "full_heads"),
        default="partial_heads",
        help="with --propose-reference-floors: 'partial_heads' (default) is the v14-r4 "
        "reference mode (untrained heads, typed/region decode forced off); 'full_heads' "
        "decodes a reference with trained heads (fresh S4: v15b step 224 000) under the "
        "policy's own decode settings",
    )
    parser.add_argument(
        "--reference-floor-metric",
        action="append",
        default=None,
        metavar="METRIC_ID",
        help="with --propose-reference-floors: a gate that gets a reference floor "
        "(repeatable); every other gate but level 6 is marked fixed. Default: the "
        "v14-r4 set (levels 1, 2, 4).",
    )
    parser.add_argument(
        "--visual-input-policy",
        type=Path,
        default=None,
        help="fresh S4: decode a page-image (7.0.0) --exact-text-cache-dir through the "
        "frozen ViT (with --page-image-store); the checkpoint's tap picks DeepStack",
    )
    parser.add_argument("--page-image-store", type=Path, default=None)
    parser.add_argument(
        "--page-text-layer",
        type=Path,
        default=None,
        help="fresh S4: JSONL of the cohort pages' text-layer text "
        "(scripts/selection_page_text_layer.py) for visible_text_retained_rate; "
        "without it the rebuilt gold text is the reference",
    )
    return parser


#: The `RealEvaluatorConfig` fields the CLI can populate directly, and the
#: argparse destination each comes from -- listed once so `_real_evaluator_config_from_args`
#: and its "are they all present" check share one source of truth.
_REAL_EVALUATOR_ARGS = (
    "cohort_manifest",
    "feature_cache_dir",
    "exact_text_cache_dir",
    "run_manifest_bindings",
    "component_lock",
    "allowlist",
    "tokenizer_artifact_set_hash",
    "expect_s1_run_manifest_hash",
    "page_raster_source",
)


def _real_evaluator_config_from_args(args: argparse.Namespace) -> RealEvaluatorConfig | None:
    """`None` unless every `_REAL_EVALUATOR_ARGS` flag was supplied -- a
    partial set is refused explicitly by `_default_build_evaluator`'s
    caller rather than silently building a half-configured evaluator."""
    values = {name: getattr(args, name) for name in _REAL_EVALUATOR_ARGS}
    if all(v is None for v in values.values()):
        return None
    missing = [f"--{name.replace('_', '-')}" for name, v in values.items() if v is None]
    if missing:
        raise SelectionRefused(
            f"partial real-evaluator configuration: missing {missing} -- supply all of "
            f"{_REAL_EVALUATOR_ARGS} together, or none of them"
        )
    return RealEvaluatorConfig(
        cohort_manifest=values["cohort_manifest"],
        feature_cache_dir=values["feature_cache_dir"],
        exact_text_cache_dir=values["exact_text_cache_dir"],
        run_manifest_bindings=values["run_manifest_bindings"],
        component_lock=values["component_lock"],
        allowlist=values["allowlist"],
        tokenizer_artifact_set_hash=values["tokenizer_artifact_set_hash"],
        expect_s1_run_manifest_hash=values["expect_s1_run_manifest_hash"],
        page_raster_source=tuple(values["page_raster_source"]),
        device=args.device,
        reference_partial_heads=getattr(args, "reference_partial_heads", False),
        headless_defaults=getattr(args, "headless_defaults", False),
        table_geometry=getattr(args, "table_geometry", False),
        recovered_image_materialization=getattr(
            args, "recovered_image_materialization", False
        ),
        story_dedup=getattr(args, "story_dedup", False),
        language_detect=getattr(args, "language_detect", False),
        page_number_fields=getattr(args, "page_number_fields", False),
        autolink=getattr(args, "autolink", False),
        runaway_row_repair=getattr(args, "runaway_row_repair", False),
        page_geometry_from_raster=getattr(args, "page_geometry_from_raster", False),
        leader_fill=getattr(args, "leader_fill", False),
        visual_input_policy=getattr(args, "visual_input_policy", None),
        page_image_store=getattr(args, "page_image_store", None),
        page_text_layer=getattr(args, "page_text_layer", None),
        prefill_chunk_tokens=getattr(args, "prefill_chunk_tokens", None),
    )


def main(
    argv: list[str] | None = None,
    *,
    build_evaluator: Callable[[argparse.Namespace], CandidateEvaluator] | None = None,
) -> int:
    args = build_arg_parser().parse_args(argv)
    policy = load_s4_selection_policy(args.policy)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.propose_reference_floors is not None:
        # RFC005-ERRATA-005 T2 (owner decision 2026-09-24): the reference
        # checkpoint's older model carries no trained typed/table_plan/
        # node_visual_aligner heads (measured, select_checkpoint's own
        # `FIXED_FLOOR_METRIC_IDS` docstring) -- decoded here with a
        # dedicated partial-heads load (`args.reference_partial_heads`,
        # `RealEvaluatorConfig.reference_partial_heads`) and with
        # typed_decode/region_decode forced off, never the policy's own
        # (production) decode settings, which those untrained heads cannot
        # honestly satisfy.
        full_heads = getattr(args, "reference_decode", "partial_heads") == "full_heads"
        args.reference_partial_heads = not full_heads
        evaluator = (build_evaluator or _default_build_evaluator)(args)
        sha = ca.file_sha256(args.propose_reference_floors)
        reference = Candidate(
            candidate_id=args.propose_reference_floors.stem,
            path=args.propose_reference_floors,
            sha256=sha,
        )
        reference_decode_policy = (
            policy
            if full_heads
            else {
                **policy,
                "decode": {**policy["decode"], "typed_decode": False, "region_decode": False},
            }
        )
        # No try/except here (owner instruction, RFC005-ERRATA-005 E3): a
        # refusal decoding the reference under this policy's settings must
        # surface exactly as it would for any other candidate, never be
        # hidden or downgraded to a missing measurement.
        result = evaluator(reference, reference_decode_policy, args.out)
        reference_metrics = getattr(args, "reference_floor_metric", None)
        proposed = (
            propose_reference_floors(policy, result)
            if reference_metrics is None
            else propose_full_heads_reference_floors(policy, result, reference_metrics)
        )
        proposed["note"] = (
            (
                "reference checkpoint decoded with trained heads under the policy's own "
                "decode settings (fresh S4, --reference-decode full_heads). "
            )
            if full_heads
            else (
                "reference checkpoint decoded with typed_decode=false, region_decode=false "
                "(RFC005-ERRATA-005 T2): its older model carries no trained typed/table_plan/"
                "node_visual_aligner heads, so those heads cannot honestly run. "
            )
            + str(proposed.get("note", ""))
        ).strip()
        validate_s4_selection_policy(proposed)
        (args.out / "proposed_policy.json").write_text(
            json.dumps(proposed, indent=2, sort_keys=True)
        )
        return 0

    if has_placeholder_floor(policy) and not args.report_only:
        raise SelectionRefused(
            f"{args.policy} still carries a DEC-035 §5 placeholder floor -- rerun with "
            "--report-only to rank candidates without declaring a best, or supply a "
            "policy whose floors are all owner-ratified"
        )
    refuse_unless_cohort_disjoint_from_eval_pages(policy, args.run_dir)
    candidates = enumerate_candidates(args.run_dir, extra=tuple(args.candidates))
    evaluator = (build_evaluator or _default_build_evaluator)(args)
    cache_tag = _chain_policy_tag(
        headless_defaults=getattr(args, "headless_defaults", False),
        story_dedup=getattr(args, "story_dedup", False),
        table_geometry=getattr(args, "table_geometry", False),
        recovered_image_materialization=getattr(
            args, "recovered_image_materialization", False
        ),
        language_detect=getattr(args, "language_detect", False),
        page_number_fields=getattr(args, "page_number_fields", False),
        autolink=getattr(args, "autolink", False),
        runaway_row_repair=getattr(args, "runaway_row_repair", False),
        page_geometry_from_raster=getattr(args, "page_geometry_from_raster", False),
        leader_fill=getattr(args, "leader_fill", False),
    )
    results = evaluate_candidates(
        candidates, policy, out_dir=args.out, evaluator=evaluator, cache_tag=cache_tag
    )
    report = build_selection_report(results, policy, report_only=args.report_only)
    (args.out / "selection_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return 0


def _default_build_evaluator(args: argparse.Namespace) -> CandidateEvaluator:
    config = _real_evaluator_config_from_args(args)
    if config is None:
        raise SelectionRefused(
            "no real-evaluator configuration supplied on the CLI (see "
            f"{_REAL_EVALUATOR_ARGS}) and no build_evaluator= injected: this CLI never "
            "guesses cohort/cache paths"
        )
    return default_candidate_evaluator(config)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
