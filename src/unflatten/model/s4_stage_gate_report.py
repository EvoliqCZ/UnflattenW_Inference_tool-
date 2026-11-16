"""RFC005 §12 Stage Gate Report for S4 (`S4_page_scale_and_assembly`,
RFC005-ERRATA-006 E6, plan W0-4).

RFC005 §12.2: "`promoted` vyžaduje uložený Stage Gate Report. Absencia
reportu znamená `blocked`, nie implicitný úspech." S4 closes on **Gate B**
only (the model track, RFC005-ERRATA-006 E6): B1 selection under a frozen
`selection_policy_hash` with "not worse than v14-r4" floors
(RFC005-ERRATA-005 E3), B2 zero silent truncation, B3 zero text gaps or
duplicates, B4 resume continuity, B5 real vs synthetic reported
separately. **Gate A** (the deterministic assembler, A1-A4) is measured
separately per E6's "Väzba stôp: variant 1" and is reported here as a
non-blocking, informational section -- it can never move `stage_status`,
only a claim about multi-page documents.

This module is a pure *consumer*: every gate reads an artefact some other
module already produced (`select_checkpoint.build_selection_report`,
`s4_plan.build_s4_matrix`, the run's own `s4_eval.jsonl`, an optional
assembler report) and never recomputes their quality metrics. A gate whose
required artefact is missing, unmeasurable, or still carries a placeholder
is `insufficient_evidence` -- treated exactly like `fail` when deciding
`stage_status` (fail closed: RFC005 §12.2's "blocked, nie implicitný
úspech" extends to a report that cannot evidence one of its own gates).

This is a **separate report type** from `stage_gate_report.py`'s
fifteen-contract-gate S1/S2 `StageGateReport`: S4's Gate B is five named
policy-level checks, not the S1/S2 runner's fifteen `GateResult`s, so
bolting S4 onto that schema would force a shape it was never designed for.
`stage_gate_report.py` and its S1/S2 policy/schema are untouched by this
module (see `test_s4_stage_gate_report.py::
test_s1_s2_stage_gate_policy_hash_unaffected_by_s4_module`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from unflatten.model.s4_selection_policy import has_placeholder_floor, s4_selection_policy_hash
from unflatten.model.stage_data_matrix import stage_data_matrix_hash

S4_STAGE_GATE_REPORT_SCHEMA_VERSION = "1.0.0"

#: RFC005-ERRATA-006 E1 (owner 2026-09-25, variant 1).
S4_STAGE_ID = "S4_page_scale_and_assembly"

#: RFC005-ERRATA-006 E6's own table order for Gate B.
B_GATE_NAMES = (
    "B1_selection",
    "B2_no_silent_truncation",
    "B3_no_text_gaps_or_duplicates",
    "B4_resume",
    "B5_real_vs_synthetic",
)

#: RFC005-ERRATA-006 E6's own table order for Gate A (informational only).
A_GATE_NAMES = (
    "A1_golden_identity",
    "A2_determinism",
    "A3_text_preservation",
    "A4_end_to_end",
)

_NON_PROMOTING_STATUSES = frozenset({"fail", "insufficient_evidence"})


class S4StageGateReportInvalid(ValueError):
    """An S4 stage gate report cannot be built from the artefacts given."""


def _file_evidence(path: Path | None) -> dict[str, Any] | None:
    """`{"path", "sha256"}` for an artefact this gate actually read, or
    `None` when there is nothing to point to -- callers filter `None` out
    rather than fabricate an evidence entry for a file that was never
    opened."""
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": f"sha256:{digest.hexdigest()}"}


def _gate(
    name: str, status: str, reason: str, evidence: Sequence[dict[str, Any] | None]
) -> dict[str, Any]:
    return {
        "gate": name,
        "status": status,
        "reason": reason,
        "evidence": [e for e in evidence if e is not None],
    }


# ---------------------------------------------------------------------------
# Gate B (model track) -- closes S4
# ---------------------------------------------------------------------------


def gate_b1_selection(
    *,
    selection_report: Mapping[str, Any],
    selection_report_path: Path | None,
    selection_policy: Mapping[str, Any],
    selection_policy_path: Path | None,
    s4_run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """RFC005-ERRATA-006 B1: the best checkpoint under `selection_policy_hash`
    passes every non-zero "not worse than v14-r4" floor (RFC005-ERRATA-005
    E3) -- read straight off `select_checkpoint.build_selection_report`'s
    own verdict, never re-derived."""
    evidence = [_file_evidence(selection_report_path), _file_evidence(selection_policy_path)]
    recomputed_policy_hash = s4_selection_policy_hash(dict(selection_policy))
    if selection_report.get("selection_policy_hash") != recomputed_policy_hash:
        return _gate(
            "B1_selection",
            "fail",
            "selection report's selection_policy_hash does not match a recompute over the "
            "supplied selection policy",
            evidence,
        )
    if recomputed_policy_hash != s4_run_manifest.get("selection_policy_hash"):
        return _gate(
            "B1_selection",
            "fail",
            "run manifest selection_policy_hash does not match the selection policy actually used",
            evidence,
        )
    if has_placeholder_floor(selection_policy):
        return _gate(
            "B1_selection",
            "insufficient_evidence",
            "selection policy still carries a placeholder floor "
            "(RFC005-ERRATA-005 E3 not yet ratified for every gate)",
            evidence,
        )
    if selection_report.get("report_only"):
        return _gate(
            "B1_selection",
            "insufficient_evidence",
            "selection report was built report_only=True; no best checkpoint was declared",
            evidence,
        )
    best = selection_report.get("best_under_selection_policy_hash")
    if best is None:
        reason = selection_report.get("no_best_reason") or "no best checkpoint declared"
        return _gate(
            "B1_selection", "fail", f"selection report declares no best: {reason}", evidence
        )
    return _gate("B1_selection", "pass", f"best_under_selection_policy_hash={best}", evidence)


def gate_b2_no_silent_truncation(
    *,
    selection_report: Mapping[str, Any],
    selection_report_path: Path | None,
    selection_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """RFC005-ERRATA-006 B2: a page that hits `max_steps` or overflows its
    budget is a failure in the report, never a success (§17.6 odrážka 5).
    `max_steps_exceeded` is a typed failure (`s1_runner.gate_13_free_
    running_no_silent_truncation`'s own branch), so a selection policy
    whose level-1 gate pins `no_typed_failure_rate` at a zero-tolerance
    floor (`floor_micro == 1_000_000`, not a placeholder) and whose winning
    candidate passes it is exactly this gate's evidence."""
    evidence = [_file_evidence(selection_report_path)]
    best = selection_report.get("best_under_selection_policy_hash")
    if best is None:
        return _gate(
            "B2_no_silent_truncation",
            "insufficient_evidence",
            "no declared best candidate (see B1) -- cannot evaluate its truncation rate",
            evidence,
        )
    level1 = next((g for g in selection_policy.get("gates", []) if g.get("level") == 1), None)
    if level1 is None or level1.get("metric_id") != "no_typed_failure_rate":
        return _gate(
            "B2_no_silent_truncation",
            "insufficient_evidence",
            "selection policy has no level-1 no_typed_failure_rate gate",
            evidence,
        )
    if level1.get("is_placeholder") or level1.get("floor_micro") != 1_000_000:
        return _gate(
            "B2_no_silent_truncation",
            "insufficient_evidence",
            "level-1 gate does not enforce a zero-tolerance (floor_micro=1000000) "
            "no_typed_failure_rate floor",
            evidence,
        )
    candidate_doc = next(
        (c for c in selection_report.get("candidates", []) if c["candidate_id"] == best), None
    )
    if candidate_doc is None:
        return _gate(
            "B2_no_silent_truncation",
            "insufficient_evidence",
            f"selection report has no candidate entry for {best!r}",
            evidence,
        )
    gate1_outcome = next((g for g in candidate_doc["gates"] if g["level"] == 1), None)
    if gate1_outcome is None:
        return _gate(
            "B2_no_silent_truncation",
            "insufficient_evidence",
            "winning candidate has no level-1 gate outcome",
            evidence,
        )
    if gate1_outcome["state"] != "pass":
        return _gate(
            "B2_no_silent_truncation",
            "fail",
            f"winning candidate's no_typed_failure_rate={gate1_outcome['value_micro']} "
            "is below the required 1000000",
            evidence,
        )
    return _gate(
        "B2_no_silent_truncation",
        "pass",
        "winning candidate has zero typed failures (max_steps_exceeded included) on the "
        "selection cohort",
        evidence,
    )


def gate_b3_no_text_gaps_or_duplicates(
    *,
    selection_policy: Mapping[str, Any],
    text_continuity_report: Mapping[str, Any] | None = None,
    text_continuity_report_path: Path | None = None,
) -> dict[str, Any]:
    """RFC005-ERRATA-006 B3: zero text gaps or duplicates on every
    strict-valid page. `unflatten.eval.text_continuity_report.build_text_
    continuity_report` is the producing artefact (a caller passes it in
    via `text_continuity_report`); this gate reads its verdict verbatim,
    never re-derives it. Without one, this stays `insufficient_evidence`
    naming the missing artefact, exactly as before this artefact existed
    -- a run that has not yet produced the report is not silently treated
    as clean."""
    if text_continuity_report is None:
        metric_ids = {g.get("metric_id") for g in selection_policy.get("gates", [])}
        if "zero_text_gap_or_duplicate_rate" in metric_ids:
            return _gate(
                "B3_no_text_gaps_or_duplicates",
                "insufficient_evidence",
                "selection policy declares a zero_text_gap_or_duplicate_rate gate, but this "
                "report does not yet know how to read its per-candidate outcome",
                [],
            )
        return _gate(
            "B3_no_text_gaps_or_duplicates",
            "insufficient_evidence",
            "missing artefact: RFC005-ERRATA-006 B3 (zero text gaps/duplicates on every "
            "strict-valid page) has no producing metric_id in s4_selection_policy and no "
            "per-page text-continuity check over the selection cohort's decode output "
            "(unflatten.eval.text_continuity_report was not supplied)",
            [],
        )

    evidence = [_file_evidence(text_continuity_report_path)]
    if text_continuity_report.get("is_placeholder"):
        return _gate(
            "B3_no_text_gaps_or_duplicates",
            "insufficient_evidence",
            "text continuity report's policy still carries PLACEHOLDER thresholds "
            "(owner has not frozen RFC005-ERRATA-006 B3)",
            evidence,
        )
    scope = text_continuity_report.get("scope")
    if scope != "strict_valid_pages":
        return _gate(
            "B3_no_text_gaps_or_duplicates",
            "insufficient_evidence",
            f"text continuity report scope is {scope!r}, not 'strict_valid_pages' -- "
            "RFC005-ERRATA-006 E6 B3 requires zero text gaps/duplicates on every "
            "strict-valid page, not on every attempted page",
            evidence,
        )
    verdict = text_continuity_report.get("verdict")
    reason = text_continuity_report.get("reason") or "no reason recorded on the report"
    if verdict not in ("pass", "fail"):
        return _gate(
            "B3_no_text_gaps_or_duplicates",
            "insufficient_evidence",
            f"text continuity report verdict is {verdict!r}, not pass/fail: {reason}",
            evidence,
        )
    return _gate("B3_no_text_gaps_or_duplicates", verdict, reason, evidence)


def gate_b4_resume(*, eval_log_path: Path | None) -> dict[str, Any]:
    """RFC005-ERRATA-006 B4: "checkpoint po nadviazaní pokračuje bez
    rozdielu" (DEC-023). Evidenced structurally from the run's own
    `s4_eval.jsonl`: `s4_train.py` truncates that log to `step <=
    start_step` on every resume before appending fresh records
    (`_load_eval_log`/`eval_log_path_for_resume`), so a resumed run whose
    steps are not strictly increasing replayed or duplicated a step -- the
    one shape a clean resume cannot produce. This never reads a metric
    VALUE out of the log (module docstring of `stage_gate_report.py`'s own
    sibling caution about `s4_eval.jsonl` applies equally here), only the
    step sequence's own shape."""
    evidence = [_file_evidence(eval_log_path)]
    if eval_log_path is None or not eval_log_path.exists():
        return _gate(
            "B4_resume",
            "insufficient_evidence",
            f"no resume/attempt ledger found at {eval_log_path}",
            evidence,
        )
    steps: list[int] = []
    for line in eval_log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        steps.append(json.loads(line)["step"])
    if not steps:
        return _gate(
            "B4_resume",
            "insufficient_evidence",
            f"{eval_log_path} carries no eval records",
            evidence,
        )
    for prev, cur in zip(steps, steps[1:], strict=False):
        if cur <= prev:
            return _gate(
                "B4_resume",
                "fail",
                f"eval log step sequence is not strictly increasing at {prev} -> {cur} "
                "(a resume replayed or duplicated a step)",
                evidence,
            )
    return _gate(
        "B4_resume",
        "pass",
        f"eval log carries {len(steps)} strictly increasing step record(s) across resumes",
        evidence,
    )


def gate_b5_real_vs_synthetic(
    *,
    stage_data_matrix: Mapping[str, Any] | None,
    stage_data_matrix_path: Path | None,
    s4_run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """RFC005-ERRATA-006 B5: real and synthetic are reported separately;
    zero synthetic must be stated explicitly, never left implicit."""
    evidence = [_file_evidence(stage_data_matrix_path)]
    if stage_data_matrix is None:
        return _gate(
            "B5_real_vs_synthetic",
            "insufficient_evidence",
            "no stage_data_matrix supplied for this run",
            evidence,
        )
    recomputed = stage_data_matrix_hash(dict(stage_data_matrix))
    if recomputed != s4_run_manifest.get("stage_data_matrix_hash"):
        return _gate(
            "B5_real_vs_synthetic",
            "fail",
            "run manifest stage_data_matrix_hash does not match a recompute over the "
            "supplied stage data matrix",
            evidence,
        )
    cohort = stage_data_matrix.get("declared_marginals", {}).get("cohort", {})
    real_micro = cohort.get("docxcorp")
    synthetic_micro = cohort.get("synthetic")
    if real_micro is None or synthetic_micro is None:
        return _gate(
            "B5_real_vs_synthetic",
            "insufficient_evidence",
            "stage data matrix declared_marginals.cohort has no docxcorp/synthetic split",
            evidence,
        )
    if synthetic_micro == 0:
        note = "run used zero synthetic mass -- 100% real (P4 docxcorp)"
    else:
        note = f"real_mass_micro={real_micro} synthetic_mass_micro={synthetic_micro}"
    return _gate(
        "B5_real_vs_synthetic",
        "pass",
        note,
        evidence,
    )


# ---------------------------------------------------------------------------
# Gate A (assembler track) -- measured separately, never blocks S4
# ---------------------------------------------------------------------------


def gate_a_assembler_section(assembler_report: Mapping[str, Any] | None) -> dict[str, Any]:
    """RFC005-ERRATA-006 E6, "Väzba stôp: variant 1" (owner 2026-09-25):
    Gate A is measured separately and blocks only multi-page claims, never
    S4's own `stage_status`. No fixed A1-A4 report producer exists yet
    (M1/M2 measurement and thresholds are still open per the errata's own
    "Stav stopy A" note), so this section is `not_measured` unless a
    caller supplies one; when supplied, each named sub-gate is reported
    verbatim, never re-derived."""
    if assembler_report is None:
        return {
            "section": "gate_a_assembler",
            "status": "not_measured",
            "reason": "no assembler report supplied; RFC005-ERRATA-006 A1-A4 thresholds "
            "(M1/M2) are not yet owner-ratified",
            "gates": [],
            "blocks": "multi-page claims only, never S4 stage_status",
        }
    gates = []
    for name in A_GATE_NAMES:
        entry = assembler_report.get(name)
        if entry is None:
            gates.append({"gate": name, "status": "not_measured", "reason": "absent from report"})
        else:
            gates.append({"gate": name, **dict(entry)})
    return {
        "section": "gate_a_assembler",
        "status": "reported",
        "gates": gates,
        "blocks": "multi-page claims only, never S4 stage_status",
    }


# ---------------------------------------------------------------------------
# The combined report
# ---------------------------------------------------------------------------


def build_s4_stage_gate_report(
    *,
    selection_report: Mapping[str, Any],
    selection_report_path: Path | None,
    selection_policy: Mapping[str, Any],
    selection_policy_path: Path | None,
    s4_run_manifest: Mapping[str, Any],
    stage_data_matrix: Mapping[str, Any] | None,
    stage_data_matrix_path: Path | None,
    eval_log_path: Path | None,
    assembler_report: Mapping[str, Any] | None = None,
    text_continuity_report: Mapping[str, Any] | None = None,
    text_continuity_report_path: Path | None = None,
) -> dict[str, Any]:
    """Build one S4 Stage Gate Report. `stage_status` is `promoted` only
    when every Gate B gate is `pass`; any non-pass gate (`fail` or
    `insufficient_evidence`) makes it `blocked`, with the offending gates
    named in `blocked_reasons` -- fail closed, RFC005 §12.2."""
    gate_b = [
        gate_b1_selection(
            selection_report=selection_report,
            selection_report_path=selection_report_path,
            selection_policy=selection_policy,
            selection_policy_path=selection_policy_path,
            s4_run_manifest=s4_run_manifest,
        ),
        gate_b2_no_silent_truncation(
            selection_report=selection_report,
            selection_report_path=selection_report_path,
            selection_policy=selection_policy,
        ),
        gate_b3_no_text_gaps_or_duplicates(
            selection_policy=selection_policy,
            text_continuity_report=text_continuity_report,
            text_continuity_report_path=text_continuity_report_path,
        ),
        gate_b4_resume(eval_log_path=eval_log_path),
        gate_b5_real_vs_synthetic(
            stage_data_matrix=stage_data_matrix,
            stage_data_matrix_path=stage_data_matrix_path,
            s4_run_manifest=s4_run_manifest,
        ),
    ]
    non_passing = [g for g in gate_b if g["status"] in _NON_PROMOTING_STATUSES]
    if non_passing:
        stage_status = "blocked"
        blocked_reasons = [f"{g['gate']}:{g['status']}:{g['reason']}" for g in non_passing]
    else:
        stage_status = "promoted"
        blocked_reasons = []

    return {
        "s4_stage_gate_report_schema_version": S4_STAGE_GATE_REPORT_SCHEMA_VERSION,
        "stage": S4_STAGE_ID,
        "run_id": s4_run_manifest.get("run_id"),
        "run_class": s4_run_manifest.get("run_class"),
        "gate_b": gate_b,
        "gate_a_assembler": gate_a_assembler_section(assembler_report),
        "blocked_reasons": blocked_reasons,
        "stage_status": stage_status,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise S4StageGateReportInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise S4StageGateReportInvalid(f"{path} is not valid JSON: {exc}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the RFC005 S4 Stage Gate Report (RFC005-ERRATA-006 E6) from "
        "already-produced artefacts."
    )
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--selection-policy", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--stage-data-matrix", type=Path, default=None)
    parser.add_argument("--eval-log", type=Path, default=None)
    parser.add_argument("--assembler-report", type=Path, default=None)
    parser.add_argument(
        "--text-continuity-report", type=Path, default=None,
        help="unflatten.eval.text_continuity_report output JSON (RFC005-ERRATA-006 B3)",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    selection_report = _load_json(args.selection_report)
    selection_policy = _load_json(args.selection_policy)
    s4_run_manifest = _load_json(args.run_manifest)
    stage_data_matrix = _load_json(args.stage_data_matrix) if args.stage_data_matrix else None
    assembler_report = _load_json(args.assembler_report) if args.assembler_report else None
    text_continuity_report = (
        _load_json(args.text_continuity_report) if args.text_continuity_report else None
    )

    report = build_s4_stage_gate_report(
        selection_report=selection_report,
        selection_report_path=args.selection_report,
        selection_policy=selection_policy,
        selection_policy_path=args.selection_policy,
        s4_run_manifest=s4_run_manifest,
        stage_data_matrix=stage_data_matrix,
        stage_data_matrix_path=args.stage_data_matrix,
        eval_log_path=args.eval_log,
        assembler_report=assembler_report,
        text_continuity_report=text_continuity_report,
        text_continuity_report_path=args.text_continuity_report,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
