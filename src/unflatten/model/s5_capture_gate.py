"""S5 capture gate (DEC-083 point 6, RFC005-ERRATA-014 E1).

Evaluates an `s5_control_eval.py` report against the frozen policy
`policies/s5_capture_gate.v17.json` and keeps the trial ledger
`s5-gate-trials.jsonl`: one line per candidate, at most `trial_budget.max_candidates`,
the first PASS is final, an exhausted budget without a PASS is `none_passed`
(never "best of the failures", never a silent extra trial).

    python -m unflatten.model.s5_capture_gate --gate G --report s5-control-report.json \\
        --candidate ID --checkpoint-sha256 sha256:... --trials s5-gate-trials.jsonl \\
        [--probe-stems captures.json]

Exit 0 PASS, 1 FAIL, 3 `none_passed` (budget spent without a PASS, or a trial refused).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXIT_PASS, EXIT_FAIL, EXIT_NONE_PASSED = 0, 1, 3
_WPI = "whole_page_image_count"
_SCAN = "scan_text_similarity_median_ratio_milli"


class TrialRefused(RuntimeError):
    """A trial the frozen budget or the first-PASS-is-final rule does not allow."""


@dataclass(frozen=True)
class GateResult:
    passed: bool
    values: dict[str, int]
    failures: list[str]


def _capture_rows(report: Mapping[str, Any], excluded: set[tuple[str, str]]) -> list[Mapping]:
    rows = [r for r in report["rows"] if r["capture"] in ("scan", "photo")
            and (r["stem"], r["capture"]) not in excluded]
    if any("text_similarity_ppm" in r and "whole_page_image" not in r for r in rows):
        raise ValueError("report rows lack `whole_page_image`: produced by an old "
                         "s5_control_eval.py, re-measure")
    return rows


def evaluate_gate(gate: Mapping[str, Any], report: Mapping[str, Any], *,
                  exclude: set[tuple[str, str]] | None = None) -> GateResult:
    """Values and verdict of one report. `exclude` is (stem, capture) pairs left out
    (the probe); a verdict is only honest with no exclusions, see the CLI."""
    rows = _capture_rows(report, exclude or set())
    gates = {x["metric_id"]: x for x in gate["gates"]}
    expected = gates[_WPI]["of"] - len(exclude or ())
    scans = [r for r in rows if r["capture"] == "scan"]
    # a scan whose Word is missing has no similarity: it counts as 0, not as absent
    ppm = [r.get("text_similarity_ppm", 0) for r in scans]
    values = {
        _WPI: sum(1 for r in rows if r.get("whole_page_image")),
        _SCAN: int(statistics.median(ppm)) // 1000 if ppm else 0,
        "captures": len(rows),
    }
    failures: list[str] = []
    sample = gate["sample"]
    if report.get("control_sample_sha256") != sample["control_sample_sha256"]:
        failures.append("control_sample_sha256: missing or differs from the frozen sample")
    identities = [(r["stem"], r["capture"]) for r in rows]
    if len(set(identities)) != len(identities):
        failures.append("captures: duplicate page/capture identities")
    for capture, field in (("scan", "scans"), ("photo", "photos")):
        count = sum(r["capture"] == capture for r in rows)
        wanted = sample[field] - sum(c == capture for _, c in (exclude or ()))
        if count != wanted:
            failures.append(f"{field}: {count} rows, expected {wanted}")
    if any(not r.get("docx_produced") or "whole_page_image" not in r
           or "text_similarity_ppm" not in r for r in rows):
        failures.append("captures: incomplete Word/render comparison; re-measure missing outputs")
    if len(rows) != expected:
        failures.append(f"captures: {len(rows)} rows, expected {expected}")
    if values[_WPI] > gates[_WPI]["max"]:
        failures.append(f"{_WPI}: {values[_WPI]} > {gates[_WPI]['max']}")
    if values[_SCAN] < gates[_SCAN]["min"]:
        failures.append(f"{_SCAN}: {values[_SCAN]} < {gates[_SCAN]['min']}")
    return GateResult(passed=not failures, values=values, failures=failures)


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ledger_result(path: Path, *, max_candidates: int) -> str:
    """`pass` | `none_passed` | `open`."""
    trials = _read_ledger(path)
    if any(t["passed"] for t in trials):
        return "pass"
    return "none_passed" if len(trials) >= max_candidates else "open"


def record_trial(path: Path, *, candidate_id: str, checkpoint_sha256: str,
                 result: GateResult, max_candidates: int) -> None:
    trials = _read_ledger(path)
    state = ledger_result(path, max_candidates=max_candidates)
    if state == "pass":
        raise TrialRefused(f"{candidate_id}: the first PASS is final, no further trial")
    if state == "none_passed":
        raise TrialRefused(f"{candidate_id}: trial budget {max_candidates} spent, "
                           "result none_passed (the owner decides)")
    if any(t["candidate_id"] == candidate_id for t in trials):
        raise TrialRefused(f"{candidate_id}: already tried, a trial is not repeated")
    line = {"candidate_id": candidate_id, "checkpoint_sha256": checkpoint_sha256,
            "values": result.values, "failures": result.failures, "passed": result.passed,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--probe-stems", type=Path, default=None,
                        help="captures.json of the morning probe: also print the values "
                             "on the captures outside it (information, not the verdict)")
    args = parser.parse_args(argv)
    gate = json.loads(args.gate.read_text())
    report = json.loads(args.report.read_text())
    budget = gate["trial_budget"]["max_candidates"]
    result = evaluate_gate(gate, report)
    summary: dict[str, Any] = {"candidate": args.candidate, "passed": result.passed,
                               "values": result.values, "failures": result.failures}
    if args.probe_stems is not None:
        probe = {(c["stem"], c["capture"])
                 for c in json.loads(args.probe_stems.read_text())["captures"]}
        outside = evaluate_gate(gate, report, exclude=probe)
        summary["outside_probe"] = outside.values
    try:
        record_trial(args.trials, candidate_id=args.candidate,
                     checkpoint_sha256=args.checkpoint_sha256, result=result,
                     max_candidates=budget)
    except TrialRefused as exc:
        print(json.dumps({**summary, "refused": str(exc),
                          "result": ledger_result(args.trials, max_candidates=budget)}))
        return EXIT_NONE_PASSED if "none_passed" in str(exc) else EXIT_FAIL
    state = ledger_result(args.trials, max_candidates=budget)
    print(json.dumps({**summary, "result": state}, indent=1))
    if result.passed:
        return EXIT_PASS
    return EXIT_NONE_PASSED if state == "none_passed" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
