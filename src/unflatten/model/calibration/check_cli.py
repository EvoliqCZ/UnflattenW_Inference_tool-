"""`python -m unflatten.model.calibration.check_cli` -- RFC005 §22.2/§22.4
one-shot check of a frozen calibration manifest against
`val_calibration_check`, dormant per DEC-047.

The check-partition pairs are read through a
`unflatten.model.locked_test_guard.LockedTestGuard` keyed at
`--guard-state-path`: the *first* run against a given guard state file
reserves it, reads the pairs and closes the state to `consumed`; any
later run against the same state file is refused
(`SecondAccessRefusedError`) per §22.2's "their only acceptance check is
a one-shot val_calibration_check" -- mechanically enforced here, not
just documented.

`candidate_mode` is **always** `review_only`: RFC005 §22.4 leaves
`auto_accept` thresholds `[OPEN]`, and DEC-047 forbids this dormant code
from carrying any threshold that could enable it. This CLI reports
diagnostics only; the recommendation and its reason are fixed until an
owner freezes a real acceptance policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from unflatten.model.calibration._io import load_json, load_pairs
from unflatten.model.calibration.calibrator import IsotonicCalibrator, TemperatureCalibrator
from unflatten.model.calibration.manifest import to_ppm
from unflatten.model.calibration.metrics import (
    aurc,
    brier_score,
    expected_calibration_error,
    risk_coverage_curve,
    wilson_interval,
)
from unflatten.model.locked_test_guard import LockedTestGuard, SecondAccessRefusedError

REVIEW_ONLY_REASON = (
    "RFC005 §22.4: auto_accept thresholds are OPEN and DEC-047 keeps this "
    "dormant code free of any threshold that could enable auto_accept; "
    "candidate_mode is fixed at review_only until an owner freezes a real "
    "acceptance policy."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibrator-state", type=Path, required=True)
    parser.add_argument("--check-pairs", type=Path, required=True)
    parser.add_argument("--guard-state-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _load_calibrator(calibrator_id: str, state: dict[str, Any]):
    if calibrator_id == "temperature":
        calibrator = TemperatureCalibrator()
    elif calibrator_id == "isotonic":
        calibrator = IsotonicCalibrator()
    else:
        raise ValueError(f"unknown calibrator_id {calibrator_id!r}")
    calibrator.load_state_dict(state)
    return calibrator


def _split_pairs(records: list[dict]) -> tuple[list[float], list[float], int]:
    scores: list[float] = []
    labels: list[float] = []
    missing = 0
    for record in records:
        if record["score"] is None:
            missing += 1
            continue
        scores.append(float(record["score"]))
        labels.append(float(record["label"]))
    return scores, labels, missing


def _build_report(
    raw_scores: list[float],
    calibrated_scores: list[float],
    labels: list[float],
    n_missing: int,
    manifest_hash: str,
) -> dict[str, Any]:
    n = len(labels)
    n_correct = int(sum(labels))
    coverage, risk = risk_coverage_curve(calibrated_scores, labels)
    lower, upper = wilson_interval(n_correct, n)
    return {
        "check_report_schema_version": "0.1.0",
        "calibration_manifest_hash": manifest_hash,
        "n_check_examples": n,
        "n_check_missing_scores": n_missing,
        "ece_raw_ppm": to_ppm(expected_calibration_error(raw_scores, labels)),
        "ece_calibrated_ppm": to_ppm(expected_calibration_error(calibrated_scores, labels)),
        "brier_raw_ppm": to_ppm(brier_score(raw_scores, labels)),
        "brier_calibrated_ppm": to_ppm(brier_score(calibrated_scores, labels)),
        "aurc_ppm": to_ppm(aurc(coverage, risk)),
        "accuracy_wilson_lower_ppm": to_ppm(lower),
        "accuracy_wilson_upper_ppm": to_ppm(upper),
        "candidate_mode": "review_only",
        "candidate_mode_reason": REVIEW_ONLY_REASON,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    manifest = load_json(args.calibration_manifest)
    calibrator_state = load_json(args.calibrator_state)
    calibrator = _load_calibrator(manifest["calibrator_id"], calibrator_state)

    if not args.guard_state_path.exists():
        LockedTestGuard.initialize(args.guard_state_path)
    guard = LockedTestGuard(args.guard_state_path)

    try:
        records = guard.reserve_and_read(lambda: load_pairs(args.check_pairs))
    except SecondAccessRefusedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    raw_scores, labels, n_missing = _split_pairs(records)
    if not raw_scores:
        print("REFUSED: no check-partition pair has a present score", file=sys.stderr)
        return 1
    calibrated_scores = calibrator.predict(raw_scores).tolist()

    report = _build_report(
        raw_scores, calibrated_scores, labels, n_missing, manifest["manifest_hash"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
