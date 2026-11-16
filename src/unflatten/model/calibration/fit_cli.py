"""`python -m unflatten.model.calibration.fit_cli` -- RFC005 §22.2 calibrator
fit, dormant per DEC-047.

Inputs (all caller-supplied paths; this CLI resolves no split itself):

- `--fit-pairs`: JSON list of `{"page_id", "score": float | null, "label": 0|1}`
  restricted by the caller to `val_calibration_fit` (`score: null` = a
  genuinely missing confidence field, excluded from fitting, counted).
- `--fit-manifest` / `--check-manifest`: the two `OOFFoldManifest` JSON
  records the fit/check partitions must be produced under (§22.2).
- `--calibrator`: `temperature` or `isotonic`.
- `--out`: where to write the resulting `CalibrationManifest` JSON. The
  fitted calibrator's exact parameters (full isotonic thresholds, not
  just the ppm-rounded summary in the hashed manifest) are written
  alongside it at `<out stem>.state.json` for `check_cli` to load.

Leakage is verified with `verify_no_leakage` *before* any fitting; a
leakage violation exits non-zero and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from unflatten.model.calibration._io import load_oof_manifest, load_pairs, oof_manifest_hash
from unflatten.model.calibration.calibrator import IsotonicCalibrator, TemperatureCalibrator
from unflatten.model.calibration.manifest import CalibrationManifest, to_ppm
from unflatten.model.calibration.metrics import brier_score, expected_calibration_error
from unflatten.model.calibration.oof_manifest import OOFLeakageError, verify_no_leakage


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-pairs", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--check-manifest", type=Path, required=True)
    parser.add_argument("--calibrator", choices=("temperature", "isotonic"), required=True)
    parser.add_argument("--partition-id", action="append", dest="partition_ids")
    parser.add_argument("--out", type=Path, required=True)
    return parser


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


def _fit_metrics(scores: list[float], labels: list[float]) -> dict[str, int]:
    return {
        "ece_ppm": to_ppm(expected_calibration_error(scores, labels)),
        "brier_ppm": to_ppm(brier_score(scores, labels)),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    fit_manifest = load_oof_manifest(args.fit_manifest)
    check_manifest = load_oof_manifest(args.check_manifest)
    try:
        verify_no_leakage(fit_manifest, check_manifest)
    except OOFLeakageError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    records = load_pairs(args.fit_pairs)
    scores, labels, n_missing = _split_pairs(records)
    if not scores:
        print("REFUSED: no fit-partition pair has a present score", file=sys.stderr)
        return 1

    metrics_before = _fit_metrics(scores, labels)

    if args.calibrator == "temperature":
        calibrator = TemperatureCalibrator()
        calibrator.fit(scores, labels)
        params_ppm = {"temperature_ppm": to_ppm(calibrator.temperature)}
    else:
        calibrator = IsotonicCalibrator()
        calibrator.fit(scores, labels)
        params_ppm = {"n_thresholds": len(scores)}

    calibrated = calibrator.predict(scores).tolist()
    metrics_after = _fit_metrics(calibrated, labels)

    partition_ids = tuple(args.partition_ids) if args.partition_ids else ("val_calibration_fit",)
    manifest = CalibrationManifest.sealed(
        calibrator_id=args.calibrator,
        calibrator_params_ppm=params_ppm,
        fit_manifest_hash=oof_manifest_hash(fit_manifest),
        check_manifest_hash=oof_manifest_hash(check_manifest),
        partition_ids=partition_ids,
        n_fit_examples=len(scores),
        n_fit_missing_scores=n_missing,
        metrics_before_ppm=metrics_before,
        metrics_after_ppm=metrics_after,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    state_path = args.out.with_name(args.out.stem + ".state.json")
    state_path.write_text(json.dumps(calibrator.state_dict(), indent=2, sort_keys=True))
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
