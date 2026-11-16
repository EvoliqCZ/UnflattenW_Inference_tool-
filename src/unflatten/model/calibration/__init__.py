"""RFC005 §22.2-22.4/§41/§53 calibration math + OOF verification.

Dormant per `docs/decisions/DEC-047-spiaci-kod-v-predstihu.md`: nothing
here is wired into the S4 v15b promoted run, the DEC-044 diagnostic
service or any hashed run identity. `auto_accept` thresholds stay
[OPEN] (§22.4) — this package computes calibration diagnostics and
verifies OOF lineage disjointness; it never ships a default that would
enable `auto_accept`.
"""

from unflatten.model.calibration.calibrator import (
    Calibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
)
from unflatten.model.calibration.confidence_report import (
    ConfidenceReportInvalid,
    build_confidence_report,
)
from unflatten.model.calibration.labels import (
    CorrectnessRecord,
    MissingGoldError,
    as_score_label_arrays,
    default_confidence_score,
    extract_correctness_labels,
    split_missing,
)
from unflatten.model.calibration.manifest import CalibrationManifest, to_ppm
from unflatten.model.calibration.metrics import (
    aurc,
    brier_score,
    expected_calibration_error,
    risk_coverage_curve,
    wilson_interval,
)
from unflatten.model.calibration.oof_manifest import (
    OOFFoldManifest,
    OOFLeakageError,
    verify_no_leakage,
)

__all__ = [
    "Calibrator",
    "IsotonicCalibrator",
    "TemperatureCalibrator",
    "aurc",
    "brier_score",
    "expected_calibration_error",
    "risk_coverage_curve",
    "wilson_interval",
    "OOFFoldManifest",
    "OOFLeakageError",
    "verify_no_leakage",
    "CalibrationManifest",
    "to_ppm",
    "CorrectnessRecord",
    "MissingGoldError",
    "as_score_label_arrays",
    "default_confidence_score",
    "extract_correctness_labels",
    "split_missing",
    "ConfidenceReportInvalid",
    "build_confidence_report",
]
