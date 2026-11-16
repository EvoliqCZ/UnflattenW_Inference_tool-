"""RFC005 §22.2-22.4 calibration manifest -- the frozen artefact `fit_cli`
writes and `check_cli` applies.

Hashing follows the `failure_taxonomy` / RFC001 CBOR-profile convention
used elsewhere in this codebase (see
`unflatten.eval.failure_taxonomy.report`): the RFC001 CBOR profile is
float-free, so every metric in the hashed core is an integer
`*_ppm` (parts-per-million) value, never a Python float. Calibrator
parameters are likewise ppm-scaled integers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unflatten.model.hashing import domain_hash

_DOMAIN = "unflatten.rfc005.calibration_manifest.v0"
CALIBRATION_MANIFEST_SCHEMA_VERSION = "0.1.0"

#: The only two calibrator ids this manifest format knows.
CALIBRATOR_IDS = ("temperature", "isotonic")

PPM = 1_000_000


def to_ppm(value: float) -> int:
    """Round a `[0, 1]`-ish float metric to an integer ppm value."""
    return round(value * PPM)


@dataclass(frozen=True, slots=True)
class CalibrationManifest:
    """A frozen, hashed record of one fit run (§22.2, §53 #48).

    `partition_ids` names the fit-partition(s) the calibrator was fit on
    (e.g. `("val_calibration_fit",)`) -- supplied by the caller, this
    module resolves no split itself. `metrics_before_ppm`/
    `metrics_after_ppm` are ECE/Brier computed on the *fit* data before
    and after calibration (the manifest's own sanity check; the
    check-partition numbers are `check_cli`'s job, not this one's).
    """

    calibration_manifest_schema_version: str
    calibrator_id: str
    calibrator_params_ppm: dict[str, int]
    fit_manifest_hash: str
    check_manifest_hash: str
    partition_ids: tuple[str, ...]
    n_fit_examples: int
    n_fit_missing_scores: int
    metrics_before_ppm: dict[str, int]
    metrics_after_ppm: dict[str, int]
    manifest_hash: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.calibrator_id not in CALIBRATOR_IDS:
            raise ValueError(
                f"calibrator_id must be one of {CALIBRATOR_IDS}, got {self.calibrator_id!r}"
            )
        if not self.partition_ids:
            raise ValueError("partition_ids must be non-empty")
        if self.n_fit_examples <= 0:
            raise ValueError(f"n_fit_examples must be positive, got {self.n_fit_examples}")
        if self.n_fit_missing_scores < 0:
            raise ValueError("n_fit_missing_scores must be non-negative")

    def core_payload(self) -> dict[str, Any]:
        """The float-free payload the manifest hash covers."""
        return {
            "calibration_manifest_schema_version": self.calibration_manifest_schema_version,
            "calibrator_id": self.calibrator_id,
            "calibrator_params_ppm": dict(sorted(self.calibrator_params_ppm.items())),
            "fit_manifest_hash": self.fit_manifest_hash,
            "check_manifest_hash": self.check_manifest_hash,
            "partition_ids": list(self.partition_ids),
            "n_fit_examples": self.n_fit_examples,
            "n_fit_missing_scores": self.n_fit_missing_scores,
            "metrics_before_ppm": dict(sorted(self.metrics_before_ppm.items())),
            "metrics_after_ppm": dict(sorted(self.metrics_after_ppm.items())),
        }

    def compute_hash(self) -> str:
        return domain_hash(_DOMAIN, self.core_payload())

    def to_dict(self) -> dict[str, Any]:
        """Core payload plus the computed `manifest_hash`."""
        payload = self.core_payload()
        payload["manifest_hash"] = self.compute_hash()
        return payload

    @classmethod
    def sealed(
        cls,
        *,
        calibrator_id: str,
        calibrator_params_ppm: dict[str, int],
        fit_manifest_hash: str,
        check_manifest_hash: str,
        partition_ids: tuple[str, ...],
        n_fit_examples: int,
        n_fit_missing_scores: int,
        metrics_before_ppm: dict[str, int],
        metrics_after_ppm: dict[str, int],
    ) -> CalibrationManifest:
        """Build a manifest and freeze its `manifest_hash` in one call."""
        manifest = cls(
            calibration_manifest_schema_version=CALIBRATION_MANIFEST_SCHEMA_VERSION,
            calibrator_id=calibrator_id,
            calibrator_params_ppm=calibrator_params_ppm,
            fit_manifest_hash=fit_manifest_hash,
            check_manifest_hash=check_manifest_hash,
            partition_ids=partition_ids,
            n_fit_examples=n_fit_examples,
            n_fit_missing_scores=n_fit_missing_scores,
            metrics_before_ppm=metrics_before_ppm,
            metrics_after_ppm=metrics_after_ppm,
        )
        object.__setattr__(manifest, "manifest_hash", manifest.compute_hash())
        return manifest
