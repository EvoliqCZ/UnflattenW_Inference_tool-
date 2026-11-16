"""RFC005 §22.2/§53 #48 OOF-fold manifest schema and leakage verifier.

§22.2 requires: an OOF producer and every ancestor must have zero weight
exposure and zero `selection_observation_exposure` to the *entire*
RFC003 leakage component it labels (not just one lineage root); the
post-hoc calibrator and its binning/thresholds are fit only on
`val_calibration_fit`, and their only acceptance check is a one-shot
`val_calibration_check`. This module verifies both rules mechanically
over caller-supplied manifests; it never resolves or creates split
partitions itself (task instruction: no new split partitions).
"""

from __future__ import annotations

from dataclasses import dataclass

FIT_SPLIT = "val_calibration_fit"
CHECK_SPLIT = "val_calibration_check"


class OOFLeakageError(ValueError):
    """A fit/check manifest pair (or a single manifest) violates §22.2/§22.4."""


@dataclass(frozen=True)
class OOFFoldManifest:
    """One OOF fold's hashed manifest (§22.2 second paragraph, §53 #48).

    `component_ids` are the RFC003 leakage-component ids this fold's
    predictions/labels cover. `weight_exposure_components` and
    `selection_observation_exposure_components` are the union, over the
    producer checkpoint and *all* its ancestors, of every leakage
    component that had any weight-update or checkpoint/early-stopping-
    selection exposure — not just the ones this fold labels.
    """

    fold_id: str
    split: str
    component_ids: frozenset[str]
    producer_checkpoint_hash: str
    producer_ancestry_hashes: tuple[str, ...]
    weight_exposure_components: frozenset[str]
    selection_observation_exposure_components: frozenset[str]

    def __post_init__(self) -> None:
        if not self.component_ids:
            raise ValueError("component_ids must be non-empty")
        if self.split not in (FIT_SPLIT, CHECK_SPLIT):
            raise ValueError(
                f"split must be {FIT_SPLIT!r} or {CHECK_SPLIT!r}, got {self.split!r}"
            )

    def has_zero_exposure(self) -> bool:
        """§22.2: producer + ancestors must have zero exposure to the
        *whole* component set this fold labels, not a partial overlap."""
        return not (
            self.weight_exposure_components & self.component_ids
        ) and not (self.selection_observation_exposure_components & self.component_ids)


def verify_no_leakage(fit_manifest: OOFFoldManifest, check_manifest: OOFFoldManifest) -> None:
    """Raise :class:`OOFLeakageError` on any §22.2/§22.4 leakage shape.

    Refuses:
    - either manifest's producer having non-zero exposure to the
      components it labels (§22.2),
    - `fit_manifest` not carrying `val_calibration_fit` or
      `check_manifest` not carrying `val_calibration_check` — the
      concrete shape of "the calibrator fitted on val_calibration_check"
      is `fit_manifest.split == check_manifest.split`,
    - the same leakage component appearing in both manifests' component
      sets ("a lineage in both fit and check").
    """
    if not fit_manifest.has_zero_exposure():
        raise OOFLeakageError(
            f"fit manifest {fit_manifest.fold_id!r} has non-zero producer/"
            "ancestor exposure to a component it labels"
        )
    if not check_manifest.has_zero_exposure():
        raise OOFLeakageError(
            f"check manifest {check_manifest.fold_id!r} has non-zero producer/"
            "ancestor exposure to a component it labels"
        )
    if fit_manifest.split != FIT_SPLIT:
        raise OOFLeakageError(
            f"fit manifest {fit_manifest.fold_id!r} split must be {FIT_SPLIT!r}, "
            f"got {fit_manifest.split!r}"
        )
    if check_manifest.split != CHECK_SPLIT:
        raise OOFLeakageError(
            f"check manifest {check_manifest.fold_id!r} split must be "
            f"{CHECK_SPLIT!r}, got {check_manifest.split!r}"
        )
    overlap = fit_manifest.component_ids & check_manifest.component_ids
    if overlap:
        raise OOFLeakageError(
            f"fit and check manifests share leakage component(s) {sorted(overlap)}"
        )
