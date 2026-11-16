"""JSON (de)serialization helpers shared by `fit_cli` and `check_cli`.

Not part of the public package API (leading underscore): both CLIs need
the same `OOFFoldManifest` JSON shape and a hash of it for the
`CalibrationManifest`'s `fit_manifest_hash`/`check_manifest_hash` fields,
and duplicating that shape in two CLI modules would be the kind of
copy-paste this codebase's own notes warn about.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from unflatten.model.calibration.oof_manifest import OOFFoldManifest
from unflatten.model.hashing import domain_hash

_OOF_MANIFEST_DOMAIN = "unflatten.rfc005.oof_fold_manifest.v0"


def oof_manifest_from_dict(payload: dict[str, Any]) -> OOFFoldManifest:
    return OOFFoldManifest(
        fold_id=payload["fold_id"],
        split=payload["split"],
        component_ids=frozenset(payload["component_ids"]),
        producer_checkpoint_hash=payload["producer_checkpoint_hash"],
        producer_ancestry_hashes=tuple(payload["producer_ancestry_hashes"]),
        weight_exposure_components=frozenset(payload["weight_exposure_components"]),
        selection_observation_exposure_components=frozenset(
            payload["selection_observation_exposure_components"]
        ),
    )


def oof_manifest_hash(manifest: OOFFoldManifest) -> str:
    payload = {
        "fold_id": manifest.fold_id,
        "split": manifest.split,
        "component_ids": sorted(manifest.component_ids),
        "producer_checkpoint_hash": manifest.producer_checkpoint_hash,
        "producer_ancestry_hashes": list(manifest.producer_ancestry_hashes),
        "weight_exposure_components": sorted(manifest.weight_exposure_components),
        "selection_observation_exposure_components": sorted(
            manifest.selection_observation_exposure_components
        ),
    }
    return domain_hash(_OOF_MANIFEST_DOMAIN, payload)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_oof_manifest(path: Path) -> OOFFoldManifest:
    return oof_manifest_from_dict(load_json(path))


def load_pairs(path: Path) -> list[dict[str, Any]]:
    """A fit/check-partition pairs file: a JSON list of
    `{"page_id": str, "score": float | null, "label": 0 | 1}` records.
    `score: null` is `score_missing` -- never imputed, excluded from
    calibrator fitting/metrics and counted separately by the caller.
    """
    records = load_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty JSON list of pair records")
    return records
