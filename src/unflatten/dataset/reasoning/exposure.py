"""Producer split-exposure computation (RFC003-ERRATA-003 E3 rule 4).

`docs/plans/reasoning-data-readiness-2026-09-28.md` §2 row e/step 2: today
`envelope_validator.validate_reasoning_corrective_rules` takes
`producer_touches_val_test` as a caller-supplied bool -- nothing computes
it from the producer checkpoint's own ancestry. This module derives it from
`checkpoint_ancestry`'s `dataset_exposures` / `exposure_ledger` /
`transitive_exposure_manifest` plus the target release's split membership,
and fails CLOSED to `"unknown"` whenever the ancestry does not carry
split-level detail -- today's universal case
(`exposure_ledger.kind == "not_produced"`, per
`checkpoint_ancestry.py:383-390`). `"unknown"` refuses exactly like a
confirmed exposure (rule 4: "producent má nulovú expozíciu ... inak
fatálna chyba buildu" -- absence of proof of a clean exposure is not proof
of a clean one).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

ExposureResult = Literal["exposed", "not_exposed", "unknown"]

_PRODUCED = "produced"


def compute_producer_split_exposure(
    producer_ancestry: Mapping[str, Any],
    *,
    release_name: str,
    val_test_ids: Iterable[str],
) -> ExposureResult:
    """`(exposed | not_exposed | unknown)` of `producer_ancestry` against
    `val_test_ids` (bundle or page ids in `release_name`'s val/test splits).

    Fails closed to `"unknown"` unless BOTH `exposure_ledger` and
    `transitive_exposure_manifest` carry `kind == "produced"` -- the
    release-level `dataset_exposures` list alone
    (`checkpoint_ancestry.py:366-382`) never has per-split/per-bundle
    detail, so it can never upgrade this past `"unknown"` on its own.
    """
    ledger = producer_ancestry.get("exposure_ledger") or {}
    transitive = producer_ancestry.get("transitive_exposure_manifest") or {}
    if ledger.get("kind") != _PRODUCED or transitive.get("kind") != _PRODUCED:
        return "unknown"

    exposed_ids = set(transitive.get("exposed_ids", ()))
    val_test_set = set(val_test_ids)
    if exposed_ids & val_test_set:
        return "exposed"
    return "not_exposed"


def producer_touches_val_test(exposure: ExposureResult) -> bool:
    """The E3-rule-4 boolean: `"unknown"` refuses exactly like `"exposed"`
    (fail-closed) -- only a positively confirmed `"not_exposed"` passes."""
    return exposure != "not_exposed"
