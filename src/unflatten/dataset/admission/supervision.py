"""§9.2 row 6 — required-unknown and `imputed_for_render` supervision.

RFC003 §34.2 states the two full-AST conditions this measures: *no
required property with `known_mask = false`* and *no `imputed_for_render`
field in the authoritative target*. Neither is aggregated anywhere in the
pipeline, so the only honest answer comes from reading the target bundles
themselves — ~20 GB on the real corpus.

The scan is therefore two-phase. Phase one is a byte prefilter: target
bundles are stored as compact canonical JSON, so a violating field is
literally the substring ``"known_mask":false`` or ``imputed_for_render``.
A bundle with neither is clean without being parsed. Phase two parses
only the flagged bundles for exact counts, which also removes the false
positive of a document whose visible TEXT contains one of those
substrings. A bundle that mentions ``known_mask`` without the compact
form is not assumed to be anything — it is parsed too.

Every read goes through `gates.store.read_object`, which verifies the
sha256 and size, so the scan doubles as byte-level re-verification of the
target lane (§9.2 row 7).

`mode="none"` skips the scan and reports `unmeasured`. It is NOT a pass:
the caller blocks on it, because §12.2 makes insufficient evidence
`blocked` rather than a post-hoc success.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from unflatten.dataset.gates.store import StoreError, read_object

#: Scan modes. `full` reads every candidate bundle; `none` reads nothing
#: and reports the row as unmeasured.
SCAN_MODES: tuple[str, ...] = ("none", "full")

_UNKNOWN_MARKER = b'"known_mask":false'
_IMPUTED_MARKER = b"imputed_for_render"
_COMPACT_KNOWN = b'"known_mask":'


def _needs_parse(data: bytes) -> bool:
    """Could this bundle carry a violation? Cheap, and never a false negative."""
    if _UNKNOWN_MARKER in data or _IMPUTED_MARKER in data:
        return True
    # Unexpected serialization: `known_mask` present but not in the
    # compact form the prefilter assumes. Parse rather than guess.
    return b"known_mask" in data and _COMPACT_KNOWN not in data


def _count(bundle: dict[str, Any]) -> tuple[int, int]:
    """`(required_unknown_fields, imputed_fields)` in one target bundle."""
    unknown = imputed = 0
    typed = bundle.get("typed")
    if not isinstance(typed, dict):
        return 0, 0
    for entry in typed.values():
        for field in (entry or {}).get("fields", []):
            if field.get("known_mask") is False:
                unknown += 1
            if field.get("origin") == _IMPUTED_MARKER.decode():
                imputed += 1
    return unknown, imputed


def scan_supervision(
    artifacts: dict[str, dict[str, Any]],
    store_dir: Path | str,
    bundle_ids: list[str],
    *,
    mode: str = "full",
) -> dict[str, Any]:
    """Measure §9.2 row 6 over `bundle_ids`' target bundles."""
    if mode not in SCAN_MODES:
        raise ValueError(f"unknown scan mode {mode!r}; expected one of {SCAN_MODES}")
    measured: dict[str, Any] = {
        "mode": mode,
        "bundles_in_scope": len(bundle_ids),
        "bundles_scanned": 0,
        "bundles_parsed": 0,
        "bundles_with_required_unknown": 0,
        "bundles_with_imputed_for_render": 0,
        "fields_required_unknown": 0,
        "fields_imputed_for_render": 0,
        "bundles_without_target_bundle": 0,
        "unreadable": [],
    }
    if mode == "none":
        return {"status": "unmeasured", "measured": measured}

    store_dir = Path(store_dir)
    for bundle_id in bundle_ids:
        ref = (artifacts.get(bundle_id) or {}).get("target_bundle")
        if ref is None:
            measured["bundles_without_target_bundle"] += 1
            continue
        try:
            data = read_object(store_dir, ref)
        except StoreError as exc:
            measured["unreadable"].append({"bundle": bundle_id, "code": exc.code})
            continue
        measured["bundles_scanned"] += 1
        if not _needs_parse(data):
            continue
        measured["bundles_parsed"] += 1
        try:
            bundle = json.loads(data)
        except json.JSONDecodeError as exc:
            measured["unreadable"].append(
                {"bundle": bundle_id, "code": f"TARGET_BUNDLE_NOT_JSON: {exc}"}
            )
            continue
        unknown, imputed = _count(bundle)
        measured["fields_required_unknown"] += unknown
        measured["fields_imputed_for_render"] += imputed
        measured["bundles_with_required_unknown"] += 1 if unknown else 0
        measured["bundles_with_imputed_for_render"] += 1 if imputed else 0

    violations = (
        measured["bundles_with_required_unknown"]
        + measured["bundles_with_imputed_for_render"]
    )
    status = "fail" if (measured["unreadable"] or violations) else "pass"
    return {"status": status, "measured": measured}
