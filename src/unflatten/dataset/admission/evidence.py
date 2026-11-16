"""Loading and verifying the evidence an admission report certifies.

The report may only speak about a release whose identity recomputes, so
the release is opened through the SAME verification the training path
uses (`dataset.samples`): `release_root_hash` over the manifest
projection, then every manifest-pinned artifact against its recorded
sha256 and size. Reimplementing that check here would let the two drift,
and the one that drifts is the one that stops refusing.

`load_release` raises `AdmissionError` carrying the checks it managed to
run, so the caller can still emit a report that says WHY the release is
not an authority instead of crashing without one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.dataset.samples import (
    RELEASE_SPLITS,
    SampleSelectionError,
    _pinned_rows,
    _release_manifest,
)

#: Split manifest files a release writes (`partial` is not a split — its
#: rows carry their own `split_name`).
MANIFEST_FILES: tuple[str, ...] = (*RELEASE_SPLITS, "partial")


class AdmissionError(ValueError):
    """The evidence base cannot be read or does not verify."""

    def __init__(self, message: str, checks: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.checks: list[dict[str, Any]] = list(checks or [])


@dataclass(frozen=True)
class ReleaseEvidence:
    """Everything the report reads out of one immutable release."""

    release_dir: Path
    manifest: dict[str, Any]
    checks: list[dict[str, Any]]
    gate_rows: list[dict[str, Any]]
    artifacts: dict[str, dict[str, Any]]
    #: `(bundle_id, split_name, lineage_root_id, manifest_file)`, one per
    #: written row — NOT deduplicated, so a bundle listed in two files is
    #: visible as the collision it is.
    membership_rows: list[tuple[str, str, str, str]]
    coverage: dict[str, Any]
    quarantine: list[dict[str, Any]]

    @property
    def membership(self) -> dict[str, tuple[str, str]]:
        """`bundle_id -> (split, lineage_root_id)`, last row wins."""
        return {row[0]: (row[1], row[2]) for row in self.membership_rows}

    @property
    def gate_rows_by_bundle(self) -> dict[str, dict[str, Any]]:
        return {row["document_bundle_id"]: row for row in self.gate_rows}


def _check(name: str, status: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail}


def _iter_manifest_refs(
    manifest: dict[str, Any], prefix: str = ""
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Every `{ref, sha256, size_bytes}` entry the manifest declares."""
    for key, value in sorted(manifest.items()):
        name = f"{prefix}{key}"
        if isinstance(value, dict) and "ref" in value and "sha256" in value:
            yield name, value
        elif isinstance(value, dict):
            yield from _iter_manifest_refs(value, f"{name}.")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "ref" in item and "sha256" in item:
                    label = item.get("split_name") or Path(item["ref"]).stem
                    yield f"{name}[{label}]", item


def _ref_check(release_dir: Path, name: str, entry: dict[str, Any]) -> dict[str, Any]:
    path = release_dir / entry["ref"]
    if not path.is_file():
        return _check(name, "fail", f"{entry['ref']} is missing")
    data = path.read_bytes()
    digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
    if digest != entry.get("sha256"):
        return _check(name, "fail", f"{entry['ref']} hashes to {digest}")
    if len(data) != entry.get("size_bytes"):
        return _check(name, "fail", f"{entry['ref']} has {len(data)} bytes")
    return _check(name, "pass", entry["sha256"])


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"{path} is not readable JSON: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise AdmissionError(f"{path} is not readable JSONL: {exc}") from exc


def load_release(release_dir: Path | str) -> ReleaseEvidence:
    """Open `release_dir`, verifying its identity before reading anything.

    Raises `AdmissionError` (with `.checks`) if the release cannot serve
    as an authority.
    """
    release_dir = Path(release_dir)
    checks: list[dict[str, Any]] = []
    try:
        manifest = _release_manifest(release_dir)
    except SampleSelectionError as exc:
        checks.append(_check("release_root_hash", "fail", str(exc)))
        raise AdmissionError(str(exc), checks) from exc
    checks.append(_check("release_root_hash", "pass", manifest["release_root_hash"]))

    pinned: dict[str, list[dict[str, Any]]] = {}
    for key in ("gate_report", "artifact_index"):
        try:
            pinned[key] = _pinned_rows(release_dir, manifest, key)
        except SampleSelectionError as exc:
            checks.append(_check(key, "fail", str(exc)))
            raise AdmissionError(str(exc), checks) from exc
        checks.append(_check(key, "pass", manifest[key]["sha256"]))

    for name, entry in _iter_manifest_refs(manifest):
        if name in ("gate_report", "artifact_index"):
            continue
        checks.append(_ref_check(release_dir, name, entry))
    failures = [c for c in checks if c["status"] == "fail"]
    if failures:
        raise AdmissionError(
            f"release {release_dir}: {len(failures)} pinned artifact(s) do not "
            "match the manifest",
            checks,
        )

    coverage_entry = manifest.get("coverage_report")
    coverage = (
        _read_json(release_dir / coverage_entry["ref"])
        if isinstance(coverage_entry, dict)
        else {}
    )
    quarantine_entry = manifest.get("quarantine_index")
    quarantine = (
        _read_jsonl(release_dir / quarantine_entry["ref"])
        if isinstance(quarantine_entry, dict)
        else []
    )
    return ReleaseEvidence(
        release_dir=release_dir,
        manifest=manifest,
        checks=checks,
        gate_rows=pinned["gate_report"],
        artifacts={
            row["document_bundle_id"]: row.get("artifacts") or {}
            for row in pinned["artifact_index"]
        },
        membership_rows=_membership_rows(release_dir),
        coverage=coverage,
        quarantine=quarantine,
    )


def _membership_rows(release_dir: Path) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for name in MANIFEST_FILES:
        for row in _read_jsonl(release_dir / "splits" / f"{name}.jsonl"):
            rows.append(
                (
                    row["authority_id"],
                    row.get("split_name", name),
                    row["lineage_root_id"],
                    name,
                )
            )
    return rows


def load_store_index(store_dir: Path | str) -> dict[str, dict[str, Any]]:
    """`bundle_id -> factory record` from the store's `index.jsonl`.

    The store is the migration report's inventory side (§9.4 line 1) and
    the only place that records `verification_tier`, `source_origin` and
    the CURRENT rights decision. It is never the training authority — the
    release is — so a divergence between the two is reported, not applied.
    """
    path = Path(store_dir) / "index.jsonl"
    if not path.is_file():
        raise AdmissionError(f"store {store_dir} has no index.jsonl")
    records: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        envelope = row.get("envelope") or {}
        bundle_id = envelope.get("document_bundle_id")
        if isinstance(bundle_id, str):
            records[bundle_id] = row
    return records
