"""Hash-verified RFC003 split reader and the bundle -> source map (DP7; torch-free).

``s2_plan.load_split_records`` walks fixed file names and silently skips a
missing one; it never checks a byte. This reader goes through the release
manifest instead: the root hash is recomputed, and every ``split_manifests[]``
entry plus ``partial_manifest`` is sha256/size checked. A pinned file that is
missing or altered raises; a split file the manifest does not pin raises too.

A real split record carries only ``authority_id, authority_kind, eligibility,
lineage_root_id`` (+ ``split_name`` in ``partial.jsonl``): there is no source on
it. :func:`load_bundle_source` derives the source of a bundle from the store
index, and a bundle the index does not know is simply absent from the map.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unflatten.dataset.samples import (
    SampleSelectionError,
    _pinned_rows_entry,
    _release_manifest,
)

PARTIAL_FILE = "partial"


def read_split_records_verified(release_dir: Path | str) -> list[dict[str, Any]]:
    """Every pinned split record, each carrying its real ``split_name``.

    Raises :class:`SampleSelectionError` for: no/foreign manifest root hash,
    a pinned file that is missing, a pinned file whose bytes differ from the
    manifest ref, no ``partial_manifest``, an unpinned ``splits/*.jsonl``, a
    ``partial.jsonl`` row without ``split_name`` and a per-split row whose own
    ``split_name`` contradicts its file."""
    release = Path(release_dir)
    manifest = _release_manifest(release)
    entries = manifest.get("split_manifests")
    if not isinstance(entries, list) or not entries:
        raise SampleSelectionError(f"release {release}: manifest pins no split_manifests")
    partial = manifest.get("partial_manifest")
    pinned_refs = {e["ref"] for e in entries if isinstance(e, dict)}
    if isinstance(partial, dict):
        pinned_refs.add(partial["ref"])
    splits_dir = release / "splits"
    on_disk = (
        {f"splits/{p.name}" for p in splits_dir.glob("*.jsonl")} if splits_dir.is_dir() else set()
    )
    if on_disk - pinned_refs:
        raise SampleSelectionError(
            f"release {release}: split file(s) {sorted(on_disk - pinned_refs)} are not "
            "pinned by the manifest"
        )
    records: list[dict[str, Any]] = []
    for entry in entries:
        name = entry.get("split_name") if isinstance(entry, dict) else None
        rows = _pinned_rows_entry(release, entry, key_label=str(name))
        for row in rows:
            own = row.get("split_name", name)
            if own != name:
                raise SampleSelectionError(
                    f"release {release}: {row.get('authority_id')!r} in {entry['ref']} "
                    f"claims split_name {own!r}"
                )
            records.append({**row, "split_name": name})
    for row in _pinned_rows_entry(release, partial, key_label=PARTIAL_FILE):
        if "split_name" not in row:
            raise SampleSelectionError(
                f"release {release}: partial row {row.get('authority_id')!r} has no "
                "split_name -- its split is never guessed"
            )
        records.append(dict(row))
    return records


#: Pinned copies of ``s4_plan``'s literals: importing ``s4_plan`` drags ``torch`` (via
#: ``exact_text_cache``), and ``data_prep`` is torch-free end to end (DEC-047 addendum A
#: v3 item 7). ``tests/model/rl/test_adapters_release.py`` holds the parity guard.
SYNTHETIC_ORIGIN = "synthetic_ast"
REAL_COHORT = "docxcorp"
SYNTHETIC_COHORT = "synthetic"


class BundleSourceRefused(SampleSelectionError):
    """A store row cannot be classified; a bundle is never assumed real."""


def load_bundle_source(store_dir: Path | str) -> Mapping[str, str]:
    """``{bundle: "docxcorp" | "synthetic"}`` from the store index.

    Torch-free re-implementation of ``s4_plan.load_synthetic_bundles`` (a row
    without ``source_origin``, or rows of one bundle that disagree, refuse) with
    the S4 cell cohort labels. A bundle absent from the index is absent from the
    map -- never defaulted to real."""
    import json

    origins: dict[str, str] = {}
    with open(Path(store_dir) / "index.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            envelope = json.loads(line)["envelope"]
            bundle = envelope["document_bundle_id"]
            origin = envelope.get("source_origin")
            if origin is None:
                raise BundleSourceRefused(
                    f"store row for {bundle!r} carries no source_origin -- a bundle "
                    "is never assumed real"
                )
            if origins.setdefault(bundle, origin) != origin:
                raise BundleSourceRefused(
                    f"store rows for {bundle!r} disagree on source_origin "
                    f"({origins[bundle]!r} vs {origin!r})"
                )
    return {
        bundle: SYNTHETIC_COHORT if origin == SYNTHETIC_ORIGIN else REAL_COHORT
        for bundle, origin in origins.items()
    }
