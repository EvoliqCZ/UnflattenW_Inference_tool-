"""Sample → training item assembly.

`samples.py` decides *which* pages a task may learn from; this module
reads *what* they contain. It is the last torch-free layer: it returns
raw PNG bytes and plain dicts, so it can be tested without an ML stack
and reused by any framework.

Every object is read through `gates.store.read_object`, which verifies
the artifact's SHA-256 before returning bytes. Training on silently
corrupted targets is worse than crashing, so no fast path skips that.

The fragment index is cached per store+ref: a document's pages are
enumerated consecutively, so the index for a 40-page document would
otherwise be parsed 40 times. The target bundle and mask projection are
cached the same way (`_cached_document_json`): they are one artifact per
document, re-read and re-verified by every page under the naive path,
and the largest bundles in the corpus run to ~156 MB parsed — the
dominant per-page cost on the corpus's largest documents (measured in
`docs/plans/NALEZ-velke-dokumenty-kvadraticka-cena.md`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from unflatten.dataset.gates.store import read_object
from unflatten.dataset.samples import PageSample


class ItemBuildError(ValueError):
    """A sample cannot be assembled into a training item."""


@dataclass(frozen=True)
class TrainingItem:
    """One page image with the labels that describe it."""

    sample: PageSample
    image_bytes: bytes
    #: Document-level Canonical Target Bundle (structural events, text
    #: spans, typed properties). Describes the whole document.
    targets: dict[str, Any] | None
    #: Page-local fragment (RFC003 §26): the slices, char ranges and
    #: event ranges that actually appear on this page. `None` when the
    #: document has no fragment index.
    fragment: dict[str, Any] | None
    #: Font-substitution supervision mask (RFC003 §29). Properties whose
    #: value is known but not visible in the pixels must not be trained.
    mask_projection: dict[str, Any] | None
    #: Observation Sidecar: per-node page geometry in `bbox_ppm`. Read
    #: only when a caller asks for it (the spatial probe does).
    observation_sidecar_data: dict[str, Any] | None = None


@lru_cache(maxsize=64)
def _fragment_entries(store_dir: str, ref_json: str) -> tuple[dict[str, Any], ...]:
    index = json.loads(read_object(Path(store_dir), json.loads(ref_json)))
    return tuple(index.get("entries", ()))


@lru_cache(maxsize=4)
def _cached_document_json(store_dir: str, ref_json: str) -> dict[str, Any]:
    """Parsed JSON for one document-level artifact, cached per store+ref.

    `ref_json` embeds the artifact's own declared sha256
    (`read_object` verifies it on the read that populates this cache),
    so two different artifacts can never share a slot and a stale entry
    can never cross documents or store revisions. `maxsize` is small,
    unlike `_fragment_entries`'s 64: this cache holds full target bundles
    and mask projections, not small indices.

    `maxsize=4`, not 8: a worker processes one document's pages
    consecutively, so a hit only ever needs the current document's
    working set, and the largest bundles parse to ~1 GB+. 4, not fewer
    (T1, `spec-node-visual-aligner-design.md`): `build_item(with_geometry=
    True)` routes FOUR document-level reads through this cache per page --
    fragment, target bundle, mask projection, observation sidecar, in that
    order -- and the fragment ref is the only one of the four that changes
    every page. With `maxsize=3` the sidecar read would evict the target
    bundle on every single page, forcing a re-read and re-hash of it for
    every page of the document -- exactly the `maxsize=2` regression below
    measured for 3 readers, reintroduced by the 4th reader this channel
    adds. With `maxsize=2` (measured regression, mp-perf2: 1.8 s/page on a
    307-page document vs. 0.03-0.4 s/page at this size) all three rotate
    out before the next page ever reaches the bundle/projection again.
    At 4, the next page's fragment miss evicts the PREVIOUS page's
    fragment -- the least recently used of the four, since the fragment
    is always fetched first within a `build_item` call and is therefore
    the oldest by the time the next page needs a slot -- leaving the
    bundle, projection and sidecar to hit.

    Callers must not mutate what they get back — every consumer of
    `TrainingItem.targets`/`.mask_projection` either only reads it or
    goes through `targets.masks.apply_masks`, which deep-copies before
    writing (see that module for why).
    """
    return json.loads(read_object(Path(store_dir), json.loads(ref_json)))


def read_document_json(
    store_dir: Path | str, ref: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Public entry point to `_cached_document_json`, for a caller
    outside `build_item` that needs the same cached object by ref
    (`model.build_cache`'s per-document mask stamp) instead of re-reading
    and re-verifying it."""
    if ref is None:
        return None
    return _cached_document_json(str(store_dir), json.dumps(ref, sort_keys=True))


def _read_json(store_dir: Path, ref: dict[str, Any] | None) -> dict[str, Any] | None:
    return read_document_json(store_dir, ref)


def build_item(
    store_dir: Path | str, sample: PageSample, *, with_geometry: bool = False
) -> TrainingItem:
    """Read every artifact `sample` refers to and assemble a training item.

    `with_geometry` additionally reads the Observation Sidecar, which is
    large and only the spatial probe needs it."""
    store_dir = Path(store_dir)
    fragment: dict[str, Any] | None = None
    if sample.fragment_index is not None:
        entries = _fragment_entries(
            str(store_dir), json.dumps(sample.fragment_index, sort_keys=True)
        )
        match = [e for e in entries if sample.page_index in e.get("page_indices", ())]
        if not match:
            raise ItemBuildError(
                f"{sample.document_bundle_id}: no fragment covers page "
                f"{sample.page_index} ({len(entries)} fragment(s) indexed)"
            )
        fragment = _read_json(store_dir, match[0]["fragment"])

    return TrainingItem(
        sample=sample,
        image_bytes=read_object(store_dir, sample.image),
        targets=_read_json(store_dir, sample.target_bundle),
        fragment=fragment,
        mask_projection=_read_json(store_dir, sample.mask_projection),
        observation_sidecar_data=(
            _read_json(store_dir, sample.observation_sidecar) if with_geometry else None
        ),
    )
