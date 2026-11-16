"""Read visual features from an EXISTING exact-text cache instead of a
structural feature cache (spec-cache-rebuild-v15).

`cache-candidate-v12` -- the feature cache `build_exact_text_cache.py`'s
`open_feature_sources` was written to read from -- was deleted from disk.
Its features are not lost: every row of an exact-text cache built from it
(`exact_text_cache.ExactTextCacheRow`) already carries `features` and
`grid_thw` verbatim, copied from that feature cache's own `CacheEntry`
(`row_from_materialized`). This module lets a NEW exact-text-cache build
read those two fields back out of an OLDER exact-text cache, so a rebuild
whose feature cache is gone can still avoid a GPU re-extraction pass.

**Measured, not assumed: `exact_text_cache.ExactTextCacheReader` cannot
open the real source.** The production source for the v15 rebuild,
`cache-exact-text-v14-r4/train`, is `cache_version` `"5.0.0"` (measured on
disk, 2026-09-23). This repository's `exact_text_cache.
EXACT_TEXT_CACHE_VERSION` is now `"6.0.0"` (the region-channel work
landing concurrently with this task, `spec-node-visual-aligner-design.md`
T1): `ExactTextCacheReader._check_family_and_completeness` refuses any
cache whose `cache_version` is not EXACTLY that string. Loosening that
check would not be enough either -- `_row_from_storage` separately refuses
a row with no `region_targets` key, and a v14-r4 row genuinely has none
(measured: its keys are exactly `{authority_target_hash,
controller_mode_ids, document_bundle_id, effective_loss_mask, features,
fragment_id, grid_thw, loss_family_ids, loss_unit_index, loss_units,
mask_projection_hash, packed_input_ids, packed_token_ids, page_index,
source_event_index, structural_allowed_local_ids,
structural_allowed_row_ptr, text_scalar_end, text_scalar_start,
typed_cardinalities, typed_targets, typed_union_arms}` -- no
`region_targets`). So this module does NOT go through
`ExactTextCacheReader` at all: it reads the manifest and shard files at
the raw storage level, verifying only what reading `features`/`grid_thw`
back out requires, and is deliberately tolerant of a cache_version that
predates a channel this module never touches.

**This is an adapter, not a shortcut around identity.** `open_feature_sources`
(`build_exact_text_cache.py`) exists to guarantee that "the features of
this page" means one thing, by refusing a source whose identity does not
match. An exact-text cache carries its own, DIFFERENT identity
(`exact_text_cache.ExactTextCacheIdentity`), and this module applies the
equivalent check over the fields that name WHICH features a row holds,
never fewer:

- `backbone_artifact_hash` -- the direct analog of `feature_cache`
  manifest identity's `backbone` + `backbone_revision` COLLAPSED into one
  hash (`build_exact_text_cache.build_identity`'s own docstring: "the
  source cache's `backbone_revision`, the hub-recorded commit the frozen
  weights came from").
- `processor_artifact_hash` -- the analog of `processor`.
- `feature_tap`, `max_pixels` -- carried under the same names.

**What does NOT compose, named rather than hidden.** `feature_cache`'s
cross-source check (`build_exact_text_cache._CROSS_SOURCE_IDENTITY_FIELDS`)
also compares `multiscale` and `merged_grid`. `ExactTextCacheIdentity` has
no field for either: the exact-text cache format never recorded them, so
a page's `multiscale`/`merged_grid` provenance is NOT verifiable through
this adapter. That is a real gap in what this source can prove, not
something this module papers over by inventing a value.

`storage_dtype` needs no comparison here (unlike `feature_cache`, whose
manifest-level `storage_dtype` can vary): `exact_text_cache.
row_from_materialized` unconditionally casts `features` to `float16`
before writing, so every exact-text cache's rows are float16 by
construction -- checked per row below regardless, since this module reads
raw storage dicts a schema check never covered.

A page present in more than one source is refused, exactly like
`open_feature_sources`. A page absent from every source is never silently
dropped: :func:`entry` returns `None` for it and a caller must count that
under a named refusal (mirrors `build_exact_text_cache.py`'s own
`SKIP_COUNTS` discipline) -- :func:`missing_page_keys` is the tool for
measuring how many pages that will be, before a build is even started.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from unflatten.model.exact_text_cache import CACHE_FAMILY, PageKey, is_page_image_identity

#: `ExactTextCacheIdentity` fields that name WHICH features a row's
#: `.features`/`.grid_thw` hold -- the exact-text-cache analogs of
#: `build_exact_text_cache._CROSS_SOURCE_IDENTITY_FIELDS`'s `backbone`,
#: `backbone_revision`, `processor`, `feature_tap`, `max_pixels` (module
#: docstring: `backbone_artifact_hash`/`processor_artifact_hash` are each
#: ONE hash where the feature-cache identity used two/one differently
#: named fields). `multiscale` and `merged_grid` have no member here on
#: purpose -- see the module docstring's "what does NOT compose".
CROSS_SOURCE_IDENTITY_FIELDS: tuple[str, ...] = (
    "backbone_artifact_hash",
    "processor_artifact_hash",
    "feature_tap",
    "max_pixels",
)

#: `cache_version` at and above which `grid_thw` is guaranteed present on
#: every row (`exact_text_cache.py`'s own "1.1.0: `ExactTextCacheRow.
#: grid_thw` added" changelog entry) -- the oldest cache this module may
#: ever admit, because `grid_thw` is exactly what it exists to read back.
_MIN_CACHE_VERSION = (1, 1, 0)

#: `cache_version` at and above which every `shard_index` entry carries a
#: `keys` list (DEC-MP-4, "4.0.0: the mandatory ... `scope` block and the
#: per-shard `keys` list") -- what this module's index is built from,
#: without loading a single shard.
_MIN_SCOPE_VERSION = (4, 0, 0)

DEFAULT_CACHE_SHARDS = 8


class ExactTextFeatureSourceMismatch(ValueError):
    """A source exact-text cache is not admissible as a feature source:
    wrong `cache_family`, too old a `cache_version` to carry `grid_thw` or
    a per-shard key list, an incomplete/tampered manifest, a shard whose
    bytes changed after the build, identity that does not match an
    already-opened source on the fields that name which features a row
    holds, or a page key already claimed by an earlier source."""


@dataclass(frozen=True, slots=True)
class FeatureOnlyEntry:
    """The two fields this module exists to read back, for one page --
    the same duck-typed shape `build_exact_text_cache.py`'s call site
    already expects (`entry.features`, `entry.grid_thw`:
    `feature_cache.CacheEntry` and this class are interchangeable there).
    """

    features: torch.Tensor
    grid_thw: torch.Tensor


def _parse_version(cache_version: Any, *, source_dir: Path | str) -> tuple[int, int, int]:
    if not isinstance(cache_version, str):
        raise ExactTextFeatureSourceMismatch(
            f"{source_dir}: manifest cache_version is {cache_version!r}, not a string"
        )
    parts = cache_version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ExactTextFeatureSourceMismatch(
            f"{source_dir}: manifest cache_version {cache_version!r} is not "
            "'X.Y.Z'"
        )
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _read_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        raise ExactTextFeatureSourceMismatch(f"{cache_dir}: no manifest.json")
    return json.loads(path.read_text())


def _checked_identity(manifest: dict[str, Any], *, source_dir: Path | str) -> dict[str, Any]:
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ExactTextFeatureSourceMismatch(
            f"{source_dir}: manifest carries no 'identity' object"
        )
    required = (*CROSS_SOURCE_IDENTITY_FIELDS, "admission")
    missing = [name for name in required if identity.get(name) is None]
    if missing:
        raise ExactTextFeatureSourceMismatch(
            f"{source_dir}: identity is missing (or null on) required field(s) "
            + ", ".join(sorted(missing))
        )
    return identity


def _checked_shard_index(
    manifest: dict[str, Any], *, source_dir: Path | str
) -> list[dict[str, Any]]:
    shard_index = manifest.get("shard_index")
    if not isinstance(shard_index, list):
        raise ExactTextFeatureSourceMismatch(
            f"{source_dir}: manifest carries no 'shard_index' list"
        )
    for shard in shard_index:
        if not isinstance(shard.get("keys"), list):
            raise ExactTextFeatureSourceMismatch(
                f"{source_dir}: shard {shard.get('name')!r} carries no 'keys' "
                "list -- the page key set cannot be reconstructed without "
                "loading every shard, and is refused rather than guessed"
            )
    return shard_index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_feature_entry(
    data: dict[str, Any], *, source_dir: Path | str, document_bundle_id: str, page_index: int
) -> FeatureOnlyEntry:
    features = data.get("features")
    grid_thw = data.get("grid_thw")
    if not isinstance(features, torch.Tensor) or features.dtype != torch.float16:
        raise ExactTextFeatureSourceMismatch(
            f"{source_dir}: {document_bundle_id} page {page_index}: features "
            f"must be a float16 tensor, got {type(features).__name__}"
        )
    if not isinstance(grid_thw, torch.Tensor) or tuple(grid_thw.shape) != (3,):
        raise ExactTextFeatureSourceMismatch(
            f"{source_dir}: {document_bundle_id} page {page_index}: grid_thw "
            "must be a (3,) tensor"
        )
    grid = grid_thw.to(torch.int64)
    if bool((grid < 1).any()):
        raise ExactTextFeatureSourceMismatch(
            f"{source_dir}: {document_bundle_id} page {page_index}: grid_thw "
            f"{grid.tolist()} has a non-positive axis"
        )
    temporal, height, width = (int(v) for v in grid)
    visual_rows = int(features.shape[0])
    if not any(
        height % merge == 0
        and width % merge == 0
        and temporal * (height // merge) * (width // merge) == visual_rows
        for merge in (1, 2, 4)
    ):
        raise ExactTextFeatureSourceMismatch(
            f"{source_dir}: {document_bundle_id} page {page_index}: grid_thw "
            f"{grid.tolist()} cannot produce the {visual_rows} visual row(s) "
            "features stores at any merge size"
        )
    return FeatureOnlyEntry(features=features, grid_thw=grid)


@dataclass(slots=True)
class RawFeatureShardReader:
    """One exact-text cache, read at the raw shard-storage level -- the
    version-tolerant counterpart of `exact_text_cache.ExactTextCacheReader`
    (module docstring: the real v14-r4 source cannot go through that
    reader at all)."""

    cache_dir: Path
    manifest: dict[str, Any]
    identity: dict[str, Any]
    #: `(document_bundle_id, page_index)` -> shard ordinal, built from the
    #: manifest's own per-shard `keys` lists -- no shard is loaded here.
    page_index: dict[PageKey, tuple[int, int]]
    cache_shards: int = DEFAULT_CACHE_SHARDS
    _loaded: OrderedDict[int, list[dict[str, Any]]] = None  # type: ignore[assignment]
    _verified: set[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._loaded is None:
            self._loaded = OrderedDict()
        if self._verified is None:
            self._verified = set()

    def _shard_rows(self, shard_ordinal: int) -> list[dict[str, Any]]:
        rows = self._loaded.get(shard_ordinal)
        if rows is not None:
            self._loaded.move_to_end(shard_ordinal)
            return rows
        shard = self.manifest["shard_index"][shard_ordinal]
        path = self.cache_dir / shard["name"]
        if shard_ordinal not in self._verified:
            actual = _sha256(path)
            expected = shard.get("sha256")
            if actual != expected:
                raise ExactTextFeatureSourceMismatch(
                    f"{path}: sha256 {actual} does not match the manifest's "
                    f"{expected}; the shard changed after the build"
                )
            self._verified.add(shard_ordinal)
        rows = torch.load(path, weights_only=True)
        declared = shard["keys"]
        if len(rows) != len(declared):
            raise ExactTextFeatureSourceMismatch(
                f"{path}: {len(rows)} row(s), manifest declares "
                f"{len(declared)} key(s)"
            )
        for position, data in enumerate(rows):
            key = [data.get("document_bundle_id"), data.get("page_index")]
            if key != list(declared[position]):
                raise ExactTextFeatureSourceMismatch(
                    f"{path} row {position} is page {tuple(key)!r}, but the "
                    f"manifest's shard key list declares "
                    f"{tuple(declared[position])!r}"
                )
        self._loaded[shard_ordinal] = rows
        while len(self._loaded) > self.cache_shards:
            self._loaded.popitem(last=False)
        return rows

    def entry(self, document_bundle_id: str, page_index: int) -> FeatureOnlyEntry | None:
        location = self.page_index.get((document_bundle_id, page_index))
        if location is None:
            return None
        shard_ordinal, row_ordinal = location
        rows = self._shard_rows(shard_ordinal)
        return _checked_feature_entry(
            rows[row_ordinal],
            source_dir=self.cache_dir,
            document_bundle_id=document_bundle_id,
            page_index=page_index,
        )


def open_raw_feature_shard_reader(
    cache_dir: Path | str, *, gradient_bearing: bool
) -> RawFeatureShardReader:
    """Open one exact-text cache for raw feature/grid_thw reads.

    Refuses (never defaults): a foreign `cache_family`; a `cache_version`
    older than `_MIN_CACHE_VERSION` (no `grid_thw` guarantee) or
    `_MIN_SCOPE_VERSION` (no per-shard `keys` list); a manifest whose
    claimed/stored/indexed entry counts disagree; a missing/null identity
    field this module needs; and, when `gradient_bearing=True`, an
    `admission` other than `"strict"` -- `ExactTextCacheReader`'s own
    rules (design §8 rule 2), reapplied here because this reader does not
    go through that class at all (module docstring).
    """
    cache_dir = Path(cache_dir)
    manifest = _read_manifest(cache_dir)
    family = manifest.get("cache_family")
    if family != CACHE_FAMILY:
        raise ExactTextFeatureSourceMismatch(
            f"{cache_dir}: cache_family {family!r} != {CACHE_FAMILY!r}"
        )
    version = _parse_version(manifest.get("cache_version"), source_dir=cache_dir)
    if version < _MIN_CACHE_VERSION:
        raise ExactTextFeatureSourceMismatch(
            f"{cache_dir}: cache_version {manifest.get('cache_version')!r} "
            f"predates {'.'.join(map(str, _MIN_CACHE_VERSION))}, before "
            "grid_thw was guaranteed present on every row"
        )
    if version < _MIN_SCOPE_VERSION:
        raise ExactTextFeatureSourceMismatch(
            f"{cache_dir}: cache_version {manifest.get('cache_version')!r} "
            f"predates {'.'.join(map(str, _MIN_SCOPE_VERSION))}, before every "
            "shard carried its own page key list"
        )
    if is_page_image_identity(manifest.get("identity") or {}):
        raise ExactTextFeatureSourceMismatch(
            f"{cache_dir}: a page-image (DEC-056) cache holds no features to read"
        )
    if manifest.get("status") != "complete":
        raise ExactTextFeatureSourceMismatch(
            f"{cache_dir}: manifest status is not 'complete'; the build was "
            "interrupted"
        )
    claimed = manifest.get("entries")
    stored = manifest.get("shard_entries")
    shard_index = _checked_shard_index(manifest, source_dir=cache_dir)
    indexed = sum(int(s["count"]) for s in shard_index)
    if not claimed == stored == indexed:
        raise ExactTextFeatureSourceMismatch(
            f"{cache_dir}: manifest claims {claimed} entries but the shards "
            f"hold {stored} and the index sums to {indexed}"
        )
    identity = _checked_identity(manifest, source_dir=cache_dir)
    if gradient_bearing and identity["admission"] != "strict":
        raise ExactTextFeatureSourceMismatch(
            f"{cache_dir}: admission {identity['admission']!r} cannot back a "
            "gradient-bearing read, which requires 'strict'"
        )
    page_index: dict[PageKey, tuple[int, int]] = {}
    for shard_ordinal, shard in enumerate(shard_index):
        for row_ordinal, (bundle, page) in enumerate(shard["keys"]):
            key = (str(bundle), int(page))
            if key in page_index:
                raise ExactTextFeatureSourceMismatch(
                    f"{cache_dir}: page key {key!r} is declared by more than "
                    "one shard"
                )
            page_index[key] = (shard_ordinal, row_ordinal)
    return RawFeatureShardReader(
        cache_dir=cache_dir, manifest=manifest, identity=identity, page_index=page_index
    )


@dataclass(frozen=True, slots=True)
class ExactTextFeatureSource:
    """Random access to one or more exact-text caches' rows, by page, for
    their `features`/`grid_thw` alone."""

    readers: tuple[RawFeatureShardReader, ...]
    index: dict[PageKey, tuple[int, int]]

    def entry(self, document_bundle_id: str, page_index: int) -> FeatureOnlyEntry | None:
        """The page's features/grid_thw, or `None` if no source holds this
        page.

        `None` is not a fallback value a caller may treat as "no
        features": it means the page must be counted under a named
        refusal, never silently skipped without a count
        (:func:`missing_page_keys`).
        """
        location = self.index.get((document_bundle_id, page_index))
        if location is None:
            return None
        reader_ordinal, row_ordinal = location
        reader = self.readers[reader_ordinal]
        # `row_ordinal` re-resolved through the reader's own key -> shard
        # index rather than trusted verbatim: identical to `FeatureSource.
        # entry`'s own reasoning, and cheap since it is a dict lookup.
        return reader.entry(document_bundle_id, page_index)


def open_exact_text_feature_sources(
    primary: Path | str,
    supplements: Sequence[Path | str] = (),
    *,
    gradient_bearing: bool,
) -> ExactTextFeatureSource:
    """Open and pin the primary exact-text cache plus any supplement
    exact-text caches as a feature source, indexed by page.

    Each source (primary first, then `supplements` in order) is opened via
    :func:`open_raw_feature_shard_reader`. `gradient_bearing` has no
    default: reading features for a build that will itself be
    gradient-bearing must say so explicitly.

    Every supplement is then additionally required to match the primary on
    `CROSS_SOURCE_IDENTITY_FIELDS` -- a mismatch there would let two
    sources' rows disagree about what "the features of this page" means
    while both still individually open. A page key present in more than
    one source is refused outright: nothing here merges or prefers one
    source's row over another's.
    """
    dirs = [primary, *supplements]
    readers: list[RawFeatureShardReader] = []
    index: dict[PageKey, tuple[int, int]] = {}
    primary_identity_values: dict[str, object] = {}
    for reader_ordinal, source_dir in enumerate(dirs):
        reader = open_raw_feature_shard_reader(source_dir, gradient_bearing=gradient_bearing)
        if reader_ordinal == 0:
            primary_identity_values = {
                field: reader.identity[field] for field in CROSS_SOURCE_IDENTITY_FIELDS
            }
        else:
            for field in CROSS_SOURCE_IDENTITY_FIELDS:
                actual = reader.identity[field]
                expected = primary_identity_values[field]
                if actual != expected:
                    raise ExactTextFeatureSourceMismatch(
                        f"{source_dir}: identity.{field} is {actual!r}, the "
                        f"primary's is {expected!r} -- a supplement source "
                        "must match the primary's feature identity"
                    )
        overlap = set(reader.page_index) & set(index)
        if overlap:
            first_key = sorted(overlap)[0]
            raise ExactTextFeatureSourceMismatch(
                f"{source_dir}: {len(overlap)} page key(s) already present in "
                f"an earlier source, e.g. {first_key!r} -- a page's features "
                "must come from exactly one source"
            )
        for key, location in reader.page_index.items():
            index[key] = (reader_ordinal, location[1])
        # `index` records only the row ordinal; the reader ordinal doubles
        # as the shard-lookup owner via `ExactTextFeatureSource.entry`
        # delegating straight back to `reader.entry`, so a stale shard
        # ordinal captured here can never go stale.
        readers.append(reader)
    return ExactTextFeatureSource(readers=tuple(readers), index=index)


def missing_page_keys(
    source: ExactTextFeatureSource, keys: Iterable[PageKey]
) -> list[PageKey]:
    """Which of `keys` no source in `source.index` holds, sorted.

    A named, countable measurement -- the tool for answering "does the
    source cover exactly what this build needs" BEFORE a build is
    started, and the same predicate a build must use to name (never
    silently drop) a page it cannot write for lack of features.
    """
    return sorted(key for key in keys if key not in source.index)
