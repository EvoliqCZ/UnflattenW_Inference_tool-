"""Persisted frozen-backbone features for decoder training.

RFC005 stages 1–3 keep the backbone frozen, which means its output for a
given page never changes and does not need recomputing every epoch. That
is what makes local training practical: the decoder trains without the
backbone resident at all (~0.7 GB instead of ~2.5 GB), and an epoch costs
a decoder forward rather than a full vision-tower forward.

The cache is sharded because one page is ~11.6 MB at 1 M pixels: a single
file for 20 000 pages would be 232 GB that has to be opened whole, while
per-page files would be 20 000 inodes for a sequential read. Shards of a
few dozen pages sit between the two.

Features are stored as float16, halving the only resource this design
actually spends. Against the bf16 tower that produces them, **that costs
range, not precision**: bf16 has 7 mantissa bits and fp16 has 10, so the
mantissa is not the issue — the exponent is. fp16's largest finite value
is 65504 and values below ~6e-8 flush to zero. No feature-magnitude
statistic exists for this backbone yet, so `add` refuses a non-finite
post-cast tensor rather than storing an Inf, and `scan_magnitudes`
measures an existing cache so the question can be settled by data.

**A cache is progressively checkpointed.** Each full shard is written and
fsynced in staging, published, then recorded by an atomic manifest
checkpoint with `status: "in_progress"`.  The manifest remains the only
authority for committed content; a published shard not yet indexed by it is
an orphan.  A clean close writes the same manifest with `status: "complete"`.
`FeatureCacheReader` accepts only that final status, because an interrupted
build must never read as a smaller completed cache.

**A cache is bound to the lineage that produced it.** Nothing about a
feature vector says which model, pixel budget, split or release emitted
it, so reading one cache with another model's decoder would train on
noise that looks like data, and a test-split cache would leak silently.
`CacheIdentity` records that lineage in the manifest.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from unflatten.model.converter_hash import CONVERTER_NAME, converter_block

#: Cache format version; bumped when the shard layout or manifest changes.
#: 0.4.0 adds per-shard row keys, converter provenance, build options, and
#: the ``in_progress`` status value (which the reader continues to refuse).
#: 0.5.0 (DEC-024 wiring) adds `CacheIdentity.capability_profile_hash` --
#: additive to `identity` only, no top-level manifest shape change. See
#: `_IDENTITY_KEYS_BEFORE_CAPABILITY_PROFILE`/`_IDENTITY_KEYS_V050`: the
#: `identity` dict's own closed key set is now PINNED PER VERSION rather
#: than derived live from `asdict(CacheIdentity())`, so an existing 0.3.0/
#: 0.4.0 cache on disk keeps validating against the 16-key set it was
#: actually written with -- adding a field to the dataclass must never by
#: itself make an old manifest fail the exact-key-set check below.
CACHE_VERSION = "0.5.0"

#: On-disk feature dtype. See the module docstring on range vs precision.
STORAGE_DTYPE = torch.float16

#: How many shards a reader keeps resident. Shards are ~512 MB, so this
#: is the memory the reader spends to stop shuffled access from
#: re-reading the cache once per sample; it also bounds the window
#: `ShardWindowSampler` may shuffle over.
DEFAULT_CACHE_SHARDS = 8

#: Subdirectory holding shards of a build that has not been published.
#: Left in place on failure: it is the evidence of what was interrupted.
STAGING_DIR = ".staging"
_BUILD_LOCK = ".build.lock"
_SHARD_NAME = re.compile(r"^shard-\d{5,}\.pt$")


class CacheIntegrityError(RuntimeError):
    """A cache does not prove it is what its manifest claims."""


class CacheBuildLockedError(RuntimeError):
    """Another process currently owns this cache build session."""


def close_inherited_build_lock(out_dir: Path | str) -> int:
    """Close inherited descriptors for ``out_dir/.build.lock`` in this process.

    Worker initializers call this after fork so a dead writer's descendants do
    not retain its advisory lock.  Descriptors opened by this process for other
    purposes are untouched.
    """
    target = str((Path(out_dir) / _BUILD_LOCK).resolve())
    closed = 0
    for name in os.listdir("/proc/self/fd"):
        try:
            descriptor = int(name)
            if os.readlink(f"/proc/self/fd/{name}") == target:
                os.close(descriptor)
                closed += 1
        except (FileNotFoundError, OSError, ValueError):
            continue
    return closed


@dataclass(frozen=True)
class _ShardVerification:
    """Disk verification result kept distinct for future tail-repair policy."""

    name: str
    is_last: bool
    valid: bool


_HEX64 = set("0123456789abcdef")
_BUILD_OPTION_KEYS = {
    "single_page_only",
    "exclude_capability_profile",
    "max_steps",
    "workers",
    "chunk",
}
_V030_KEYS = {
    "cache_version",
    "status",
    "entries",
    "shard_entries",
    "visual_dim",
    "shards",
    "shard_size",
    "storage_dtype",
    "identity",
    "shard_index",
}
_V040_KEYS = _V030_KEYS | {"converter", "build_options", "build_options_initial"}
#: 0.5.0 adds no top-level manifest key -- only `identity` gains one, see
#: `_IDENTITY_KEYS_V050` below.
_V050_KEYS = _V040_KEYS

#: The two cohort-restriction pairs a manifest may carry -- never both
#: (`build_cache.py`'s `--only-bundles` and `--only-pages` are themselves
#: mutually exclusive, and a manifest mirrors that).
_ONLY_BUNDLES_KEYS = {"only_bundles_sha256", "only_bundles_count"}
_ONLY_PAGES_KEYS = {"only_pages_sha256", "only_pages_count"}

#: `identity`'s own closed key set, PINNED PER VERSION (not derived live
#: from `asdict(CacheIdentity())` the way this set used to be computed --
#: that made every existing on-disk manifest fail the moment a field was
#: added to the dataclass, since its identity dict would then be smaller
#: than what the live class demands). This is the set every 0.3.0/0.4.0
#: cache was actually written with.
_IDENTITY_KEYS_BEFORE_CAPABILITY_PROFILE = {
    "backbone",
    "backbone_revision",
    "revision_source",
    "processor",
    "feature_tap",
    "max_pixels",
    "split",
    "release_root_hash",
    "task_kind",
    "multiscale",
    "merged_grid",
    "token_map_version",
    "grammar_hash",
    "grammar_profile",
    "target_projector_version",
    "admission",
}
#: 0.5.0's `identity` key set: the 16 above plus `capability_profile_hash`
#: (DEC-024 wiring).
_IDENTITY_KEYS_V050 = _IDENTITY_KEYS_BEFORE_CAPABILITY_PROFILE | {"capability_profile_hash"}


def _is_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CacheIntegrityError(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def _validate_build_options(
    options: Any, *, error_type: type[Exception] = CacheIntegrityError
) -> None:
    def fail(message: str) -> None:
        raise error_type(message)

    if not isinstance(options, dict):
        fail(f"build_options must be a dict, got {options!r}")
    keys = set(options)
    if keys != _BUILD_OPTION_KEYS:
        fail(f"build_options keys must be exactly {sorted(_BUILD_OPTION_KEYS)}, got {sorted(keys)}")
    if not isinstance(options["single_page_only"], bool):
        fail("build_options.single_page_only must be bool")
    if not isinstance(options["exclude_capability_profile"], bool):
        fail("build_options.exclude_capability_profile must be bool")
    for key, minimum in (("max_steps", 1), ("workers", 0), ("chunk", 1)):
        if not _is_int(options[key], minimum=minimum):
            fail(f"build_options.{key} must be an int >= {minimum}, got {options[key]!r}")


def _validate_converter(
    converter: Any, *, error_type: type[Exception] = CacheIntegrityError
) -> None:
    def fail(message: str) -> None:
        raise error_type(message)

    if not isinstance(converter, dict) or set(converter) != {"name", "version", "source_sha256"}:
        fail("converter must have exactly name, version, and source_sha256")
    if converter["name"] != CONVERTER_NAME:
        fail(f"converter.name must be {CONVERTER_NAME!r}, got {converter['name']!r}")
    if not isinstance(converter["version"], str) or not converter["version"]:
        fail(f"converter.version must be a nonempty string, got {converter['version']!r}")
    if not _is_sha256(converter["source_sha256"]):
        source_sha256 = converter["source_sha256"]
        fail(f"converter.source_sha256 must be 64 lowercase hex characters, got {source_sha256!r}")


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the versioned on-disk cache manifest before it is trusted."""
    _require(isinstance(manifest, dict), f"manifest must be a dict, got {manifest!r}")
    version = manifest.get("cache_version")
    _require(
        version in {"0.3.0", "0.4.0", "0.5.0"},
        f"cache_version must be '0.3.0', '0.4.0' or '0.5.0', got {version!r}",
    )
    # The per-shard row index, converter provenance and build-options block
    # all arrived together in 0.4.0 (module docstring) and are unaffected
    # by 0.5.0's identity-only addition -- both versions carry them. Named
    # for the FEATURE, not restated per version, so a future version that
    # keeps this shape does not need a fourth copy of `version == "0.4.0"`.
    has_row_index = version in {"0.4.0", "0.5.0"}
    if version == "0.5.0":
        reserved = _V050_KEYS
    elif version == "0.4.0":
        reserved = _V040_KEYS
    else:
        reserved = _V030_KEYS
    for key in manifest:
        allowed_extension = version in {"0.4.0", "0.5.0"} and key.startswith("extensions.")
        _require(
            key in reserved
            or key in _ONLY_BUNDLES_KEYS
            or key in _ONLY_PAGES_KEYS
            or allowed_extension,
            f"manifest has unknown top-level key {key!r} for cache_version {version}",
        )
    required = reserved
    missing = sorted(required - set(manifest))
    if missing:
        raise CacheIntegrityError(f"manifest missing required top-level key {missing[0]!r}")
    only_bundles_keys = {key for key in _ONLY_BUNDLES_KEYS if key in manifest}
    only_pages_keys = {key for key in _ONLY_PAGES_KEYS if key in manifest}
    _require(
        not only_bundles_keys or only_bundles_keys == _ONLY_BUNDLES_KEYS,
        "only_bundles_sha256 and only_bundles_count must be present together",
    )
    _require(
        not only_pages_keys or only_pages_keys == _ONLY_PAGES_KEYS,
        "only_pages_sha256 and only_pages_count must be present together",
    )
    _require(
        not (only_bundles_keys and only_pages_keys),
        "only_bundles_* and only_pages_* must never both be present",
    )
    only_keys = only_bundles_keys
    _require(
        manifest["status"] in {"complete", "in_progress"},
        f"status must be 'complete' or 'in_progress', got {manifest['status']!r}",
    )
    for key in ("entries", "shard_entries", "shards"):
        _require(_is_int(manifest[key]), f"{key} must be an int >= 0, got {manifest[key]!r}")
    _require(
        _is_int(manifest["shard_size"], minimum=1),
        f"shard_size must be an int >= 1, got {manifest['shard_size']!r}",
    )
    visual_dim = manifest["visual_dim"]
    _require(
        visual_dim is None or _is_int(visual_dim, minimum=1),
        f"visual_dim must be an int >= 1 or null, got {visual_dim!r}",
    )
    _require(
        manifest["storage_dtype"] == "float16",
        f"storage_dtype must be 'float16', got {manifest['storage_dtype']!r}",
    )
    identity = manifest["identity"]
    # Pinned per version (see `_IDENTITY_KEYS_V050`'s docstring), never
    # derived live from `asdict(CacheIdentity())`: that would make an
    # existing 0.3.0/0.4.0 manifest fail the instant a field is added to
    # the dataclass, since its stored identity would then be smaller than
    # what the live class demands.
    identity_keys = (
        _IDENTITY_KEYS_V050 if version == "0.5.0" else _IDENTITY_KEYS_BEFORE_CAPABILITY_PROFILE
    )
    identity_value = sorted(identity) if isinstance(identity, dict) else identity
    _require(
        isinstance(identity, dict) and set(identity) == identity_keys,
        f"identity must have exactly {len(identity_keys)} CacheIdentity keys, "
        f"got {identity_value!r}",
    )
    if only_keys:
        _require(
            _is_sha256(manifest["only_bundles_sha256"]),
            "only_bundles_sha256 must be 64 lowercase hex characters",
        )
        _require(
            _is_int(manifest["only_bundles_count"], minimum=1),
            f"only_bundles_count must be an int >= 1, got {manifest['only_bundles_count']!r}",
        )
    if only_pages_keys:
        _require(
            _is_sha256(manifest["only_pages_sha256"]),
            "only_pages_sha256 must be 64 lowercase hex characters",
        )
        _require(
            _is_int(manifest["only_pages_count"], minimum=1),
            f"only_pages_count must be an int >= 1, got {manifest['only_pages_count']!r}",
        )
    if has_row_index:
        _validate_converter(manifest["converter"])
        _validate_build_options(manifest["build_options"])
        _validate_build_options(manifest["build_options_initial"])

    index = manifest["shard_index"]
    _require(isinstance(index, list), f"shard_index must be a list, got {index!r}")
    all_rows: list[tuple[str, int]] = []
    counts: list[int] = []
    for position, record in enumerate(index):
        _require(
            isinstance(record, dict), f"shard_index[{position}] must be a dict, got {record!r}"
        )
        expected_keys = {"name", "count", "sha256"} | ({"rows"} if has_row_index else set())
        actual_keys = sorted(record)
        _require(
            set(record) == expected_keys,
            f"shard_index[{position}] keys must be exactly {sorted(expected_keys)}, "
            f"got {actual_keys}",
        )
        name = record["name"]
        expected_name = f"shard-{position:05d}.pt"
        _require(
            name == expected_name,
            f"shard_index[{position}].name must be {expected_name!r}, got {name!r}",
        )
        count = record["count"]
        _require(
            _is_int(count, minimum=1) and count <= manifest["shard_size"],
            f"shard_index[{position}].count must be in [1, shard_size], got {count!r}",
        )
        _require(
            _is_sha256(record["sha256"]),
            f"shard_index[{position}].sha256 must be 64 lowercase hex characters; "
            f"got {record['sha256']!r}",
        )
        counts.append(count)
        if position < len(index) - 1:
            _require(
                count == manifest["shard_size"],
                f"non-last shard {name} must have count == shard_size "
                f"({manifest['shard_size']}), got {count}",
            )
        if has_row_index:
            rows = record["rows"]
            _require(
                isinstance(rows, list), f"shard_index[{position}].rows must be a list, got {rows!r}"
            )
            parsed: list[tuple[str, int]] = []
            for row in rows:
                _require(
                    isinstance(row, list)
                    and len(row) == 2
                    and isinstance(row[0], str)
                    and bool(row[0])
                    and _is_int(row[1]),
                    f"shard_index[{position}].rows contains invalid row {row!r}",
                )
                parsed.append((row[0], row[1]))
            _require(
                parsed == sorted(parsed),
                f"shard_index[{position}].rows must be sorted by (bundle_id, page_index)",
            )
            _require(
                len(set(parsed)) == len(parsed),
                f"shard_index[{position}].rows has duplicate row key",
            )
            _require(
                len(parsed) == count,
                f"shard_index[{position}].rows has {len(parsed)} rows but count is {count}",
            )
            all_rows.extend(parsed)
    count_message = (
        f"entries ({manifest['entries']}), shard_entries ({manifest['shard_entries']}), "
        f"and shard counts ({sum(counts)}) must agree"
    )
    _require(
        manifest["entries"] == manifest["shard_entries"] == sum(counts),
        count_message,
    )
    _require(
        manifest["shards"] == len(index),
        f"shards ({manifest['shards']}) must equal len(shard_index) ({len(index)})",
    )
    if has_row_index:
        _require(
            sum(counts) == len(all_rows),
            f"shard counts ({sum(counts)}) must equal total rows ({len(all_rows)})",
        )
        _require(len(set(all_rows)) == len(all_rows), "shard_index rows must be globally unique")


@dataclass(frozen=True)
class CacheIdentity:
    """What a cache's features are bound to.

    `backbone_revision` is best-effort: no loader in this repository
    passes `revision=`, so a model ID does not pin weights.
    `revision_source` says where the recorded revision came from —
    `"unavailable"` means the cache is pinned to a *name*, not to bytes,
    and callers must not read it as a guarantee.
    """

    backbone: str | None = None
    backbone_revision: str | None = None
    revision_source: str = "unavailable"
    #: Processor / feature-tap identity (config digest, not a name).
    processor: str | None = None
    #: The named feature tap, for a backbone that exposes more than one
    #: (Granite Vision). Two taps of one checkpoint are two different
    #: feature sets under the same model id and processor config, and this
    #: is the only field that says which. `visual_dim` in the manifest
    #: proper does separate Granite's two taps today, but by width rather
    #: than by lineage — two taps of equal width would read as the same
    #: cache — so it is not a substitute. Enforced by
    #: `train.SHARED_LINEAGE_FIELDS`.
    feature_tap: str | None = None
    max_pixels: int | None = None
    split: str | None = None
    release_root_hash: str | None = None
    task_kind: str | None = None
    multiscale: bool = False
    merged_grid: bool = False
    #: `events_vocab.TOKEN_MAP_VERSION` — the event ALPHABET's version.
    #: Named for what it holds: it says nothing about the controller.
    token_map_version: str | None = None
    #: `grammar.controller_hash()` — source identity of the §18.4
    #: pushdown controller, derived at run time so a phase-logic or
    #: parent-child change moves it without anyone bumping a constant.
    grammar_hash: str | None = None
    #: RFC004 §18.4 capability profile the grammar controller runs under
    #: -- `word-core-v0`, `word-visual-v1` or `word-visual-v1.1`, whichever
    #: this build actually used (not a fixed example: every build stamps
    #: its own). A profile, not a version — named for what it is, because a
    #: field whose name claims more than it holds is compared as though the
    #: claim were true.
    grammar_profile: str | None = None
    #: `capability_profile.CapabilityProfile.profile_hash` this build was
    #: bound to (DEC-024 wiring) -- `None` for a cache built by the
    #: historical S1 feature-build reproduction (no `--capability-profile`
    #: given; grammar stays the literal `"word-visual-v1"` above, exclusion
    #: stays `PRIMARY_S1_CAPABILITY_PROFILE`'s). A cache whose exclusion set
    #: and grammar differ by profile cannot be told apart from one that
    #: doesn't without this field; added in `CACHE_VERSION` 0.5.0, whose
    #: `identity` key set is pinned separately from 0.3.0/0.4.0's (see
    #: `_IDENTITY_KEYS_V050`) so an existing cache's smaller identity keeps
    #: validating against the set it was actually written with.
    capability_profile_hash: str | None = None
    #: `dataset.targets.bundle.PROJECTOR_VERSION`: the §27 projector's
    #: derivation version, which does move when the target materializer
    #: changes.
    target_projector_version: str | None = None
    #: Weakest admission this cache accepted: `"strict"` (RFC005 §9.2
    #: training admission) or `"dev_only"` (dev/research inventory, an
    #: explicit opt-in). A dev-only cache must never be mistaken for a
    #: training one later, so it says which it is.
    #:
    #: NOT the `admission_tier` of RFC003-ERRATA-001 E5 — that design was
    #: implemented and reverted, and no gate enum grew a value. This
    #: records a fact about the BUILD (which flag the operator passed),
    #: not a run's purpose; RFC005 §9.3 `research | smoke |
    #: coverage-building` is the vocabulary for the latter, and using it
    #: here would state an intent nobody verified.
    admission: str = "strict"


@dataclass
class CacheEntry:
    """One page: its visual features and the decoder targets for it.

    `grid_thw` is `None` for every backbone track that predates it (the
    probe/decoder-training tracks this cache was originally built for never
    needed it) and a `(3,)` long tensor `(temporal, height, width)` for the
    candidate track, which does: `candidate_loader.QwenCandidate.inject_visual`
    needs it, unchanged, as its native mRoPE positional index (§5.1's
    `BackboneAdapter` seam) -- a cache that stored the feature tensor and
    dropped the grid would make native injection impossible downstream, not
    merely inconvenient. `CacheEntry(**row)` reading an OLDER shard that
    never wrote this key still works: the default applies.
    """

    document_bundle_id: str
    page_index: int
    features: torch.Tensor
    target_ids: torch.Tensor
    input_ids: torch.Tensor
    allowed: torch.Tensor
    grid_thw: torch.Tensor | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    """Durably flush a regular file before it becomes authoritative."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    """Durably flush a directory entry created by ``os.replace``."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class FeatureCacheWriter:
    """Write bounded shards and atomically checkpoint each published shard."""

    def __init__(
        self,
        out_dir: Path | str,
        *,
        shard_size: int = 32,
        identity: CacheIdentity | None = None,
        converter: dict[str, Any] | None = None,
        build_options: dict[str, Any] | None = None,
        only_bundles_fields: dict[str, Any] | None = None,
        extra_manifest: dict[str, Any] | None = None,
        on_checkpoint: Callable[[str, int, int], None] | None = None,
    ) -> None:
        if not _is_int(shard_size, minimum=1):
            raise ValueError(f"shard_size must be an int >= 1, got {shard_size!r}")
        if converter is None:
            converter = converter_block(Path(__file__).resolve().parents[3])
        _validate_converter(converter, error_type=ValueError)
        if build_options is None:
            build_options = {
                "single_page_only": False,
                "exclude_capability_profile": False,
                "max_steps": 10_000,
                "workers": 0,
                "chunk": 8,
            }
        _validate_build_options(build_options, error_type=ValueError)
        only_bundles_fields = dict(only_bundles_fields or {})
        only_keys = set(only_bundles_fields)
        if not only_keys <= (_ONLY_BUNDLES_KEYS | _ONLY_PAGES_KEYS):
            unexpected = sorted(only_keys - (_ONLY_BUNDLES_KEYS | _ONLY_PAGES_KEYS))[0]
            raise ValueError(f"only_bundles_fields has unknown key {unexpected!r}")
        only_bundles_keys = only_keys & _ONLY_BUNDLES_KEYS
        only_pages_keys = only_keys & _ONLY_PAGES_KEYS
        if only_bundles_keys and only_pages_keys:
            raise ValueError("only_bundles_* and only_pages_* must never both be present")
        if only_bundles_keys and only_bundles_keys != _ONLY_BUNDLES_KEYS:
            raise ValueError("only_bundles_sha256 and only_bundles_count must be present together")
        if only_pages_keys and only_pages_keys != _ONLY_PAGES_KEYS:
            raise ValueError("only_pages_sha256 and only_pages_count must be present together")
        if only_bundles_keys:
            if not _is_sha256(only_bundles_fields["only_bundles_sha256"]):
                raise ValueError("only_bundles_sha256 must be 64 lowercase hex characters")
            if not _is_int(only_bundles_fields["only_bundles_count"], minimum=1):
                raise ValueError("only_bundles_count must be an int >= 1")
        if only_pages_keys:
            if not _is_sha256(only_bundles_fields["only_pages_sha256"]):
                raise ValueError("only_pages_sha256 must be 64 lowercase hex characters")
            if not _is_int(only_bundles_fields["only_pages_count"], minimum=1):
                raise ValueError("only_pages_count must be an int >= 1")
        extra_manifest = dict(extra_manifest or {})
        for key in extra_manifest:
            if not key.startswith("extensions."):
                raise ValueError(f"extra_manifest key {key!r} must start with 'extensions.'")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._lock_fd: int | None = None
        self.cleanup_report: list[dict[str, str]] = []
        self._acquire_lock()
        self.staging = self.out_dir / STAGING_DIR
        self.shard_size = shard_size
        self.identity = identity or CacheIdentity()
        self.converter = dict(converter)
        self.build_options = dict(build_options)
        self.build_options_initial = dict(self.build_options)
        self.only_bundles_fields = only_bundles_fields
        self.extra_manifest = extra_manifest
        #: Entries handed to `add`; the manifest compares it against the
        #: rows that actually reached a shard.
        self.entries = 0
        self.visual_dim: int | None = None
        self._buffer: list[dict[str, Any]] = []
        self._shards: list[dict[str, Any]] = []
        self._committed_keys: set[tuple[str, int]] = set()
        self._committed_keys_snapshot: frozenset[tuple[str, int]] | None = None
        self._buffer_keys: set[tuple[str, int]] = set()
        self._tail_rows: list[dict[str, Any]] | None = None
        self._tail_name: str | None = None
        self._repair_needed = False
        self._repair_rows: list[tuple[str, int]] | None = None
        self._tail_shard_recomputed = False
        self._tail_rows_recomputed = 0
        self._on_checkpoint = on_checkpoint
        self._closed = False
        self._resumed = False

        try:
            manifest_path = self.out_dir / "manifest.json"
            if manifest_path.is_symlink():
                raise CacheIntegrityError(
                    f"cache integrity failure: conflicting symlink {manifest_path}"
                )
            if manifest_path.exists():
                self._resume(manifest_path)
            elif any(self.out_dir.glob("shard-*.pt")):
                raise CacheIntegrityError(
                    "cache integrity failure: published shards without manifest under "
                    f"{self.out_dir}; "
                    "refusing to guess identity; rebuild or restore the manifest"
                )
            self._cleanup_interrupted_artifacts()
            self.staging.mkdir(parents=True, exist_ok=True)
            if not manifest_path.exists():
                self._checkpoint("in_progress")
        except BaseException:
            self._release_lock()
            raise

    def _acquire_lock(self) -> None:
        """Acquire the one non-blocking kernel lock for this writer session."""
        path = self.out_dir / _BUILD_LOCK
        if path.is_symlink():
            raise CacheIntegrityError(f"cache integrity failure: conflicting symlink {path}")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            if exc.errno == getattr(os, "ELOOP", 40):
                raise CacheIntegrityError(
                    f"cache integrity failure: conflicting symlink {path}"
                ) from exc
            raise
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise CacheBuildLockedError(
                f"cache build locked: {self.out_dir}; another writer holds .build.lock"
            ) from exc
        except BaseException:
            os.close(descriptor)
            raise
        self._lock_fd = descriptor

    def _release_lock(self) -> None:
        """Release and close the session descriptor without unlinking its path."""
        if getattr(self, "_lock_fd", None) is None:
            return
        descriptor, self._lock_fd = self._lock_fd, None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __del__(self) -> None:
        """Best-effort release for abandoned writers in a normal interpreter."""
        self._release_lock()

    def _report_cleanup(self, action: str, path: Path, reason: str) -> None:
        relative = str(path.relative_to(self.out_dir))
        self.cleanup_report.append({"action": action, "path": relative, "reason": reason})
        if action == "kept":
            print(f"[cache] WARNING: preserving unknown cache entry {relative}", file=sys.stderr)

    def _cleanup_interrupted_artifacts(self) -> None:
        """Delete only known uncommitted artifacts after manifest validation."""
        self._validate_cleanup_paths()

        if self.staging.exists():
            for path in sorted(self.staging.iterdir()):
                if _SHARD_NAME.fullmatch(path.name) and path.is_file():
                    path.unlink()
                    self._report_cleanup("deleted", path, "staging shard")
                elif not _SHARD_NAME.fullmatch(path.name):
                    self._report_cleanup("kept", path, "unknown staging entry")
            if not any(self.staging.iterdir()):
                self.staging.rmdir()

        indexed = {record["name"] for record in self._shards}
        for path in sorted(self.out_dir.iterdir()):
            if path.name in {"manifest.json", STAGING_DIR, _BUILD_LOCK}:
                continue
            if path.name == "manifest.json.tmp" and path.is_file():
                path.unlink()
                self._report_cleanup("deleted", path, "manifest temporary")
            elif _SHARD_NAME.fullmatch(path.name) and path.name not in indexed and path.is_file():
                path.unlink()
                self._report_cleanup("deleted", path, "unindexed shard")
            elif path.name != "manifest.json.tmp" and not _SHARD_NAME.fullmatch(path.name):
                self._report_cleanup("kept", path, "unknown cache entry")

    def _validate_cleanup_paths(self) -> None:
        """Refuse every symlink before cleanup can mutate the cache directory."""
        conflicts: list[Path] = []
        if self.staging.is_symlink():
            conflicts.append(self.staging)
        elif self.staging.exists():
            for path in sorted(self.staging.iterdir()):
                if path.is_symlink():
                    conflicts.append(path)

        for path in sorted(self.out_dir.iterdir()):
            if path != self.staging and path.is_symlink():
                conflicts.append(path)
        if conflicts:
            raise CacheIntegrityError(
                f"cache integrity failure: conflicting symlink {conflicts[0]}"
            )

    @property
    def committed_keys(self) -> frozenset[tuple[str, int]]:
        """Cached authoritative-key snapshot, refreshed only after a shard commit."""
        if self._committed_keys_snapshot is None:
            self._committed_keys_snapshot = frozenset(self._committed_keys)
        return self._committed_keys_snapshot

    @property
    def committed_entries(self) -> int:
        """Number of manifest-authoritative entries, excluding the RAM buffer."""
        return sum(int(shard["count"]) for shard in self._shards)

    @property
    def repair_rows(self) -> list[tuple[str, int]] | None:
        """Manifest-authoritative tail keys that must be restored, in order."""
        return None if self._repair_rows is None else list(self._repair_rows)

    @property
    def tail_shard_recomputed(self) -> bool:
        """Whether this writer session durably restored a mismatched tail."""
        return self._tail_shard_recomputed

    @property
    def tail_rows_recomputed(self) -> int:
        """Rows durably recomputed for a repaired tail in this session."""
        return self._tail_rows_recomputed

    def _resume(self, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text())
        # Validation intentionally precedes all resume-specific comparisons.
        validate_manifest(manifest)
        if manifest["cache_version"] != CACHE_VERSION:
            raise CacheIntegrityError(
                "legacy manifest without row index; rebuild or run an explicit migration"
            )
        if manifest["shard_size"] != self.shard_size:
            raise CacheIntegrityError(
                "resume refused: shard_size mismatch: "
                f"stored={manifest['shard_size']}, requested={self.shard_size}"
            )
        stored_converter = manifest["converter"]["source_sha256"]
        current_converter = self.converter["source_sha256"]
        if stored_converter != current_converter:
            raise CacheIntegrityError(
                "resume refused: converter source mismatch: "
                f"stored={stored_converter[:12]}, current={current_converter[:12]}; "
                "the code that wrote this cache differs; use a new output directory"
            )
        stored_identity = manifest["identity"]
        requested_identity = asdict(self.identity)
        for field in asdict(CacheIdentity()):
            if stored_identity[field] != requested_identity[field]:
                raise CacheIntegrityError(
                    f"resume refused: CacheIdentity mismatch: {field}: "
                    f"stored={stored_identity[field]!r}, requested={requested_identity[field]!r}; "
                    "use a new output directory"
                )
        stored_only = {
            key: manifest[key]
            for key in (_ONLY_BUNDLES_KEYS | _ONLY_PAGES_KEYS)
            if key in manifest
        }
        if stored_only != self.only_bundles_fields:
            requested_only = self.only_bundles_fields
            families = []
            if _ONLY_BUNDLES_KEYS & (stored_only.keys() | requested_only.keys()):
                families.append("only_bundles")
            if _ONLY_PAGES_KEYS & (stored_only.keys() | requested_only.keys()):
                families.append("only_pages")
            selection_name = "/".join(families) if families else "only_bundles"
            raise CacheIntegrityError(
                f"resume refused: {selection_name} selection mismatch: "
                f"stored={stored_only or None!r}, requested={requested_only or None!r}; "
                "use a new output directory"
            )
        results = [
            self._verify_committed_shard(record, i == len(manifest["shard_index"]) - 1)
            for i, record in enumerate(manifest["shard_index"])
        ]
        failed = next((result for result in results if not result.valid), None)
        if failed is not None:
            if not failed.is_last:
                raise CacheIntegrityError(
                    f"cache integrity failure: non-last shard {failed.name} does not match "
                    "manifest; "
                    "possible corruption; preserve directory and investigate manually"
                )
        self._shards = [dict(record) for record in manifest["shard_index"]]
        self._committed_keys = {
            (row[0], row[1]) for record in self._shards for row in record["rows"]
        }
        self.entries = self.committed_entries
        self.visual_dim = manifest["visual_dim"]
        # Keep this run's requested settings for the next checkpoint while
        # retaining the first run as the immutable drift-audit anchor.
        self.converter = dict(manifest["converter"])
        self.build_options_initial = dict(manifest["build_options_initial"])
        self.extra_manifest = {
            key: value for key, value in manifest.items() if key.startswith("extensions.")
        }
        if failed is not None:
            self._repair_needed = True
            self._repair_rows = [tuple(row) for row in self._shards[-1]["rows"]]
        elif self._shards and self._shards[-1]["count"] < self.shard_size:
            self._tail_name = self._shards[-1]["name"]
            loaded_tail = torch.load(self.out_dir / self._tail_name, weights_only=True)
            self._tail_rows = [self._copy_row(row) for row in loaded_tail]
        self._resumed = True

    def _verify_committed_shard(self, record: dict[str, Any], is_last: bool) -> _ShardVerification:
        path = self.out_dir / record["name"]
        try:
            if not path.is_file() or _sha256(path) != record["sha256"]:
                return _ShardVerification(record["name"], is_last, False)
            rows = torch.load(path, weights_only=True)
            keys = sorted((row["document_bundle_id"], row["page_index"]) for row in rows)
            valid = len(rows) == record["count"] and keys == [tuple(row) for row in record["rows"]]
            return _ShardVerification(record["name"], is_last, valid)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            return _ShardVerification(record["name"], is_last, False)

    def add(self, entry: CacheEntry) -> None:
        """Append one page to the cache."""
        self._ensure_open()
        if self._repair_needed:
            raise ValueError("tail repair required before add; call restore_tail() first")
        key = (entry.document_bundle_id, entry.page_index)
        if key in self._committed_keys or key in self._buffer_keys:
            raise ValueError(f"duplicate cache row key {key!r}")
        if self._resumed and not self._buffer and self._shards:
            # A complete cache is made writable only when new work actually
            # arrives; a no-op open/close must preserve its manifest bytes.
            if cache_manifest(self.out_dir)["status"] == "complete":
                self._checkpoint("in_progress")
            self._resumed = False
        row = self._normalize_entry(entry)
        dim = int(row["features"].shape[-1])
        if self.visual_dim is None:
            self.visual_dim = dim
        elif dim != self.visual_dim:
            raise ValueError(
                f"visual_dim {dim} does not match the cache's {self.visual_dim}; "
                "one cache holds one backbone's features"
            )
        self._buffer.append(row)
        self._buffer_keys.add(key)
        self.entries += 1
        if len(self._buffer) >= self._buffer_capacity():
            self._flush()

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("cannot write to a closed FeatureCacheWriter")

    def _normalize_entry(self, entry: CacheEntry) -> dict[str, Any]:
        features = entry.features.detach().to(STORAGE_DTYPE).cpu()
        if not bool(torch.isfinite(features).all()):
            raise ValueError(
                f"{entry.document_bundle_id} page {entry.page_index}: features are not finite "
                f"after the cast to {STORAGE_DTYPE} (fp16 holds at most 65504); storing them "
                "would cache an Inf as data"
            )
        grid_thw = entry.grid_thw.detach().cpu().clone() if entry.grid_thw is not None else None
        if grid_thw is not None and grid_thw.shape != (3,):
            raise ValueError(
                f"{entry.document_bundle_id} page {entry.page_index}: grid_thw must be a (3,) "
                f"(temporal, height, width) tensor, got shape {tuple(grid_thw.shape)}"
            )
        return {
            "document_bundle_id": entry.document_bundle_id,
            "page_index": entry.page_index,
            "features": features.clone(),
            "target_ids": entry.target_ids.detach().cpu().clone(),
            "input_ids": entry.input_ids.detach().cpu().clone(),
            "allowed": entry.allowed.detach().cpu().clone(),
            "grid_thw": grid_thw,
        }

    def _copy_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Give the serializer fresh, non-aliased tensor storage deterministically."""
        return {
            key: row[key].clone() if isinstance(row[key], torch.Tensor) else row[key]
            for key in (
                "document_bundle_id",
                "page_index",
                "features",
                "target_ids",
                "input_ids",
                "allowed",
                "grid_thw",
            )
        }

    def _buffer_capacity(self) -> int:
        if self._tail_rows is None:
            return self.shard_size
        return self.shard_size - len(self._tail_rows)

    def _flush(self) -> None:
        if not self._buffer:
            return
        tail_mode = self._tail_rows is not None
        rows = self._tail_rows + self._buffer if tail_mode else self._buffer
        name = self._tail_name if tail_mode else f"shard-{len(self._shards):05d}.pt"
        assert name is not None
        path = self.staging / name
        torch.save(rows, path)
        _fsync_file(path)
        shard = {
            "name": name,
            "count": len(rows),
            "sha256": _sha256(path),
            "rows": sorted([row["document_bundle_id"], row["page_index"]] for row in rows),
        }
        os.replace(path, self.out_dir / name)
        _fsync_dir(self.out_dir)
        # Retain this record before checkpointing: if that atomic replace
        # fails, the next checkpoint in this process re-emits the shard.
        if tail_mode:
            self._shards[-1] = shard
        else:
            self._shards.append(shard)
        self._committed_keys.update(self._buffer_keys)
        self._committed_keys_snapshot = None
        self._buffer = []
        self._buffer_keys = set()
        self._tail_rows = None
        self._tail_name = None
        self._checkpoint("in_progress")

    def restore_tail(self, entries_in_manifest_order: list[CacheEntry]) -> None:
        """Durably restore exactly the manifest-authoritative mismatched tail."""
        self._ensure_open()
        if not self._repair_needed or self._repair_rows is None:
            raise ValueError("tail repair is not pending")
        keys = [(entry.document_bundle_id, entry.page_index) for entry in entries_in_manifest_order]
        if keys != self._repair_rows:
            raise ValueError(
                "tail repair entries must exactly match manifest rows in manifest order"
            )
        rows = [self._normalize_entry(entry) for entry in entries_in_manifest_order]
        if any(int(row["features"].shape[-1]) != self.visual_dim for row in rows):
            raise ValueError("tail repair visual_dim does not match the cache")
        name = self._shards[-1]["name"]
        path = self.staging / name
        torch.save(rows, path)
        _fsync_file(path)
        os.replace(path, self.out_dir / name)
        _fsync_dir(self.out_dir)
        self._shards[-1] = {
            "name": name,
            "count": len(rows),
            "sha256": _sha256(self.out_dir / name),
            "rows": [[bundle_id, page_index] for bundle_id, page_index in self._repair_rows],
        }
        self._checkpoint("in_progress")
        self._repair_needed = False
        self._repair_rows = None
        self._tail_shard_recomputed = True
        self._tail_rows_recomputed = len(rows)
        if len(rows) < self.shard_size:
            self._tail_rows = rows
            self._tail_name = name

    def _manifest(self, *, status: str) -> dict[str, Any]:
        committed_entries = sum(shard["count"] for shard in self._shards)
        return {
            "cache_version": CACHE_VERSION,
            "status": status,
            "entries": committed_entries,
            "shard_entries": committed_entries,
            "visual_dim": self.visual_dim,
            "shards": len(self._shards),
            "shard_size": self.shard_size,
            "shard_index": self._shards,
            "storage_dtype": str(STORAGE_DTYPE).removeprefix("torch."),
            "identity": asdict(self.identity),
            "converter": self.converter,
            "build_options": self.build_options,
            "build_options_initial": self.build_options_initial,
            **self.only_bundles_fields,
            **self.extra_manifest,
        }

    def _write_manifest_tmp(self, manifest: dict[str, Any]) -> Path:
        temporary = self.out_dir / "manifest.json.tmp"
        with temporary.open("w") as handle:
            handle.write(json.dumps(manifest, indent=1, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        return temporary

    def _checkpoint(self, status: str) -> None:
        manifest = self._manifest(status=status)
        validate_manifest(manifest)
        temporary = self._write_manifest_tmp(manifest)
        os.replace(temporary, self.out_dir / "manifest.json")
        _fsync_dir(self.out_dir)
        if self._on_checkpoint is not None:
            self._on_checkpoint(status, manifest["entries"], manifest["shards"])

    def close(self) -> None:
        """Flush and complete the cache; retries after any close attempt are no-ops."""
        if self._closed:
            return
        try:
            if self._repair_needed:
                raise CacheIntegrityError("tail repair required before close; call restore_tail()")
            self._flush()
            self._checkpoint("complete")
            if self.staging.exists() and not any(self.staging.iterdir()):
                self.staging.rmdir()
        finally:
            self._closed = True
            self._release_lock()

    def __enter__(self) -> FeatureCacheWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        # An exception in flight means the build did not finish. Publishing
        # anything here is how a partial cache used to read as a small one.
        if exc_type is None:
            self.close()
        else:
            self._closed = True
            self._release_lock()


def cache_manifest(cache_dir: Path | str) -> dict[str, Any]:
    """Read a cache's manifest."""
    path = Path(cache_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no manifest under {cache_dir}")
    return json.loads(path.read_text())


def scan_magnitudes(cache_dir: Path | str) -> list[dict[str, Any]]:
    """Max absolute feature magnitude per shard, as a diagnostic.

    Deliberately reads the shard files directly: it must work on a v0.1
    cache and on the staging directory of an interrupted build, which is
    exactly where the question of feature range needs answering.
    """
    report = []
    for path in sorted(Path(cache_dir).glob("shard-*.pt")):
        rows = torch.load(path, weights_only=True)
        peak = max((float(row["features"].abs().max()) for row in rows), default=0.0)
        report.append({"shard": path.name, "rows": len(rows), "max_abs": peak})
    return report


class FeatureCacheReader:
    """Random access over a sharded feature cache.

    Every shard is hashed **once per reader**, before its first read, and
    the last `cache_shards` shards stay resident. Hashing on every access
    instead was correct for a one-shot read and pathological inside a
    training loop: it put a second 512 MB pass on top of every 512 MB
    load, and shuffled access made that per sample. What the check still
    proves is what it always proved of a run — that the bytes this reader
    trained on hashed to what the manifest claims; what it no longer
    detects is a shard rewritten *during* a run, which is a case no
    caller has and which the manifest cannot describe anyway.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        expect_backbone: str | None = None,
        cache_shards: int = DEFAULT_CACHE_SHARDS,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = cache_manifest(self.cache_dir)
        validate_manifest(self.manifest)
        self._check_complete()
        self.identity = self.manifest.get("identity", {})
        backbone = self.identity.get("backbone")
        if expect_backbone is not None and backbone != expect_backbone:
            raise ValueError(
                f"cache was built with backbone {backbone!r}, "
                f"not {expect_backbone!r}; features are not interchangeable"
            )
        if cache_shards < 1:
            raise ValueError(f"cache_shards must be positive, got {cache_shards}")
        self.shard_size = self.manifest["shard_size"]
        self.cache_shards = cache_shards
        self._index = {s["name"]: s for s in self.manifest["shard_index"]}
        #: Resident shards, least-recently-used first.
        self._loaded: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        #: Shards whose bytes this reader has already hashed.
        self._verified: set[int] = set()
        #: Counters for the throughput property: an epoch of shuffled
        #: access must cost O(shards), not O(samples).
        self.shard_loads = 0
        self.shard_hashes = 0

    def _check_complete(self) -> None:
        if self.manifest.get("status") != "complete":
            raise CacheIntegrityError(
                f"{self.cache_dir}: manifest has no status 'complete' "
                f"(cache_version {self.manifest.get('cache_version')!r}); the "
                "build was interrupted or predates the v0.2 transaction"
            )
        claimed = int(self.manifest["entries"])
        stored = int(self.manifest["shard_entries"])
        indexed = sum(int(s["count"]) for s in self.manifest["shard_index"])
        if not claimed == stored == indexed:
            raise CacheIntegrityError(
                f"{self.cache_dir}: manifest claims {claimed} entries but the "
                f"shards hold {stored} and the index sums to {indexed}"
            )

    def __len__(self) -> int:
        return int(self.manifest["entries"])

    def rows_of_shard(self, shard: int) -> list[dict[str, Any]]:
        """The resident rows of one shard by ordinal, loaded (and
        hash-verified against the manifest) on first use.

        A caller that already knows which shard holds a key -- the
        manifest's per-shard `rows` key lists are validated to be exactly
        the shard's keys, in order -- does not need the positional
        `index // shard_size` arithmetic of `__getitem__` at all. That is
        what lets `build_exact_text_cache.open_feature_source` build a
        key→shard index from the manifest JSON instead of reading every
        entry of the cache.
        """
        if not 0 <= shard < len(self.manifest["shard_index"]):
            raise IndexError(shard)
        return self._shard_rows(shard)

    def _shard_rows(self, shard: int) -> list[dict[str, Any]]:
        rows = self._loaded.get(shard)
        if rows is not None:
            self._loaded.move_to_end(shard)
            return rows
        name = f"shard-{shard:05d}.pt"
        path = self.cache_dir / name
        expected = self._index[name]
        if shard not in self._verified:
            actual = _sha256(path)
            if actual != expected["sha256"]:
                raise CacheIntegrityError(
                    f"{path}: sha256 {actual} does not match the manifest's "
                    f"{expected['sha256']}; the shard changed after the build"
                )
            self._verified.add(shard)
            self.shard_hashes += 1
        rows = torch.load(path, weights_only=True)
        self.shard_loads += 1
        if len(rows) != int(expected["count"]):
            raise CacheIntegrityError(
                f"{path}: {len(rows)} rows, manifest says {expected['count']}"
            )
        self._loaded[shard] = rows
        while len(self._loaded) > self.cache_shards:
            self._loaded.popitem(last=False)
        return rows

    def __getitem__(self, index: int) -> CacheEntry:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        rows = self._shard_rows(index // self.shard_size)
        row = rows[index % self.shard_size]
        return CacheEntry(**row)


if __name__ == "__main__":  # pragma: no cover
    import sys

    # Diagnostic only: `python -m unflatten.model.feature_cache <cache-dir>`
    # answers the question D4 leaves open — how large do features get?
    print(json.dumps(scan_magnitudes(sys.argv[1]), indent=1))
