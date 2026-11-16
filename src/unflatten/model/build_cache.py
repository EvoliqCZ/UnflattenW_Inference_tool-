"""Build a feature cache for decoder training.

Runs the frozen backbone once over a split and persists, per page, the
visual features plus the decoder targets derived from that document's
Canonical Target Bundle. Training then never touches the backbone.

Pages whose targets cannot be encoded (a structural stream the grammar
refuses) are skipped with a typed count rather than crashing the build —
one malformed document must not cost hours of cached work — but the count
is reported, because a large one means the bridge is broken, not the data.

**Candidate-track cost, measured (2026-09-04) -- the first CPU-only
estimate was misleading, not just conservative.** `--use-candidate-loader`
(below) runs the real M2/M3 dense candidate instead of the probe backbone.
A first CPU-only measurement (3 pages, 19.4 s/page) extrapolated to ~7.1
CPU-hours for the full ~1314-page single-page train cohort (DEC-007 D1's
measured 23.6% fraction of release-v10's 5569-doc train split). That
number was inflated by two real bugs, both now fixed:
`candidate_loader.load_qwen_candidate` never moved the model to `device`
(fixed: it now does, after freezing), and the candidate's vision tower
never got the `patch_embed`-as-matmul rewrite `probe/features.py` already
needed for this exact tower (fixed: `load_qwen_candidate` now calls
`probe.backbones.optimize_patch_embed` after the device move).

**Dated note on the denominator (2026-09-05).** "release-v10's 5569-doc
train split" is `wc -l splits/train.jsonl` -- one manifest, not the cohort
this function enumerates. `samples._split_membership` merges every manifest
in `_MANIFEST_FILES`, and `partial.jsonl` carries its real split in
`split_name`: measured by a full pass on 2026-09-05,
`iter_page_samples(release-v10, m2a-v5, "train", "structure_v0",
require_privacy_review=True)` yields **91,170 page samples over 14,644
documents (5,569 from train.jsonl + 9,075 from partial.jsonl), of which
3,556 are single-page** (2,400 + 1,156). The ~1314 above is therefore a
different quantity from anything that cohort measures; anyone sizing a full
build should use `docs/RUNBOOK.md` §4g's table, not this paragraph.

A real GPU build (150 pages, `--single-page-only --exclude-capability-profile`,
release-v10/m2a-v5) measured **0.256 s/page, 4.25 MB/page** -- ~76x the
CPU number, from the fix, not from the device alone. Extrapolated (not
separately measured) full single-page cohort: minutes, not hours; low
single-digit GB. The full build is still NOT run by default: S1's fifteen
gates (RFC005 §14.4) are mechanical and none of them gets more true with
more data (DEC-007 D8 pins no thresholds/denominators for S1 on purpose),
so the S1 canary uses a small (~100-200 single-page document) cache sized
to exercise the gates, not to be representative -- that discipline no
longer needs the cost as its reason, but the team lead's decision to keep
it stands regardless. The full cohort is S2's, when a quality claim first
needs a denominator. See `docs/RUNBOOK.md` §4f for the runnable command
and the same numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import torch

from unflatten.dataset.gates.store import read_object
from unflatten.dataset.items import build_item
from unflatten.dataset.samples import iter_page_samples
from unflatten.dataset.targets.bundle import PROJECTOR_VERSION
from unflatten.dataset.targets.bundle import target_bundle_hash as compute_target_bundle_hash
from unflatten.dataset.targets.masks import (
    MaskAstMismatchError,
    StaleMaskProjectionError,
    UnassessedMaskPointerError,
    apply_masks,
    mask_projection_hash,
    masked_typed_for_document,
    unsupervised_placements,
    withdraw_placement_counts,
)
from unflatten.model.capability_profile import (
    PRIMARY_S1_CAPABILITY_PROFILE,
    CapabilityProfile,
    resolve_capability_profile_by_id,
)
from unflatten.model.events_vocab import TOKEN_MAP_VERSION
from unflatten.model.feature_cache import CacheEntry, CacheIdentity, FeatureCacheWriter
from unflatten.model.grammar import GrammarViolation, controller_hash
from unflatten.model.prefetch import ordered_prefetch
from unflatten.model.targets import (
    DEFAULT_PROFILE,
    IGNORE_INDEX,
    allowed_mask_sequence,
    declares_supervision_scope,
    decoder_targets,
    supervised_event_indices,
)


class OnlyBundlesInvalid(ValueError):
    """The `--only-bundles` file is unreadable, empty, or matches zero
    members of the cohort it was meant to restrict -- a silent empty cache
    is the failure mode this repo pays for."""


def load_only_bundles(path: Path | str) -> set[str]:
    """Read a `--only-bundles` file: one `document_bundle_id` per line,
    blank lines and `#`-prefixed comments ignored.

    Refuses (:class:`OnlyBundlesInvalid`, naming the file) when the file
    cannot be read at all, or holds no ids once comments and blank lines
    are stripped -- both would otherwise silently build an empty cache.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise OnlyBundlesInvalid(f"--only-bundles: cannot read {path}: {exc}") from exc
    ids = {
        stripped
        for line in text.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    }
    if not ids:
        raise OnlyBundlesInvalid(f"--only-bundles: {path} holds no bundle ids")
    return ids


class OnlyPagesInvalid(ValueError):
    """The `--only-pages` file is unreadable, empty, malformed, or names a
    page absent from the enumerated cohort -- the list was computed over
    this exact release/split, so a miss means the wrong release, split or
    list, not a legitimately absent page (unlike `--only-bundles`)."""


def load_only_pages(path: Path | str) -> set[tuple[str, int]]:
    """Read a `--only-pages` file: JSON `[[bundle_id, page_index], ...]`.

    Refuses (:class:`OnlyPagesInvalid`, naming the file) when the file
    cannot be read, does not parse as JSON, is empty, or holds an entry
    that is not a 2-element `[str, int]` list.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise OnlyPagesInvalid(f"--only-pages: cannot read {path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OnlyPagesInvalid(f"--only-pages: {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise OnlyPagesInvalid(f"--only-pages: {path} must hold a nonempty JSON list")
    keys: set[tuple[str, int]] = set()
    for position, entry in enumerate(payload):
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or isinstance(entry[1], bool)
            or not isinstance(entry[1], int)
        ):
            raise OnlyPagesInvalid(
                f"--only-pages: {path}[{position}] must be [bundle_id: str, page_index: int], "
                f"got {entry!r}"
            )
        keys.add((entry[0], entry[1]))
    return keys


def only_pages_manifest_fields(only_pages: set[tuple[str, int]]) -> dict[str, Any]:
    """`{only_pages_sha256, only_pages_count}` for the manifest, mirroring
    `only_bundles_manifest_fields`: sha256 over the sorted
    `"{bundle}\\t{page}"` lines, so two callers handing in the same keys in
    a different order agree."""
    ordered = sorted(f"{bundle}\t{page}" for bundle, page in only_pages)
    digest = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
    return {"only_pages_sha256": digest, "only_pages_count": len(ordered)}


@dataclass(frozen=True)
class _BuildPlan:
    """CPU-only build scope, deliberately available before backbone loading."""

    samples: list[Any]
    split_samples: int
    cohort: int
    resume: bool
    manifest_entries: int


def resolve_limit(limit: int | None, remaining: int) -> tuple[int, str]:
    """Resolve a run's new-row cap, matching the exact-text cache precedent."""
    if limit is None:
        return remaining, "all"
    if limit <= 0:
        raise ValueError(
            "--limit must be a positive page count or absent "
            f"(absent = the whole remaining cohort), got {limit}"
        )
    return min(limit, remaining), "limited"


def _plan_build(
    release_dir: Path,
    store_dir: Path,
    out_dir: Path,
    *,
    split: str,
    task_kind: str,
    allow_dev_only: bool,
    single_page_only: bool,
    only_bundles: set[str] | None,
    only_bundles_path: Path | str | None,
    only_pages: set[tuple[str, int]] | None = None,
    only_pages_path: Path | str | None = None,
    multi_page_only: bool = False,
) -> _BuildPlan:
    """Enumerate and select the cohort without loading or touching a model.

    `multi_page_only` (`s2_gate.cohort`'s `multi_page_pages` slice, DEC-033
    §12) restricts the cohort to `PageSample.page_count > 1` -- the
    complement of `single_page_only`. Combining both is refused: neither
    filter can be the other's subset of itself."""
    if single_page_only and multi_page_only:
        raise ValueError("single_page_only and multi_page_only are mutually exclusive")
    samples = list(
        iter_page_samples(
            release_dir, store_dir, split, task_kind, require_privacy_review=not allow_dev_only
        )
    )
    if only_pages is not None:
        present = {(sample.document_bundle_id, sample.page_index) for sample in samples}
        missing = sorted(only_pages - present)
        if missing:
            first = ", ".join(f"{bundle}:{page}" for bundle, page in missing[:5])
            raise OnlyPagesInvalid(
                f"--only-pages: {len(missing)} of {len(only_pages)} requested page(s) "
                f"from {only_pages_path or '<file>'} not found in split "
                f"{split!r}/{task_kind!r} -- first offending: {first}"
            )
        samples = [
            sample
            for sample in samples
            if (sample.document_bundle_id, sample.page_index) in only_pages
        ]
        print(
            f"[cache] --only-pages: restricting {split!r}/{task_kind!r} to "
            f"{len(only_pages)} requested page(s) ({len(samples)} kept)",
            flush=True,
        )
    elif only_bundles is not None:
        present = {sample.document_bundle_id for sample in samples}
        matched = only_bundles & present
        missing = only_bundles - present
        if not matched:
            raise OnlyBundlesInvalid(
                f"--only-bundles: 0 of {len(only_bundles)} requested bundle id(s) "
                f"from {only_bundles_path or '<file>'} were found in split "
                f"{split!r}/{task_kind!r} -- refusing to build an empty cache"
            )
        samples = [sample for sample in samples if sample.document_bundle_id in only_bundles]
        survived_suffix = ""
        if single_page_only:
            samples = [sample for sample in samples if sample.page_count == 1]
            survived_suffix = f", {len(samples)} of {len(matched)} survived single_page_only"
            if not samples:
                raise OnlyBundlesInvalid(
                    f"--only-bundles: {len(matched)} of {len(only_bundles)} requested "
                    f"bundle id(s) from {only_bundles_path or '<file>'} matched split "
                    f"{split!r}/{task_kind!r}, but none survived single_page_only -- "
                    "refusing to build an empty cache"
                )
        print(
            f"[cache] --only-bundles: restricting {split!r}/{task_kind!r} to "
            f"{len(only_bundles)} requested bundle id(s) ({len(matched)} matched, "
            f"{len(missing)} not found in the split{survived_suffix})",
            flush=True,
        )
    elif single_page_only:
        samples = [sample for sample in samples if sample.page_count == 1]
    elif multi_page_only:
        samples = [sample for sample in samples if sample.page_count > 1]

    manifest_path = out_dir / "manifest.json"
    manifest_entries = 0
    if manifest_path.exists():
        manifest_entries = int(json.loads(manifest_path.read_text()).get("entries", 0))
    return _BuildPlan(samples, len(samples), len(samples), manifest_path.exists(), manifest_entries)


def only_bundles_manifest_fields(only_bundles: set[str]) -> dict[str, Any]:
    """`{only_bundles_sha256, only_bundles_count}` for the manifest, so a
    cohort-restricted cache can never pass as a full one. The hash is over
    the sorted id list so two callers handing in the same ids in a
    different order agree."""
    ordered = sorted(only_bundles)
    digest = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
    return {"only_bundles_sha256": digest, "only_bundles_count": len(ordered)}


def _hub_root() -> Path:
    home = os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"
    return Path(home) / "hub"


#: A git commit sha, as huggingface_hub names a snapshot directory.
_SNAPSHOT_SHA = re.compile(r"^[0-9a-f]{40}$")


def resolve_backbone_revision(
    model_id: str, *, hub_root: Path | None = None
) -> tuple[str | None, str]:
    """Best-effort weight identity for `model_id`, offline.

    No loader in this repository passes `revision=`, so the model ID
    pins a name and not bytes. The local hub cache does record which
    commit it resolved, and that is the strongest claim available
    without a network call. A SHA-pinned download leaves no `refs/main`
    file, but its snapshot directory is *named by* the commit sha — the
    same fact one level deeper, so a single such directory is accepted
    with a source that says exactly where it was read from. Zero or
    several commit directories is not a pin: which weights generated
    the features is then genuinely unknown, and the identity stays
    `"unavailable"` — an honest gap, not a fabricated one.
    """
    root = hub_root if hub_root is not None else _hub_root()
    model_dir = root / f"models--{model_id.replace('/', '--')}"
    ref = model_dir / "refs" / "main"
    try:
        revision = ref.read_text().strip()
    except OSError:
        revision = ""
    if revision:
        return revision, "local_hf_cache_ref"
    try:
        names = os.listdir(model_dir / "snapshots")
    except OSError:
        names = []
    snapshots = [name for name in names if _SNAPSHOT_SHA.match(name)]
    if len(snapshots) == 1:
        return snapshots[0], "local_hf_snapshot_dir"
    return None, "unavailable"


def cache_identity(
    backbone: Any,
    release_dir: Path,
    *,
    split: str,
    task_kind: str = "structure_v0",
    max_pixels: int | None = None,
    use_multiscale: bool = False,
    at_merged: bool = False,
    admission: str = "strict",
    capability_profile: CapabilityProfile | None = None,
) -> CacheIdentity:
    """Everything a cache's features are bound to, for the manifest.

    `capability_profile` is `None` for the historical S1 feature-build
    reproduction (see `prepare_page`): `grammar_profile` records the
    literal `targets.DEFAULT_PROFILE` and `capability_profile_hash` stays
    `None`, exactly what every cache built before DEC-024 recorded. An
    explicit profile records its own `ast_profile` and `profile_hash`
    instead.
    """
    try:
        manifest = json.loads((Path(release_dir) / "release-manifest.json").read_text())
        release_root_hash = manifest.get("release_root_hash")
    except (OSError, json.JSONDecodeError):
        release_root_hash = None

    processor = getattr(backbone, "processor", None)
    digest = None
    if processor is not None and hasattr(processor, "to_json_string"):
        digest = hashlib.sha256(processor.to_json_string().encode("utf-8")).hexdigest()[:16]

    revision, source = resolve_backbone_revision(backbone.model_id)
    return CacheIdentity(
        backbone=backbone.model_id,
        backbone_revision=revision,
        revision_source=source,
        processor=digest,
        feature_tap=getattr(backbone, "tap", None),
        max_pixels=max_pixels,
        split=split,
        release_root_hash=release_root_hash,
        task_kind=task_kind,
        multiscale=use_multiscale,
        merged_grid=at_merged,
        token_map_version=TOKEN_MAP_VERSION,
        grammar_hash=controller_hash(),
        grammar_profile=(
            DEFAULT_PROFILE if capability_profile is None else capability_profile.ast_profile
        ),
        target_projector_version=PROJECTOR_VERSION,
        admission=admission,
        capability_profile_hash=(
            None if capability_profile is None else capability_profile.profile_hash
        ),
    )


@lru_cache(maxsize=64)
def _sidecar_font_pack_hash(store_dir: str, ref_json: str) -> str | None:
    sidecar = json.loads(read_object(Path(store_dir), json.loads(ref_json)))
    return (sidecar.get("renderer") or {}).get("font_pack_hash")


def render_font_pack_hash(store_dir: Path, sample: Any) -> str | None:
    """The font pack the pixels of this page were actually rendered with.

    Read from the Observation Sidecar's `renderer` block, which records
    the render that produced the raster. Cached per document: a sidecar
    is large and a document averages 5.84 pages.
    """
    ref = getattr(sample, "observation_sidecar", None)
    if ref is None:
        return None
    return _sidecar_font_pack_hash(str(store_dir), json.dumps(ref, sort_keys=True))


class _DocumentMaskedTypedCache:
    """`_document_masked_typed`'s cache, keyed by content identity rather
    than by store ref: `(target_bundle_hash, mask_projection_hash,
    font_pack_hash)`. `functools.lru_cache` cannot key on the bundle/
    projection dicts themselves (unhashable) without also holding them
    alive as the key, so this is a small hand-rolled LRU instead -- same
    `maxsize` reasoning as `items._cached_document_json` (holds full
    bundles' typed surfaces, one document's worth at a time in the
    common consecutive-pages-per-worker case).

    Exposes `.cache_clear()` for the same reason every `lru_cache` in
    this codebase does: a test that reuses one fixture bundle's content
    identity across scenarios with different monkeypatched behaviour
    must not see an earlier scenario's cached result.
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._data: dict[tuple[Any, ...], dict[str, Any]] = {}

    def get_or_compute(self, key, compute):
        cached = self._data.get(key)
        if cached is not None:
            return cached
        result = compute()
        if len(self._data) >= self._maxsize:
            self._data.pop(next(iter(self._data)))
        self._data[key] = result
        return result

    def cache_clear(self) -> None:
        self._data.clear()


#: `maxsize=2`, not 8: a worker processes one document's pages
#: consecutively, so a hit only ever needs the current document.
_document_masked_typed_cache = _DocumentMaskedTypedCache(maxsize=2)


def _document_masked_typed(
    target_bundle: dict[str, Any],
    mask_projection: dict[str, Any] | None,
    *,
    render_font_pack_hash: str | None,
) -> dict[str, Any]:
    """The document-scoped half of §29 masking (`masks.masked_typed_for_document`),
    cached per document so its pages share ONE font-projection stamp
    instead of each paying `apply_masks`' whole-bundle deep copy and hash
    (measured 1.37 s + 1.06 s per page on a 68 MB bundle;
    `docs/plans/NALEZ-velke-dokumenty-kvadraticka-cena.md`).

    Keyed on `target_bundle["target_bundle_hash"]` and
    `masks.mask_projection_hash(mask_projection)` — the bundle's and
    projection's own content identities — rather than on a store ref, so
    this works for any caller that already has the parsed objects in
    hand (real pages via `items.build_item`'s own cache, or a test double
    that never touches the store at all). Every bundle `build_target_bundle`
    produces carries `target_bundle_hash` already (§27) and validates it
    recomputes, so the field is trusted there at zero cost; it is
    recomputed here ONLY as a fallback for a bundle that omits it (a bare
    test fixture, never a real one) -- a missing key must not become two
    unrelated documents silently sharing one cache slot.
    """
    key = (
        target_bundle.get("target_bundle_hash")
        or compute_target_bundle_hash(target_bundle),
        mask_projection_hash(mask_projection) if mask_projection is not None else None,
        render_font_pack_hash,
    )
    return _document_masked_typed_cache.get_or_compute(
        key,
        lambda: masked_typed_for_document(
            target_bundle, mask_projection, render_font_pack_hash=render_font_pack_hash
        ),
    )


def _page_withdrawal_counts(
    target_bundle: dict[str, Any],
    fragment: dict[str, Any],
    *,
    mask_projection: dict[str, Any] | None,
    render_font_pack_hash: str | None,
) -> tuple[int, int, int]:
    """`(nodes, flipped, typed_fields_total)` for one page, the only
    three things `prepare_page` reads off `withdraw_placements`' bundle
    -- computed against the document-scoped, cached font stamp
    (`_document_masked_typed`) instead of re-deriving and re-hashing the
    whole document's typed surface for every page.
    """
    stamped_typed = _document_masked_typed(
        target_bundle, mask_projection, render_font_pack_hash=render_font_pack_hash
    )
    total = sum(len(entry["fields"]) for entry in stamped_typed.values())
    nodes, flipped = withdraw_placement_counts(stamped_typed, fragment)
    return nodes, flipped, total


def withdraw_placements(
    bundle: dict[str, Any],
    fragment: dict[str, Any],
    *,
    mask_projection: dict[str, Any] | None = None,
    render_font_pack_hash: str | None = None,
) -> tuple[dict[str, Any], int, int, int]:
    """Apply every supervision withdrawal this page's labels are subject to.

    Returns `(bundle, nodes, flipped, typed_fields_total)`. The three
    counts are not the same kind of number, and only one of them is
    evidence that the withdrawal ran:

    - `flipped` is a **diff** of before against after — typed fields this
      call actually changed. It goes to zero if the withdrawal is removed,
      which is the failure mode this count exists to catch.
    - `nodes` is **re-derived from the fragment**: unresolved owner nodes,
      including ones `apply_masks` skips because they carry no `typed`
      entry (`targets/masks.py`). It would stay correct with the
      withdrawal deleted, so it reports scope, not effect.
    - `typed_fields_total` is the size of the typed surface after masking.

    Order is load-bearing and `apply_masks` owns it: the font projection
    re-derives `supervision_mask` from `known AND observable` for the
    pointers it assesses, so the placement withdrawal must run after it.
    Both run over one deep copy and cost one `target_bundle_hash`.

    `render_font_pack_hash` is required with no default: a projection
    derived under another font pack matches on `ast_hash`, validates, and
    masks the wrong runs. A mismatch — or a caller that cannot name its
    render — raises.
    """
    withdrawn, flipped = apply_masks(
        bundle,
        mask_projection,
        [fragment],
        render_font_pack_hash=render_font_pack_hash,
    )
    total = sum(len(entry["fields"]) for entry in withdrawn.get("typed", {}).values())
    nodes = len(unsupervised_placements([fragment]))
    return withdrawn, nodes, flipped, total


def owns_content(fragment: dict[str, Any]) -> bool:
    """Does this page own any structural event? (OWNERSHIP axis.)

    Ownership is partition bookkeeping: every event has exactly one
    owner whatever its evidence. This answers only that. Whether those
    events may be TRAINED on is the other axis, and
    :func:`owns_supervised_content` answers it — conflating the two is
    what caused three defects in this file's history, so the pair is
    deliberately two predicates and not one.

    Its probe counterpart is
    :func:`probe.labels.fragment_owns_supervised_structure`, which sits
    on the SUPERVISION axis and is therefore the counterpart of
    `owns_supervised_content`, not of this function. There is no
    ownership-axis predicate in the probe: it has no use for one.

    A fragment with no owned events means the page's nodes could not be
    placed by the observation geometry — not that the page is blank. On
    the imported corpus that is ~50 % of pages, and sampling their rasters
    found 11-18 % ink on seven of eight: real text. Caching them as empty
    would train the model to emit nothing for a full page. RFC005 states
    the general rule: a missing label is never encoded as a value.
    """
    from unflatten.model.encode_fragment import _owned_event_indices

    return bool(_owned_event_indices(fragment))


def owns_supervised_content(fragment: dict[str, Any]) -> bool:
    """Does any owned event survive the withdrawal? (SUPERVISION axis.)

    The stronger of the pair: a page can own plenty and supervise none
    of it. :func:`owns_content` answers the ownership question and this
    one answers whether anything here may reach a loss, which is why
    both exist and why neither is redundant.

    The envelope's own tokens (`FRAGMENT_BEGIN`, the resume chain, the
    suspends, `END_FRAGMENT`) belong to no node and are always scored, so
    a page whose every owned event is `unresolved` would still train the
    sequence "this page contains nothing" — the same false label
    `owns_content` refuses, arrived at by a different route.
    """
    from unflatten.model.encode_fragment import _owned_event_indices

    return bool(_owned_event_indices(fragment) & supervised_event_indices(fragment))


def has_continuation_presentation(fragment: dict[str, Any], structural: dict[str, Any]) -> bool:
    """Does this page present (design §4.1) a continuation it does not own?

    True exactly when `encode_fragment._presented_text_indices` is
    non-empty: a `char_ranges` grant for a node whose `op == "text"` event
    this page's `structural_event_ranges` do not cover. Design §4.1 names
    this as a NEW occasion for a grammar/length/capability refusal that
    could not arise before continuation presentation existed (P4), so
    T-9's counters keep it apart from the pre-existing bucket rather than
    letting it inflate a baseline measured before this page shape existed.
    """
    from unflatten.model.encode_fragment import (
        _owned_event_indices,
        _presented_text_indices,
    )

    owned = _owned_event_indices(fragment)
    events = list(structural.get("events", ()))
    return bool(_presented_text_indices(fragment, events, owned))


def _uses_excluded_capability(
    target_ids: Any, input_ids: Any, *, capability_profile: CapabilityProfile
) -> bool:
    """Does this page's decoded structural stream use a node type
    `capability_profile.capability_excluded` masks out?

    `target_ids`/`input_ids` are `targets.decoder_targets`' structural
    local-id tensors, drawn from `events_vocab.TOKEN_IDS` -- the SAME
    global alphabet `capability_profile.py` derives
    `excluded_structural_local_ids` from (one alphabet spans every
    profile, `events_vocab.py`'s own docstring), so membership is checked
    directly with no re-encoding.

    **This checks the DECODED per-page stream, not the raw target
    bundle -- and that distinction only stops mattering under a specific,
    measured condition.** RFC003 §29.3 withdrawal can drop a node from
    the decoded stream while the node (and its pixels) still exist in the
    document's raw target bundle; checking only the decoded stream could
    then silently admit a page whose pixels contain excluded content the
    model is never told to account for (RFC005 §14.6 forbids exactly
    this). A stricter, raw-bundle-reading version of this check existed
    and was retired as a duplicate ONLY after measuring the two against
    each other: **0/1500, then 0/3257, single-page `train` candidates
    from `release-v10`/`m2a-v5` on 2026-09-04** disagreed in either
    direction, and every page that never reached this decoded check at
    all (grammar/skip failures) also carried an excluded type in its raw
    bundle. The reason it measured zero is structural, not incidental:
    the cohort this runs against is `single_page_only`-filtered, so a
    document's raw structural stream and its one page's fragment
    describe the same content, and `prepare_page` skips the WHOLE page
    (`skipped_unplaced`/`skipped_no_supervision_scope`) rather than
    silently omitting one node from an otherwise-cached page. **That
    reasoning breaks once a multi-page cohort exists** -- a node could
    then genuinely live on a different page than the one being checked --
    so this function must be re-measured against a raw-bundle check
    before `single_page_only=False` is ever combined with
    `exclude_capability_profile=True` for real training data.
    `scripts/census/capability_exclusion_agreement.py` is that
    re-measurement tool -- re-derivable evidence, not a number that only
    lives in a chat transcript.
    """
    excluded = capability_profile.excluded_structural_local_ids
    return bool(excluded & set(target_ids.tolist()) or excluded & set(input_ids.tolist()))


#: Image modes whose `tobytes()` rebuilds through `Image.frombytes`.
#: Anything else travels as its encoded PNG and is decoded by the
#: consumer: a palette image's raw bytes are indices, and rebuilding it
#: without the palette would hand the tower different pixels.
_RAW_IMAGE_MODES = frozenset({"RGB", "RGBA", "L"})

#: Summary keys `prepare_page` can name as the reason a page is not cached.
#:
#: T-9 (design §4.1): every `..._continuation` key below is a SUBSET of its
#: parent, never a separate bucket -- a page counted under `skipped_grammar`
#: because it presents (`has_continuation_presentation`) a continuation it
#: does not structurally own increments BOTH `skipped_grammar` AND
#: `skipped_grammar_continuation`. `RUNBOOK`'s `skipped_grammar` ≈ 5 %
#: baseline and `scripts/census/capability_exclusion_agreement.py` (P13)
#: both read the parent label; a page that would have been counted there
#: before continuation presentation existed must still land there --
#: partitioning it into the new key instead would make a multi-page build
#: look like an improvement over that baseline for the wrong reason.
SKIP_COUNTS: tuple[str, ...] = (
    "skipped_unencodable",
    "skipped_grammar",
    "skipped_grammar_continuation",
    "skipped_unplaced",
    "skipped_too_long",
    "skipped_too_long_continuation",
    "skipped_fully_unsupervised",
    "skipped_no_observation",
    "skipped_no_supervision_scope",
    # Not a `prepare_page` skip -- decided in `build`'s own loop, after
    # `prepare_page` has already decoded `target_ids`/`input_ids`, against
    # `capability_profile.PRIMARY_S1_CAPABILITY_PROFILE.excluded_structural_local_ids`.
    # DEC-007 D1.1: inline_image (and the rest of that exclusion set) stays
    # OUT of the active S1 profile even though it is in the ratified node
    # scope -- a page whose structural stream uses it is not S1 cohort
    # material and must never be silently included.
    "skipped_capability_excluded",
    "skipped_capability_excluded_continuation",
)

#: `page.skip` reasons T-9 attributes a `_continuation` subset for. Only
#: the causes design §4.1 names (grammar, length) -- `skipped_unencodable`,
#: `skipped_unplaced`, `skipped_fully_unsupervised`, `skipped_no_observation`
#: and `skipped_no_supervision_scope` are not new occasions for continuation
#: presentation and keep no split. `skipped_capability_excluded` is
#: decided outside `prepare_page` (`build`'s own loop) and gets the same
#: subset treatment there directly, not through this set.
_CONTINUATION_ATTRIBUTABLE_SKIPS = frozenset({"skipped_grammar", "skipped_too_long"})


@dataclass
class PreparedPage:
    """One page's CPU-side work, ready for the backbone.

    Every field is picklable: this crosses a process boundary when the
    build runs with workers.
    """

    document_bundle_id: str
    page_index: int
    #: Summary counter to increment instead of caching this page, or None.
    skip: str | None = None
    #: Deltas this page contributes whether or not it is cached — the
    #: withdrawal counts, which are accrued for skipped pages too.
    counters: dict[str, int] = field(default_factory=dict)
    #: Set once `bundle["structural"]` is available (`has_continuation_presentation`).
    #: `build`'s loop reads this to attribute a `skipped_capability_excluded`
    #: refusal, which is decided outside `prepare_page` (T-9).
    has_continuation: bool = False
    target_ids: Any = None
    input_ids: Any = None
    allowed: Any = None
    #: `(mode, size, pixels)`, or `(None, None, png_bytes)`.
    raster: tuple[str | None, tuple[int, int] | None, bytes] | None = None
    #: A mask failure raised while preparing. Re-raised by the consumer at
    #: this page's position, because the message names how far the build
    #: got and a worker cannot know that.
    error: Exception | None = None

    def image(self) -> Any:
        """The page raster, rebuilt from whichever form it travelled in."""
        from PIL import Image

        mode, size, data = self.raster or (None, None, b"")
        if mode is None:
            image = Image.open(io.BytesIO(data))
            image.load()
            return image
        return Image.frombytes(mode, size, data)


def prepare_page(
    store_dir: Path,
    sample: Any,
    *,
    max_steps: int | None,
    capability_profile: CapabilityProfile | None = None,
) -> PreparedPage:
    """Everything one page needs before the GPU: labels, targets, raster.

    This is the whole CPU cost of a page — ~360 ms of it, against ~87 ms
    of backbone forward — and it is a pure function of the store and the
    sample, which is what lets it run in a worker process.

    A page that must not be cached comes back with `skip` naming the
    counter, never as an exception: one malformed document must not cost
    hours of cached work. The three mask failures are the exception, and
    they travel in `error` for the consumer to raise in order.

    `capability_profile` is `None` by default: **the historical S1
    feature-build reproduction**, byte-identical to every build before
    DEC-024 -- the structural grammar stays the literal
    `targets.DEFAULT_PROFILE` (`"word-visual-v1"`), because no ratified
    capability profile declares that `ast_profile` (they all declare
    `"word-visual-v1.1"`, ratified after this cache format existed) and
    deriving the grammar from one would silently change what an unflagged
    invocation builds. Passing an explicit `capability_profile` derives the
    grammar from `capability_profile.ast_profile` instead -- a NEW kind of
    build, not "the S1 profile with a different name": no profile id maps
    to the historical `"word-visual-v1"` grammar.
    """
    from PIL import Image

    grammar_profile = (
        DEFAULT_PROFILE if capability_profile is None else capability_profile.ast_profile
    )

    page = PreparedPage(document_bundle_id=sample.document_bundle_id, page_index=sample.page_index)
    item = build_item(store_dir, sample)
    fragment = item.fragment
    if not (item.targets or {}).get("structural") or fragment is None:
        # Page-local targets: a sample is one page, so the document
        # stream would ask page 1 to predict page 7.
        page.skip = "skipped_unencodable"
        return page
    # A node whose page was inferred rather than observed has typed
    # labels describing pixels nobody can point at. The withdrawal
    # happens HERE, at the only boundary where the dataset's labels enter
    # the model path (RFC003 §26, D2).
    #
    # Below reads only `nodes`/`fields`/`total` and the UNCHANGED
    # `structural` stream off what `withdraw_placements` would have
    # returned -- masks.py never touches "structural" -- so it is read
    # straight off `item.targets` instead of paying for a page that
    # nothing here consumes: `_page_withdrawal_counts` computes the same
    # three counts against a document-scoped, cached font stamp rather
    # than deep-copying and re-hashing the whole document's typed surface
    # on every page (`docs/plans/NALEZ-velke-dokumenty-kvadraticka-cena.md`).
    try:
        nodes, fields, total = _page_withdrawal_counts(
            item.targets,
            fragment,
            mask_projection=item.mask_projection,
            render_font_pack_hash=render_font_pack_hash(store_dir, sample),
        )
        bundle = item.targets
    except (
        MaskAstMismatchError,
        StaleMaskProjectionError,
        UnassessedMaskPointerError,
    ) as exc:
        # A projection that disagrees with its own render, with the AST it
        # is applied to, or with its own assessed pointers means artifacts
        # that must agree do not: store integrity, not a property of this
        # page. All three abort the build.
        page.error = exc
        return page
    page.counters = {
        "withdrawn_typed_nodes": nodes,
        "withdrawn_typed_fields": fields,
        "typed_fields_total": total,
    }
    if not fragment.get("slices"):
        # O-3 (`docs/plans/NALEZ-o3-strany-bez-pozorovani.md`): an empty
        # `slices` list is vacuously `declares_supervision_scope() is
        # False` too, but it is not the same page. It is a post-§26
        # fragment the aligner placed no observation on at all — not a
        # fragment written before §26. Conflating the two blamed a stale
        # schema for 11,773 real pages that are actually unplaced.
        page.skip = "skipped_no_observation"
        return page
    if not declares_supervision_scope(fragment):
        # Ownership without a supervision scope (a fragment written before
        # §26 page_evidence). Neither reading is honest, so it is refused
        # and counted, never guessed.
        page.skip = "skipped_no_supervision_scope"
        return page
    if not owns_content(fragment):
        # geometry could not place this page's nodes; storing it as an
        # empty page would be a false label, not a missing one
        page.skip = "skipped_unplaced"
        return page
    # Computed once here rather than lazily inside each except/skip branch
    # below: every one of them may need it, and it is cheap (two set
    # comprehensions over this page's own slices and events).
    page.has_continuation = has_continuation_presentation(fragment, bundle["structural"])
    try:
        target_ids, input_ids = decoder_targets(
            bundle["structural"], fragment=fragment, profile=grammar_profile
        )
        allowed = allowed_mask_sequence(
            bundle["structural"], fragment=fragment, profile=grammar_profile
        )
    except GrammarViolation:
        # The fragment protocol needs a resume chain this encoder cannot
        # always reconstruct when a page's owned events form disjoint
        # spans (~5 % of pages). Counted separately: it is a known encoder
        # limit, not malformed data. `page.skip` keeps its old label
        # regardless of `page.has_continuation` -- `build`'s loop is what
        # ADDS the `_continuation` increment (T-9), so the RUNBOOK's
        # `skipped_grammar` baseline and `capability_exclusion_agreement.py`
        # (P13), which both read this label, see no change on a page this
        # refusal would already have counted before continuation
        # presentation existed.
        page.skip = "skipped_grammar"
        return page
    except Exception:  # noqa: BLE001 - one typed count, not a crash
        page.skip = "skipped_unencodable"
        return page
    if max_steps is not None and target_ids.shape[0] > max_steps:
        # Same rule as the `GrammarViolation` branch above: the label
        # does not change, `build`'s loop adds the split.
        page.skip = "skipped_too_long"
        return page
    # Counted off the emitted tensor, not re-derived from the fragment: a
    # count derived from the fragment would survive the removal of the
    # withdrawal it is meant to prove.
    scored = int((target_ids != IGNORE_INDEX).sum())
    page.counters["withdrawn_structural_steps"] = int(target_ids.shape[0]) - scored
    if scored == 0 or not owns_supervised_content(fragment):
        # Nothing of this page's content survives the withdrawal: it would
        # cost ~11 MB to teach an empty page.
        page.skip = "skipped_fully_unsupervised"
        return page
    page.target_ids, page.input_ids, page.allowed = target_ids, input_ids, allowed
    image = Image.open(io.BytesIO(item.image_bytes))
    image.load()
    page.raster = (
        (image.mode, image.size, image.tobytes())
        if image.mode in _RAW_IMAGE_MODES
        else (None, None, item.image_bytes)
    )
    return page


def prepare_pages(
    samples: list[Any],
    *,
    store_dir: Path,
    max_steps: int,
    capability_profile: CapabilityProfile | None = None,
) -> list[PreparedPage]:
    """`prepare_page` over consecutive samples — one unit of worker work.

    Chunked rather than per-page so a document's pages land in the same
    process, where the fragment index and the Observation Sidecar are
    behind an `lru_cache`. `capability_profile` is a frozen dataclass of
    strings/frozensets and picklable for the worker pool with no special
    handling.
    """
    return [
        prepare_page(store_dir, s, max_steps=max_steps, capability_profile=capability_profile)
        for s in samples
    ]


def _cache_entry(
    backbone: Any, page: PreparedPage, *, at_merged: bool, use_multiscale: bool
) -> CacheEntry:
    """Extract one prepared page into the writer's stable row representation."""
    features = backbone.extract(page.image())
    visual = (
        features.dense(at_merged=at_merged, include_levels=use_multiscale)
        if use_multiscale or at_merged
        else features.visual_features
    )
    return CacheEntry(
        document_bundle_id=page.document_bundle_id,
        page_index=page.page_index,
        features=visual,
        target_ids=page.target_ids,
        input_ids=page.input_ids,
        allowed=page.allowed,
        grid_thw=torch.tensor(features.grid_thw, dtype=torch.long),
    )


def build(
    backbone: Any,
    release_dir: Path,
    store_dir: Path,
    out_dir: Path,
    *,
    split: str = "train",
    task_kind: str = "structure_v0",
    limit: int | None = None,
    max_steps: int = 4096,
    shard_size: int = 32,
    max_pixels: int | None = None,
    use_multiscale: bool = False,
    at_merged: bool = False,
    allow_dev_only: bool = False,
    progress_every: int = 100,
    workers: int = 0,
    chunk: int = 8,
    single_page_only: bool = False,
    exclude_capability_profile: bool = False,
    only_bundles: set[str] | None = None,
    only_bundles_path: Path | str | None = None,
    only_pages: set[tuple[str, int]] | None = None,
    only_pages_path: Path | str | None = None,
    capability_profile: CapabilityProfile | None = None,
    _plan: _BuildPlan | None = None,
    _intent_emitted: bool = False,
) -> dict[str, Any]:
    """Cache one split's pages.

    `allow_dev_only` admits records whose privacy review is still open.
    RFC005 §9.2 classifies that as BLOCKED for training, so the default
    refuses them and the refusal is an exception, not a shorter cache:
    a build that silently drops most of a release is the failure this
    admission boundary exists to prevent. Pass it for a smoke or
    coverage run over dev/research inventory, deliberately.

    `workers` prepares pages in that many processes while the backbone
    runs here. It changes what waits for what and nothing else: the same
    pages are cached, in the same order, with the same bytes (see
    `prefetch.ordered_prefetch`). Zero keeps every page in this process.

    `single_page_only` restricts the cohort to `PageSample.page_count == 1`
    documents -- the S1 canary's cohort (DEC-007). `exclude_capability_profile`
    additionally skips (never silently caches) any page whose decoded
    structural stream uses a node type the chosen `capability_profile`
    excludes -- checked AFTER `prepare_page` decodes `target_ids`/`input_ids`
    but BEFORE the backbone forward, so an excluded page never spends a
    forward pass it cannot use. Both default off: the probe/decoder-training
    tracks this cache predates never needed either restriction.

    `capability_profile` is `None` by default: the historical S1
    feature-build reproduction (see `prepare_page`'s docstring) -- grammar
    stays the literal `targets.DEFAULT_PROFILE` (`"word-visual-v1"`) and
    `exclude_capability_profile` (if set) checks against
    `PRIMARY_S1_CAPABILITY_PROFILE`, byte-identical to every build before
    DEC-024. An explicit profile derives the grammar from its own
    `ast_profile` and, if `exclude_capability_profile` is set, checks
    against ITS OWN exclusion set instead -- under `TYPED_WIDENED_V1` that
    excludes nothing, so the flag becomes a no-op.

    `only_bundles` restricts the cohort to exactly those
    `document_bundle_id`s regardless of `split`/`task_kind` membership --
    an A/B across two store revisions needs to pin the cohort by id, because
    the RFC003 split allocator is RNG-free but greedy and one changed
    document can scramble val/test membership. A file whose ids match none
    of the split is refused (:class:`OnlyBundlesInvalid`); ids in the file
    the split does not have are not an error (the cohort may legitimately
    have moved to val/test) but are counted in the printed intent line.

    `only_pages` restricts the cohort to exactly those
    `(document_bundle_id, page_index)` keys, filtered AFTER enumeration so
    release order is kept. Unlike `only_bundles`, any requested key absent
    from the enumerated split refuses the whole build
    (:class:`OnlyPagesInvalid`): the list was computed over this exact
    release/split, so a miss means the wrong release, split or list, not a
    legitimately absent page. Mutually exclusive with `only_bundles` and
    `single_page_only` -- the page list is already explicit.
    """
    if only_pages is not None and (only_bundles is not None or single_page_only):
        raise ValueError("only_pages is mutually exclusive with only_bundles/single_page_only")
    started = datetime.now(UTC)
    started_clock = time.monotonic()
    plan = _plan or _plan_build(
        release_dir,
        store_dir,
        out_dir,
        split=split,
        task_kind=task_kind,
        allow_dev_only=allow_dev_only,
        single_page_only=single_page_only,
        only_bundles=only_bundles,
        only_bundles_path=only_bundles_path,
        only_pages=only_pages,
        only_pages_path=only_pages_path,
    )
    # Validate before any writer/backbone work in the programmatic API too.
    resolve_limit(limit, plan.cohort)
    validate_planning_options(max_steps=max_steps, chunk=chunk, workers=workers)
    written = 0
    attempted = 0
    counts = dict.fromkeys(
        (
            *SKIP_COUNTS,
            "withdrawn_typed_nodes",
            "withdrawn_typed_fields",
            "typed_fields_total",
            "withdrawn_structural_steps",
        ),
        0,
    )
    identity = cache_identity(
        backbone,
        release_dir,
        split=split,
        task_kind=task_kind,
        max_pixels=max_pixels,
        use_multiscale=use_multiscale,
        at_merged=at_merged,
        admission="dev_only" if allow_dev_only else "strict",
        capability_profile=capability_profile,
    )
    if only_pages is not None:
        only_bundles_fields = only_pages_manifest_fields(only_pages)
    elif only_bundles is not None:
        only_bundles_fields = only_bundles_manifest_fields(only_bundles)
    else:
        only_bundles_fields = None
    build_options = _requested_build_options(
        single_page_only=single_page_only,
        exclude_capability_profile=exclude_capability_profile,
        max_steps=max_steps,
        workers=workers,
        chunk=chunk,
    )

    def checkpoint(status: str, entries: int, shards: int) -> None:
        if status == "in_progress":
            print(
                f"[cache] checkpoint: status={status} entries={entries} shards={shards}",
                flush=True,
            )

    with FeatureCacheWriter(
        out_dir,
        shard_size=shard_size,
        identity=identity,
        build_options=build_options,
        only_bundles_fields=only_bundles_fields,
        on_checkpoint=checkpoint,
    ) as writer:
        build_options_changed = _build_options_diff(writer.build_options_initial, build_options)
        serialized_options_diff = _serialized_build_options_diff(build_options_changed)
        already_cached = writer.committed_entries
        committed_keys = writer.committed_keys
        remaining_samples = [
            sample
            for sample in plan.samples
            if (sample.document_bundle_id, sample.page_index) not in committed_keys
        ]
        remaining = len(remaining_samples)
        attempt_target, _ = resolve_limit(limit, remaining)
        if not _intent_emitted:
            print(
                _intent_line(
                    plan,
                    split=split,
                    task_kind=task_kind,
                    already_cached=already_cached,
                    remaining=remaining,
                    limit=limit,
                    build_options_changed=serialized_options_diff,
                ),
                flush=True,
            )
            if build_options_changed:
                print(
                    "[cache] WARNING: BUILD OPTIONS CHANGED "
                    f"{serialized_options_diff}; committed rows are retained; this cache is a "
                    "union of run selections; Ctrl-C to abort",
                    flush=True,
                )
        if plan.resume:
            print(
                f"[cache] resume validated: committed={already_cached}, "
                f"tail_repair={str(writer.repair_rows is not None).lower()}",
                flush=True,
            )
        if writer.repair_rows is not None:
            by_key = {
                (sample.document_bundle_id, sample.page_index): sample
                for sample in iter_page_samples(
                    release_dir,
                    store_dir,
                    split,
                    task_kind,
                    require_privacy_review=not allow_dev_only,
                )
            }
            repair_entries = []
            for key in writer.repair_rows:
                sample = by_key.get(key)
                if sample is None:
                    raise ValueError(f"tail repair source missing committed row {key!r}")
                page = prepare_page(
                    store_dir, sample, max_steps=None, capability_profile=capability_profile
                )
                if page.error is not None:
                    raise type(page.error)(f"tail repair {key!r}: {page.error}") from page.error
                if page.skip is not None:
                    raise ValueError(
                        f"tail repair source {key!r} is no longer cacheable: {page.skip}"
                    )
                repair_entries.append(
                    _cache_entry(backbone, page, at_merged=at_merged, use_multiscale=use_multiscale)
                )
            writer.restore_tail(repair_entries)
        pages = ordered_prefetch(
            remaining_samples,
            partial(
                prepare_pages,
                store_dir=store_dir,
                max_steps=max_steps,
                capability_profile=capability_profile,
            ),
            workers=workers,
            chunk=chunk,
        )
        for page in pages:
            if written >= attempt_target:
                break
            attempted += 1
            if page.error is not None:
                # Caught in the worker, raised here: only this loop knows
                # how far the build got, and the class is preserved so the
                # cause names itself without anyone reading a traceback.
                raise type(page.error)(
                    f"{page.document_bundle_id} page {page.page_index}, "
                    f"after {written} page(s) cached: {page.error}"
                ) from page.error
            for name, delta in page.counters.items():
                counts[name] += delta
            if page.skip is not None:
                counts[page.skip] += 1
                # T-9 (design §4.1): `<reason>_continuation` is a SUBSET of
                # `<reason>`, not a separate bucket -- both increment, so
                # the RUNBOOK's `skipped_grammar` baseline and
                # `capability_exclusion_agreement.py`'s skip-reason census
                # (P13) see the same total they always did, with the
                # continuation-caused share now also visible on its own.
                if page.has_continuation and page.skip in _CONTINUATION_ATTRIBUTABLE_SKIPS:
                    counts[f"{page.skip}_continuation"] += 1
                continue
            if exclude_capability_profile and _uses_excluded_capability(
                page.target_ids,
                page.input_ids,
                capability_profile=capability_profile or PRIMARY_S1_CAPABILITY_PROFILE,
            ):
                counts["skipped_capability_excluded"] += 1
                if page.has_continuation:
                    counts["skipped_capability_excluded_continuation"] += 1
                continue

            writer.add(
                _cache_entry(backbone, page, at_merged=at_merged, use_multiscale=use_multiscale)
            )
            written += 1
            if progress_every and written % progress_every == 0:
                print(f"[cache] {written}/{attempt_target} strán", flush=True)

    finished = datetime.now(UTC)
    total_after = writer.committed_entries
    tail_shard_recomputed = writer.tail_shard_recomputed
    tail_rows_recomputed = writer.tail_rows_recomputed
    return {
        "written": written,
        "split_samples": plan.split_samples,
        "cohort": plan.cohort,
        "remaining": remaining,
        "attempt_target": attempt_target,
        "limit": limit,
        "attempted": attempted,
        "already_cached": already_cached,
        "total_after": total_after,
        "tail_shard_recomputed": tail_shard_recomputed,
        "tail_rows_recomputed": tail_rows_recomputed,
        "build_options_changed": build_options_changed,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": time.monotonic() - started_clock,
        # `skipped_fully_unsupervised` is distinct from every other skip
        # on purpose: a page dropped there was marked task-ready by the
        # release, and a silent drop is how a corpus loses its hardest
        # pages without anyone noticing. The withdrawal counts are the
        # honest cost of not fabricating supervision.
        **counts,
        "out_dir": str(out_dir),
        "identity": asdict(identity),
        **(only_bundles_fields or {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.build_cache", description=__doc__
    )
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("store_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--backbone", default=None)
    parser.add_argument(
        "--tap",
        default=None,
        help="feature tap, for a backbone that exposes several "
        "(Granite Vision: 'siglip4' / 'projected8')",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--task-kind", default="structure_v0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--multiscale", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="processes preparing pages while the backbone runs (0: none). "
        "Page preparation costs ~4x the backbone forward, so a serial "
        "build leaves the GPU idle; this changes only what waits for "
        "what, not which pages are cached, their order, or their bytes.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=8,
        help="consecutive pages per worker task; keeps a document's pages "
        "in one process, where its sidecar and fragment index are cached",
    )
    parser.add_argument(
        "--merged-grid",
        action="store_true",
        help="cache the coarse merged-token grid (a quarter of the rows) "
        "instead of the patch grid. Independent of --multiscale: alone "
        "it caches the final layer on the coarse grid, so grid "
        "resolution and feature richness can be compared one at a time",
    )
    parser.add_argument(
        "--allow-dev-only",
        action="store_true",
        help="admit records whose privacy review is still open. RFC005 "
        "§9.2 blocks those from training: a cache built with this "
        "flag is dev/research inventory and the manifest says so.",
    )
    parser.add_argument(
        "--use-candidate-loader",
        action="store_true",
        help="extract features through candidate_loader.QwenCandidate (the "
        "full frozen M2/M3 dense candidate, DEC-007 D3's exact pin) "
        "instead of the probe's vision-tower-only backbone. This is "
        "the only path whose features are provably what "
        "QwenDenseCandidateModel would compute for the same pixels -- "
        "see candidate_backbone.py's module docstring. Incompatible "
        "with --backbone/--tap/--max-pixels/--multiscale/--merged-grid: "
        "the candidate track has no tap, no pixel-budget knob and no "
        "multiscale levels to select.",
    )
    parser.add_argument(
        "--single-page-only",
        action="store_true",
        help="restrict the cohort to single-page documents (page_count == 1) "
        "-- the S1 contract canary's cohort, DEC-007.",
    )
    parser.add_argument(
        "--exclude-capability-profile",
        action="store_true",
        help="skip (never cache) a page whose decoded structural stream "
        "uses a node type the chosen --capability-profile excludes (with "
        "no --capability-profile, that is PRIMARY_S1_CAPABILITY_PROFILE, "
        "e.g. inline_image, DEC-007 D1.1) -- checked after decoding, "
        "before the backbone forward, so an excluded page never spends "
        "a forward pass it cannot use.",
    )
    parser.add_argument(
        "--capability-profile",
        default=None,
        help="capability_profile.CapabilityProfile.profile_id to build "
        "under (e.g. 's2-typed-widened-v1', DEC-024). Absent (the default) "
        "reproduces the historical S1 feature build byte-for-byte: grammar "
        "stays the literal 'word-visual-v1' and --exclude-capability-profile "
        "checks against PRIMARY_S1_CAPABILITY_PROFILE. An explicit id "
        "derives the grammar from that profile's ast_profile instead (every "
        "ratified profile declares 'word-visual-v1.1') and, with "
        "--exclude-capability-profile, checks against that profile's own "
        "exclusion set -- no profile id ever maps to the historical grammar.",
    )
    parser.add_argument(
        "--only-bundles",
        type=Path,
        default=None,
        help="file of document_bundle_id (one per line, '#' comments and "
        "blank lines ignored) restricting the cohort to exactly those "
        "ids -- pins a cohort across two store revisions for an A/B, "
        "since the RFC003 split allocator can silently reshuffle "
        "val/test membership when one document changes.",
    )
    parser.add_argument(
        "--only-pages",
        type=Path,
        default=None,
        help="JSON file of [[document_bundle_id, page_index], ...] restricting "
        "the cohort to exactly those pages, filtered after enumeration in "
        "release order. Unlike --only-bundles, any requested page absent "
        "from the enumerated split refuses the whole build -- the list was "
        "computed over this exact release/split. Mutually exclusive with "
        "--only-bundles/--single-page-only.",
    )
    args = parser.parse_args(argv)

    capability_profile = (
        None
        if args.capability_profile is None
        else resolve_capability_profile_by_id(args.capability_profile)
    )
    only_bundles = load_only_bundles(args.only_bundles) if args.only_bundles is not None else None
    only_pages = load_only_pages(args.only_pages) if args.only_pages is not None else None
    if only_pages is not None and (only_bundles is not None or args.single_page_only):
        parser.error("--only-pages is mutually exclusive with --only-bundles/--single-page-only")
    if args.use_candidate_loader:
        conflicting = {
            "--backbone": args.backbone,
            "--tap": args.tap,
            "--max-pixels": args.max_pixels,
            "--multiscale": args.multiscale,
            "--merged-grid": args.merged_grid,
        }
        given = [flag for flag, value in conflicting.items() if value]
        if given:
            parser.error(f"--use-candidate-loader is incompatible with {', '.join(given)}")
    plan = _plan_build(
        args.release_dir,
        args.store_dir,
        args.out_dir,
        split=args.split,
        task_kind=args.task_kind,
        allow_dev_only=args.allow_dev_only,
        single_page_only=args.single_page_only,
        only_bundles=only_bundles,
        only_bundles_path=args.only_bundles,
        only_pages=only_pages,
        only_pages_path=args.only_pages,
    )
    planning_remaining = max(0, plan.cohort - plan.manifest_entries)
    # This validates an invalid user value before any model loader can run.
    resolve_limit(args.limit, planning_remaining)
    validate_planning_options(max_steps=args.max_steps, chunk=args.chunk, workers=args.workers)
    requested_build_options = _requested_build_options(
        single_page_only=args.single_page_only,
        exclude_capability_profile=args.exclude_capability_profile,
        max_steps=args.max_steps,
        workers=args.workers,
        chunk=args.chunk,
    )
    build_options_changed = _manifest_build_options_diff(args.out_dir, requested_build_options)
    serialized_options_diff = _serialized_build_options_diff(build_options_changed)
    print(
        _intent_line(
            plan,
            split=args.split,
            task_kind=args.task_kind,
            already_cached=plan.manifest_entries,
            remaining=planning_remaining,
            limit=args.limit,
            build_options_changed=serialized_options_diff,
        ),
        flush=True,
    )
    if build_options_changed:
        print(
            "[cache] WARNING: BUILD OPTIONS CHANGED "
            f"{serialized_options_diff}; committed rows are retained; this cache is a union "
            "of run selections; Ctrl-C to abort",
            flush=True,
        )
    if plan.resume:
        print(
            "[cache] resume identity validation pending backbone load; no cache data changed",
            flush=True,
        )

    if args.use_candidate_loader:
        from unflatten.model.candidate_backbone import load_candidate_vision_backbone

        backbone = load_candidate_vision_backbone(device=args.device)
        print(f"[cache] candidate backbone {backbone.model_id}", flush=True)
    else:
        from unflatten.probe.features import DEFAULT_BACKBONE, FrozenVisionBackbone

        model_id = args.backbone or DEFAULT_BACKBONE
        backbone = FrozenVisionBackbone(
            model_id, device=args.device, max_pixels=args.max_pixels, tap=args.tap
        )
        tap_tag = f" tap {backbone.tap}" if backbone.tap else ""
        print(
            f"[cache] backbone {model_id}{tap_tag}, dim {backbone.feature_dim}",
            flush=True,
        )
    summary = build(
        backbone,
        args.release_dir,
        args.store_dir,
        args.out_dir,
        split=args.split,
        task_kind=args.task_kind,
        limit=args.limit,
        max_steps=args.max_steps,
        shard_size=args.shard_size,
        max_pixels=args.max_pixels,
        use_multiscale=args.multiscale,
        at_merged=args.merged_grid,
        allow_dev_only=args.allow_dev_only,
        workers=args.workers,
        chunk=args.chunk,
        single_page_only=args.single_page_only,
        exclude_capability_profile=args.exclude_capability_profile,
        only_bundles=only_bundles,
        only_bundles_path=args.only_bundles,
        only_pages=only_pages,
        only_pages_path=args.only_pages,
        capability_profile=capability_profile,
        _plan=plan,
        _intent_emitted=True,
    )
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def _build_options_diff(
    initial: dict[str, Any], requested: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return requested operational drift against the immutable first-run anchor."""
    return {
        field: {"from": initial[field], "to": requested[field]}
        for field in sorted(requested)
        if initial[field] != requested[field]
    }


def _serialized_build_options_diff(diff: dict[str, dict[str, Any]]) -> str:
    """Serialize the audit diff deterministically for one-line logs."""
    return json.dumps(diff, sort_keys=True, separators=(",", ":")) if diff else "none"


def _requested_build_options(
    *,
    single_page_only: bool,
    exclude_capability_profile: bool,
    max_steps: int,
    workers: int,
    chunk: int,
) -> dict[str, Any]:
    return {
        "single_page_only": single_page_only,
        "exclude_capability_profile": exclude_capability_profile,
        "max_steps": max_steps,
        "workers": workers,
        "chunk": chunk,
    }


def _manifest_build_options_diff(
    out_dir: Path, requested: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Read the first-run audit anchor before the CLI loads a backbone."""
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    initial = json.loads(manifest_path.read_text()).get("build_options_initial")
    return _build_options_diff(initial, requested) if isinstance(initial, dict) else {}


def validate_planning_options(*, max_steps: int, chunk: int, workers: int) -> None:
    """Reject invalid CPU and target bounds before any model-dependent work."""
    if max_steps < 1:
        raise ValueError(f"max_steps must be a positive step count, got {max_steps}")
    if chunk < 1:
        raise ValueError(f"chunk must be a positive page count, got {chunk}")
    if workers < 0:
        raise ValueError(f"workers must be a non-negative process count, got {workers}")


def _intent_line(
    plan: _BuildPlan,
    *,
    split: str,
    task_kind: str,
    already_cached: int,
    remaining: int,
    limit: int | None,
    build_options_changed: str = "none",
) -> str:
    """Return the pre-load build intent, including audited option drift."""
    attempt_target, reason = resolve_limit(limit, remaining)
    prefix = (
        f"[cache] {split}/{task_kind}: {plan.split_samples} sample(s), "
        f"{plan.cohort} cohort page(s); already_cached={already_cached}, remaining={remaining}; "
    )
    if reason == "all":
        return (
            f"{prefix}writing all remaining {remaining} page(s); limit=all; "
            f"build_options_changed={build_options_changed}"
        )
    return (
        f"{prefix}stopping after at most {attempt_target} NEW written page(s) "
        f"(--limit {limit}); build_options_changed={build_options_changed}"
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
