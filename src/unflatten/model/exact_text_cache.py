"""Exact-text cache v1: writer/reader for `MaterializedSequence` rows
(design §8, §16 criterion 11).

A separate cache **family** from the structural feature cache
(`feature_cache.py`, `cache_version` `"0.2.0"`, no `cache_family` key at
all). Old rows there cannot reconstruct text/scalar authority, so there
is no auto-upgrade path -- a structural cache must be refused, loudly,
not silently read as a shorter or emptier exact-text one. The manifest
carries an explicit `cache_family` string so that refusal does not rely
on `cache_version` alone: two families could coincidentally pick the
same version string, and a missing/foreign `cache_family` is a clearer
signal than "wrong version" for the row shapes a cache like this expects.

**Admission is checked, not merely recorded** (design §8 "Privacy
admission binding", RFC003-ERRATA-001 E5). `ExactTextCacheIdentity`
carries `admission` because the cache rows may have been built with
`--allow-dev-only` over records whose privacy review is still open
(mirrors `build_cache.py:410-450`). A reader opened for a
gradient-bearing consumer states that intent explicitly via the
`gradient_bearing` constructor argument -- there is no default, because
defaulting it to `False` would make "forgot to pass it" silently
diagnostic-safe for a caller that actually intended to train. Reading
then refuses:

- a missing/null `admission` on the cache, always, whether or not the
  read is gradient-bearing -- an unlabelled cache can be dev/research
  inventory just as easily as a reviewed one, and nothing here may
  assume the safer label on its behalf (design §8 rule 5);
- `admission == "dev_only"` when `gradient_bearing=True` (rule 2).

This module does not implement the full `RunManifestBindingVerifier`
(`run_binding.py`, a separate task): it has no `s1_run_manifest` to read
`required_admission` from, and does not invent one. What it does own is
the narrower, cache-local half of the same refusal -- a `dev_only` or
unlabelled cache must never satisfy a gradient-bearing reader, with or
without a Run Manifest in the picture -- which is why `gradient_bearing`
is a parameter here at all, not left for the (still unbuilt) binding
verifier to be the only place that knows to ask.

**Scope is declared, and completeness is compared** (DEC-MP-4, design §9).
`ExactTextCacheIdentity` says what the rows are bound to and nothing about
which pages are present, so a cohort-restricted cache and a full one share
one identity. Every cache therefore carries a mandatory
:class:`ExactTextCacheScope` block -- refused at open by exact equality,
like `cache_version` -- naming its selection kind and hashing the
`(document_bundle_id, page_index)` pairs actually written. That answers
"what exists". It cannot answer "is this the cohort my run intends to
train on", because a hash of the contents agrees with itself whatever the
contents are; only a comparison against an independent expectation can,
which is the optional `expectation` argument: given the pinned `E` and
approved exclusions `X`, the reader becomes a view over exactly `E \\ X`
and refuses a cache that does not cover it. Keys in `C \\ E` stay
physically legal and unreachable as training examples.

Ships no `nn.Module`, no weights, no optimizer, and authorizes no
gradient update: only dataclasses, transactional shard I/O, and pure
verification functions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch

from unflatten.dataset.targets.properties import normalize_pointer
from unflatten.model.composite_grammar import CompositeGrammarController, replay
from unflatten.model.composite_vocab import CompositeVocabulary
from unflatten.model.exact_text import validate_byte_frame_scalars
from unflatten.model.loss_unit import (
    KNOWN_REASON_CODES,
    EffectiveLossMask,
    EventRangeLocator,
    Locator,
    LossUnit,
    Rfc003OwnershipProof,
    TextRangeLocator,
    compute_loss_unit_id,
    loss_unit_payload,
)
from unflatten.model.property_head_registry import (
    PropertyHeadRegistryError,
    head_for_pointer,
)
from unflatten.model.region_targets import (
    RegionTarget,
    region_target_from_payload,
    region_target_payload,
)
from unflatten.model.training_materializer import MaterializedSequence
from unflatten.model.typed_property_targets import (
    # `_UNION_DISCRIMINANT` and `_split_container_index` are imported
    # private on purpose: the discriminant name and the container split are
    # ONE mechanism, owned by the module that writes these records. A
    # second copy here would be a second rule, free to stop matching.
    _UNION_DISCRIMINANT,
    TypedPropertyTarget,
    TypedRepeatedCardinality,
    TypedUnionArm,
    _split_container_index,
    typed_target_from_payload,
    typed_target_payload,
)

#: Distinguishes this family from the structural feature cache
#: (`feature_cache.CACHE_VERSION`), whose manifest carries no such key.
CACHE_FAMILY = "exact_text_v1"

#: Cache format version; bumped when the row/manifest shape changes.
#: 1.1.0: `ExactTextCacheRow.grid_thw` added. Required, with no default:
#: `candidate_loader.QwenCandidate.inject_visual` needs the backbone's own
#: mRoPE positional index and it cannot be recovered from the stored
#: features (`feature_cache.CacheEntry.grid_thw`: dropping it "would make
#: native injection impossible downstream, not merely inconvenient"), so a
#: 1.0.0 row is refused when read rather than defaulted to a grid nobody
#: measured.
#: 2.0.0: `ExactTextCacheIdentity.head_registry_hash` added, required with
#: no default (DEC-008 addendum A); a 1.1.0 cache has no such field and is
#: refused rather than read as one with a null/unbound head registry.
#: 3.0.0: the typed-property channel (`typed_targets`,
#: `typed_cardinalities`, `typed_union_arms`) added. Bumping this alone
#: refused nothing until 3.0.0: the version was written into every
#: manifest and never compared, appearing only inside an error message.
#: `_check_family_and_completeness` now tests it for exact equality, at
#: open -- semver-major compatibility would invent an unratified policy,
#: and relying on the row-level refusal alone fires only on the first
#: shard READ, after the reader has already reported a valid-looking
#: N-entry cache.
#: 4.0.0: the mandatory DEC-MP-4 `scope` block and the per-shard `keys`
#: list. The missing-`scope` refusal already catches a 3.0.0 manifest, so
#: this bump is not what makes the two formats distinguishable at read
#: time -- it is what NAMES them apart: two incompatible manifest shapes
#: sharing one version string are two artifacts whose identity cannot be
#: told from the identity itself, which is the state this project's
#: identity discipline exists to prevent.
#: 5.0.0 (R-1, `docs/plans/NALEZ-velke-dokumenty-kvadraticka-cena.md`
#: review): `ExactTextCacheScope.single_page_only`/`.limit` added, and
#: `selection_kind` gained a third value, `"restricted"`. A 4.0.0 scope
#: recorded only `only_bundles` restriction; `single_page_only=True`
#: (the DEFAULT cohort filter) or `--limit` silently declared
#: `"full_split"` on disk for a cache that held neither the whole split
#: nor a named `only_bundles` cohort. A 4.0.0 manifest has neither new
#: key and is refused rather than read as unrestricted.
#: 6.0.0 (`spec-node-visual-aligner-design.md` T1): the `region_targets`
#: channel added -- RFC004 §21 raw region observations per `(node_id,
#: page)`, one record per fragment slice on the page. Required, with no
#: default (the typed-channel 3.0.0 precedent): a 5.0.0 row has no such
#: channel at all, and reading that as "a page with zero regions" would be
#: indistinguishable from a page that genuinely observed none of its
#: nodes -- exactly the distinction this channel exists to carry.
EXACT_TEXT_CACHE_VERSION = "6.0.0"

#: 7.0.0 (DEC-056 addendum A, v16): the page-image row -- no `features`,
#: a `page_image` ArtifactRef (the release's own pinned PNG, sha256 + size
#: verified on read) instead; the frozen ViT computes the visual features
#: during training. Written ONLY by a cache whose identity says so
#: (`is_page_image_identity`), so every feature cache keeps 6.0.0 and its
#: exact bytes, and a reader never has to guess which row shape a cache
#: holds: the version is chosen by the identity and checked against it.
EXACT_TEXT_CACHE_VERSION_PAGE_IMAGE = "7.0.0"

#: `ExactTextCacheIdentity.feature_tap` of a page-image cache:
#: `"onfly-visual:<visual_input_policy_hash>"`. The hash is
#: `onfly_visual.VisualInputPolicy.policy_hash()` (ViT revision, DeepStack
#: taps, compute dtype, processor artifact, pixel budget, patch geometry):
#: what replaces the feature-cache id as the thing the rows are bound to.
#: Carried in the existing `feature_tap` slot rather than a new identity
#: field, so `asdict(identity)` -- hashed by the run manifests, censuses
#: and aug stores -- is unchanged for every existing cache.
PAGE_IMAGE_FEATURE_TAP_PREFIX = "onfly-visual:"


def is_page_image_identity(identity: Any) -> bool:
    """Whether an identity (`ExactTextCacheIdentity` or its manifest dict)
    names a page-image (7.0.0) cache."""
    if isinstance(identity, Mapping):
        tap = identity.get("feature_tap")
    else:
        tap = identity.feature_tap
    return isinstance(tap, str) and tap.startswith(PAGE_IMAGE_FEATURE_TAP_PREFIX)


def expected_cache_version(identity: Any) -> str:
    """The one `cache_version` a cache with this identity may carry."""
    if is_page_image_identity(identity):
        return EXACT_TEXT_CACHE_VERSION_PAGE_IMAGE
    return EXACT_TEXT_CACHE_VERSION

#: Mirrors `feature_cache.DEFAULT_CACHE_SHARDS`: shards a reader keeps
#: resident between accesses.
DEFAULT_CACHE_SHARDS = 8

#: Mirrors `feature_cache.STAGING_DIR`.
STAGING_DIR = ".staging"

type Admission = Literal["strict", "dev_only"]

_ADMISSION_VALUES: frozenset[str] = frozenset({"strict", "dev_only"})


class ExactTextCacheIncompatible(RuntimeError):
    """A cache does not prove it is what a gradient-bearing (or diagnostic)
    reader requires: wrong family/version, incomplete/tampered manifest,
    a row that fails hash/count/shape/replay verification, or an
    admission that does not clear the boundary this reader was opened
    with (design §8, §9.3). M1-internal, never the unratified
    `EXACT_TEXT_CACHE_INCOMPATIBLE` Issue code (design §9.1, RR-3)."""


@dataclass(frozen=True, slots=True)
class ExactTextCacheIdentity:
    """What one exact-text cache's rows are bound to (design §8).

    Every field is required: an absent field on disk means the cache was
    built by something else (a foreign artifact schema) or by a partial
    write, and both must be refused rather than defaulted.
    """

    dataset_release_root_hash: str
    split: str
    task_kind: str
    admission: Admission
    target_projector_version: str
    target_bundle_schema_version: str
    fragment_schema_version: str
    mask_projection_schema_version: str
    backbone_artifact_hash: str
    processor_artifact_hash: str
    feature_tap: str
    max_pixels: int
    composite_vocabulary_hash: str
    tokenizer_artifact_set_hash: str
    qwen_text_allowlist_hash: str
    composite_grammar_hash: str
    capability_profile_hash: str
    training_target_materializer_hash: str
    loss_unit_schema_hash: str
    stage_policy_hash: str
    conditioning_policy_hash: str
    runtime_limits_policy_hash: str
    component_lock_hash: str
    head_registry_hash: str


@dataclass(frozen=True, slots=True)
class ExactTextCacheRow:
    """One page/fragment's materialized sequence plus its visual features
    and the identifiers that bind it to an RFC003 sample (design §8 shard
    row)."""

    document_bundle_id: str
    fragment_id: str | None
    page_index: int
    authority_target_hash: str
    mask_projection_hash: str
    #: float16 [V, D]; `None` exactly on a page-image (7.0.0) row, whose
    #: visual features are computed from `page_image` during training.
    features: torch.Tensor | None
    #: `(temporal, height, width)` PATCH grid (pre-merge), int64 `(3,)` --
    #: `feature_cache.CacheEntry.grid_thw` verbatim, copied from the same
    #: entry as `features`. Never derived: 875 merged tokens are consistent
    #: with `(1, 70, 50)` AND `(1, 50, 70)`, and both occur in the real
    #: candidate cache, so a grid inferred from the token count would be a
    #: transposed page nobody could detect.
    grid_thw: torch.Tensor  # int64 [3]
    packed_token_ids: torch.Tensor  # int64 [T]
    packed_input_ids: torch.Tensor  # int64 [T]
    effective_loss_mask: torch.Tensor  # bool [T]
    loss_family_ids: torch.Tensor  # uint8 [T]
    controller_mode_ids: torch.Tensor  # uint8 [T]
    structural_allowed_row_ptr: torch.Tensor  # int32 [T+1]
    structural_allowed_local_ids: torch.Tensor  # uint8 [N]
    loss_unit_index: torch.Tensor  # int32 [T]
    loss_units: tuple[LossUnit, ...]
    text_scalar_start: torch.Tensor  # int32 [T], -1 for non-text
    text_scalar_end: torch.Tensor  # int32 [T], -1 for non-text
    source_event_index: torch.Tensor  # int32 [T], -1 for protocol/non-event
    #: RFC004 §24/§26 typed property supervision for this page, flat:
    #: one target per assigned `(node_id, pointer)` leaf, one cardinality
    #: per repeated-container instance, one arm per union-container
    #: instance. Empty tuples mean "this page owns no typed target" -- a
    #: statement, which is why a shard that never carried the channel is
    #: refused rather than read as one (`_row_from_storage`).
    typed_targets: tuple[TypedPropertyTarget, ...]
    typed_cardinalities: tuple[TypedRepeatedCardinality, ...]
    typed_union_arms: tuple[TypedUnionArm, ...]
    #: RFC004 §21 raw region-observation supervision for this page
    #: (design T1): one record per fragment slice, `k` observed boxes each
    #: (0, 1 or many -- never unioned, never capped). Empty tuple means
    #: "this page owns no node with a slice at all", which cannot happen
    #: for a written row; a shard that never carried the channel is
    #: refused rather than read as one (`_row_from_storage`).
    region_targets: tuple[RegionTarget, ...]
    #: DEC-056: the page PNG's `ArtifactRef` dict (`ref`, `sha256`,
    #: `size_bytes`, `mime_type`) on a page-image row, `None` on a feature
    #: row. For a page-image row `grid_thw` is the grid the preprocessing
    #: policy derives from this PNG's size, and training refuses a processor
    #: output that disagrees with it.
    page_image: dict[str, Any] | None = None


def _checked_grid_thw(
    grid_thw: torch.Tensor | None, *, document_bundle_id: str, page_index: int
) -> torch.Tensor:
    """`grid_thw` as an int64 `(3,)` CPU tensor, or a refusal.

    Absence is refused, never defaulted: this is the one field the stored
    features cannot replace, so a row that reaches disk without it is a
    row `inject_visual` can never use.
    """
    if grid_thw is None:
        raise ExactTextCacheIncompatible(
            f"{document_bundle_id} page {page_index}: no grid_thw. It is the "
            "backbone's own mRoPE positional index, cannot be recovered from "
            "the stored features, and is therefore refused rather than "
            "defaulted (feature_cache.CacheEntry.grid_thw)"
        )
    tensor = grid_thw.detach().to(torch.int64).cpu()
    if tuple(tensor.shape) != (3,):
        raise ExactTextCacheIncompatible(
            f"{document_bundle_id} page {page_index}: grid_thw must be a (3,) "
            f"(temporal, height, width) tensor, got shape {tuple(tensor.shape)}"
        )
    return tensor


def row_from_materialized(
    sequence: MaterializedSequence,
    *,
    document_bundle_id: str,
    fragment_id: str | None,
    page_index: int,
    authority_target_hash: str,
    mask_projection_hash: str,
    features: torch.Tensor | None,
    grid_thw: torch.Tensor | None,
    typed_targets: tuple[TypedPropertyTarget, ...],
    typed_cardinalities: tuple[TypedRepeatedCardinality, ...],
    typed_union_arms: tuple[TypedUnionArm, ...],
    region_targets: tuple[RegionTarget, ...],
    page_image: dict[str, Any] | None = None,
) -> ExactTextCacheRow:
    """Build a shard row from a materializer's output plus page context.

    `grid_thw` is required and may not be `None`: it must be the
    `feature_cache.CacheEntry.grid_thw` of the SAME entry `features` was
    copied from (see the field's own comment on why it cannot be derived).

    The three typed arguments are REQUIRED, with no default: an empty
    typed channel is a statement ("this page owns no typed target") and a
    caller must make it, because a default would let a build that never
    derived the channel at all write rows nothing downstream can tell
    apart from pages that genuinely carry none.

    `region_targets` is REQUIRED, with no default, on the same precedent:
    a caller that never derived the region channel must not be able to
    write a row indistinguishable from one whose page genuinely has a
    slice for every node and zero observations for all of them.

    `page_image` (DEC-056): exactly one of `features`/`page_image` is
    given -- a page-image row carries the PNG reference instead of features.

    `text_scalar_start`/`text_scalar_end`/`source_event_index` are derived
    here, one token at a time, from each unpadded position's `LossUnit`
    locator -- `sequence.loss_units` holds exactly one entry per unpadded
    token, in order (`training_materializer.materialize`), so position
    `i < len(loss_units)` reads `loss_units[i].locator` directly and any
    padded tail gets the `-1` sentinel.
    """
    if (features is None) == (page_image is None):
        raise ExactTextCacheIncompatible(
            f"{document_bundle_id} page {page_index}: exactly one of features "
            "(feature row) and page_image (page-image row) must be given"
        )
    total_len = int(sequence.packed_token_ids.shape[0])
    scalar_start = [-1] * total_len
    scalar_end = [-1] * total_len
    source_event = [-1] * total_len
    for i, unit in enumerate(sequence.loss_units):
        if isinstance(unit.locator, TextRangeLocator):
            scalar_start[i] = unit.locator.scalar_start
            scalar_end[i] = unit.locator.scalar_end
        else:
            source_event[i] = unit.locator.event_start
    return ExactTextCacheRow(
        grid_thw=_checked_grid_thw(
            grid_thw, document_bundle_id=document_bundle_id, page_index=page_index
        ),
        document_bundle_id=document_bundle_id,
        fragment_id=fragment_id,
        page_index=page_index,
        authority_target_hash=authority_target_hash,
        mask_projection_hash=mask_projection_hash,
        features=None if features is None else features.detach().to(torch.float16).cpu(),
        packed_token_ids=sequence.packed_token_ids,
        packed_input_ids=sequence.packed_input_ids,
        effective_loss_mask=sequence.effective_loss_mask,
        loss_family_ids=sequence.loss_family_ids,
        controller_mode_ids=sequence.controller_mode_ids,
        structural_allowed_row_ptr=sequence.structural_allowed_row_ptr,
        structural_allowed_local_ids=sequence.structural_allowed_local_ids,
        loss_unit_index=sequence.loss_unit_index,
        loss_units=sequence.loss_units,
        text_scalar_start=torch.tensor(scalar_start, dtype=torch.int32),
        text_scalar_end=torch.tensor(scalar_end, dtype=torch.int32),
        source_event_index=torch.tensor(source_event, dtype=torch.int32),
        typed_targets=tuple(typed_targets),
        typed_cardinalities=tuple(typed_cardinalities),
        typed_union_arms=tuple(typed_union_arms),
        region_targets=tuple(region_targets),
        page_image=None if page_image is None else dict(page_image),
    )


# ---------------------------------------------------------------------------
# Scope and expectation (DEC-MP-4)
# ---------------------------------------------------------------------------

#: One page's identity inside a cache: `(document_bundle_id, page_index)`.
type PageKey = tuple[str, int]

#: How the cohort a cache was built over was chosen. Closed vocabulary: a
#: `selection_kind` this module does not know is refused rather than
#: recorded, because the whole point of the declaration is that a reader
#: can decide what it is looking at without asking a human.
#:
#: `"restricted"` (R-1, 5.0.0): the cohort was narrowed by
#: `single_page_only` and/or `--limit`, but not by a named
#: `only_bundles` list -- distinct from `"only_bundles"` because a reader
#: cannot reproduce this cohort from a bundle-id file, and distinct from
#: `"full_split"` because it is not the whole split.
SELECTION_KINDS: frozenset[str] = frozenset(
    {"full_split", "only_bundles", "restricted"}
)


@dataclass(frozen=True, slots=True)
class ExactTextCacheScope:
    """What cohort one cache actually holds (design §9, DEC-MP-4).

    `ExactTextCacheIdentity` says what the rows are bound to; it says
    nothing about *which* pages were written, so a cohort-restricted cache
    and a full one share one identity and were, until this block existed,
    indistinguishable to every automatic check.

    `page_key_set_sha256` is over the sorted `(document_bundle_id,
    page_index)` pairs **actually written** -- not over the requested
    cohort, which is what `only_bundles_sha256` records. The two answer
    different questions ("what was asked for" vs "what is here") and a
    build that silently skips pages makes them disagree.

    A content hash still only answers "what exists". Whether that is the
    cohort a run intends to train on is not decidable from here, and is
    what :class:`CacheExpectation` exists for.

    `single_page_only`/`limit` (R-1, 5.0.0) are the build's OTHER two
    cohort-narrowing knobs (`build_exact_text_cache.build`'s own
    parameters of the same name), recorded here for the same reason
    `only_bundles_sha256`/`only_bundles_count` are: a reader must be able
    to see every restriction that produced this cache from the manifest
    alone, never assume the default `single_page_only=True` or an absent
    `--limit` from `selection_kind` alone.
    """

    selection_kind: str
    only_bundles_sha256: str | None
    only_bundles_count: int | None
    page_key_set_sha256: str
    page_key_count: int
    single_page_only: bool = False
    limit: int | None = None


def page_key_set_sha256(keys: Iterable[PageKey]) -> str:
    """sha256 over the sorted `(document_bundle_id, page_index)` pairs.

    Sorted, so two callers holding the same pages in a different order
    agree (`build_cache.only_bundles_manifest_fields`'s convention);
    duplicates are kept, so a cache that wrote one page twice does not
    hash as though it had not.
    """
    ordered = sorted((str(bundle), int(page)) for bundle, page in keys)
    payload = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheExpectation:
    """The pinned cohort a gradient-bearing read must be given: the
    expected page key set `E` and the approved exclusions `X` (design §9
    item 2).

    Not a boolean on `ExactTextCacheIdentity` and not a registry: this is
    a payload frozen alongside the `s1_run_manifest`
    (`freeze_s1_run_manifest.freeze`) and bound to it by hash
    (`run_binding.compute_exact_text_cache_expectation_hash`), so that
    "which pages should this run see" is pinned in the same artifact and
    by the same mechanism as every other binding the run declares.

    `X` is checked to be a subset of `E`: exclusions may only remove pages
    somebody pinned, never absorb arbitrary failures to make a read pass.
    """

    expected_page_keys: tuple[PageKey, ...]
    approved_exclusions: tuple[PageKey, ...] = ()

    def payload(self) -> dict[str, Any]:
        """The hashable/serializable form -- lists of `[bundle, page]`,
        in the pinned order, because the order IS what the dataset yields."""
        return {
            "expected_page_keys": [[b, p] for b, p in self.expected_page_keys],
            "approved_exclusions": [[b, p] for b, p in self.approved_exclusions],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CacheExpectation:
        """Parse a frozen expectation payload, refusing anything that is
        not a list of two-element `[str, int]` pairs -- a malformed pin is
        refused, never coerced."""
        keys = _parse_page_keys(payload, "expected_page_keys")
        exclusions = _parse_page_keys(payload, "approved_exclusions")
        return cls(expected_page_keys=keys, approved_exclusions=exclusions)

    def target_keys(self) -> tuple[PageKey, ...]:
        """`E \\ X`, in `E`'s pinned order."""
        excluded = set(self.approved_exclusions)
        return tuple(key for key in self.expected_page_keys if key not in excluded)


def _parse_page_keys(payload: Mapping[str, Any], field: str) -> tuple[PageKey, ...]:
    raw = payload.get(field)
    if raw is None and field == "approved_exclusions":
        return ()
    if not isinstance(raw, list):
        raise ExactTextCacheIncompatible(
            f"expectation.{field} must be a list of [document_bundle_id, "
            f"page_index] pairs, got {type(raw).__name__}"
        )
    keys: list[PageKey] = []
    for item in raw:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], int)
            or isinstance(item[1], bool)
        ):
            raise ExactTextCacheIncompatible(
                f"expectation.{field} entry {item!r} is not a "
                "[document_bundle_id: str, page_index: int] pair"
            )
        keys.append((item[0], item[1]))
    return tuple(keys)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _recompute_loss_unit_id(payload: dict[str, Any]) -> str:
    """Recompute `loss_unit_id` via `loss_unit.compute_loss_unit_id`, so a
    mutated row (payload changed, id left stale) is detected -- and so
    this file carries no second copy of `loss_unit.py`'s domain string
    to fall out of sync with it."""
    payload_without_id = {k: v for k, v in payload.items() if k != "loss_unit_id"}
    return compute_loss_unit_id(payload_without_id)


def _locator_from_payload(payload: dict[str, Any]) -> Locator:
    if "text_span_id" in payload:
        return TextRangeLocator(
            text_span_id=payload["text_span_id"],
            scalar_start=payload["scalar_start"],
            scalar_end=payload["scalar_end"],
        )
    return EventRangeLocator(
        event_start=payload["event_start"], event_end=payload["event_end"]
    )


def _loss_unit_from_payload(payload: dict[str, Any]) -> LossUnit:
    proof = payload["ownership_proof"]
    mask = payload["effective_mask"]
    # `loss_unit.schema.json` leaves `reason_codes` as free-form strings
    # (D2 2026-09-07, `loss_unit.py`'s own `KNOWN_REASON_CODES` docstring),
    # so an unrecognized code round-trips the schema silently -- the one
    # place that registry has a runtime consumer at all is here, refusing
    # a row that carries a code no M1 code path is known to emit rather
    # than reading it through unchecked.
    for code in mask["reason_codes"]:
        if code not in KNOWN_REASON_CODES:
            raise ExactTextCacheIncompatible(
                f"loss_unit_id={payload['loss_unit_id']!r} carries unknown reason "
                f"code {code!r}, not in KNOWN_REASON_CODES {KNOWN_REASON_CODES!r}"
            )
    return LossUnit(
        loss_unit_id=payload["loss_unit_id"],
        loss_unit_kind=payload["loss_unit_kind"],
        output_locator=payload["output_locator"],
        attachment=payload["attachment"],
        loss_family=payload["loss_family"],
        stage_enabled=payload["stage_enabled"],
        owner_scope=payload["owner_scope"],
        owner_fragment_id=payload["owner_fragment_id"],
        locator=_locator_from_payload(payload["locator"]),
        authority_target_hash=payload["authority_target_hash"],
        projection_hash=payload["projection_hash"],
        mask_projection_hash=payload["mask_projection_hash"],
        ownership_proof=Rfc003OwnershipProof(
            document_bundle_id=proof["document_bundle_id"],
            target_bundle_hash=proof["target_bundle_hash"],
            fragment_id=proof["fragment_id"],
            fragment_schema_version=proof["fragment_schema_version"],
            fragment_sequence_index=proof["fragment_sequence_index"],
            node_id=proof["node_id"],
            structure_owner=proof["structure_owner"],
            typed_target_owner=proof["typed_target_owner"],
            text_range_owner=proof["text_range_owner"],
            page_evidence=proof["page_evidence"],
            owned_char_ranges=tuple(tuple(r) for r in proof["owned_char_ranges"]),
            ownership_projection_hash=proof["ownership_projection_hash"],
        ),
        effective_mask=EffectiveLossMask(
            mask_projection_hash=mask["mask_projection_hash"],
            known_mask=mask["known_mask"],
            observable_mask=mask["observable_mask"],
            supervision_mask=mask["supervision_mask"],
            ownership_enabled=mask["ownership_enabled"],
            stage_enabled=mask["stage_enabled"],
            effective=mask["effective"],
            reason_codes=tuple(mask["reason_codes"]),
        ),
        composite_vocabulary_hash=payload["composite_vocabulary_hash"],
        composite_grammar_hash=payload["composite_grammar_hash"],
        capability_profile_hash=payload["capability_profile_hash"],
        head_registry_hash=payload["head_registry_hash"],
    )


def _row_to_storage(row: ExactTextCacheRow) -> dict[str, Any]:
    # A feature row keeps exactly its pre-DEC-056 key set and order; a
    # page-image row stores `page_image` in the `features` slot's place.
    visual: dict[str, Any] = (
        {"features": row.features}
        if row.page_image is None
        else {"page_image": dict(row.page_image)}
    )
    return {
        "document_bundle_id": row.document_bundle_id,
        "fragment_id": row.fragment_id,
        "page_index": row.page_index,
        "authority_target_hash": row.authority_target_hash,
        "mask_projection_hash": row.mask_projection_hash,
        **visual,
        "grid_thw": row.grid_thw,
        "packed_token_ids": row.packed_token_ids,
        "packed_input_ids": row.packed_input_ids,
        "effective_loss_mask": row.effective_loss_mask,
        "loss_family_ids": row.loss_family_ids,
        "controller_mode_ids": row.controller_mode_ids,
        "structural_allowed_row_ptr": row.structural_allowed_row_ptr,
        "structural_allowed_local_ids": row.structural_allowed_local_ids,
        "loss_unit_index": row.loss_unit_index,
        "loss_units": [loss_unit_payload(unit) for unit in row.loss_units],
        "text_scalar_start": row.text_scalar_start,
        "text_scalar_end": row.text_scalar_end,
        "source_event_index": row.source_event_index,
        "typed_targets": [typed_target_payload(t) for t in row.typed_targets],
        "typed_cardinalities": [
            typed_target_payload(t) for t in row.typed_cardinalities
        ],
        "typed_union_arms": [typed_target_payload(t) for t in row.typed_union_arms],
        "region_targets": [region_target_payload(t) for t in row.region_targets],
    }


def _checked_typed_records(data: dict[str, Any], key: str) -> tuple[Any, ...]:
    """The stored typed records under `key`, or a refusal.

    Absence is refused, never defaulted to an empty tuple: an empty typed
    channel is a claim about the page ("it owns no typed target"), and a
    shard written before the channel existed makes no such claim.
    """
    records = data.get(key)
    if records is None:
        raise ExactTextCacheIncompatible(
            f"{data.get('document_bundle_id')} page {data.get('page_index')}: "
            f"the row carries no {key!r} channel. A cache built before "
            f"{EXACT_TEXT_CACHE_VERSION} has none, and reading that as a page "
            "with zero typed targets would be indistinguishable from a page "
            "that genuinely has none; it is refused, not defaulted"
        )
    return tuple(typed_target_from_payload(payload) for payload in records)


def _checked_region_records(data: dict[str, Any], key: str) -> tuple[RegionTarget, ...]:
    """The stored region records under `key`, or a refusal.

    Absence is refused, never defaulted to an empty tuple, on the
    `_checked_typed_records` precedent: an empty region channel is a
    claim about the page, and a shard written before the channel existed
    (cache_version < 6.0.0) makes no such claim.
    """
    records = data.get(key)
    if records is None:
        raise ExactTextCacheIncompatible(
            f"{data.get('document_bundle_id')} page {data.get('page_index')}: "
            f"the row carries no {key!r} channel. A cache built before "
            f"{EXACT_TEXT_CACHE_VERSION} has none, and reading that as a page "
            "with zero region targets would be indistinguishable from a page "
            "that genuinely has none; it is refused, not defaulted"
        )
    return tuple(region_target_from_payload(payload) for payload in records)


def _row_from_storage(data: dict[str, Any]) -> ExactTextCacheRow:
    return ExactTextCacheRow(
        document_bundle_id=data["document_bundle_id"],
        fragment_id=data["fragment_id"],
        page_index=data["page_index"],
        authority_target_hash=data["authority_target_hash"],
        mask_projection_hash=data["mask_projection_hash"],
        # A page-image row (7.0.0) has `page_image` and no `features`; a
        # feature row must carry `features` (KeyError otherwise, as before).
        features=None if "page_image" in data else data["features"],
        page_image=data.get("page_image"),
        # `.get`, then refused: a 1.0.0 shard has no `grid_thw` key at all
        # and a 1.1.0 one must never carry a null. Both are the same
        # unusable row, and both are named rather than raising KeyError or
        # round-tripping a `None` a consumer would only discover at
        # `inject_visual` time.
        grid_thw=_checked_grid_thw(
            data.get("grid_thw"),
            document_bundle_id=data["document_bundle_id"],
            page_index=data["page_index"],
        ),
        packed_token_ids=data["packed_token_ids"],
        packed_input_ids=data["packed_input_ids"],
        effective_loss_mask=data["effective_loss_mask"],
        loss_family_ids=data["loss_family_ids"],
        controller_mode_ids=data["controller_mode_ids"],
        structural_allowed_row_ptr=data["structural_allowed_row_ptr"],
        structural_allowed_local_ids=data["structural_allowed_local_ids"],
        loss_unit_index=data["loss_unit_index"],
        loss_units=tuple(_loss_unit_from_payload(p) for p in data["loss_units"]),
        text_scalar_start=data["text_scalar_start"],
        text_scalar_end=data["text_scalar_end"],
        source_event_index=data["source_event_index"],
        # `.get`, then refused, on the `_checked_grid_thw` precedent: a
        # 2.0.0 shard has no typed channel at all, and reading it as "a
        # page with zero typed targets" is indistinguishable from a page
        # that genuinely has none -- which is the whole distinction this
        # channel exists to carry.
        typed_targets=_checked_typed_records(data, "typed_targets"),
        typed_cardinalities=_checked_typed_records(data, "typed_cardinalities"),
        typed_union_arms=_checked_typed_records(data, "typed_union_arms"),
        region_targets=_checked_region_records(data, "region_targets"),
    )


def _verify_page_image_ref(row: ExactTextCacheRow) -> None:
    """A page-image row's `page_image` must be a well-formed PNG
    `ArtifactRef` -- the store read (`read_object`) is what verifies the
    bytes against it."""
    ref = row.page_image or {}
    sha = ref.get("sha256")
    well_formed = (
        isinstance(ref.get("ref"), str)
        and isinstance(sha, str)
        and sha.startswith("sha256:")
        and len(sha) == len("sha256:") + 64
        and isinstance(ref.get("size_bytes"), int)
        and ref.get("mime_type") == "image/png"
    )
    if not well_formed:
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: page_image {ref!r} is "
            "not a PNG ArtifactRef (ref, sha256, size_bytes, mime_type image/png)"
        )


def _require_row_kind_matches(row: ExactTextCacheRow, identity: Any) -> None:
    """A page-image identity holds only page-image rows and a feature
    identity only feature rows -- a mixed cache would hand the trainer a
    row it cannot feed."""
    if (row.page_image is not None) != is_page_image_identity(identity):
        kind = "page-image" if row.page_image is not None else "feature"
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: a {kind} row in a cache "
            "whose identity says otherwise (feature_tap decides the row kind)"
        )


def _verify_row_integrity(row: ExactTextCacheRow) -> None:
    """Shape, hash and ownership-cross-link checks (design §8, §16
    criterion 11's "valid hashes, counts, shapes"). Raises
    `ExactTextCacheIncompatible` naming the first mismatch found.

    Deliberately checks the loss-unit-side hashes/links rather than only
    the tensors: a row with intact tensors but a hand-edited `LossUnit`
    payload (the mutation this exists to catch) has every tensor shape
    still correct.
    """
    total_len = int(row.packed_token_ids.shape[0])
    for name, tensor in (
        ("packed_input_ids", row.packed_input_ids),
        ("effective_loss_mask", row.effective_loss_mask),
        ("loss_family_ids", row.loss_family_ids),
        ("controller_mode_ids", row.controller_mode_ids),
        ("loss_unit_index", row.loss_unit_index),
        ("text_scalar_start", row.text_scalar_start),
        ("text_scalar_end", row.text_scalar_end),
        ("source_event_index", row.source_event_index),
    ):
        if int(tensor.shape[0]) != total_len:
            raise ExactTextCacheIncompatible(
                f"{row.document_bundle_id} page {row.page_index}: {name} has "
                f"length {tensor.shape[0]}, packed_token_ids has {total_len}"
            )
    row_ptr = row.structural_allowed_row_ptr
    if int(row_ptr.shape[0]) != total_len + 1:
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: "
            f"structural_allowed_row_ptr has length {row_ptr.shape[0]}, "
            f"expected {total_len + 1}"
        )
    if int(row_ptr[0]) != 0 or int(row_ptr[-1]) != int(
        row.structural_allowed_local_ids.shape[0]
    ):
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: "
            "structural_allowed_row_ptr does not bound "
            "structural_allowed_local_ids"
        )
    if bool((row_ptr[1:] < row_ptr[:-1]).any()):
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: "
            "structural_allowed_row_ptr is not monotonic"
        )

    grid = row.grid_thw
    if tuple(grid.shape) != (3,) or grid.dtype != torch.int64:
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: grid_thw must be "
            f"an int64 (3,) tensor, got {row.grid_thw.dtype} {tuple(grid.shape)}"
        )
    if bool((grid < 1).any()):
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: grid_thw "
            f"{grid.tolist()} has a non-positive axis; it is a patch grid, not "
            "an offset"
        )
    temporal, height, width = (int(v) for v in grid)
    if (row.features is None) == (row.page_image is None):
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: a row carries exactly "
            "one of features and page_image"
        )
    visual_rows: int | None = None
    if row.page_image is not None:
        # The page-image row's grid is checked against the processor output
        # at training time (`onfly_visual`); here only its reference shape.
        _verify_page_image_ref(row)
    else:
        visual_rows = int(row.features.shape[0])
    if visual_rows is not None and not any(
        height % merge == 0
        and width % merge == 0
        and temporal * (height // merge) * (width // merge) == visual_rows
        for merge in (1, 2, 4)
    ):
        # Secondary, deliberately: the primary guarantee is provenance --
        # `build_exact_text_cache` copies `features` and `grid_thw` out of
        # ONE `CacheEntry`, so they cannot be paired across pages there.
        # This catches a row whose grid was later replaced by a page with a
        # different visual-token count (the real 150-page candidate cache
        # holds five distinct `(grid, V)` pairs). It CANNOT catch a
        # transposed grid of equal count -- `(1, 70, 50)` and `(1, 50, 70)`
        # both yield 875 rows and both occur in that cache -- which is why
        # provenance, not this check, is what the builder relies on.
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: grid_thw "
            f"{grid.tolist()} cannot produce the {visual_rows} visual row(s) "
            "this row stores at any merge size; the grid describes a "
            "different page than the features"
        )

    seq_len = len(row.loss_units)
    index = row.loss_unit_index
    if seq_len > total_len:
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: {seq_len} "
            f"loss_units exceeds the {total_len}-token row"
        )
    expected_index = list(range(seq_len)) + [-1] * (total_len - seq_len)
    if index.tolist() != expected_index:
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: "
            "loss_unit_index is not the contiguous [0, seq_len) prefix "
            "of -1-padded positions one LossUnit per unpadded token requires"
        )

    for i, unit in enumerate(row.loss_units):
        payload = loss_unit_payload(unit)
        recomputed = _recompute_loss_unit_id(payload)
        if recomputed != unit.loss_unit_id:
            raise ExactTextCacheIncompatible(
                f"{row.document_bundle_id} page {row.page_index}: "
                f"loss_units[{i}].loss_unit_id {unit.loss_unit_id!r} does not "
                f"match its own payload (recomputed {recomputed!r}); the row "
                "was mutated after it was built"
            )
        proof = unit.ownership_proof
        if proof.document_bundle_id != row.document_bundle_id:
            raise ExactTextCacheIncompatible(
                f"{row.document_bundle_id} page {row.page_index}: "
                f"loss_units[{i}] ownership_proof.document_bundle_id "
                f"{proof.document_bundle_id!r} does not match the row"
            )
        if proof.fragment_id != row.fragment_id:
            raise ExactTextCacheIncompatible(
                f"{row.document_bundle_id} page {row.page_index}: "
                f"loss_units[{i}] ownership_proof.fragment_id "
                f"{proof.fragment_id!r} does not match the row's "
                f"{row.fragment_id!r}"
            )
        if unit.mask_projection_hash != row.mask_projection_hash:
            raise ExactTextCacheIncompatible(
                f"{row.document_bundle_id} page {row.page_index}: "
                f"loss_units[{i}].mask_projection_hash does not match the row"
            )
        if isinstance(unit.locator, TextRangeLocator):
            if (
                int(row.text_scalar_start[i]) != unit.locator.scalar_start
                or int(row.text_scalar_end[i]) != unit.locator.scalar_end
                or int(row.source_event_index[i]) != -1
            ):
                raise ExactTextCacheIncompatible(
                    f"{row.document_bundle_id} page {row.page_index}: "
                    f"text_scalar_start/end or source_event_index at position "
                    f"{i} disagrees with loss_units[{i}]'s TextRangeLocator"
                )
        else:
            if (
                int(row.source_event_index[i]) != unit.locator.event_start
                or int(row.text_scalar_start[i]) != -1
                or int(row.text_scalar_end[i]) != -1
            ):
                raise ExactTextCacheIncompatible(
                    f"{row.document_bundle_id} page {row.page_index}: "
                    f"source_event_index or text_scalar_start/end at position "
                    f"{i} disagrees with loss_units[{i}]'s EventRangeLocator"
                )
    for i in range(seq_len, total_len):
        if (
            int(row.text_scalar_start[i]) != -1
            or int(row.text_scalar_end[i]) != -1
            or int(row.source_event_index[i]) != -1
        ):
            raise ExactTextCacheIncompatible(
                f"{row.document_bundle_id} page {row.page_index}: padded "
                f"position {i} does not carry the -1 sentinel"
            )
    _verify_typed_channel(row)


def _typed_prefix(row: ExactTextCacheRow, record: Any) -> str:
    return (
        f"{row.document_bundle_id} page {row.page_index}: node "
        f"{record.node_id!r}"
    )


def _verify_typed_anchor(row: ExactTextCacheRow, record: Any) -> None:
    """Check 1: the anchor is a token of THIS row that THIS node owns.

    Raised, never clamped: an out-of-range anchor is not a near miss to be
    pulled back into range, and an in-range one belonging to another node
    is what a "nearest structural token" rule would produce -- a typed
    property supervised at the wrong node's step.
    """
    index = record.anchor_token_index
    units = row.loss_units
    if not 0 <= index < len(units):
        raise ExactTextCacheIncompatible(
            f"{_typed_prefix(row, record)} has anchor_token_index {index}, "
            f"outside [0, {len(units)}) -- refused, never clamped into range"
        )
    owner = units[index].ownership_proof.node_id
    if owner != record.node_id:
        raise ExactTextCacheIncompatible(
            f"{_typed_prefix(row, record)} has anchor_token_index {index}, "
            f"whose loss unit is owned by node {owner!r}. The anchor must be "
            "the node's OWN opening structural event, never a neighbour"
        )


def _verify_typed_masks(row: ExactTextCacheRow, record: Any) -> None:
    """Check 3: `supervision ⇒ known ∧ observable` (RFC003 §29.3)."""
    if record.supervision_mask and not (
        record.known_mask and record.observable_mask
    ):
        raise ExactTextCacheIncompatible(
            f"{_typed_prefix(row, record)} carries supervision_mask=True with "
            f"known_mask={record.known_mask} and "
            f"observable_mask={record.observable_mask} -- RFC003 §29.3 requires "
            "supervision_mask to imply both"
        )


def _verify_typed_target(row: ExactTextCacheRow, target: TypedPropertyTarget) -> None:
    """Checks 1-4 for one leaf target."""
    _verify_typed_anchor(row, target)
    _verify_typed_masks(row, target)
    pointer = normalize_pointer(target.node_property_pointer)
    try:
        derived = head_for_pointer(target.node_type, pointer)
    except PropertyHeadRegistryError as error:
        # Check 2, the unknown-pointer half: a stored head_id the registry
        # cannot re-derive is registry drift, not a pointer to route around.
        raise ExactTextCacheIncompatible(
            f"{_typed_prefix(row, target)} pointer "
            f"{target.node_property_pointer!r} on {target.node_type!r} has no "
            f"head in the property head registry: {error}"
        ) from error
    if derived != target.head_id:
        raise ExactTextCacheIncompatible(
            f"{_typed_prefix(row, target)} pointer "
            f"{target.node_property_pointer!r} stores head_id "
            f"{target.head_id!r}, but the registry derives {derived!r} -- the "
            "stored value is verified against the registry, never trusted"
        )
    # Check 4, in ONE direction only. The converse (`value is None ⇒ not
    # known`) is deliberately not asserted: `paragraph:/style/borders =
    # null` is a real "no borders" label, and asserting it would refuse
    # every node without borders.
    if not target.known_mask and target.value is not None:
        raise ExactTextCacheIncompatible(
            f"{_typed_prefix(row, target)} pointer "
            f"{target.node_property_pointer!r} has known_mask=False but "
            f"carries the value {target.value!r}; an unknown label is null"
        )


def _verify_typed_cardinality(
    row: ExactTextCacheRow, record: TypedRepeatedCardinality
) -> None:
    """Check 5: `count` against the indices the row's own leaves carry.

    The stored indices must be exactly `{0..count-1}`. A gap is refused,
    never filled and never re-read as `max(index) + 1` -- a hole in a tab
    stop or column list is a label this build cannot state.
    """
    _verify_typed_anchor(row, record)
    _verify_typed_masks(row, record)
    indices = set()
    for target in row.typed_targets:
        if target.node_id != record.node_id:
            continue
        split = _split_container_index(target.node_property_pointer)
        if split is not None and split[0] == record.container_pointer:
            indices.add(split[1])
    if sorted(indices) != list(range(record.count)):
        raise ExactTextCacheIncompatible(
            f"{_typed_prefix(row, record)} container "
            f"{record.container_pointer!r} stores count={record.count} against "
            f"the stored leaf indices {sorted(indices)}; a repeated container's "
            "indices must be exactly {0..count-1}"
        )


def _verify_typed_union_arm(row: ExactTextCacheRow, record: TypedUnionArm) -> None:
    """Check 6: the arm is in its domain and IS the stored mode value."""
    _verify_typed_anchor(row, record)
    _verify_typed_masks(row, record)
    if record.arm not in record.arm_domain:
        raise ExactTextCacheIncompatible(
            f"{_typed_prefix(row, record)} union {record.container_pointer!r} "
            f"stores the arm {record.arm!r}, outside its arm_domain "
            f"{record.arm_domain}"
        )
    pointer = f"{record.container_pointer}/{_UNION_DISCRIMINANT}"
    mode = next(
        (
            target
            for target in row.typed_targets
            if target.node_id == record.node_id
            and target.node_property_pointer == pointer
        ),
        None,
    )
    if mode is None:
        raise ExactTextCacheIncompatible(
            f"{_typed_prefix(row, record)} union {record.container_pointer!r} "
            f"has no stored {pointer!r} target to name its arm; a union never "
            "gets a default arm"
        )
    if mode.value != record.arm:
        raise ExactTextCacheIncompatible(
            f"{_typed_prefix(row, record)} union {record.container_pointer!r} "
            f"stores the arm {record.arm!r} against a {pointer!r} value of "
            f"{mode.value!r}; the arm IS the mode"
        )


def _verify_typed_channel(row: ExactTextCacheRow) -> None:
    """The six typed-channel checks (design §3.5), each failable on a
    hand-edited row."""
    for target in row.typed_targets:
        _verify_typed_target(row, target)
    for cardinality in row.typed_cardinalities:
        _verify_typed_cardinality(row, cardinality)
    for arm in row.typed_union_arms:
        _verify_typed_union_arm(row, arm)


def verify_row_replay(
    row: ExactTextCacheRow,
    *,
    vocabulary: CompositeVocabulary,
    controller: CompositeGrammarController,
    capability_profile_id: str,
) -> None:
    """Design §8 reader check: "every ID unpacks; every stream replays
    under bound controller/capability". Replays only the unpadded prefix
    (`len(row.loss_units)` tokens) -- the padded tail is filled with a
    repeated BOS id (`training_materializer.materialize`), which is not a
    legal continuation after the controller reaches `"terminated"` and is
    not meant to be one.

    Also applies `exact_text.validate_byte_frame_scalars` to the unpacked
    stream: `replay` proves grammar/UTF-8-well-formedness, not Run-control
    policy -- a byte frame carrying U+0009/000A/000D is grammar-legal and
    valid UTF-8, but `encode_text` never emits it. Without this check a
    mutated or foreign cache row could carry one and pass this reader
    while `exact_text.decode_tokens` would refuse the same text.
    """
    seq_len = len(row.loss_units)
    prefix = row.packed_token_ids[:seq_len].tolist()
    try:
        tokens = [vocabulary.unpack(int(packed_id)) for packed_id in prefix]
    except Exception as exc:  # noqa: BLE001 - re-raised typed, with row context
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: a packed id "
            f"does not unpack under the bound vocabulary: {exc}"
        ) from exc
    try:
        replay(controller, capability_profile_id, tokens)
    except Exception as exc:  # noqa: BLE001 - re-raised typed, with row context
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: the token "
            f"stream does not replay under the bound controller/capability: "
            f"{exc}"
        ) from exc
    try:
        validate_byte_frame_scalars(tokens)
    except Exception as exc:  # noqa: BLE001 - re-raised typed, with row context
        raise ExactTextCacheIncompatible(
            f"{row.document_bundle_id} page {row.page_index}: a byte frame "
            f"decodes to a lone surrogate or illegal Run control scalar: {exc}"
        ) from exc


def assemble_shard(
    staging_dir: Path | str, ordinal: int, positions: list[int]
) -> tuple[str, int, str]:
    """Assemble one shard from its worker-written row files -- the
    `--workers` path, out of process.

    Each materialize worker wrote its page's storage dict to
    `staging/rows/row-<cohort_position>.pt`; this loads the listed
    positions in order, saves them as `staging/shard-<ordinal>.pt`,
    hashes the shard, and deletes the row files. Returns
    `(name, count, sha256)` for the parent's `ExactTextCacheWriter.
    commit_shard`.

    The PARENT commits, not this function: the manifest's shard order
    must be the parent's cohort order, and a pool worker committing
    directly would race both the manifest and the sibling shards.
    """
    rows_dir = Path(staging_dir) / "rows"
    rows: list[dict[str, Any]] = []
    for position in positions:
        path = rows_dir / f"row-{position:06d}.pt"
        (row,) = torch.load(path, weights_only=True)
        rows.append(row)
        path.unlink()
    name = f"shard-{ordinal:05d}.pt"
    shard_path = Path(staging_dir) / name
    torch.save(rows, shard_path)
    return name, len(rows), _sha256(shard_path)


#: The manifest's own fixed schema keys (`_manifest`, below) -- refused by
#: `ExactTextCacheWriter.record_summary` so a caller cannot silently
#: overwrite e.g. `identity` under a name that looks like extra summary
#: data (D2, 2026-09-07).
_FIXED_MANIFEST_KEYS = frozenset(
    {
        "cache_family",
        "cache_version",
        "status",
        "entries",
        "shard_entries",
        "shards",
        "shard_size",
        "shard_index",
        "identity",
        "scope",
    }
)


class ExactTextCacheWriter:
    """Write `ExactTextCacheRow`s into bounded shards, transactionally.

    Full shards are published with an atomic ``in_progress`` manifest so an
    interrupted build can resume without recomputing them. Readers still
    require ``status: "complete"``; only this writer consumes checkpoints.
    """

    def __init__(
        self,
        out_dir: Path | str,
        *,
        identity: ExactTextCacheIdentity,
        selection_kind: str,
        only_bundles_sha256: str | None = None,
        only_bundles_count: int | None = None,
        single_page_only: bool = False,
        limit: int | None = None,
        shard_size: int = 32,
        checkpoint_every_shards: int = 1,
        extra_manifest: dict[str, Any] | None = None,
    ) -> None:
        if (
            not isinstance(checkpoint_every_shards, int)
            or isinstance(checkpoint_every_shards, bool)
            or checkpoint_every_shards < 1
        ):
            raise ValueError("checkpoint_every_shards must be a positive integer")
        self.checkpoint_every_shards = checkpoint_every_shards
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.staging = self.out_dir / STAGING_DIR
        self.identity = identity
        #: DEC-MP-4: the scope declaration is MANDATORY, so `selection_kind`
        #: has no default -- a writer that could omit it would go on
        #: producing caches indistinguishable from a full one, which is the
        #: whole defect this block closes.
        if selection_kind not in SELECTION_KINDS:
            raise ValueError(
                f"selection_kind {selection_kind!r} is not one of "
                f"{sorted(SELECTION_KINDS)} -- an unknown scope kind is refused, "
                "not recorded"
            )
        only_bundles_restricted = (
            only_bundles_sha256 is not None or only_bundles_count is not None
        )
        if selection_kind == "only_bundles" and not (
            only_bundles_sha256 is not None and only_bundles_count is not None
        ):
            raise ValueError(
                "selection_kind='only_bundles' requires both only_bundles_sha256 "
                f"and only_bundles_count, got {only_bundles_sha256!r} / "
                f"{only_bundles_count!r} -- a restricted cohort that cannot name "
                "what restricted it is not a declaration"
            )
        if selection_kind != "only_bundles" and only_bundles_restricted:
            raise ValueError(
                f"selection_kind={selection_kind!r} must carry no "
                "only_bundles_sha256 / only_bundles_count: a cache that names a "
                "restricting cohort is not full_split or restricted"
            )
        # R-1 (5.0.0): `single_page_only`/`limit` are the other two
        # cohort-narrowing knobs -- `"full_split"` claims neither applies,
        # `"restricted"` requires at least one of them to, so the two
        # readings can never both describe the same build.
        narrowed = single_page_only or limit is not None
        if selection_kind == "full_split" and narrowed:
            raise ValueError(
                "selection_kind='full_split' must carry single_page_only=False "
                f"and limit=None, got single_page_only={single_page_only!r} / "
                f"limit={limit!r} -- a cache built over a narrowed cohort is "
                "not a full split"
            )
        if selection_kind == "restricted" and not narrowed:
            raise ValueError(
                "selection_kind='restricted' requires single_page_only=True "
                "and/or a limit -- a restriction that names nothing that "
                "restricted it is not a declaration"
            )
        self.selection_kind = selection_kind
        self.only_bundles_sha256 = only_bundles_sha256
        self.only_bundles_count = only_bundles_count
        self.single_page_only = single_page_only
        self.limit = limit
        #: `(document_bundle_id, page_index)` per written row, in write
        #: order -- the pairs ACTUALLY written, which is what `scope`
        #: hashes. Never the requested cohort.
        self._keys: list[PageKey] = []
        self._shard_keys: list[PageKey] = []
        self.shard_size = shard_size
        #: Top-level manifest keys beyond the fixed schema below -- e.g.
        #: `only_bundles_sha256`/`only_bundles_count`
        #: (build_exact_text_cache.py's `--only-bundles`), so a
        #: cohort-restricted cache can never pass as a full one. Absent by
        #: default: no caller sees a new key.
        self.extra_manifest = dict(extra_manifest) if extra_manifest else {}
        self.entries = 0
        self._buffer: list[dict[str, Any]] = []
        self._shards: list[dict[str, Any]] = []
        #: `commit_shard` guard: once a short shard is committed, nothing
        #: else may follow -- the reader's positional arithmetic requires
        #: every non-last shard to be full.
        self._partial_last = False
        self._closed = False
        self._resumed = False

        manifest_path = self.out_dir / "manifest.json"
        if manifest_path.exists():
            self._resume(manifest_path)
        elif any(self.out_dir.glob("shard-*.pt")):
            raise ExactTextCacheIncompatible(
                f"{self.out_dir}: published shards exist without a manifest; "
                "resume refuses to guess their identity"
            )
        if self.staging.exists():
            if self.staging.is_symlink():
                raise ExactTextCacheIncompatible(
                    f"{self.staging}: staging directory is a symlink"
                )
            shutil.rmtree(self.staging)
        self.staging.mkdir()

    @property
    def resumed(self) -> bool:
        return self._resumed

    @property
    def committed_entries(self) -> int:
        return sum(int(shard["count"]) for shard in self._shards)

    @property
    def committed_shards(self) -> int:
        return len(self._shards)

    @property
    def committed_metadata(self) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = []
        for shard in self._shards:
            rows = torch.load(self.out_dir / shard["name"], weights_only=True)
            for row in rows:
                metadata = row.get("_build_metadata")
                if not isinstance(metadata, dict):
                    raise ExactTextCacheIncompatible(
                        f"{shard['name']}: committed row has no resume metadata"
                    )
                records.append(metadata)
        return tuple(records)

    def _resume(self, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("cache_family") != CACHE_FAMILY:
            raise ExactTextCacheIncompatible("resume refused: cache_family mismatch")
        if manifest.get("cache_version") != expected_cache_version(self.identity):
            raise ExactTextCacheIncompatible("resume refused: cache_version mismatch")
        if manifest.get("status") not in {"in_progress", "complete"}:
            raise ExactTextCacheIncompatible("resume refused: invalid manifest status")
        if manifest.get("identity") != asdict(self.identity):
            raise ExactTextCacheIncompatible("resume refused: exact-text cache identity mismatch")
        if manifest.get("shard_size") != self.shard_size:
            raise ExactTextCacheIncompatible("resume refused: shard_size mismatch")
        requested_scope = {
            "selection_kind": self.selection_kind,
            "only_bundles_sha256": self.only_bundles_sha256,
            "only_bundles_count": self.only_bundles_count,
            "single_page_only": self.single_page_only,
            "limit": self.limit,
        }
        stored_scope = manifest.get("scope") or {}
        for key, value in requested_scope.items():
            if stored_scope.get(key) != value:
                raise ExactTextCacheIncompatible(
                    f"resume refused: scope.{key} mismatch: "
                    f"stored={stored_scope.get(key)!r}, requested={value!r}"
                )
        for key, value in self.extra_manifest.items():
            if key != "build_progress" and manifest.get(key) != value:
                raise ExactTextCacheIncompatible(
                    f"resume refused: manifest field {key!r} mismatch"
                )
        shards = manifest.get("shard_index")
        if not isinstance(shards, list):
            raise ExactTextCacheIncompatible("resume refused: shard_index is not a list")
        keys: list[PageKey] = []
        for shard in shards:
            path = self.out_dir / shard["name"]
            if not path.is_file() or _sha256(path) != shard["sha256"]:
                raise ExactTextCacheIncompatible(
                    f"resume refused: committed shard {shard.get('name')!r} is missing or corrupt"
                )
            rows = torch.load(path, weights_only=True)
            shard_keys = [(row["document_bundle_id"], row["page_index"]) for row in rows]
            expected = [tuple(key) for key in shard.get("keys", ())]
            if len(rows) != shard.get("count") or shard_keys != expected:
                raise ExactTextCacheIncompatible(
                    f"resume refused: committed shard {shard['name']!r} row index mismatch"
                )
            keys.extend(shard_keys)
        self._shards = [dict(shard) for shard in shards]
        self._keys = keys
        self.entries = len(keys)
        self.extra_manifest = {
            key: value for key, value in manifest.items() if key not in _FIXED_MANIFEST_KEYS
        }
        self._resumed = True

    def add(self, row: ExactTextCacheRow, *, build_metadata: dict[str, Any] | None = None) -> None:
        """Append one row. Verified before buffering: a row that fails
        its own shape/hash checks must never reach disk looking valid."""
        if self._partial_last:
            raise ValueError(
                "add() after a committed partial shard would put a full "
                "shard after a short one -- the reader's positional "
                "arithmetic cannot survive that"
            )
        _verify_row_integrity(row)
        _require_row_kind_matches(row, self.identity)
        stored = _row_to_storage(row)
        if build_metadata is not None:
            stored["_build_metadata"] = build_metadata
        self._buffer.append(stored)
        self._shard_keys.append((row.document_bundle_id, row.page_index))
        self._keys.append((row.document_bundle_id, row.page_index))
        self.entries += 1
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def commit_shard(
        self,
        name: str,
        *,
        count: int,
        sha256: str,
        keys: list[tuple[str, int]],
    ) -> None:
        """Register a shard whose bytes the `--workers` assembly pool
        already wrote into `staging` under `name` (and hashed there, out
        of process, in parallel with the materialize pool).

        The `add()` path buffers rows and flushes them itself, so the two
        paths never mix: a non-empty `add()` buffer is refused here, and
        a partial shard may only ever be the LAST one committed, because
        the reader's `index // shard_size` arithmetic assumes every shard
        before the last is full. `close()` publishes whatever the parent
        committed, in the order it committed it.
        """
        if self._closed:
            raise ValueError("commit_shard called after close()")
        if self._buffer:
            raise ValueError(
                "commit_shard cannot mix with the add() buffer -- a shard "
                "comes either from buffered rows or from a committed file, "
                "never both"
            )
        if self._partial_last:
            raise ValueError(
                "a partial shard may only ever be the last committed -- "
                "a short shard in the middle breaks the reader's positional "
                "arithmetic"
            )
        count_ok = isinstance(count, int) and not isinstance(count, bool)
        if not count_ok or not 0 < count <= self.shard_size:
            raise ValueError(
                f"count must be an int in 1..{self.shard_size}, got {count!r}"
            )
        normalized: list[tuple[str, int]] = []
        for key in keys:
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(f"key must be (document_bundle_id, page_index), got {key!r}")
            bundle, page = key
            if not isinstance(bundle, str) or not isinstance(page, int) or isinstance(page, bool):
                raise ValueError(
                    f"key must be (document_bundle_id, page_index), got {key!r}"
                )
            normalized.append((bundle, int(page)))
        if len(normalized) != count:
            raise ValueError(f"keys must have exactly {count} entries, got {len(normalized)}")
        self._shards.append(
            {
                "name": name,
                "count": count,
                "sha256": sha256,
                # Per-shard, in row order: the same field the `add()`
                # path records, so a reader of the manifest cannot tell
                # the two build paths apart.
                "keys": [[bundle, page] for bundle, page in normalized],
            }
        )
        self._keys.extend(normalized)
        self.entries = len(self._keys)
        if count < self.shard_size:
            self._partial_last = True
        os.replace(self.staging / name, self.out_dir / name)
        if count == self.shard_size:
            self._checkpoint_full_shard()

    def _flush(self) -> None:
        if not self._buffer:
            return
        name = f"shard-{len(self._shards):05d}.pt"
        path = self.staging / name
        torch.save(self._buffer, path)
        self._shards.append(
            {
                "name": name,
                "count": len(self._buffer),
                "sha256": _sha256(path),
                # Per-shard, in row order: this is what makes the cache's
                # key set `C` readable without loading every shard, and
                # what a loaded shard's rows are checked against.
                "keys": [[bundle, page] for bundle, page in self._shard_keys],
            }
        )
        self._buffer = []
        self._shard_keys = []
        os.replace(path, self.out_dir / name)
        if self._shards[-1]["count"] == self.shard_size:
            self._checkpoint_full_shard()

    def _checkpoint_full_shard(self) -> None:
        # Publish the first prefix so interrupted builds always have a
        # resumable identity. Later published files beyond this prefix
        # are replayed after interruption; close() always commits all rows.
        if len(self._shards) == 1 or len(self._shards) % self.checkpoint_every_shards == 0:
            self._checkpoint("in_progress")

    def scope(self) -> ExactTextCacheScope:
        """The scope declaration for what has been written so far."""
        return ExactTextCacheScope(
            selection_kind=self.selection_kind,
            only_bundles_sha256=self.only_bundles_sha256,
            only_bundles_count=self.only_bundles_count,
            page_key_set_sha256=page_key_set_sha256(self._keys),
            page_key_count=len(self._keys),
            single_page_only=self.single_page_only,
            limit=self.limit,
        )

    def _manifest(self, status: str = "complete") -> dict[str, Any]:
        return {
            "scope": asdict(self.scope()),
            "cache_family": CACHE_FAMILY,
            "cache_version": expected_cache_version(self.identity),
            "status": status,
            "entries": self.entries,
            "shard_entries": sum(s["count"] for s in self._shards),
            "shards": len(self._shards),
            "shard_size": self.shard_size,
            "shard_index": self._shards,
            "identity": asdict(self.identity),
            **self.extra_manifest,
        }

    def _checkpoint(self, status: str) -> None:
        temporary = self.out_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(self._manifest(status), indent=1, sort_keys=True))
        os.replace(temporary, self.out_dir / "manifest.json")

    def record_summary(self, key: str, value: Any) -> None:
        """Add one caller-supplied build-summary field to the manifest
        (e.g. `build_exact_text_cache.py`'s `min_text_coverage`/
        `text_coverage`/`text_coverage_rows`, D2 2026-09-07) -- refused for
        any of :data:`_FIXED_MANIFEST_KEYS` (the manifest's own schema, so
        a caller cannot silently overwrite e.g. `identity`) and refused
        once `close()` has already written the manifest to disk, since a
        later call could never be reflected in it."""
        if self._closed:
            raise ValueError(
                f"record_summary({key!r}, ...) called after close() -- the manifest is "
                "already written and cannot be amended"
            )
        if key in _FIXED_MANIFEST_KEYS:
            raise ValueError(
                f"record_summary: {key!r} is a fixed manifest key ({sorted(_FIXED_MANIFEST_KEYS)}) "
                "and cannot be overwritten this way"
            )
        self.extra_manifest[key] = value

    def close(self) -> None:
        self._flush()
        rows_dir = self.staging / "rows"
        if rows_dir.is_dir():
            # The `--workers` staging directory: a row file whose page was
            # written but never assembled (an early `--limit` stop) is a
            # legitimate orphan, and an uncleaned directory would keep
            # `staging` from rmdir-ing below.
            shutil.rmtree(rows_dir)
        self._checkpoint("complete")
        self.staging.rmdir()
        self._closed = True

    def __enter__(self) -> ExactTextCacheWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        if exc_type is None:
            self.close()


def cache_manifest(cache_dir: Path | str) -> dict[str, Any]:
    """Read a cache's manifest, whatever family it claims to be."""
    path = Path(cache_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no manifest under {cache_dir}")
    return json.loads(path.read_text())


def _identity_from_manifest(manifest: dict[str, Any]) -> ExactTextCacheIdentity:
    payload = manifest.get("identity")
    if not isinstance(payload, dict):
        raise ExactTextCacheIncompatible(
            "manifest carries no 'identity' object; cannot bind this cache "
            "to a lineage"
        )
    missing = [
        f.name for f in fields(ExactTextCacheIdentity) if payload.get(f.name) is None
    ]
    if missing:
        raise ExactTextCacheIncompatible(
            "identity is missing (or null on) required field(s): "
            + ", ".join(sorted(missing))
            + " -- an absent/null identity field is refused, never defaulted"
        )
    admission = payload["admission"]
    if admission not in _ADMISSION_VALUES:
        raise ExactTextCacheIncompatible(
            f"identity.admission {admission!r} is not one of {sorted(_ADMISSION_VALUES)}"
        )
    return ExactTextCacheIdentity(
        **{f.name: payload[f.name] for f in fields(ExactTextCacheIdentity)}
    )


def _scope_from_manifest(manifest: dict[str, Any]) -> ExactTextCacheScope:
    """The mandatory DEC-MP-4 `scope` block, refused at open by exact
    equality of its key set -- the way `cache_version` is refused.

    A cache built before DEC-MP-4 carries no `scope` and is refused rather
    than read as though it were a full one: "no declaration" is precisely
    the state in which a partial cache and a full cache are
    indistinguishable, so it cannot be the state a reader tolerates.
    """
    payload = manifest.get("scope")
    if not isinstance(payload, dict):
        raise ExactTextCacheIncompatible(
            "manifest carries no 'scope' object (DEC-MP-4): a cache that does "
            "not declare which pages it holds is refused at open, never read "
            "as a full one. Rebuild it"
        )
    expected_keys = {f.name for f in fields(ExactTextCacheScope)}
    if set(payload) != expected_keys:
        missing = sorted(expected_keys - set(payload))
        unexpected = sorted(set(payload) - expected_keys)
        raise ExactTextCacheIncompatible(
            f"manifest 'scope' has key set {sorted(payload)}, expected exactly "
            f"{sorted(expected_keys)} (missing: {missing}, unexpected: "
            f"{unexpected}) -- refused by exact equality, never partially read"
        )
    selection_kind = payload["selection_kind"]
    if selection_kind not in SELECTION_KINDS:
        raise ExactTextCacheIncompatible(
            f"scope.selection_kind {selection_kind!r} is not one of "
            f"{sorted(SELECTION_KINDS)}"
        )
    digest = payload["page_key_set_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ExactTextCacheIncompatible(
            f"scope.page_key_set_sha256 {digest!r} is not a 64-character hex "
            "digest"
        )
    count = payload["page_key_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ExactTextCacheIncompatible(
            f"scope.page_key_count {count!r} is not a non-negative integer"
        )
    only_hash = payload["only_bundles_sha256"]
    only_count = payload["only_bundles_count"]
    only_bundles_restricted = only_hash is not None or only_count is not None
    if selection_kind == "only_bundles" and not (
        only_hash is not None and only_count is not None
    ):
        raise ExactTextCacheIncompatible(
            "scope.selection_kind='only_bundles' but only_bundles_sha256 / "
            f"only_bundles_count are {only_hash!r} / {only_count!r} -- a "
            "restricted cohort must name what restricted it"
        )
    if selection_kind != "only_bundles" and only_bundles_restricted:
        raise ExactTextCacheIncompatible(
            f"scope.selection_kind={selection_kind!r} yet the scope names a "
            f"restricting cohort ({only_hash!r} / {only_count!r}) -- the two "
            "claims contradict each other"
        )
    # R-1 (5.0.0): the other two cohort-narrowing knobs, same cross-check
    # the writer applies at construction (`ExactTextCacheWriter.__init__`).
    single_page_only = payload["single_page_only"]
    if not isinstance(single_page_only, bool):
        raise ExactTextCacheIncompatible(
            f"scope.single_page_only {single_page_only!r} is not a boolean"
        )
    limit = payload["limit"]
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool)):
        raise ExactTextCacheIncompatible(
            f"scope.limit {limit!r} is not an integer or null"
        )
    narrowed = single_page_only or limit is not None
    if selection_kind == "full_split" and narrowed:
        raise ExactTextCacheIncompatible(
            "scope.selection_kind='full_split' yet single_page_only="
            f"{single_page_only!r} / limit={limit!r} -- the two claims "
            "contradict each other"
        )
    if selection_kind == "restricted" and not narrowed:
        raise ExactTextCacheIncompatible(
            "scope.selection_kind='restricted' but single_page_only=False "
            "and limit=None -- a restriction that names nothing that "
            "restricted it is not a declaration"
        )
    return ExactTextCacheScope(
        **{f.name: payload[f.name] for f in fields(ExactTextCacheScope)}
    )


class ExactTextCacheReader:
    """Random access over a sharded exact-text cache.

    `gradient_bearing` is required, with no default, so a caller cannot
    silently get the weaker (diagnostic-only) admission check by omission
    (design §8 rule 2). Every shard's bytes are hashed once per reader
    (mirroring `feature_cache.FeatureCacheReader`); every row is
    integrity-checked each time its shard is loaded into residence.

    Not all refusals fire at open. The shard-key-vs-row binding check
    (`_shard_rows`) only runs when a given shard is first loaded, which for
    any shard beyond the first can happen mid-epoch, not at construction --
    do not rely on an open-time guarantee for that specific check.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        gradient_bearing: bool,
        cache_shards: int = DEFAULT_CACHE_SHARDS,
        expectation: CacheExpectation | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.gradient_bearing = gradient_bearing
        self.manifest = cache_manifest(self.cache_dir)
        self._check_family_and_completeness()
        self.identity = _identity_from_manifest(self.manifest)
        self.scope = _scope_from_manifest(self.manifest)
        self._check_admission()
        if cache_shards < 1:
            raise ValueError(f"cache_shards must be positive, got {cache_shards}")
        self.shard_size = self.manifest["shard_size"]
        self.cache_shards = cache_shards
        self._index = {s["name"]: s for s in self.manifest["shard_index"]}
        self._loaded: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self._verified: set[int] = set()
        #: `C`: every key the cache physically holds, in row order.
        self.page_keys = self._check_scope()
        self.expectation = expectation
        #: Row indices this reader exposes. Without an expectation that is
        #: every row; with one it is exactly `E \ X`, in `E`'s pinned order
        #: -- which is how a key in `C \ E` is kept from ever becoming a
        #: training example (design §9 item 2).
        self._selection: tuple[int, ...] | None = (
            None if expectation is None else self._check_expectation(expectation)
        )

    def _check_family_and_completeness(self) -> None:
        family = self.manifest.get("cache_family")
        if family != CACHE_FAMILY:
            raise ExactTextCacheIncompatible(
                f"{self.cache_dir}: cache_family {family!r} != {CACHE_FAMILY!r} "
                "-- this is not an exact-text cache (a structural feature "
                "cache, for instance, carries no cache_family key at all) "
                "and must be refused, not read as a shorter one"
            )
        version = self.manifest.get("cache_version")
        expected_version = expected_cache_version(self.manifest.get("identity") or {})
        if version != expected_version:
            raise ExactTextCacheIncompatible(
                f"{self.cache_dir}: cache_version {version!r} != "
                f"{expected_version!r}. Refused at open, by exact "
                "equality: an older cache carries no typed-property channel "
                "(pre-3.0.0) or no scope declaration (pre-4.0.0), and a "
                "reader that discovered that only on the first shard read "
                "would already have reported a valid-looking cache of "
                f"{self.manifest.get('entries')!r} entries. Rebuild it"
            )
        if self.manifest.get("status") != "complete":
            raise ExactTextCacheIncompatible(
                f"{self.cache_dir}: manifest has no status 'complete' "
                f"(cache_version {self.manifest.get('cache_version')!r}); "
                "the build was interrupted"
            )
        claimed = int(self.manifest["entries"])
        stored = int(self.manifest["shard_entries"])
        indexed = sum(int(s["count"]) for s in self.manifest["shard_index"])
        if not claimed == stored == indexed:
            raise ExactTextCacheIncompatible(
                f"{self.cache_dir}: manifest claims {claimed} entries but "
                f"the shards hold {stored} and the index sums to {indexed}"
            )

    def _check_admission(self) -> None:
        if self.gradient_bearing and self.identity.admission != "strict":
            raise ExactTextCacheIncompatible(
                f"{self.cache_dir}: admission {self.identity.admission!r} "
                "cannot satisfy a gradient-bearing read, which requires "
                "'strict' (RFC003-ERRATA-001 E5; design §8 rule 2)"
            )

    def _check_scope(self) -> tuple[PageKey, ...]:
        """Reconstruct `C` from the shard index and check it against the
        scope declaration.

        Two independent bindings, because either alone is defeatable: the
        declared keys must reproduce `scope.page_key_set_sha256` (so an
        edited key list is caught at open even though the manifest is not
        itself hashed), and every loaded shard's rows are checked against
        the keys declared for that shard (`_shard_rows`, so an edited key
        list plus a recomputed hash is caught when the rows are read).
        """
        keys: list[PageKey] = []
        for shard in self.manifest["shard_index"]:
            declared = shard.get("keys")
            if not isinstance(declared, list):
                raise ExactTextCacheIncompatible(
                    f"{self.cache_dir}: shard_index entry {shard.get('name')!r} "
                    "carries no 'keys' list (DEC-MP-4); the cache's page key set "
                    "cannot be reconstructed and is refused, not guessed"
                )
            if len(declared) != int(shard["count"]):
                raise ExactTextCacheIncompatible(
                    f"{self.cache_dir}: shard_index entry {shard.get('name')!r} "
                    f"declares {len(declared)} key(s) for {shard['count']} row(s)"
                )
            for item in declared:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or not isinstance(item[0], str)
                    or not isinstance(item[1], int)
                    or isinstance(item[1], bool)
                ):
                    raise ExactTextCacheIncompatible(
                        f"{self.cache_dir}: shard_index key {item!r} is not a "
                        "[document_bundle_id: str, page_index: int] pair"
                    )
                keys.append((item[0], item[1]))
        if len(keys) != self.scope.page_key_count:
            raise ExactTextCacheIncompatible(
                f"{self.cache_dir}: scope.page_key_count "
                f"{self.scope.page_key_count} but the shard index declares "
                f"{len(keys)} key(s)"
            )
        if len(keys) != int(self.manifest["entries"]):
            raise ExactTextCacheIncompatible(
                f"{self.cache_dir}: scope declares {len(keys)} page key(s) but "
                f"the manifest claims {self.manifest['entries']} entries -- a "
                "cache whose scope does not describe its own rows is refused"
            )
        digest = page_key_set_sha256(keys)
        if digest != self.scope.page_key_set_sha256:
            raise ExactTextCacheIncompatible(
                f"{self.cache_dir}: the declared page keys hash to {digest}, "
                f"scope.page_key_set_sha256 says "
                f"{self.scope.page_key_set_sha256} -- the key list was edited "
                "after the build, or was never the list that was written"
            )
        return tuple(keys)

    def _check_expectation(self, expectation: CacheExpectation) -> tuple[int, ...]:
        """Refuse unless the cache covers the pinned cohort, and return the
        row indices for exactly `E \\ X` in `E`'s pinned order (design §9
        item 2).

        What each clause is for:

        - `X ⊆ E` -- exclusions may only remove pages somebody pinned. An
          exclusion outside `E` would let a run absorb an arbitrary
          failure by naming it.
        - duplicates in `E` or `X` -- a pin that names one page twice
          cannot say what "exactly `E \\ X`" means.
        - `E \\ X` non-empty -- an expectation that excludes everything
          would be satisfied by any cache at all, including an empty one.
        - `(E \\ X) \\ C` empty -- the coverage question a content hash
          cannot answer.
        - each target key resolves to exactly ONE row -- "with no
          duplicates" is about the training dataset, and a cache holding a
          page twice would otherwise silently double its weight.
        """
        expected = expectation.expected_page_keys
        exclusions = expectation.approved_exclusions
        if len(set(expected)) != len(expected):
            raise ExactTextCacheIncompatible(
                "the pinned expectation names the same page more than once "
                "(expected_page_keys has duplicates); it cannot define a "
                "training set"
            )
        if len(set(exclusions)) != len(exclusions):
            raise ExactTextCacheIncompatible(
                "the pinned expectation names the same approved exclusion more "
                "than once"
            )
        outside = sorted(set(exclusions) - set(expected))
        if outside:
            raise ExactTextCacheIncompatible(
                f"approved_exclusions is not a subset of expected_page_keys: "
                f"{len(outside)} key(s) outside E, first {outside[0]!r} -- an "
                "exclusion may only remove a page the expectation pinned, never "
                "absorb an arbitrary failure"
            )
        targets = expectation.target_keys()
        if not targets:
            raise ExactTextCacheIncompatible(
                "the pinned expectation leaves no page after its approved "
                "exclusions (E \\ X is empty); an empty training set is "
                "satisfied by every cache and proves nothing"
            )
        positions: dict[PageKey, list[int]] = {}
        for index, key in enumerate(self.page_keys):
            positions.setdefault(key, []).append(index)
        missing = [key for key in targets if key not in positions]
        if missing:
            raise ExactTextCacheIncompatible(
                f"{self.cache_dir}: the cache does not hold {len(missing)} of "
                f"{len(targets)} expected page(s) (first missing: "
                f"{missing[0]!r}); this cache's scope is "
                f"{self.scope.selection_kind!r} over "
                f"{self.scope.page_key_count} page(s) -- a partial cache cannot "
                "satisfy a wider expectation and is refused, not silently "
                "trained on"
            )
        duplicated = [key for key in targets if len(positions[key]) > 1]
        if duplicated:
            raise ExactTextCacheIncompatible(
                f"{self.cache_dir}: {len(duplicated)} expected page(s) occur "
                f"more than once in the cache (first: {duplicated[0]!r}); the "
                "training set must be exactly E \\ X with no duplicates"
            )
        return tuple(positions[key][0] for key in targets)

    def __len__(self) -> int:
        if self._selection is not None:
            return len(self._selection)
        return int(self.manifest["entries"])

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
                raise ExactTextCacheIncompatible(
                    f"{path}: sha256 {actual} does not match the manifest's "
                    f"{expected['sha256']}; the shard changed after the build"
                )
            self._verified.add(shard)
        rows = torch.load(path, weights_only=True)
        if len(rows) != int(expected["count"]):
            raise ExactTextCacheIncompatible(
                f"{path}: {len(rows)} rows, manifest says {expected['count']}"
            )
        declared = expected["keys"]
        for position, data in enumerate(rows):
            row = _row_from_storage(data)
            _verify_row_integrity(row)
            _require_row_kind_matches(row, self.manifest.get("identity") or {})
            key = [row.document_bundle_id, row.page_index]
            if key != list(declared[position]):
                raise ExactTextCacheIncompatible(
                    f"{path} row {position} is page {tuple(key)!r}, but the "
                    f"manifest's shard key list declares "
                    f"{tuple(declared[position])!r} -- the scope declaration "
                    "does not describe the rows it claims to"
                )
        self._loaded[shard] = rows
        while len(self._loaded) > self.cache_shards:
            self._loaded.popitem(last=False)
        return rows

    def __getitem__(self, index: int) -> ExactTextCacheRow:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._selection is not None:
            # An expectation makes this reader a view over `E \ X`: the
            # caller's index addresses the pinned cohort, never the
            # physical row order, so a row in `C \ E` has no index at all.
            index = self._selection[index]
        rows = self._shard_rows(index // self.shard_size)
        return _row_from_storage(rows[index % self.shard_size])
