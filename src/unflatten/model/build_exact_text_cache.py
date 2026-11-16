"""Build an exact-text cache (`exact_text_cache.py`) from the object store.

The producer that was missing. `ExactTextCacheWriter`/`ExactTextCacheReader`
were written and tested with no caller outside their own test file, so no
exact-text cache existed anywhere: the only cache on disk
(`cache-candidate-s1`, built by `build_cache.py`) holds `events_vocab`
structural ids and not one Qwen text token, because its targets come from
`targets.decoder_targets`. That cache is what `s1_runner` reads to evaluate
gates; it is not what an exact-text gradient run trains from.

This module assembles, it does not invent: `dataset.items.build_item` reads
the page, `text_authority.build_text_span_authorities` reshapes the real
text spans, `TrainingTargetMaterializer.materialize` produces the sequence,
`exact_text_cache.row_from_materialized` builds the row, and
`verify_row_replay` plus the writer's own `_verify_row_integrity` verify it
before it reaches disk. A page that cannot be materialized is counted under
a named skip and never replaced by a fabricated one.

**Visual features are READ, not re-extracted** (`open_feature_sources`).
They are the same per-page `float16 [V, D]` tensors `cache-candidate-s1`
already holds, keyed by `(document_bundle_id, page_index)`; re-running the
frozen candidate would cost a GPU pass and could differ, so the features
this cache stores are byte-identical to the ones the gate run already used.
The row's `grid_thw` (cache version 1.1.0) is copied from the SAME
`CacheEntry` object, never looked up a second time: `inject_visual` needs
the backbone's own mRoPE index, it cannot be derived from the visual token
count (`(1, 70, 50)` and `(1, 50, 70)` both give 875 rows and both occur in
that cache), and a grid paired with another page's features would scatter
the visual rows wrongly with nothing to notice. Carrying it on the row is
also what lets an S1 run ship ONE cache instead of two.
Because a read is only as good as its provenance,
:func:`verify_feature_cache_identity` refuses a source cache whose backbone
revision, processor digest, storage dtype, release root hash, split, task
kind or privacy admission is not the one this build claims. The cohort is
therefore exactly the feature cache's pages -- a page with no cached
features is not built with fabricated ones, it is not attempted.

**Page-image mode (DEC-056 addendum A, v16): no feature source at all.**
`build(visual_input_policy=...)` (CLI `--page-image-policy`, positional
`feature_cache_dir` given as `-`) writes a 7.0.0 cache whose rows carry the
page PNG's `ArtifactRef` instead of features (`PageImageSource`): the frozen
ViT computes the features during training (`onfly_visual`). The cohort is
the release's own pages; each row's `grid_thw` is what the policy's
preprocessing derives from the PNG size, and the identity binds the policy
(`feature_tap = "onfly-visual:<policy_hash>"`, its backbone/processor
hashes and pixel budget) instead of a feature-cache id. Absent, nothing
changes.

**What this module cannot bind, and does not pretend to.**
`ExactTextCacheIdentity` requires seven hashes this build does not compute
for itself: `composite_grammar_hash`, `loss_unit_schema_hash`,
`stage_policy_hash`, `conditioning_policy_hash`,
`runtime_limits_policy_hash`, `component_lock_hash`, `head_registry_hash`.
Every existing caller (the materializer, `loss_unit.build_loss_unit`, both
verifiers) takes them as arguments. They are supplied by the caller through
a required `--bindings` JSON file with no defaults (:func:`load_bindings`),
and a file that omits one, adds an unknown one, or carries a malformed
value is refused rather than completed.

All seven now have real producers (2026-09-05/06), so a `--bindings` file
need no longer carry a placeholder for any of them::

    composite_grammar_hash      composite_grammar.build_composite_grammar_descriptor
                                -> composite_grammar.composite_grammar_hash
    loss_unit_schema_hash       loss_unit.loss_unit_schema_hash()
    runtime_limits_policy_hash  runtime_limits.build_runtime_limits_policy
                                -> runtime_limits.runtime_limits_policy_hash
    component_lock_hash         component_lock.build_exact_text_s1_component_lock
                                -> manifests.component_lock_hash
    stage_policy_hash           stage_policy.build_stage_policy
                                -> stage_policy.stage_policy_hash
    conditioning_policy_hash    conditioning_policy.build_conditioning_policy
                                -> conditioning_policy.conditioning_policy_hash
    head_registry_hash          property_head_registry.head_registry_hash (a
                                niladic producer function)
                                -- owner-invoked via the bindings script only;
                                   never called from this module or any other
                                   in-repo caller (binding a non-null value
                                   re-materializes the whole cache)

**A producer is not a policy, and the last two are the clearest case.**
`stage_policy` and `conditioning_policy` now have a schema and a producer,
but neither has a *value*: RFC005 §26.4 requires every `λ_f` to be
explicit, versioned and stored in a loss policy, and §26.5 requires each
run to declare one of `gold_prefix` / `predicted_prefix_stop_gradient` /
`versioned_mixed_policy` per attachment. Those remain owner content
decisions, and both producers refuse to default them -- a λ missing for a
family, or an attachment with no declared mode, is refused by name rather
than filled in with the likely answer. So this build's refusal behaviour
is unchanged: a cache built with placeholder bindings is still NOT
gradient-admissible, and it is the owner supplying values, not this
module, that changes that.

`qwen_text_allowlist_hash`, `composite_vocabulary_hash` and
`tokenizer_artifact_set_hash` are NOT in that file: those have real
producers (`qwen_text_allowlist.build_qwen_text_binding` over the DEC-007
D3-pinned tokenizer) and a placeholder for them would be refused later by
`RunManifestBindingVerifier` anyway.
`training_target_materializer_hash` is derived here, really, from
`training_materializer.build_materializer_descriptor`.

**Two identity fields do not compose and are recorded, not hidden.**
`ExactTextCacheIdentity.feature_tap` and `.max_pixels` are non-nullable
(`_identity_from_manifest` refuses a null field, on purpose), but the
candidate track genuinely has neither: `build_cache.py` records
`feature_tap: null` and `max_pixels: null` for it because the candidate
exposes no named tap and no pixel-budget knob. Rather than write a null
the reader would refuse or a number nobody chose, this module records the
absence explicitly as :data:`NO_FEATURE_TAP` / :data:`NO_MAX_PIXELS` and
reports both under `identity_gaps` in the build summary, so a reader of
the artifact learns it from the artifact.

Ships no `nn.Module`, no optimizer, and authorizes no gradient update: it
writes a cache, and starting a run is the owner's decision.
"""

from __future__ import annotations

import argparse
import ctypes
import io
import json
import multiprocessing as mp
import re
import shutil
import signal
import statistics
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from unflatten.dataset.items import build_item
from unflatten.dataset.samples import iter_page_samples
from unflatten.dataset.targets.bundle import PROJECTOR_VERSION
from unflatten.dataset.targets.masks import withdraw_placements_typed
from unflatten.model import composite_vocab as cv
from unflatten.model import exact_text_feature_source as etfs
from unflatten.model import onfly_visual, page_projection
from unflatten.model import page_image_cache_reuse as reuse
from unflatten.model.boundary_projector import (
    ScalarBoundaryProjector,
    TextRangeBoundaryInvalid,
)
from unflatten.model.build_cache import (
    OnlyBundlesInvalid,
    _document_masked_typed,
    has_continuation_presentation,
    load_only_bundles,
    only_bundles_manifest_fields,
    render_font_pack_hash,
)
from unflatten.model.capability_profile import (
    BOUND_CAPABILITY_PROFILE,
    STANDALONE_PAGE_PROFILE_IDS,
    CapabilityProfile,
    resolve_capability_profile_by_id,
)
from unflatten.model.composite_grammar import (
    CompositeGrammarController,
    CompositeGrammarViolation,
    IncompleteCompositeStream,
    build_composite_grammar_descriptor,
    composite_grammar_hash,
)
from unflatten.model.conditioning_policy import (
    conditioning_policy_hash,
    load_conditioning_policy,
)
from unflatten.model.coverage_cause import (
    SUB_FLOOR_CLASSES,
    PageNodeShare,
    classify_sub_floor_page,
)
from unflatten.model.exact_text import (
    ByteFrameInvalid,
    TextEncodingBudgetExceeded,
    TextInvalidUnicode,
    TextNotNfc,
    TextTokenizerRoundtripFailed,
)
from unflatten.model.exact_text_cache import (
    CACHE_FAMILY,
    ExactTextCacheIdentity,
    ExactTextCacheIncompatible,
    ExactTextCacheRow,
    ExactTextCacheWriter,
    _row_from_storage,
    _row_to_storage,
    _verify_row_integrity,
    assemble_shard,
    row_from_materialized,
    verify_row_replay,
)
from unflatten.model.feature_cache import CacheEntry, FeatureCacheReader
from unflatten.model.loss_unit import TextRangeLocator, loss_unit_schema_hash
from unflatten.model.qwen_text_allowlist import QwenTextBinding
from unflatten.model.region_targets import build_region_targets
from unflatten.model.release_lineage_proof import (
    feature_cache_manifest_sha256,
    verify_proof_hash,
)
from unflatten.model.runtime_limits import (
    load_runtime_limits_policy,
    runtime_limits_policy_hash,
)
from unflatten.model.stage_policy import load_stage_policy, stage_policy_hash
from unflatten.model.text_authority import (
    TextAuthorityAlignmentError,
    TextCoverage,
    build_text_span_authorities,
    text_coverage,
)
from unflatten.model.training_materializer import (
    TrainingTargetMaterializer,
    build_materializer_descriptor,
    materializer_descriptor_hash,
)
from unflatten.model.typed_property_targets import (
    # `_anchor_token_indices` is imported private on purpose, same reason
    # `region_targets.py` does: the anchor-token mechanism is owned by
    # this module, and a second copy would be a second rule.
    _anchor_token_indices,
    build_typed_property_targets,
    page_piece_kinds,
    withdraw_page_piece_typed,
)
from unflatten.model.v16_target_policy import (
    V16_SKIP_COUNTS,
    TargetPolicy,
    grant_whitespace_runs,
    load_page_exclusions,
    projection_kwargs,
    rehome_inherited_stories,
    resolve_target_policy,
    spill_signature,
    unowned_ink,
)

#: The one capability profile this build binds -- named once so the grammar
#: it verifies, the controller it materializes rows under, the materializer
#: descriptor and the cache identity cannot name different profiles. A build
#: that got that wrong would write a cache whose identity claims one profile
#: while its rows were materialized under another, and that survives on disk.
#: The default a build falls back to when `--capability-profile` (DEC-024,
#: :func:`main`) is not given -- not the only profile a build can bind.
#: Passing `--capability-profile s2-typed-widened-v1` is not "silently
#: picking a profile": `main` resolves the id through `resolve_capability_
#: profile_by_id` (fail-closed on an unknown id) and the RESOLVED profile's
#: hash is what `build_identity` records, so the cache's identity always
#: names the profile it was actually built under, explicit or defaulted.
#:
#: Owned by `capability_profile` (the module that owns the profiles) and
#: imported here as the default (R4, `ZOSTAVA-typovany-head-modul.md`): a
#: build CLI module is a consumer of a profile constant, not its owner.

#: Identity hashes this build does not compute for itself (module
#: docstring). Supplied through `--bindings`, never defaulted. All six now
#: have producers; `stage_policy_hash` and `conditioning_policy_hash`
#: additionally need owner-chosen CONTENT before their producer can be
#: called at all.
REQUIRED_BINDING_HASHES: tuple[str, ...] = (
    "component_lock_hash",
    "composite_grammar_hash",
    "conditioning_policy_hash",
    "head_registry_hash",
    "loss_unit_schema_hash",
    "runtime_limits_policy_hash",
    "stage_policy_hash",
)

#: Summary keys naming why a page in the cohort was not written. Every
#: attempted page is either written or counted under exactly one of these
#: -- except `skipped_grammar_continuation` (T-9, design §4.1), which is a
#: SUBSET of `skipped_grammar`, not a separate bucket: a page refused by
#: the same `CompositeGrammarViolation`/`IncompleteCompositeStream` while
#: presenting a continuation it does not structurally own
#: (`build_cache.has_continuation_presentation`) increments BOTH keys, so
#: `skipped_grammar`'s total is unchanged by this split and a reader who
#: sums `SKIP_COUNTS` for a total must exclude it.
#:
#: `skipped_no_observation` is the O-3 population
#: (`docs/plans/NALEZ-o3-strany-bez-pozorovani.md`): a fragment with an
#: empty `slices` list, meaning the aligner placed no observation on this
#: page at all -- disjoint from `skipped_unencodable` (no structural
#: targets/fragment at all) and from every later reason, which all require
#: `slices` to be non-empty to even reach that check. `build_cache.py` has
#: carried this label since `cd3c5da`; this builder did not, so an
#: empty-slice page here fell through to whatever the adapter/materializer
#: happened to say about it instead of being named.
SKIP_COUNTS: tuple[str, ...] = (
    "skipped_unencodable",
    "skipped_no_observation",
    "skipped_schema_mismatch",
    "skipped_typed_mask",
    "skipped_text_authority",
    "skipped_text_coverage",
    "skipped_grammar",
    "skipped_grammar_continuation",
    "skipped_text_encoding",
    "skipped_replay",
    "skipped_realisation_mismatch",
    "skipped_projection_invariant",
    "skipped_no_cached_features",
    # T1 (design `spec-node-visual-aligner-design.md`): the page's
    # Observation Sidecar could not be read at all (no artifact, or no
    # entry for this page) -- distinct from `skipped_no_observation`
    # (an empty fragment `slices` list), which requires a sidecar-free
    # decision the fragment already made. A page counted here got no
    # chance to derive its region channel, so it must not be written
    # with a fabricated empty one.
    "skipped_no_sidecar",
)

#: The `SKIP_COUNTS` members that are subsets rather than partition members
#: (T-9): summing `SKIP_COUNTS` to reconstruct "how many pages were not
#: written" must exclude these, or a continuation-caused grammar refusal
#: (counted under both `skipped_grammar` and `skipped_grammar_continuation`)
#: is counted twice.
SKIP_COUNTS_SUBSETS = frozenset({"skipped_grammar_continuation"})

#: `ExactTextCacheIdentity.feature_tap` for the candidate track, which
#: exposes no named tap (module docstring). An explicit name, because the
#: reader refuses the null `build_cache.py` records.
NO_FEATURE_TAP = "candidate_native_no_tap"

#: `ExactTextCacheIdentity.max_pixels` for a build with no pixel budget
#: pinned -- the candidate track has no such knob (module docstring).
NO_MAX_PIXELS = 0

#: Bounds `exact_text.encode_text`'s tokenizer probes per text range.
DEFAULT_TEXT_ENCODING_BUDGET_OPS = 10_000

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

#: Materialization failures that are a property of ONE page, counted and
#: skipped (`build_cache.py`'s convention). Anything else propagates: a
#: builder bug must not be recorded as a corpus statistic.
_TEXT_ENCODING_ERRORS = (
    ByteFrameInvalid,
    TextEncodingBudgetExceeded,
    TextInvalidUnicode,
    TextNotNfc,
    TextTokenizerRoundtripFailed,
    TextRangeBoundaryInvalid,
)


class BindingsInvalid(ValueError):
    """The `--bindings` file does not supply exactly the hashes this build
    cannot compute for itself, in the `sha256:<64hex>` form every consumer
    of `ExactTextCacheIdentity` expects."""


class FeatureCacheMismatch(ValueError):
    """The source feature cache is not the one this build claims to be
    reading features from (backbone, revision, processor, dtype, release,
    split, task kind or privacy admission)."""


class TargetIdentityMismatch(ValueError):
    """A fragment and its target bundle disagree about `target_bundle_hash`
    (RFC003-ERRATA-001 E4 identity closure) -- store integrity, not a
    property of the page, so it aborts the build."""


@dataclass(frozen=True, slots=True)
class SchemaVersions:
    """The artifact schema versions one exact-text cache claims. Cache-wide:
    a page whose versions differ cannot be described by this identity and is
    skipped, never stored under a version it does not have."""

    target_bundle: str
    fragment: str
    mask_projection: str


@dataclass(frozen=True, slots=True)
class TypedChannelCounts:
    """What one page's typed withdrawal actually did.

    `flipped` is a DIFF of before against after (`targets.masks.
    withdraw_placements_typed`): it goes to zero if the withdrawal is removed,
    which is the failure mode this count exists to catch. Its sibling
    `nodes` count is deliberately NOT carried here -- by its own
    docstring it stays correct with the withdrawal deleted, so it reports
    scope, not effect, and recording it beside `flipped` would invite a
    reader to treat the two as the same kind of evidence.

    `node_not_opened_here` is `build_typed_property_targets`' fourth
    return value: typed-target-owning nodes with no opening structural
    event in this page's token stream (a resumed node), which therefore
    get no typed target on this page.
    """

    flipped: int
    node_not_opened_here: int


@dataclass(frozen=True, slots=True)
class RegionChannelCounts:
    """What one page's region channel actually derived (T1).

    `node_not_opened_here` is `build_region_targets`' second return
    value, on the `TypedChannelCounts` precedent: nodes with a slice on
    this page but no anchor (a resumed node), which therefore get no
    region target here.
    """

    node_not_opened_here: int


@dataclass(frozen=True, slots=True)
class MaterializerBundle:
    """One vocabulary/controller/materializer triple plus the hashes bound
    into every `LossUnit` it emits. Built once per build so the identity
    and the rows cannot name different grammars."""

    vocabulary: cv.CompositeVocabulary
    controller: CompositeGrammarController
    materializer: TrainingTargetMaterializer
    capability_profile_id: str


# ---------------------------------------------------------------------------
# Inputs this build cannot compute for itself
# ---------------------------------------------------------------------------


def load_bindings(path: Path | str) -> dict[str, str]:
    """Read the `--bindings` JSON file (module docstring).

    Requires exactly :data:`REQUIRED_BINDING_HASHES`, each `sha256:<64hex>`.
    A missing key is refused rather than defaulted; an unknown key is
    refused rather than ignored, because a caller who spelled one wrong
    would otherwise get a silently defaulted binding under a name that
    looks bound.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise BindingsInvalid(f"cannot read bindings file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BindingsInvalid(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BindingsInvalid(f"{path} must hold a JSON object, got {type(payload).__name__}")

    missing = sorted(set(REQUIRED_BINDING_HASHES) - set(payload))
    if missing:
        raise BindingsInvalid(
            f"{path} is missing required binding hash(es): {', '.join(missing)} -- "
            "no producer in this repository computes them, and this build refuses "
            "to default them"
        )
    unknown = sorted(set(payload) - set(REQUIRED_BINDING_HASHES))
    if unknown:
        raise BindingsInvalid(
            f"{path} carries unknown key(s): {', '.join(unknown)} -- expected exactly "
            f"{', '.join(REQUIRED_BINDING_HASHES)}"
        )
    for name in REQUIRED_BINDING_HASHES:
        value = payload[name]
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            raise BindingsInvalid(
                f"{path}: {name} must be 'sha256:<64 hex>', got {value!r}"
            )
    return {name: payload[name] for name in REQUIRED_BINDING_HASHES}


def check_native_vocab_size(native_vocab_size: int, *, tokenizer_len: int) -> None:
    """Refuse a `native_vocab_size` narrower than the tokenizer.

    The candidate's embedding width (151936) and the tokenizer's length
    (151669) are different numbers, and using the second packs Qwen ids
    into a range 267 entries too short: a silent wrong-range index, not an
    error, for every id above the tokenizer's length.
    """
    if native_vocab_size < tokenizer_len:
        raise ValueError(
            f"native_vocab_size {native_vocab_size} is narrower than the tokenizer's "
            f"{tokenizer_len} ids -- it must be the candidate's embedding width "
            "(`QwenCandidate.vocab_size`, config.json text_config.vocab_size), never "
            "`len(tokenizer)`, or every id above the tokenizer length packs into the "
            "wrong range with no error"
        )


def candidate_native_vocab_size() -> int:
    """The DEC-007 D3-pinned candidate's embedding width, read from the
    pinned snapshot's `config.json`.

    Reads the config, never the 8.3 GB weights: this build needs the width,
    not the model, and the only other source of it (`QwenCandidate.vocab_size`)
    costs a full load.
    """
    from huggingface_hub import try_to_load_from_cache

    from unflatten.model.qwen_tokenizer_artifacts import (
        QWEN_CANDIDATE_MODEL_ID,
        QWEN_CANDIDATE_REVISION,
        TokenizerArtifactSetInvalid,
    )

    cached = try_to_load_from_cache(
        QWEN_CANDIDATE_MODEL_ID, "config.json", revision=QWEN_CANDIDATE_REVISION
    )
    if not isinstance(cached, str):
        raise TokenizerArtifactSetInvalid(
            f"no local config.json for {QWEN_CANDIDATE_MODEL_ID}@{QWEN_CANDIDATE_REVISION} "
            f"in the hub cache (try_to_load_from_cache returned {cached!r})"
        )
    config = json.loads(Path(cached).read_text())
    vocab_size = (config.get("text_config") or {}).get("vocab_size")
    if not isinstance(vocab_size, int):
        raise TokenizerArtifactSetInvalid(
            f"{cached}: text_config.vocab_size is {vocab_size!r}, not an integer"
        )
    return vocab_size


# ---------------------------------------------------------------------------
# Visual features: read from the existing candidate cache
# ---------------------------------------------------------------------------


def _lineage_proof_accepts(
    manifest: dict[str, Any],
    *,
    source_release_root_hash: str | None,
    source_manifest_sha256: str | None,
    release_root_hash: str | None,
    split: str,
    release_lineage_proof: Mapping[str, Any] | None,
) -> bool:
    """Does `release_lineage_proof` admit this source in place of a direct
    `release_root_hash` match (design section 8, T11 follow-up)?

    Five independent checks, each with its own mutant (design R2): the
    proof's `from` must name this source's own release, its `to` must name
    THIS build's release, its `proof_hash` must recompute, `source_manifest_
    sha256` must equal ONE `feature_cache_dirs[i].manifest_sha256` entry
    (never just a same-sized cache -- a swapped directory with the same key
    COUNT used to pass), and THAT entry's own `keys_checked` must equal
    this manifest's own key count. Checking against that one entry, not the
    proof-wide `keys_checked` total (a UNION across every dir the proof
    covers), is what lets one proof built over several feature cache
    directories admit every one of them.
    """
    if release_lineage_proof is None:
        return False
    proof = release_lineage_proof
    if source_release_root_hash != proof.get("from_release_root_hash"):
        return False
    if proof.get("to_release_root_hash") != release_root_hash:
        return False
    if proof.get("split") != split:
        return False
    if not verify_proof_hash(proof):
        return False
    entry = next(
        (
            d
            for d in proof.get("feature_cache_dirs") or []
            if d.get("manifest_sha256") == source_manifest_sha256
        ),
        None,
    )
    if entry is None:
        return False
    source_index_len = sum(
        len(shard.get("rows") or []) for shard in manifest.get("shard_index") or []
    )
    return entry.get("keys_checked") == source_index_len


def _normalize_proofs(
    release_lineage_proof: Mapping[str, Any] | Sequence[Any] | None,
    release_lineage_proof_path: Any,
) -> list[tuple[Any, Mapping[str, Any]]]:
    """`(path, proof)` pairs in match order.

    Accepts a single proof mapping (historic behaviour, unchanged),
    a sequence of mappings (paths then apply to all), or a sequence
    of `(path, proof)` pairs (what :func:`build` passes). Anything
    else is a caller error, never silently half-applied.
    """
    if release_lineage_proof is None:
        return []
    if isinstance(release_lineage_proof, Mapping):
        return [(release_lineage_proof_path, release_lineage_proof)]
    pairs: list[tuple[Any, Mapping[str, Any]]] = []
    for item in release_lineage_proof:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[1], Mapping)
        ):
            pairs.append((item[0], item[1]))
        elif isinstance(item, Mapping):
            pairs.append((release_lineage_proof_path, item))
        else:
            raise ValueError(
                "release_lineage_proof entries must be mappings or "
                f"(path, mapping) pairs, got {type(item).__name__}"
            )
    return pairs


def _match_proof(
    proofs: Sequence[tuple[Any, Mapping[str, Any]]],
    *,
    manifest: Mapping[str, Any],
    source_manifest_sha256: str | None,
    release_root_hash: str | None,
    split: str,
) -> tuple[Any, Mapping[str, Any] | None]:
    """First `(path, proof)` accepting this source, else `(None, None)`.

    A source admitted by nothing falls through to the direct-match path
    of :func:`verify_feature_cache_identity`, exactly as if no proof had
    been given at all.
    """
    identity = manifest.get("identity") or {}
    for path, proof in proofs:
        if _lineage_proof_accepts(
            manifest,
            source_release_root_hash=identity.get("release_root_hash"),
            source_manifest_sha256=source_manifest_sha256,
            release_root_hash=release_root_hash,
            split=split,
            release_lineage_proof=proof,
        ):
            return path, proof
    return None, None


def verify_feature_cache_identity(
    manifest: dict[str, Any],
    *,
    release_root_hash: str | None,
    split: str,
    task_kind: str,
    admission: str,
    release_lineage_proof: Mapping[str, Any] | None = None,
    source_manifest_sha256: str | None = None,
) -> bool:
    """Refuse a source feature cache that is not the one this build claims.

    Reading features instead of re-extracting them is only cheaper, not
    equivalent, unless the source is pinned: features from another
    backbone revision, another release, another split or a `dev_only`
    admission would enter this cache under an identity that says otherwise.

    `release_lineage_proof`, when given, lets a source whose
    `release_root_hash` names an EARLIER release stand in for this build's
    own release, but only through :func:`_lineage_proof_accepts` -- a
    proof is a claim about specific pages, not a blanket override.
    `source_manifest_sha256` (the caller's hash of this source's own
    `manifest.json`, `open_feature_sources`) is what pins the match to
    THIS directory rather than any cache with the same key count. Returns
    whether this source was admitted via the proof (for the caller to
    record `feature_sources[i].release_lineage_proof`).
    """
    identity = manifest.get("identity") or {}
    for name in ("backbone", "backbone_revision", "processor"):
        if not identity.get(name):
            raise FeatureCacheMismatch(
                f"source feature cache has no {name}: its features are pinned to "
                "nothing this identity could claim (a null backbone_revision means "
                "'unavailable', which is an honest gap and not a binding)"
            )
    storage_dtype = manifest.get("storage_dtype")
    if storage_dtype != "float16":
        raise FeatureCacheMismatch(
            f"source feature cache storage_dtype is {storage_dtype!r}, not 'float16' "
            "-- ExactTextCacheRow.features is float16 and a cast from anything else "
            "would change the values this build claims to have copied"
        )
    source_release_root_hash = identity.get("release_root_hash")
    lineage_used = False
    if source_release_root_hash != release_root_hash:
        if not _lineage_proof_accepts(
            manifest,
            source_release_root_hash=source_release_root_hash,
            source_manifest_sha256=source_manifest_sha256,
            release_root_hash=release_root_hash,
            split=split,
            release_lineage_proof=release_lineage_proof,
        ):
            raise FeatureCacheMismatch(
                f"source feature cache release_root_hash {source_release_root_hash!r} "
                f"!= this build's {release_root_hash!r}: those are features of another "
                "release (pass --release-lineage-proof to admit it via a proven lineage)"
            )
        lineage_used = True
    if identity.get("split") != split:
        raise FeatureCacheMismatch(
            f"source feature cache split {identity.get('split')!r} != {split!r}"
        )
    if identity.get("task_kind") != task_kind:
        raise FeatureCacheMismatch(
            f"source feature cache task_kind {identity.get('task_kind')!r} != {task_kind!r}"
        )
    source_admission = identity.get("admission")
    if admission == "strict" and source_admission != "strict":
        raise FeatureCacheMismatch(
            f"source feature cache admission {source_admission!r} cannot back a "
            "'strict' exact-text cache (RFC005 §9.2, RFC003-ERRATA-001 E5)"
        )
    return lineage_used


def verify_feature_grammar_binding(
    manifest: dict[str, Any],
    *,
    source_dir: Path | str,
    capability_profile: CapabilityProfile,
    legacy: Mapping[Path, str],
) -> dict[str, Any]:
    """Refuse a source feature cache whose grammar does not match the
    capability profile this build claims, unless the caller has explicitly
    flagged it as a legacy source.

    A source conforms iff BOTH `identity.grammar_profile ==
    capability_profile.ast_profile` and `identity.capability_profile_hash ==
    capability_profile.profile_hash`. Every pre-DEC-024 feature cache
    records both as `null`, so this is a real refusal for the S1 canary
    caches, not a theoretical one -- they need `--legacy-feature-grammar`.

    `legacy` maps a source directory to the operator's stated reason; a
    key is matched by `Path.resolve()` equality, never by substring, so
    `.../train` cannot cover `.../train-textframe-v11` by accident. A
    source that conforms but is ALSO named in `legacy` is refused too: a
    stale flag that covers nothing is worse than none, because it reads as
    evidence for a check that never ran.
    """
    identity = manifest.get("identity") or {}
    grammar_profile = identity.get("grammar_profile")
    capability_profile_hash = identity.get("capability_profile_hash")
    conforms = (
        grammar_profile == capability_profile.ast_profile
        and capability_profile_hash == capability_profile.profile_hash
    )
    resolved = Path(source_dir).resolve()
    legacy_reason = next(
        (reason for dir_, reason in legacy.items() if Path(dir_).resolve() == resolved),
        None,
    )
    if conforms and legacy_reason is not None:
        raise FeatureCacheMismatch(
            f"{source_dir}: --legacy-feature-grammar names this source, but it "
            f"already conforms to grammar_profile={capability_profile.ast_profile!r} "
            f"/ capability_profile_hash={capability_profile.profile_hash!r} -- the "
            "flag covers nothing and its reason is stale"
        )
    if not conforms and legacy_reason is None:
        raise FeatureCacheMismatch(
            f"{source_dir}: grammar_profile={grammar_profile!r} / "
            f"capability_profile_hash={capability_profile_hash!r} does not match "
            f"this build's grammar_profile={capability_profile.ast_profile!r} / "
            f"capability_profile_hash={capability_profile.profile_hash!r} -- pass "
            f"--legacy-feature-grammar {source_dir} '<reason>' to admit it "
            "explicitly"
        )
    return {
        "dir": str(source_dir),
        "entries": manifest.get("entries"),
        "grammar_profile": grammar_profile,
        "capability_profile_hash": capability_profile_hash,
        "legacy_reason": legacy_reason,
    }


#: Compared across every supplement against the primary's `identity`
#: block (`open_feature_sources`). `storage_dtype` is a TOP-LEVEL manifest
#: key, not part of `identity`, and is compared separately.
_CROSS_SOURCE_IDENTITY_FIELDS = (
    "backbone",
    "backbone_revision",
    "processor",
    "feature_tap",
    "max_pixels",
    "multiscale",
    "merged_grid",
)


@dataclass(frozen=True, slots=True)
class FeatureSource:
    """Random access to one or more source caches' entries by page.

    `index` maps `(document_bundle_id, page_index)` to `(reader_ordinal,
    shard_ordinal)`: which reader in `readers` holds the page, and that
    reader's SHARD ordinal (its `shard_index` position), not a flat entry
    position -- `validate_manifest` guarantees each shard's `rows` key
    list is exactly that shard's keys, so the shard is the unit of lookup.
    """

    readers: tuple[FeatureCacheReader, ...]
    index: dict[tuple[str, int], tuple[int, int]]
    #: One `verify_feature_grammar_binding` record per source, in the same
    #: order as `readers` -- what `build()` records to the manifest as
    #: `feature_sources` (T3, design section 3).
    grammar_records: tuple[dict[str, Any], ...] = ()

    def entry(self, document_bundle_id: str, page_index: int) -> CacheEntry | None:
        """The whole `CacheEntry`, not just its features.

        Features and `grid_thw` travel together and must be taken from ONE
        entry: a second lookup by id could pair one page's grid with
        another page's features, and `_verify_row_integrity`'s
        token-count check cannot see that when the two pages have the same
        visual-token count (`(1, 70, 50)` and `(1, 50, 70)` both yield 875,
        and both occur in the real candidate cache). Returning the entry is
        what makes that pairing unrepresentable here rather than merely
        unlikely.
        """
        location = self.index.get((document_bundle_id, page_index))
        if location is None:
            return None
        reader_ordinal, shard_ordinal = location
        reader = self.readers[reader_ordinal]
        for row in reader.rows_of_shard(shard_ordinal):
            if (
                row["document_bundle_id"] == document_bundle_id
                and int(row["page_index"]) == page_index
            ):
                return CacheEntry(**row)
        raise FeatureCacheMismatch(
            f"{document_bundle_id} page {page_index}: the source manifest's "
            f"shard {shard_ordinal} key list names the page but the loaded "
            "shard does not hold it -- a corrupt source cache is refused, "
            "never defaulted"
        )


@dataclass(frozen=True, slots=True)
class PageImageEntry:
    """`PageImageSource.entry`'s result: the `CacheEntry` fields
    `_materialize_core` reads, with no features and the PNG reference."""

    grid_thw: torch.Tensor
    page_image: dict[str, Any]
    features: None = None


@dataclass(frozen=True, slots=True)
class PageImageSource:
    """DEC-056: the "feature source" of a page-image (7.0.0) build -- it
    holds no features. `index` maps every cohort page to its release-pinned
    PNG `ArtifactRef`; `entry` reads the PNG (sha256 + size verified by
    `read_object`) only to derive the row's `grid_thw` from its size under
    `policy`, exactly as the processor will at training time."""

    policy: onfly_visual.VisualInputPolicy
    store_dir: Path
    index: dict[tuple[str, int], dict[str, Any]]
    grammar_records: tuple[dict[str, Any], ...] = ()

    def entry(self, document_bundle_id: str, page_index: int) -> PageImageEntry | None:
        from PIL import Image

        from unflatten.dataset.gates.store import read_object

        ref = self.index.get((document_bundle_id, page_index))
        if ref is None:
            return None
        if ref.get("mime_type") != "image/png":
            raise FeatureCacheMismatch(
                f"{document_bundle_id} page {page_index}: page image is "
                f"{ref.get('mime_type')!r}, not image/png"
            )
        data = read_object(Path(self.store_dir), dict(ref))
        width, height = Image.open(io.BytesIO(data)).size
        return PageImageEntry(
            grid_thw=torch.tensor(self.policy.grid_thw_for(width, height), dtype=torch.long),
            page_image=dict(ref),
        )


def page_image_source_record(policy: onfly_visual.VisualInputPolicy) -> dict[str, Any]:
    """What a page-image build records as its one `feature_sources` entry."""
    return {
        "kind": "page_image",
        "visual_input_policy": policy.payload(),
        "visual_input_policy_hash": policy.policy_hash(),
    }


def open_feature_sources(
    primary: Path | str,
    supplements: Sequence[Path | str] = (),
    *,
    release_root_hash: str | None,
    split: str,
    task_kind: str,
    admission: str,
    capability_profile: CapabilityProfile = BOUND_CAPABILITY_PROFILE,
    legacy_feature_grammar: Mapping[Path, str] | None = None,
    release_lineage_proof: Mapping[str, Any] | Sequence[Any] | None = None,
    release_lineage_proof_path: Path | str | Sequence[Path | str] | None = None,
) -> FeatureSource:
    """Open and pin the primary feature cache plus any supplement caches,
    indexed by page.

    Each source (primary first, then `supplements` in order) passes
    `verify_feature_cache_identity` on its own. Every supplement is then
    additionally required to match the primary on the fields that name
    WHICH features a page's row holds (`_CROSS_SOURCE_IDENTITY_FIELDS`,
    plus the top-level `storage_dtype`) -- a mismatch there would let two
    sources' rows disagree about what "the features of this page" means
    while both still pass the single-source identity check. A page key
    present in more than one source is refused outright: nothing here
    merges or prefers one source's row over another's.

    The index is built from each manifest's per-shard `rows` key lists --
    a JSON read, validated by `validate_manifest` to be exactly each
    shard's keys. The old implementation read every entry of the cache
    (every 3.2 TB of features on the 2026-09-14 build) just to learn which
    position held which key, which cost ~65 minutes before a single page
    was attempted.

    Each source is also checked by :func:`verify_feature_grammar_binding`
    against `capability_profile` -- a source whose `grammar_profile`/
    `capability_profile_hash` do not match is refused unless it is named
    in `legacy_feature_grammar` (T3, design section 3). The per-source
    records land on the returned `FeatureSource.grammar_records`.

    `release_lineage_proof`, when given, is forwarded to
    `verify_feature_cache_identity` for every source (T11, design section
    8): a source whose own `release_root_hash` names an earlier release is
    admitted iff a proof accepts it -- one proof built over several
    feature cache directories (`feature_cache_dirs`) admits every one of
    them, each matched to its own entry by `manifest.json` sha256 (T11
    follow-up: never by key count alone, and never against the proof-wide
    total, which is a union across every named dir). Several proofs may
    be given (repeatable `--release-lineage-proof`): each source is
    matched against the FIRST proof that accepts it, which is what lets
    sources of different provenance (each naming its own earlier
    release) share one build. A source admitted this way gets its
    grammar record annotated with `release_lineage_proof:
    {path, proof_hash, from, to, manifest_sha256, keys_checked}`, the last
    two naming THIS source's own matched entry. A single mapping behaves
    exactly as before.
    """
    proofs = _normalize_proofs(release_lineage_proof, release_lineage_proof_path)
    dirs = [primary, *supplements]
    legacy = legacy_feature_grammar or {}
    readers: list[FeatureCacheReader] = []
    grammar_records: list[dict[str, Any]] = []
    index: dict[tuple[str, int], tuple[int, int]] = {}
    primary_identity: dict[str, Any] = {}
    primary_storage_dtype: Any = None
    for reader_ordinal, feature_cache_dir in enumerate(dirs):
        reader = FeatureCacheReader(feature_cache_dir)
        source_manifest_sha256 = (
            feature_cache_manifest_sha256(feature_cache_dir)
            if proofs
            else None
        )
        matched_path, matched_proof = _match_proof(
            proofs,
            manifest=reader.manifest,
            source_manifest_sha256=source_manifest_sha256,
            release_root_hash=release_root_hash,
            split=split,
        )
        lineage_used = verify_feature_cache_identity(
            reader.manifest,
            release_root_hash=release_root_hash,
            split=split,
            task_kind=task_kind,
            admission=admission,
            release_lineage_proof=matched_proof,
            source_manifest_sha256=source_manifest_sha256,
        )
        grammar_record = verify_feature_grammar_binding(
            reader.manifest,
            source_dir=feature_cache_dir,
            capability_profile=capability_profile,
            legacy=legacy,
        )
        if lineage_used:
            assert matched_proof is not None
            matched_entry = next(
                (
                    d
                    for d in matched_proof.get("feature_cache_dirs") or []
                    if d.get("manifest_sha256") == source_manifest_sha256
                ),
                {},
            )
            grammar_record = {
                **grammar_record,
                "release_lineage_proof": {
                    "path": str(matched_path),
                    "proof_hash": matched_proof.get("proof_hash"),
                    "from": matched_proof.get("from_release_root_hash"),
                    "to": matched_proof.get("to_release_root_hash"),
                    "manifest_sha256": source_manifest_sha256,
                    "keys_checked": matched_entry.get("keys_checked"),
                },
            }
        grammar_records.append(grammar_record)
        identity = reader.manifest.get("identity") or {}
        storage_dtype = reader.manifest.get("storage_dtype")
        if reader_ordinal == 0:
            primary_identity = identity
            primary_storage_dtype = storage_dtype
        else:
            for field in _CROSS_SOURCE_IDENTITY_FIELDS:
                if identity.get(field) != primary_identity.get(field):
                    raise FeatureCacheMismatch(
                        f"{feature_cache_dir}: {field} is {identity.get(field)!r}, "
                        f"the primary's is {primary_identity.get(field)!r} -- a "
                        "supplement source must match the primary's feature "
                        "identity"
                    )
            if storage_dtype != primary_storage_dtype:
                raise FeatureCacheMismatch(
                    f"{feature_cache_dir}: storage_dtype is {storage_dtype!r}, "
                    f"the primary's is {primary_storage_dtype!r} -- a "
                    "supplement source must match the primary's feature "
                    "identity"
                )
        local_index: dict[tuple[str, int], int] = {}
        for shard_ordinal, shard in enumerate(reader.manifest["shard_index"]):
            for bundle, page in shard["rows"]:
                local_index[(bundle, int(page))] = shard_ordinal
        overlap = set(local_index) & set(index)
        if overlap:
            first_key = sorted(overlap)[0]
            raise FeatureCacheMismatch(
                f"{feature_cache_dir}: {len(overlap)} page key(s) already "
                f"present in an earlier source, e.g. {first_key!r} -- a "
                "page's features must come from exactly one source"
            )
        for key, shard_index_ordinal in local_index.items():
            index[key] = (reader_ordinal, shard_index_ordinal)
        readers.append(reader)
    return FeatureSource(
        readers=tuple(readers), index=index, grammar_records=tuple(grammar_records)
    )


def feature_cache_keys(feature_cache_dir: Path | str) -> set[tuple[str, int]]:
    """Every `(document_bundle_id, page_index)` key a feature cache's
    manifest indexes -- a manifest-only read, the same discipline
    `open_feature_sources` uses for its own index."""
    reader = FeatureCacheReader(feature_cache_dir)
    return {
        (bundle, int(page))
        for shard in reader.manifest["shard_index"]
        for bundle, page in shard["rows"]
    }


def _read_manifest(cache_dir: Path) -> dict[str, Any] | None:
    """Read manifest.json from a cache directory, or return None if not found.
    Does not validate -- used for auto-detection of cache type."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _is_exact_text_cache_dir(cache_dir: Path | str) -> bool:
    """Auto-detect if a directory holds an exact-text cache (not a structural
    feature cache). Returns True iff manifest.json exists and its `cache_family`
    equals `CACHE_FAMILY` ("exact_text_v1").

    This is what lets `scripts/v15_chain.sh` pass `cache-exact-text-v14-r4`
    (an exact-text cache) as the 3rd argument without a flag, when the
    original design expected only a structural feature cache there.
    """
    manifest = _read_manifest(Path(cache_dir))
    if manifest is None:
        return False
    return manifest.get("cache_family") == CACHE_FAMILY


def supplement_keys(
    *,
    release_dir: Path | str,
    store_dir: Path | str,
    split: str,
    task_kind: str,
    feature_cache_dirs: Sequence[Path | str],
    allow_dev_only: bool = False,
    single_page_only: bool = False,
) -> list[tuple[str, int]]:
    """Pages this build would admit that no named feature cache holds yet.

    Diffs against the FEATURE cache keys (:func:`feature_cache_keys`),
    never the exact-text cache -- the exact-text cache does not exist yet
    on a first build, and a page becomes a legitimate build target the
    moment its features exist, independent of whether it was ever
    materialized into an exact-text row before (design section 8).
    """
    admitted: set[tuple[str, int]] = set()
    for sample in iter_page_samples(
        release_dir, store_dir, split, task_kind,
        require_privacy_review=not allow_dev_only,
    ):
        if single_page_only and sample.page_count != 1:
            continue
        admitted.add((sample.document_bundle_id, sample.page_index))
    have: set[tuple[str, int]] = set()
    for feature_cache_dir in feature_cache_dirs:
        have |= feature_cache_keys(feature_cache_dir)
    return sorted(admitted - have)


class MixedCacheTypeMismatch(FeatureCacheMismatch):
    """Primary and supplement source caches are of different types: one is a
    structural feature cache, the other is an exact-text cache. All sources
    must be of the same type for auto-detection to work."""


def _grammar_record_from_exact_text_manifest(
    cache_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Build a grammar_record-like dict from an exact-text cache manifest.

    The structural `open_feature_sources()` produces `grammar_records` for
    `feature_sources` in the output manifest. When reading from an exact-text
    cache (which has no `grammar_profile` / `legacy_feature_grammar` in the
    same sense), we produce an analogous record from what we can read.
    """
    identity = manifest.get("identity") or {}
    return {
        "dir": str(cache_dir),
        "entries": manifest.get("entries"),
        "source_type": "exact_text_cache",
        "capability_profile_hash": identity.get("capability_profile_hash"),
        "backbone_artifact_hash": identity.get("backbone_artifact_hash"),
        "processor_artifact_hash": identity.get("processor_artifact_hash"),
        "feature_tap": identity.get("feature_tap"),
        "max_pixels": identity.get("max_pixels"),
        "admission": identity.get("admission"),
        "cache_version": manifest.get("cache_version"),
    }


def _check_all_sources_same_type(
    primary: Path, supplements: Sequence[Path]
) -> tuple[bool, Path | None]:
    """Check that all source caches are of the same type.

    Returns `(is_exact_text, first_mismatch)` where:
    - `is_exact_text`: True if primary is an exact-text cache
    - `first_mismatch`: None if all match, else the first differing dir
    """
    dirs = [primary, *supplements]
    primary_is_et = _is_exact_text_cache_dir(primary)
    for cache_dir in dirs[1:]:
        cache_is_et = _is_exact_text_cache_dir(cache_dir)
        if cache_is_et != primary_is_et:
            return (primary_is_et, cache_dir)
    return (primary_is_et, None)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


#: The owner-frozen runtime-limits policy `composite_grammar_hash` is
#: computed over. It ships in this repository, and `build_s1_bindings.py`
#: loads it from exactly here -- which is what lets this build recompute
#: the §4.5 descriptor instead of trusting the hash it is handed.
RUNTIME_LIMITS_POLICY_PATH = (
    Path(__file__).parent / "policies" / "s1_runtime_limits_policy.json"
)


def verify_composite_grammar_binding(
    *,
    controller: CompositeGrammarController,
    bindings: dict[str, str],
    profile: CapabilityProfile = BOUND_CAPABILITY_PROFILE,
) -> None:
    """Refuse a `--bindings` whose `composite_grammar_hash` does not
    describe the grammar this build will actually replay.

    Without this the failure is silent and survives on disk: the rows are
    materialized under the live controller (bound included, DEC-016 C) and
    stamped with whatever hash the bindings file carries, so a stale
    bindings file -- the obvious thing to reach for mid-rebuild -- produces
    a cache whose identity names a grammar it was never built under.

    The policy document is read from the repository rather than supplied,
    and cross-checked against the `runtime_limits_policy_hash` this build
    already receives, so it cannot silently verify against a policy the
    bindings were not built from. A `--runtime-limits-policy` input would
    let a build run against an out-of-tree policy; that is a separate
    change, and until it exists an out-of-tree policy is refused rather
    than worked around.
    """
    missing = [
        name
        for name in ("composite_grammar_hash", "runtime_limits_policy_hash")
        if not bindings.get(name)
    ]
    if missing:
        raise BindingsInvalid(
            f"cannot verify the grammar this build replays: {', '.join(missing)} "
            "absent or empty. An unverifiable binding is refused, never skipped"
        )
    policy = load_runtime_limits_policy(RUNTIME_LIMITS_POLICY_PATH)
    supplied_policy_hash = bindings["runtime_limits_policy_hash"]
    actual_policy_hash = runtime_limits_policy_hash(policy)
    if actual_policy_hash != supplied_policy_hash:
        raise BindingsInvalid(
            f"runtime_limits_policy_hash {supplied_policy_hash} does not match "
            f"{RUNTIME_LIMITS_POLICY_PATH.name}'s own {actual_policy_hash} -- this "
            "build cannot verify composite_grammar_hash against a policy document "
            "it does not have, and will not skip the check instead"
        )
    expected = composite_grammar_hash(
        build_composite_grammar_descriptor(
            controller=controller,
            profile=profile,
            runtime_limits_policy=policy,
        )
    )
    supplied = bindings["composite_grammar_hash"]
    if expected != supplied:
        raise BindingsInvalid(
            f"composite_grammar_hash {supplied} does not describe the grammar this "
            f"build replays, which hashes to {expected}. The rows would be "
            "materialized under one grammar and stamped with another. Regenerate "
            "the bindings (build_s1_bindings.py) -- see docs/RUNBOOK.md, "
            "'composite_grammar_hash sa 2026-09-09 pohol'"
        )


#: The owner-frozen policy documents whose hashes `--bindings` supplies.
#: They ship in this repository and `build_s1_bindings.py` reads them from
#: exactly here, which is what lets this build re-derive the hashes rather
#: than trust the ones it is handed.
STAGE_POLICY_PATH = Path(__file__).parent / "policies" / "s1_stage_policy.json"
CONDITIONING_POLICY_PATH = (
    Path(__file__).parent / "policies" / "s1_conditioning_policy.json"
)


def _component_lock_residue(bindings: Mapping[str, Any], bindings_path: Path) -> list[str]:
    """Everything that can be checked about `component_lock_hash`, and a
    statement of what cannot.

    **The residue, stated rather than left for a reader to discover:** the
    component lock is assembled outside this repository, so there is no
    in-repo producer to compare against. A `component_lock_hash` that is
    *stale but well formed* -- correct shape, provenance sibling present,
    naming artefacts this build was never assembled from -- **passes this
    check.** That hole is real, is not closed here, and is the reason the
    other six keys are re-derived instead of shape-checked: a shape check
    is what this key is stuck with, not what the others settle for.
    """
    failures: list[str] = []
    supplied = bindings.get("component_lock_hash")
    if "component_lock_hash" not in bindings:
        failures.append(
            "component_lock_hash is absent from the bindings file (absent, not "
            "null) -- it has no in-repo producer, so an absent value cannot be "
            "reconstructed and is refused"
        )
    elif not isinstance(supplied, str) or not _SHA256_RE.match(supplied):
        failures.append(
            f"component_lock_hash must be 'sha256:<64 hex>', got {supplied!r}"
        )
    provenance = bindings_path.with_suffix(".provenance.json")
    if not provenance.is_file():
        failures.append(
            f"no provenance sibling {provenance.name} beside {bindings_path.name} -- "
            "component_lock_hash is produced outside this repository, so the "
            "provenance document is the only evidence it was produced at all"
        )
    return failures


def verify_bindings_freshness(
    *,
    bindings: Mapping[str, Any],
    bindings_path: Path | str,
    controller: CompositeGrammarController,
    capability_profile: CapabilityProfile = BOUND_CAPABILITY_PROFILE,
) -> None:
    """Refuse a `--bindings` file that no longer describes this repository.

    Six of the seven keys have in-repo producers and are **re-derived and
    compared**: `loss_unit_schema_hash`, `stage_policy_hash`,
    `conditioning_policy_hash` and `head_registry_hash` here, plus
    `composite_grammar_hash` and `runtime_limits_policy_hash` delegated to
    :func:`verify_composite_grammar_binding`. The seventh,
    `component_lock_hash`, is out-of-repo; see :func:`_component_lock_residue`
    for what that leaves open.

    Motivated by a measured disagreement on disk, not a caution: of the
    three bindings files in the operator's data directory, only
    `bindings-s1-c2.json` matches the code. The other two carry a
    `loss_unit_schema_hash` this repository stopped producing and **omit
    `head_registry_hash` entirely** -- six keys, not seven. Presence is
    therefore asserted before equality: a `.get()` returning `None` cannot
    distinguish an absent key from a stored null, and it is the absent key
    that is on disk.

    Every failure is collected and reported together. An operator who
    fixes one cause, re-runs a night-long build and meets the next has
    paid twice for one file.

    **Invariant -- derive to compare, never derive to bind.** The derived
    values below reach exactly two places: an equality test and an error
    message. None of them is written to an artefact, to
    :class:`ExactTextCacheIdentity`, or to a manifest. That is what makes
    this function's carve-out from
    `test_policy_producers.py::test_typed_modules_do_not_call_or_import_the_head_registry_hash_producer`
    principled rather than convenient: a value that only ever reaches a
    comparison cannot make any binding self-satisfying -- it can only make
    a mismatch louder. A site that derived *and then bound* would be one
    site too, which is why `M13`, not the site count, is what guards this.

    Refuses, never warns. A build proceeding on stale bindings produces a
    cache whose identity names artefacts it was never built under, and
    that survives on disk long after the run is forgotten.
    """
    # Imported here rather than at module scope so the single carved-out
    # derivation site is this function, not the module: a module-level
    # import would make every function in this file a potential binder.
    from unflatten.model.property_head_registry import PROPERTY_HEAD_REGISTRY_HASH

    bindings_path = Path(bindings_path)
    rederived: tuple[tuple[str, str, str], ...] = (
        (
            "loss_unit_schema_hash",
            loss_unit_schema_hash(),
            "unflatten/model/loss_unit.py::loss_unit_schema_hash",
        ),
        (
            "stage_policy_hash",
            stage_policy_hash(load_stage_policy(STAGE_POLICY_PATH)),
            f"unflatten/model/stage_policy.py::stage_policy_hash over "
            f"{STAGE_POLICY_PATH.name}",
        ),
        (
            "conditioning_policy_hash",
            conditioning_policy_hash(load_conditioning_policy(CONDITIONING_POLICY_PATH)),
            f"unflatten/model/conditioning_policy.py::conditioning_policy_hash over "
            f"{CONDITIONING_POLICY_PATH.name}",
        ),
        (
            "head_registry_hash",
            PROPERTY_HEAD_REGISTRY_HASH,
            "unflatten/model/property_head_registry.py::PROPERTY_HEAD_REGISTRY_HASH",
        ),
    )

    failures: list[str] = []
    for name, derived, producer in rederived:
        if name not in bindings:
            failures.append(
                f"{name} is absent from the bindings file -- absent, not null; "
                f"{producer} derives {derived}, and this build refuses to supply "
                "a key the file does not carry"
            )
            continue
        supplied = bindings[name]
        if supplied != derived:
            failures.append(
                f"{name} {supplied!r} does not match the {derived} this "
                f"repository derives from {producer}"
            )
    failures.extend(_component_lock_residue(bindings, bindings_path))
    if failures:
        raise BindingsInvalid(
            f"{bindings_path} is stale against this repository "
            f"({len(failures)} cause(s)): "
            + "; ".join(failures)
            + " -- regenerate the bindings (build_s1_bindings.py, see "
            "docs/RUNBOOK.md). A build on stale bindings writes a cache whose "
            "identity names artefacts it was never built under"
        )
    verify_composite_grammar_binding(
        controller=controller, bindings=dict(bindings), profile=capability_profile
    )


def build_materializer_bundle(
    *,
    binding: QwenTextBinding,
    tokenizer: Any,
    bindings: dict[str, str],
    text_encoding_budget_ops: int = DEFAULT_TEXT_ENCODING_BUDGET_OPS,
    capability_profile: CapabilityProfile = BOUND_CAPABILITY_PROFILE,
    target_policy: TargetPolicy | None = None,
) -> MaterializerBundle:
    """The vocabulary, controller and materializer this build runs with.

    One triple per build, so the identity and the rows cannot name
    different grammars, vocabularies or capability profiles.

    `capability_profile` defaults to `BOUND_CAPABILITY_PROFILE` -- every
    existing caller keeps today's behaviour unchanged. DEC-024's large
    build and S2 pass `TYPED_WIDENED_V1_CAPABILITY_PROFILE` explicitly.
    """
    profile = capability_profile
    profile_id = profile.profile_id
    vocabulary = cv.make_composite_vocabulary(
        tokenizer_artifact_set_hash=binding.tokenizer_artifact_set_hash,
        allowlist_payload=binding.allowlist,
    )
    controller = CompositeGrammarController(
        vocabulary=vocabulary,
        capability_profile_id=profile_id,
        structural_profile=profile.ast_profile,
        capability_excluded_local_ids=profile.excluded_structural_local_ids,
    )
    verify_composite_grammar_binding(controller=controller, bindings=bindings, profile=profile)
    materializer = TrainingTargetMaterializer(
        vocabulary=vocabulary,
        controller=controller,
        composite_grammar_hash=bindings["composite_grammar_hash"],
        capability_profile_id=profile_id,
        capability_profile_hash=profile.profile_hash,
        text_encoding_budget_ops=text_encoding_budget_ops,
        tokenizer=tokenizer,
        materializer_hash=training_target_materializer_hash(
            binding=binding,
            bindings=bindings,
            capability_profile=profile,
            target_policy=target_policy,
        ),
        head_registry_hash=bindings["head_registry_hash"],
        target_policy=target_policy,
    )
    return MaterializerBundle(
        vocabulary=vocabulary,
        controller=controller,
        materializer=materializer,
        capability_profile_id=profile_id,
    )


def training_target_materializer_hash(
    *,
    binding: QwenTextBinding,
    bindings: dict[str, str],
    capability_profile: CapabilityProfile = BOUND_CAPABILITY_PROFILE,
    target_policy: TargetPolicy | None = None,
) -> str:
    """Really derived, not supplied: the materializer descriptor's own hash
    over this build's vocabulary, grammar, profile and LossUnit schema --
    and, when selected, the v16 target policy (DEC-055), so a v16 cache can
    never share this hash with a pre-v16 one."""
    return materializer_descriptor_hash(
        build_materializer_descriptor(
            loss_unit_schema_hash=bindings["loss_unit_schema_hash"],
            composite_vocabulary_hash=binding.composite_vocabulary_hash,
            composite_grammar_hash=bindings["composite_grammar_hash"],
            capability_profile_hash=capability_profile.profile_hash,
            capability_profile_id=capability_profile.profile_id,
            target_policy=None if target_policy is None else target_policy.descriptor(),
        )
    )


def build_identity(
    *,
    feature_manifest: dict[str, Any],
    binding: QwenTextBinding,
    bindings: dict[str, str],
    release_root_hash: str | None,
    split: str,
    task_kind: str,
    admission: str,
    schema_versions: SchemaVersions,
    capability_profile: CapabilityProfile = BOUND_CAPABILITY_PROFILE,
    target_policy: TargetPolicy | None = None,
    visual_input_policy: onfly_visual.VisualInputPolicy | None = None,
) -> ExactTextCacheIdentity:
    """Everything this cache's rows are bound to (design §8).

    A structural feature cache names the feature producer as
    `backbone_revision`/`processor`; an exact-text cache used as a feature
    source already carries the normalized `backbone_artifact_hash`/
    `processor_artifact_hash` pair. Both paths are identity-preserving.

    `visual_input_policy` (DEC-056, page-image build): the four visual
    fields come from the policy instead -- `feature_tap` becomes
    `"onfly-visual:<policy_hash>"` -- and `feature_manifest` is not read.
    """
    if visual_input_policy is not None:
        feature_identity: dict[str, Any] = {
            "backbone_artifact_hash": visual_input_policy.backbone_artifact_hash,
            "processor_artifact_hash": visual_input_policy.processor_artifact_hash,
            "feature_tap": visual_input_policy.feature_tap(),
            "max_pixels": visual_input_policy.max_pixels,
        }
    else:
        feature_identity = feature_manifest.get("identity") or {}
    return ExactTextCacheIdentity(
        dataset_release_root_hash=release_root_hash,
        split=split,
        task_kind=task_kind,
        admission=admission,
        target_projector_version=PROJECTOR_VERSION,
        target_bundle_schema_version=schema_versions.target_bundle,
        fragment_schema_version=schema_versions.fragment,
        mask_projection_schema_version=schema_versions.mask_projection,
        backbone_artifact_hash=feature_identity.get("backbone_artifact_hash")
        or feature_identity["backbone_revision"],
        processor_artifact_hash=feature_identity.get("processor_artifact_hash")
        or feature_identity["processor"],
        feature_tap=feature_identity.get("feature_tap") or NO_FEATURE_TAP,
        max_pixels=feature_identity.get("max_pixels") or NO_MAX_PIXELS,
        composite_vocabulary_hash=binding.composite_vocabulary_hash,
        tokenizer_artifact_set_hash=binding.tokenizer_artifact_set_hash,
        qwen_text_allowlist_hash=binding.qwen_text_allowlist_hash,
        composite_grammar_hash=bindings["composite_grammar_hash"],
        capability_profile_hash=capability_profile.profile_hash,
        training_target_materializer_hash=training_target_materializer_hash(
            binding=binding,
            bindings=bindings,
            capability_profile=capability_profile,
            target_policy=target_policy,
        ),
        loss_unit_schema_hash=bindings["loss_unit_schema_hash"],
        stage_policy_hash=bindings["stage_policy_hash"],
        conditioning_policy_hash=bindings["conditioning_policy_hash"],
        runtime_limits_policy_hash=bindings["runtime_limits_policy_hash"],
        component_lock_hash=bindings["component_lock_hash"],
        head_registry_hash=bindings["head_registry_hash"],
    )


# ---------------------------------------------------------------------------
# One page
# ---------------------------------------------------------------------------


def page_schema_versions(
    targets: dict[str, Any] | None,
    fragment: dict[str, Any] | None,
    mask_projection: dict[str, Any] | None,
) -> SchemaVersions | None:
    """The three schema versions one page declares, or `None` when it does
    not declare all three.

    Separate from :func:`row_for_page` because the FIRST page a build
    reaches is what fixes the cache-wide identity: a page missing one of
    these must not fix it to a null that every later page then mismatches,
    which would leave a build of one skip and 149 "schema mismatches"
    looking like a corpus property.
    """
    versions = SchemaVersions(
        target_bundle=(targets or {}).get("target_schema_version"),
        fragment=(fragment or {}).get("fragment_schema_version"),
        mask_projection=(mask_projection or {}).get("mask_projection_schema_version"),
    )
    if None in (versions.target_bundle, versions.fragment, versions.mask_projection):
        return None
    return versions


def _piece_kind_counts(pieces: dict[str, Any]) -> dict[str, int]:
    """DEC-033 design §4 ledger field `pieces`: how many of this page's
    present nodes are a continuation piece of a node split across pages,
    by kind. `"whole"` (the common case -- not split here) is deliberately
    excluded: the field answers "how much of this page is a split piece",
    not a census of every node."""
    counts = {"first": 0, "middle": 0, "last": 0}
    for kind in pieces.values():
        if kind in counts:
            counts[kind] += 1
    return counts


def _table_pieces_continuation(fragment: dict[str, Any], pieces: dict[str, Any]) -> int:
    """DEC-033 design §4 ledger field `table_pieces_continuation`: how
    many `table` nodes present on this page are themselves a continuation
    piece (`page_pieces.piece_kind` reads a table's OWN `continuation`,
    same as any other node -- only a `row`'s kind is overridden to its
    table's). A separate, narrower count from `pieces` above because a
    table split across pages is the one case DEC-033 §4 calls out by
    name ("Tabuľka rozdelená cez stranu sú dve tabuľky")."""
    return sum(
        1
        for entry in fragment.get("slices", ())
        if entry.get("node_type") == "table" and pieces.get(entry["node_id"]) != "whole"
    )


def _synthesized_ancestor_opens() -> int:
    """DEC-033 design §4 ledger field `synthesized_ancestor_opens`: the
    number of container `_OPEN` tokens this page emits for a node that is
    not itself observed/present here but was opened anyway because some
    descendant needed it -- the `reconcile()`/bootstrap-prewalk behaviour
    of `encode_fragment.encode_fragment_with_origins` (the pre-DEC-033
    fragment-window protocol).

    Always 0 under `STANDALONE_V1_PROFILE_ID`: `encode_standalone_
    fragment_with_origins` does no reconciliation at all -- it opens
    exactly the nodes `fragment.slices` names, per its own docstring ("No
    stack, no reconciliation"), and `present` is already ancestor-closed
    (`envelope.py`'s descendant-union page rule always leaves
    `ancestor_context` empty, design §3). There is nothing this function
    could ever count for this profile; it is not a placeholder for a
    later computation, it is the true value.
    """
    return 0


def _sidecar_page_observations(
    observation_sidecar: dict[str, Any] | None, page_index: int
) -> list[dict[str, Any]] | None:
    """This page's `sidecar["pages"][page_index]["observations"]`, or
    `None` when the sidecar cannot be resolved to an entry for
    `page_index` -- refused by the caller as `skipped_no_sidecar`, never
    read as "this page observed nothing" (T1)."""
    if observation_sidecar is None:
        return None
    pages = observation_sidecar.get("pages")
    if not isinstance(pages, list) or not (0 <= page_index < len(pages)):
        return None
    page = pages[page_index]
    if page.get("page_index") != page_index:
        return None
    return page.get("observations", [])


def _apply_target_policy(
    policy: TargetPolicy,
    ledger_row: dict[str, Any],
    *,
    fragment: dict[str, Any],
    structural: dict[str, Any],
    text_spans: Sequence[Any],
    observation_sidecar: dict[str, Any] | None,
    page_index: int,
    listed_reason: str | None,
    whitespace_runs_granted: int,
    stories_rehomed: int = 0,
) -> str | None:
    """Record this page's `v16-targets.v1` facts on its ledger row (mutated,
    key `target_policy`) and return the skip reason the policy refuses the
    page with, else `None`. Every fact is recorded whether or not the page
    is refused -- a refusal must stay auditable (which list entry, what
    trailing-row count) and V1c is measurement-only until its floor is set.
    Order: the audited list, then the `spill` signature, then V1c."""
    is_spill, trailing_rows = spill_signature(
        fragment, structural, observation_sidecar or {}, page_index
    )
    ink_chars, ink_blocks = unowned_ink(text_spans, fragment, structural)
    ledger_row["target_policy"] = {
        "whitespace_runs_granted": whitespace_runs_granted,
        "stories_rehomed": stories_rehomed,
        "unowned_ink_chars": ink_chars,
        "unowned_ink_blocks": ink_blocks,
        "spill_signature": is_spill,
        "trailing_textless_rows": trailing_rows,
        "listed": listed_reason,
    }
    if listed_reason is not None:
        return "skipped_defect_listed"
    if policy.spill_refusal and is_spill:
        return "skipped_defect_spill"
    if policy.refuses_unowned_ink(ink_chars):
        return "skipped_unowned_ink"
    return None


def row_for_page(
    *,
    document_bundle_id: str,
    page_index: int,
    targets: dict[str, Any] | None,
    fragment: dict[str, Any] | None,
    features: torch.Tensor | None,
    grid_thw: torch.Tensor | None,
    mask_projection: dict[str, Any] | None,
    mask_projection_hash: str,
    mask_projection_schema_version: str | None,
    page_font_pack_hash: str | None,
    bound: MaterializerBundle,
    schema_versions: SchemaVersions,
    observation_sidecar: dict[str, Any] | None,
    min_text_coverage: float = 0.0,
    listed_reason: str | None = None,
    page_image: dict[str, Any] | None = None,
) -> tuple[
    ExactTextCacheRow | None,
    str | None,
    TextCoverage | None,
    TypedChannelCounts | None,
    dict[str, Any] | None,
    RegionChannelCounts | None,
]:
    """`(row, None, coverage, typed, ledger, region)` for a page that
    materializes and verifies, else `(None, skip_reason, coverage, typed,
    ledger, region)` naming one of :data:`SKIP_COUNTS`.
    `coverage` is `None` whenever a page is skipped before its text
    authority could be built at all (`skipped_no_observation`,
    `skipped_unencodable`, `skipped_schema_mismatch`, `skipped_typed_mask`,
    `skipped_text_authority`) -- there is no text-span authority yet to
    measure coverage over. `typed` is `None` whenever a page is skipped
    before the typed channel was derived. `ledger` (DEC-033 §5/§6/§10,
    T6 wiring) is `None` for every profile but `STANDALONE_V1_PROFILE_ID`,
    and for that profile is `None` only when skipped before the page
    projection could run (the same point `typed` goes `None` at).

    `observation_sidecar` is the WHOLE document's Observation Sidecar
    (`items.TrainingItem.observation_sidecar_data`, T1) -- this page's
    entry is `observation_sidecar["pages"][page_index]`. `region` is
    `None` whenever a page is skipped before the region channel was
    derived (the same point `typed` goes `None` at); a page whose
    sidecar cannot be resolved to an entry for `page_index` is refused as
    `skipped_no_sidecar` rather than written with a fabricated empty
    region channel.

    A page whose realised text ranges (what `materialize` actually emitted)
    do not equal its granted policy ranges (what `text_spans` authorized) is
    refused as `skipped_realisation_mismatch` -- a span granted characters
    but never realised would otherwise be silently dropped while
    `text_coverage` still counts it as granted (R-3, design §4.3).

    A page whose `text_coverage.below(min_text_coverage)` is refused
    BEFORE `materialize` runs at all (`skipped_text_coverage`, D2
    2026-09-07: 101/1,176 real cache rows had granted/store < 0.5) -- the
    default `min_text_coverage=0.0` never withdraws a page this way:
    `below` is `ratio < floor`, `ratio` is never negative, and a `None`
    ratio (zero store chars) never withdraws.

    The masks ARE applied here, through the same primitives the one
    production withdrawal path is built from
    (`build_cache._document_masked_typed` -> `targets/masks.masked_typed_for_document`
    for the document-scoped font stamp, `targets/masks.withdraw_placements_typed`
    for this page's placement withdrawal) -- split so a document's pages
    share ONE font-projection stamp instead of each re-deriving and
    re-hashing the whole `target_bundle` (`docs/plans/NALEZ-velke-dokumenty-kvadraticka-cena.md`).
    The result is read for ONE thing: the typed-property supervision
    surface the typed channel is derived from. Calling the existing
    withdrawal primitives rather than reproducing them is deliberate -- a
    second implementation of an existing withdrawal is what diverges.

    The masked bundle's own `target_bundle_hash` was always DISCARDED
    here (it is a TaskViewProjection-scoped derivative and differs from
    the canonical bundle's hash by design, `masks.apply_placement_mask`)
    -- which is exactly why this can skip computing it at all: binding it
    would break the E4 identity closure between the bundle hash this row
    binds and the one the fragment names, so every identity-bearing read
    below -- `authority_target_hash`, every `LossUnit`, the ownership
    projection hash, the fragment cross-check -- stays on the canonical
    `targets["target_bundle_hash"]`.

    Text/structural supervision withdrawal is a different axis and still
    reaches the rows through the fragment's own
    `supervision_mask`/`page_evidence`, which `build_text_span_authorities`
    and `supervised_event_indices` read directly.

    `page_font_pack_hash` is the `font_pack_hash` of the render whose
    pixels this page's mask projection was derived for
    (`build_cache.render_font_pack_hash`); it is resolved by the caller,
    which is the only place the store and the sample are in scope. A page
    whose render cannot be named is counted under `skipped_typed_mask`
    and never built with an unmasked typed surface: a projection from
    another font pack matches on `ast_hash`, validates, and masks the
    wrong runs.
    """
    structural = (targets or {}).get("structural")
    if not structural or fragment is None:
        return None, "skipped_unencodable", None, None, None, None

    if not fragment.get("slices"):
        # O-3: the aligner placed no observation on this page at all --
        # a different defect than "no structural targets" above, and one
        # that must not fall through to a later, unrelated skip reason
        # (SKIP_COUNTS docstring).
        return None, "skipped_no_observation", None, None, None, None

    if (
        targets.get("target_schema_version") != schema_versions.target_bundle
        or fragment.get("fragment_schema_version") != schema_versions.fragment
        or mask_projection_schema_version != schema_versions.mask_projection
    ):
        return None, "skipped_schema_mismatch", None, None, None, None

    target_bundle_hash = targets["target_bundle_hash"]
    if fragment.get("target_bundle_hash") != target_bundle_hash:
        raise TargetIdentityMismatch(
            f"{document_bundle_id} page {page_index}: fragment names "
            f"target_bundle_hash {fragment.get('target_bundle_hash')!r}, the target "
            f"bundle carries {target_bundle_hash!r} -- the two artifacts describe "
            "different documents (RFC003-ERRATA-001 E4)"
        )
    ownership_projection_hash = target_bundle_hash

    if page_font_pack_hash is None:
        # `masked_typed_for_document` requires the render's pack hash with
        # no default. Refusing here rather than one frame lower keeps the
        # absence a counted page property, while the three projection
        # failures below stay build aborts.
        return None, "skipped_typed_mask", None, None, None, None
    stamped_typed = _document_masked_typed(
        targets, mask_projection, render_font_pack_hash=page_font_pack_hash
    )
    masked_typed, flipped = withdraw_placements_typed(stamped_typed, fragment)

    try:
        text_spans = build_text_span_authorities(
            document_bundle_id=document_bundle_id,
            target_bundle_hash=target_bundle_hash,
            text=targets.get("text") or {"spans": []},
            fragment=fragment,
            ownership_projection_hash=ownership_projection_hash,
        )
    except TextAuthorityAlignmentError:
        return None, "skipped_text_authority", None, None, None, None

    # DEC-055 V1a (`v16-targets.v1` only): widen the grants of whitespace
    # runs framed on both sides by words this page owns, BEFORE the page
    # projection and `materialize` read `text_spans`, so V1b, the coverage
    # floor and the emitted stream all see the same grants.
    target_policy = bound.materializer.target_policy
    whitespace_runs_granted = 0
    if target_policy is not None and target_policy.whitespace_run_grant:
        text_spans, whitespace_runs_granted = grant_whitespace_runs(
            text_spans, fragment, structural
        )

    # DEC-033 §5/§6/§10 (T6 wiring): the page projection is computed ONCE,
    # here, and threaded into both the typed withdrawal below and
    # `materialize` -- the encoder, the structural supervision and the
    # typed channel all withdraw the SAME dropped set by construction,
    # never three separately-computed ones that could drift.
    dropped_node_ids: frozenset[str] = frozenset()
    rehome: dict[str, str] = {}
    ledger_row: dict[str, Any] | None = None
    if bound.capability_profile_id in STANDALONE_PAGE_PROFILE_IDS:
        try:
            dropped_node_ids, ledger_row = page_projection.project(
                fragment, structural, text_spans, **projection_kwargs(target_policy)
            )
        except page_projection.PageProjectionInvariantViolation as error:
            # A counted, named skip, never a crash: one such page raised
            # out of a pool worker and killed the whole v13b train build.
            # The acceptance step requires this bucket to be 0, so the
            # skip hides nothing; the page key goes on the log line.
            print(
                f"[exact-text-cache] skipped_projection_invariant: "
                f"{document_bundle_id} page {page_index}: {error}",
                flush=True,
            )
            return None, "skipped_projection_invariant", None, None, None, None
        # DEC-033 §3 / V1d (`v16-targets.v1` only): an inherited header/footer
        # moves into the page's own section; the story-only section is
        # dropped like any other projected node (typed and structural
        # supervision withdraw it below and in `materialize`).
        if target_policy is not None and target_policy.story_rehome:
            rehome, rehome_dropped = rehome_inherited_stories(
                fragment, structural, targets.get("typed") or {}
            )
            dropped_node_ids = dropped_node_ids | rehome_dropped
        # `page_piece_kinds` is O(n) and called exactly ONCE per page: its
        # result feeds both the typed withdrawal below and the ledger's
        # own `pieces`/`table_pieces_continuation` counts, never
        # recomputed for the second use.
        pieces = page_piece_kinds(fragment, structural)
        # DEC-033 §10 / RFC003-ERRATA-002 E2 (T7): page-split pieces lose
        # the pointers they cannot observe, on top of the placement
        # withdrawal above; nodes the projection dropped lose every
        # pointer, on top of that.
        masked_typed, typed_withdrawal_counts = withdraw_page_piece_typed(
            masked_typed, pieces, dropped_node_ids
        )
        ledger_row = {
            "document_bundle_id": document_bundle_id,
            "page_index": page_index,
            **ledger_row,
            "pieces": _piece_kind_counts(pieces),
            "table_pieces_continuation": _table_pieces_continuation(fragment, pieces),
            "synthesized_ancestor_opens": _synthesized_ancestor_opens(),
            "typed_withdrawn": typed_withdrawal_counts,
        }
        if target_policy is not None:
            refusal = _apply_target_policy(
                target_policy,
                ledger_row,
                fragment=fragment,
                structural=structural,
                text_spans=text_spans,
                observation_sidecar=observation_sidecar,
                page_index=page_index,
                listed_reason=listed_reason,
                whitespace_runs_granted=whitespace_runs_granted,
                stories_rehomed=len(rehome),
            )
            if refusal is not None:
                return None, refusal, None, None, ledger_row, None

    coverage = text_coverage(text_spans)
    if coverage.below(min_text_coverage):
        return None, "skipped_text_coverage", coverage, None, ledger_row, None

    try:
        sequence = bound.materializer.materialize(
            document_bundle_id=document_bundle_id,
            target_bundle_hash=target_bundle_hash,
            structural=structural,
            fragment=fragment,
            text_spans=text_spans,
            mask_projection_hash=mask_projection_hash,
            structural_ownership_projection_hash=ownership_projection_hash,
            structural_projection_hash=target_bundle_hash,
            dropped_node_ids=dropped_node_ids,
            rehome=rehome or None,
        )
    except (CompositeGrammarViolation, IncompleteCompositeStream):
        # T-9 (design §4.1): the label stays `skipped_grammar` regardless
        # of continuation presentation -- `build`'s loop is what ADDS the
        # `skipped_grammar_continuation` subset increment, so this page
        # still lands in the pre-existing bucket, same as it always did.
        return None, "skipped_grammar", coverage, None, ledger_row, None
    except _TEXT_ENCODING_ERRORS:
        return None, "skipped_text_encoding", coverage, None, ledger_row, None

    # R-3 (design §4.3): the adapter has no `structural` clause enforcing
    # "a granted span has exactly one op == 'text' event" -- a span the
    # fragment grants characters for but whose structural text event
    # `_interleave_text` cannot find (`training_materializer.py:633-683`)
    # is silently dropped from the token stream while `text_coverage`
    # above still counts it as granted. Cheap because it is a set
    # comparison over what `materialize` already built, not a re-derivation
    # of grant realisation (`tests/model/grant_realisation.py`'s full six
    # clauses are the authoritative, expensive definition; this is the one
    # clause -- "nothing granted went missing" -- worth paying for on every
    # build).
    granted_keys = {
        (pr.text_span_id, pr.scalar_start, pr.scalar_end)
        for pr in ScalarBoundaryProjector().project(text_spans)
    }
    realised_keys = {
        (unit.locator.text_span_id, unit.locator.scalar_start, unit.locator.scalar_end)
        for unit in sequence.loss_units
        if isinstance(unit.locator, TextRangeLocator)
    }
    if realised_keys != granted_keys:
        return None, "skipped_realisation_mismatch", coverage, None, ledger_row, None

    # T1 (design `spec-node-visual-aligner-design.md`): resolved before the
    # channel is derived, so a page whose sidecar cannot be resolved is
    # refused by name rather than written with a fabricated empty region
    # channel indistinguishable from "genuinely observed nothing".
    page_observations = _sidecar_page_observations(observation_sidecar, page_index)
    if page_observations is None:
        return None, "skipped_no_sidecar", coverage, None, ledger_row, None

    typed_targets, cardinalities, union_arms, not_opened_here = (
        build_typed_property_targets(
            masked_typed=masked_typed,
            fragment=fragment,
            sequence=sequence,
            structural=structural,
            capability_profile_id=bound.capability_profile_id,
        )
    )
    typed = TypedChannelCounts(flipped=flipped, node_not_opened_here=not_opened_here)

    region_targets, region_not_opened_here = build_region_targets(
        observations=page_observations,
        slices=fragment.get("slices", ()),
        anchors=_anchor_token_indices(sequence, structural),
    )
    region = RegionChannelCounts(node_not_opened_here=region_not_opened_here)

    row = row_from_materialized(
        sequence,
        document_bundle_id=document_bundle_id,
        fragment_id=fragment.get("fragment_id"),
        page_index=page_index,
        authority_target_hash=target_bundle_hash,
        mask_projection_hash=mask_projection_hash,
        features=features,
        # Refused, not defaulted, by `_checked_grid_thw`: a source cache
        # older than `CACHE_VERSION` 0.3.0 carries no grid, and a row
        # without one is a row `inject_visual` can never use.
        grid_thw=grid_thw,
        typed_targets=typed_targets,
        typed_cardinalities=cardinalities,
        typed_union_arms=union_arms,
        region_targets=region_targets,
        page_image=page_image,
    )
    try:
        verify_row_replay(
            row,
            vocabulary=bound.vocabulary,
            controller=bound.controller,
            capability_profile_id=bound.capability_profile_id,
        )
    except ExactTextCacheIncompatible:
        return None, "skipped_replay", coverage, typed, ledger_row, region
    return row, None, coverage, typed, ledger_row, region


def validate_min_text_coverage(min_text_coverage: float) -> None:
    """Refuse a `--min-text-coverage` outside the closed unit interval --
    a ratio floor above 1.0 or below 0.0 can never be a coverage ratio."""
    if not (0.0 <= min_text_coverage <= 1.0):
        raise ValueError(
            f"--min-text-coverage must be within [0, 1], got {min_text_coverage}"
        )


def page_node_shares(
    document_bundle_id: str,
    targets: dict[str, Any],
    fragment: dict[str, Any],
    *,
    target_policy: TargetPolicy | None = None,
) -> tuple[PageNodeShare, ...]:
    """One page's `PageNodeShare`s (P14 part 2, `coverage_cause`): for every
    text span present on this page, how many characters THIS page was
    granted and the span's full store length.

    Calls `build_text_span_authorities` a second time on data `row_for_page`
    already read for this same page -- pure Python over already-parsed
    dicts, not a second store read or a second tokenizer pass -- rather
    than widen `row_for_page`'s return contract, which ~20 existing call
    sites unpack positionally as a 4-tuple.
    """
    authorities = build_text_span_authorities(
        document_bundle_id=document_bundle_id,
        target_bundle_hash=targets["target_bundle_hash"],
        text=targets.get("text") or {"spans": []},
        fragment=fragment,
        ownership_projection_hash=targets["target_bundle_hash"],
    )
    if target_policy is not None and target_policy.whitespace_run_grant:
        # the same V1a grants `row_for_page` materialized, or the coverage
        # causes would be classified on a different grant set
        authorities, _granted = grant_whitespace_runs(
            authorities, fragment, targets.get("structural") or {}
        )
    return tuple(
        PageNodeShare(
            node_id=authority.node_id,
            full_length=len(authority.text),
            granted_chars=sum(
                end - start for grant in authority.grants for start, end in grant.char_ranges
            ),
        )
        for authority in authorities
    )


def document_coverage_causes(
    rows: list[dict[str, Any]], *, min_text_coverage: float
) -> dict[tuple[str, int], str]:
    """`(document_bundle_id, page_index) -> cause` for every row below
    `min_text_coverage` whose `node_shares` are present (P14 part 2,
    `unflatten.model.coverage_cause`). A row missing `node_shares` (older
    callers, or a row this build never attached them to) is silently
    absent from the returned mapping -- its caller decides what that means
    (`text_coverage_summary`: excluded from the five cause keys but still
    counted in `rows_below_floor`; `build`: no `coverage_cause` is
    persisted on that row).

    The ONE place this per-document classification runs: both
    `text_coverage_summary` (to count) and `build` (to attach `coverage_
    cause` to a row before `node_shares` -- a per-page-per-node breakdown
    too large to persist into the manifest at corpus scale -- is stripped)
    call this function rather than repeating the grouping logic.
    """
    by_document: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_document.setdefault(row["document_bundle_id"], []).append(row)

    causes: dict[tuple[str, int], str] = {}
    for doc_rows in by_document.values():
        page_shares: dict[int, tuple[PageNodeShare, ...]] = {}
        node_page_chars: dict[str, dict[int, int]] = {}
        for row in doc_rows:
            if row.get("node_shares") is None:
                continue
            shares = tuple(PageNodeShare(**share) for share in row["node_shares"])
            page_shares[row["page_index"]] = shares
            for share in shares:
                node_page_chars.setdefault(share.node_id, {})[row["page_index"]] = (
                    share.granted_chars
                )
        for row in doc_rows:
            shares = page_shares.get(row["page_index"])
            if shares is None:
                continue
            coverage = TextCoverage(row["granted_chars"], row["store_chars"])
            if not coverage.below(min_text_coverage):
                continue
            result = classify_sub_floor_page(shares, node_page_chars, min_text_coverage)
            causes[(row["document_bundle_id"], row["page_index"])] = result["cause"]
    return causes


def text_coverage_summary(
    rows: list[dict[str, Any]], *, min_text_coverage: float
) -> dict[str, Any]:
    """Aggregate `text_coverage_rows` (each `{document_bundle_id, page_index,
    granted_chars, store_chars, node_shares, ...}`) into the manifest's
    `text_coverage` summary: how many rows this build's floor actually
    excluded, how many pages have no store text at all (an undefined
    ratio), the min/median ratio among the rows that DO have one, and (P14
    part 2) the same split-run-vs-real-misalignment cause breakdown P14
    measured over the corpus (`unflatten.model.coverage_cause`), so a build
    reports it directly rather than leaving it to a separate census.

    The cause breakdown is computed per DOCUMENT (`document_coverage_
    causes`): a row missing `node_shares` (older callers, or rows this
    build never attached them to) is still counted in `rows_below_floor`
    but contributes nothing to the five `below_min_text_coverage_*` keys,
    which is why those five sum to at most, not always exactly,
    `rows_below_floor`.
    """
    # One `TextCoverage` per row, shared by the ratio statistics and the
    # below-floor count below: a second construction of the same pair is a
    # second reading of the same authority data.
    coverages = [
        TextCoverage(row["granted_chars"], row["store_chars"]) for row in rows
    ]
    ratios = [c.ratio for c in coverages]
    defined = [r for r in ratios if r is not None]

    cause_counts: dict[str, int] = dict.fromkeys(SUB_FLOOR_CLASSES, 0)
    for cause in document_coverage_causes(rows, min_text_coverage=min_text_coverage).values():
        cause_counts[cause] += 1

    return {
        "rows_below_floor": sum(
            1 for c in coverages if c.below(min_text_coverage)
        ),
        "rows_store_chars_zero": sum(1 for row in rows if row["store_chars"] == 0),
        "min_ratio": min(defined) if defined else None,
        "median_ratio": statistics.median(defined) if defined else None,
        **{f"below_min_text_coverage_{name}": count for name, count in cause_counts.items()},
    }


def finalize_text_coverage_rows(
    rows: list[dict[str, Any]], *, min_text_coverage: float
) -> None:
    """Mutate `rows` IN PLACE into the shape actually persisted (manifest
    `text_coverage_rows` and this build's returned summary dict, both the
    same list objects): strip the per-page-per-node `node_shares` --
    an intermediate for `document_coverage_causes`/`text_coverage_summary`
    only, which at corpus scale (~90,000 pages, tens of text nodes each)
    would put on the order of hundreds of MB of JSON into a manifest every
    cache open reads -- and attach the already-computed `coverage_cause`
    to every row below `min_text_coverage` whose cause could be classified
    (P14 part 2). A row not below the floor, or one `document_coverage_
    causes` could not classify (no `node_shares` to begin with), gets no
    `coverage_cause` key at all -- never a placeholder value.
    """
    causes = document_coverage_causes(rows, min_text_coverage=min_text_coverage)
    for row in rows:
        row.pop("node_shares", None)
        cause = causes.get((row["document_bundle_id"], row["page_index"]))
        if cause is not None:
            row["coverage_cause"] = cause


#: Ledger row keys summed straight into `derivation_ledger` totals --
#: every count `page_projection.project`'s ledger produces except the
#: two run-length maxima, which are histograms instead (a sum of a
#: bound is not the bound), and `typed_withdrawn`, which is its own
#: per-reason-code breakdown.
_DERIVATION_LEDGER_SUM_KEYS: tuple[str, ...] = (
    "empty_trailing_dropped",
    "empty_interior_runs_capped",
    "empty_interior_dropped",
    "tab_runs_capped",
    "tabs_dropped",
    "kept_for_container_floor",
    "trailing_breaks_dropped",
)


def derivation_ledger_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """DEC-033 §5/§6/§10 totals over `derivation_ledger_rows` (T6 wiring):
    sums for every drop/cap counter, `max_empty_run_after`/
    `max_tab_run_after` histograms (DEC-031 needs the distribution, not
    just its max, to re-measure a guard constant), and `typed_withdrawn`
    summed per reason code and pointer across every row's own
    `{reason_code: {pointer: count}}` breakdown (`typed_property_targets.
    withdraw_page_piece_typed`'s design §4 ledger shape)."""
    totals: dict[str, Any] = {key: 0 for key in _DERIVATION_LEDGER_SUM_KEYS}
    max_empty_run_after: dict[int, int] = {}
    max_tab_run_after: dict[int, int] = {}
    typed_withdrawn: dict[str, dict[str, int]] = {}
    for row in rows:
        for key in _DERIVATION_LEDGER_SUM_KEYS:
            totals[key] += row[key]
        max_empty_run_after[row["max_empty_run_after"]] = (
            max_empty_run_after.get(row["max_empty_run_after"], 0) + 1
        )
        max_tab_run_after[row["max_tab_run_after"]] = (
            max_tab_run_after.get(row["max_tab_run_after"], 0) + 1
        )
        for reason, by_pointer in row["typed_withdrawn"].items():
            totals_by_pointer = typed_withdrawn.setdefault(reason, {})
            for pointer, count in by_pointer.items():
                totals_by_pointer[pointer] = totals_by_pointer.get(pointer, 0) + count
    return {
        "pages": len(rows),
        **totals,
        "max_empty_run_after_histogram": {
            str(k): v for k, v in sorted(max_empty_run_after.items())
        },
        "max_tab_run_after_histogram": {
            str(k): v for k, v in sorted(max_tab_run_after.items())
        },
        "typed_withdrawn": typed_withdrawn,
    }


def _target_policy_summary(
    policy: TargetPolicy,
    cohort: Sequence[Any],
    ledger_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """The manifest's `target_policy` block (DEC-055): the descriptor this
    cache was built under, how much of the exclusion list this cohort
    actually contained (a list that matches nothing must be visible as
    such), and the per-rule totals over every page the policy saw
    (written or refused)."""
    exclusions = policy.page_exclusions
    cohort_keys = {(sample.lineage_root_id, sample.page_index) for sample in cohort}
    in_cohort = (
        sum(1 for key in exclusions.entries if key in cohort_keys)
        if exclusions is not None
        else 0
    )
    facts = [row["target_policy"] for row in ledger_rows if "target_policy" in row]
    ink = sorted(fact["unowned_ink_chars"] for fact in facts)

    def quantile(q: float) -> int | None:
        return ink[min(len(ink) - 1, int(q * len(ink)))] if ink else None

    return {
        "descriptor": policy.descriptor(),
        "page_exclusions": {
            "sha256": exclusions.sha256 if exclusions is not None else None,
            "entries": len(exclusions) if exclusions is not None else 0,
            "in_cohort": in_cohort,
            "not_in_cohort": (len(exclusions) - in_cohort) if exclusions is not None else 0,
        },
        "pages": len(facts),
        "whitespace_runs_granted": sum(f["whitespace_runs_granted"] for f in facts),
        "withdrawn_run_chains_collapsed": sum(
            row.get("withdrawn_run_chains_collapsed", 0) for row in ledger_rows
        ),
        "withdrawn_runs_collapsed": sum(
            row.get("withdrawn_runs_collapsed", 0) for row in ledger_rows
        ),
        "stories_rehomed": sum(f.get("stories_rehomed", 0) for f in facts),
        "withdrawn_whitespace_runs_dropped": sum(
            row.get("withdrawn_whitespace_runs_dropped", 0) for row in ledger_rows
        ),
        "spill_signature_pages": sum(1 for f in facts if f["spill_signature"]),
        "spill_signature_listed_pages": sum(
            1 for f in facts if f["spill_signature"] and f["listed"] is not None
        ),
        "listed_pages_seen": sum(1 for f in facts if f["listed"] is not None),
        "unowned_ink": {
            "pages_with_any": sum(1 for n in ink if n > 0),
            "chars_total": sum(ink),
            "p50": quantile(0.5),
            "p99": quantile(0.99),
            "p999": quantile(0.999),
            "max": ink[-1] if ink else None,
        },
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _release_root_hash(release_dir: Path) -> str | None:
    manifest = json.loads((Path(release_dir) / "release-manifest.json").read_text())
    return manifest.get("release_root_hash")


def _real_binding(native_vocab_size: int | None) -> tuple[QwenTextBinding, Any, int]:
    """The DEC-007 D3-pinned tokenizer, its allowlist and the candidate's
    embedding width -- the real ones, never a placeholder (`build_cache`'s
    `--use-candidate-loader` pins the same lineage for the features)."""
    from unflatten.model import qwen_text_allowlist as qtal
    from unflatten.model import qwen_tokenizer_artifacts as qta

    tokenizer, snapshot_dir = qta.load_pinned_tokenizer()
    artifact_set_hash = qta.tokenizer_artifact_set_hash(
        snapshot_dir,
        model_id=qta.QWEN_CANDIDATE_MODEL_ID,
        revision=qta.QWEN_CANDIDATE_REVISION,
    )
    width = native_vocab_size if native_vocab_size is not None else candidate_native_vocab_size()
    check_native_vocab_size(width, tokenizer_len=len(tokenizer))
    binding = qtal.build_qwen_text_binding(
        tokenizer=tokenizer,
        tokenizer_artifact_set_hash=artifact_set_hash,
        native_vocab_size=width,
    )
    return binding, tokenizer, width


def _selection_kind(
    *, only_bundles: set[str] | None, single_page_only: bool, limit: int | None
) -> str:
    """R-1 (design §9, DEC-MP-4): the manifest's `scope.selection_kind`,
    as a function of ALL THREE cohort-narrowing knobs this build has --
    not only `only_bundles`. `single_page_only` defaults to `True` and
    `--limit` narrows the cohort exactly as much as `--only-bundles`
    does; declaring `"full_split"` while either one is active claimed a
    default single-page-only or `--limit`-truncated build was the whole
    split, permanently, on disk.

    `only_bundles` takes priority when set -- a named cohort is the more
    specific declaration, whether or not it also happens to be
    single-page-only or limited.
    """
    if only_bundles is not None:
        return "only_bundles"
    if single_page_only or limit is not None:
        return "restricted"
    return "full_split"


def resolve_limit(limit: int | None, *, cohort_size: int) -> tuple[int, str]:
    """`(pages this run will write at most, why)`.

    `None` means the whole cohort and is the DEFAULT: a numeric default
    here would make "build everything" unsayable and would silently stop a
    full-corpus build at a number the operator never chose, reporting a
    complete-looking cache (`.claude/skills/check-store` records the same
    shape costing hours: `m2c --workers` defaulting to 1 on a 28-core
    machine). `0` is refused rather than read as either "all" or "none" --
    both readings are plausible and guessing builds the wrong cache in
    silence.
    """
    if limit is not None and limit < 1:
        raise ValueError(
            f"--limit must be a positive page count or absent (absent = the whole "
            f"cohort), got {limit} -- 0 is refused because it reads equally well "
            "as 'all' and as 'none'"
        )
    if limit is None:
        return cohort_size, "all"
    return min(limit, cohort_size), f"--limit {limit}"


def cohort_intent(
    *,
    split: str,
    task_kind: str,
    split_samples: int,
    single_page_samples: int,
    cohort_size: int,
    attempt_target: int,
    reason: str,
    min_text_coverage: float = 0.0,
) -> str:
    """The line printed BEFORE the loop: what this run will do, and which
    filter narrowed the corpus to it.

    Printed up front on purpose. A run that stops early because it was told
    to must say so while the operator can still fix it; the same fact
    arriving in the final count is read as the size of the corpus. Names
    `min_text_coverage` too (D2, 2026-09-07) -- a page-count floor and a
    per-page coverage floor both narrow the written cohort, so both belong
    in the same up-front line.
    """
    scope = (
        f"writing all {cohort_size}"
        if reason == "all"
        else f"stopping after {attempt_target} written page(s) of {cohort_size} ({reason})"
    )
    return (
        f"[exact-text-cache] {split}/{task_kind}: {split_samples} sample(s), "
        f"{single_page_samples} single-page, {cohort_size} eligible after filters; "
        f"{scope}; min_text_coverage={min_text_coverage}"
    )


# ---------------------------------------------------------------------------
# --workers: per-page materialization over a fork pool (m2b pattern)
# ---------------------------------------------------------------------------


#: Set once per `build()` call, alongside the phase-A pin: a fork-context
#: pool worker takes only the SAMPLE -- these globals are what the fork
#: already inherited, never re-pickled per task (the `_ALIGN_JOBS` pattern
#: from `migration/m2b.py`). `_WORKER_STATE` is
#: `(bound, source, store_dir, schema_versions, min_text_coverage)`.
_WORKER_STATE: tuple[Any, ...] | None = None


def _assembly_pool_size(workers: int) -> int:
    """The assembly pool saves and hashes ~1.3 GB shards: a few big
    jobs, not one helper per materialize worker. Scaled off `--workers`
    so a small build does not fork more helpers than materializers.

    The old sizing was `min(6, workers // 8)`. Six was right when a build
    ran 24 materialize workers; at 88 it made the assembly the whole
    build. Measured 2026-09-23 on the v15 train cache: materialisation
    finished, **52,007 rows sat staged and the box was 93.4 % idle**,
    because six processes packing 32 rows apiece manage ~28 shards a
    minute — a shard being a 1.3 GB read plus a 1.3 GB write.

    `workers // 4` with a ceiling of 16 keeps small builds exactly where
    they were (24 workers still gives 6) and lets a big one use the disk
    instead of one core. The ceiling is the disk, not the CPU: 16
    assemblers moving 2.6 GB each already ask for more than the ~3 GB/s
    this array delivers, so raising it further would only queue.
    """
    return max(2, min(16, workers // 4))


def _materialize_worker_init() -> None:
    """PDEATHSIG: a killed parent must not leave orphan workers burning
    CPU forever (same pattern as `migration/m2b.py`). And a worker must
    not fan torch across all cores under dozens of siblings. Shared by
    the materialize pool and the assembly pool: both are short-lived,
    fork-context children of `build()`."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGKILL)
    except OSError:
        pass
    torch.set_num_threads(1)
    if _WORKER_STATE is not None:
        source = _WORKER_STATE[1]
        if isinstance(source, (FeatureSource, etfs.ExactTextFeatureSource)):
            # The cohort is in feature-cache order, so a worker's shards
            # arrive nearly sequentially: a small LRU is enough, and 48
            # workers each keeping 8 shards resident (~10 GB apiece on
            # the 2026-09-14 cache) would push the box toward its RAM
            # ceiling. One shard ahead is the only re-read a sequential
            # walk can ever take. Every reader (primary and supplements)
            # gets the same cap.
            for reader in source.readers:
                reader.cache_shards = 2


def _materialize_chunksize(
    pages: int, workers: int, *, page_image_mode: bool = False
) -> int:
    """Small reassignable page-image batches; contiguous feature-cache runs.

    Page-image mode reads individual PNGs, not feature shards. Cap its
    batches at 32 pages so workers that finish can take more work instead
    of idling behind a slow, cohort-sized run on another worker.

    Feature reads are the cost of this build, and they are shard-sized: a
    source shard holds 32 pages and is read whole (measured 2026-09-23 on
    `cache-exact-text-v14-r4`: 1.4 GB per shard, 2.5 s to verify plus
    4.1 s to load). A worker that walks the cohort CONTIGUOUSLY touches
    each shard once, in order, and the two-shard LRU is enough.

    `chunksize=workers` did the opposite: 90,786 pages became 2,270
    chunks of 40 handed round-robin, so each worker collected ~57 runs
    scattered across the whole cohort and re-read the same shards over
    and over. Measured mid-build: 1,685 GB read for 1,984 pages written,
    against an 87 GB floor -- 19x amplification, and a projected 7 hours
    against the 1h05 the previous build took from a feature cache.

    Submission order is untouched: `Executor.map` yields in submission
    order for any chunksize, so the writer's page sequence -- and with it
    the cache identity -- stays exactly the cohort's, as Phase B promises.
    """
    if workers <= 1:
        return 1
    size = max(1, -(-pages // workers))
    return min(32, size) if page_image_mode else size


_REUSE_PARENT: reuse.VerifiedParent | None = None


def _materialize_core(
    *,
    bound: Any,
    source: FeatureSource,
    store_dir: Path,
    schema_versions: SchemaVersions,
    min_text_coverage: float,
    rows_dir: Path | None,
    position: int | None,
    sample: Any,
) -> tuple[Any, ...]:
    """One page, whichever process computes it: the shared core of both
    entry points below and of Phase A's in-process pin.

    `rows_dir is None` (the `workers == 1` path): returns the legacy
    `(row, skip, coverage, typed, ledger, node_shares, continuation,
    region)` tuple and the writer buffers the row as before.

    Otherwise this process writes the full row to
    `rows_dir/row-<cohort_position>.pt` and returns only a small record
    `(position, written, bundle, page, skip, coverage, typed, ledger,
    node_shares, continuation, region)`. The ~41 MB row never crosses IPC:
    it is the parent holding every row and flushing shards one-by-one that
    starved the workers to ~5 pages/s at high `--workers` (2026-09-14),
    and it is the row file that lets a separate assembly pool save and
    hash shards in parallel with materialization.
    """
    retained = (
        _REUSE_PARENT.retained((sample.document_bundle_id, sample.page_index))
        if _REUSE_PARENT is not None
        else None
    )
    entry = (
        None if retained is not None else source.entry(sample.document_bundle_id, sample.page_index)
    )
    if retained is not None:
        stored, metadata, skip, continuation = retained
        row = _row_from_storage(stored) if stored is not None else None
        coverage = (
            TextCoverage(**metadata["coverage"]) if metadata["coverage"] is not None else None
        )
        typed = TypedChannelCounts(**metadata["typed"]) if metadata["typed"] is not None else None
        ledger = metadata["ledger"]
        node_shares = metadata["node_shares"]
        region = (
            RegionChannelCounts(**metadata["region"]) if metadata["region"] is not None else None
        )
    elif entry is None:
        skip: str | None = "skipped_no_cached_features"
        row: Any = None
        coverage: Any = None
        typed: Any = None
        ledger: Any = None
        node_shares: Any = None
        continuation = False
        region: Any = None
    else:
        # `with_geometry=True` (T1): every page needs its Observation Sidecar
        # entry to derive the region channel, not only the spatial probe.
        item = build_item(store_dir, sample, with_geometry=True)
        mask_projection = item.mask_projection or {}
        mask_projection_hash = (sample.mask_projection or {}).get("sha256")
        if mask_projection_hash is not None:
            row, skip, coverage, typed, ledger, region = row_for_page(
                document_bundle_id=sample.document_bundle_id,
                page_index=sample.page_index,
                targets=item.targets,
                fragment=item.fragment,
                features=entry.features,
                grid_thw=entry.grid_thw,
                page_image=getattr(entry, "page_image", None),
                mask_projection=item.mask_projection,
                mask_projection_hash=mask_projection_hash,
                mask_projection_schema_version=mask_projection.get(
                    "mask_projection_schema_version"
                ),
                page_font_pack_hash=render_font_pack_hash(store_dir, sample),
                bound=bound,
                schema_versions=schema_versions,
                observation_sidecar=item.observation_sidecar_data,
                min_text_coverage=min_text_coverage,
                listed_reason=(
                    None
                    if bound.materializer.target_policy is None
                    else bound.materializer.target_policy.listed_reason(
                        sample.lineage_root_id, sample.page_index
                    )
                ),
            )
            node_shares = (
                [
                    asdict(share)
                    for share in page_node_shares(
                        sample.document_bundle_id,
                        item.targets,
                        item.fragment,
                        target_policy=bound.materializer.target_policy,
                    )
                ]
                if coverage is not None
                else None
            )
            continuation = skip == "skipped_grammar" and has_continuation_presentation(
                item.fragment, (item.targets or {}).get("structural") or {}
            )
        else:
            # No mask projection artifact means no hash to bind. Nothing here
            # may substitute one: every LossUnit carries it. Reported exactly
            # as the serial build counts it, so the parent's counters never
            # depend on which process computed the page.
            skip = "skipped_unencodable"
            row = None
            coverage = None
            typed = None
            ledger = None
            node_shares = None
            continuation = False
            region = None
    if row is None:
        if rows_dir is None:
            return (row, skip, coverage, typed, ledger, node_shares, continuation, region)
        return (
            position,
            False,
            sample.document_bundle_id,
            sample.page_index,
            skip,
            coverage,
            typed,
            ledger,
            node_shares,
            continuation,
            region,
        )
    # The `add()` path verifies inside the writer; the row-file path has
    # no writer to verify for it, so the verifier runs HERE -- a row that
    # fails its own shape/hash checks must never reach disk looking valid.
    _verify_row_integrity(row)
    if rows_dir is None:
        return (row, skip, coverage, typed, ledger, node_shares, continuation, region)
    stored = _row_to_storage(row)
    stored["_build_metadata"] = _build_metadata(
        position=position,
        document_bundle_id=sample.document_bundle_id,
        page_index=sample.page_index,
        coverage=coverage,
        typed=typed,
        ledger=ledger,
        node_shares=node_shares,
        region=region,
    )
    torch.save([stored], rows_dir / f"row-{position:06d}.pt")
    return (
        position,
        True,
        sample.document_bundle_id,
        sample.page_index,
        None,
        coverage,
        typed,
        ledger,
        node_shares,
        False,
        region,
    )


def _build_metadata(
    *,
    position: int | None,
    document_bundle_id: str,
    page_index: int,
    coverage: Any,
    typed: Any,
    ledger: Any,
    node_shares: Any,
    region: Any,
) -> dict[str, Any]:
    """Small JSON-safe record needed to restore manifest summaries on resume."""
    return {
        "position": position,
        "document_bundle_id": document_bundle_id,
        "page_index": page_index,
        "coverage": asdict(coverage) if coverage is not None else None,
        "typed": asdict(typed) if typed is not None else None,
        "ledger": ledger,
        "node_shares": node_shares,
        "region": asdict(region) if region is not None else None,
    }


def _materialize_worker(sample: Any) -> tuple[Any, ...]:
    """`workers == 1` in-process entry point: takes only `sample`;
    everything else is the fork-inherited `_WORKER_STATE` (the 5-tuple
    without a `rows_dir` -- the serial path never writes row files).

    Returns `(row, skip, coverage, typed, ledger, node_shares,
    continuation, region)` -- the serial build's per-page result,
    unchanged.
    """
    assert _WORKER_STATE is not None, "build() did not pin the worker state"
    bound, source, store_dir, schema_versions, min_text_coverage = _WORKER_STATE
    return _materialize_core(
        bound=bound,
        source=source,
        store_dir=store_dir,
        schema_versions=schema_versions,
        min_text_coverage=min_text_coverage,
        rows_dir=None,
        position=None,
        sample=sample,
    )


def _materialize_worker_payload(payload: tuple[int, Any]) -> tuple[Any, ...]:
    """`workers > 1` pool entry point: takes only `(cohort_position,
    sample)`; everything else is the fork-inherited `_WORKER_STATE` (the
    6-tuple carrying the `rows_dir`). The row file is written here and
    only the record crosses IPC -- see `_materialize_core`."""
    assert _WORKER_STATE is not None, "build() did not pin the worker state"
    bound, source, store_dir, schema_versions, min_text_coverage, rows_dir = (
        _WORKER_STATE
    )
    position, sample = payload
    return _materialize_core(
        bound=bound,
        source=source,
        store_dir=store_dir,
        schema_versions=schema_versions,
        min_text_coverage=min_text_coverage,
        rows_dir=rows_dir,
        position=position,
        sample=sample,
    )


def _assembly_job(job: tuple[int, list[int], Path]) -> tuple[str, int, str]:
    """Assembly-pool entry point: the job pickles, the work does not.

    Assembles one shard from its worker-written row files (see
    `exact_text_cache.assemble_shard`), returning
    `(name, count, sha256)` for the parent to commit in cohort order.
    """
    ordinal, positions, staging_dir = job
    return assemble_shard(staging_dir, ordinal, positions)


def build(
    *,
    release_dir: Path,
    store_dir: Path,
    feature_cache_dir: Path | None,
    out_dir: Path,
    bindings: dict[str, str],
    supplement_feature_cache_dirs: Sequence[Path] | None = None,
    bindings_path: Path | str,
    split: str = "train",
    task_kind: str = "structure_v0",
    limit: int | None = None,
    shard_size: int = 32,
    checkpoint_every_shards: int = 1,
    allow_dev_only: bool = False,
    single_page_only: bool = True,
    native_vocab_size: int | None = None,
    text_encoding_budget_ops: int = DEFAULT_TEXT_ENCODING_BUDGET_OPS,
    progress_every: int = 25,
    only_bundles: set[str] | None = None,
    only_bundles_path: Path | str | None = None,
    min_text_coverage: float = 0.0,
    capability_profile: CapabilityProfile = BOUND_CAPABILITY_PROFILE,
    legacy_feature_grammar: Mapping[Path, str] | None = None,
    release_lineage_proof_path: Path | str | Sequence[Path | str] | None = None,
    workers: int = 1,
    target_policy: TargetPolicy | None = None,
    visual_input_policy: onfly_visual.VisualInputPolicy | None = None,
    reuse_page_image_cache: Path | None = None,
    reuse_parent_page_exclusions: Sequence[Path] | None = None,
    reuse_parent_build_config: Path | None = None,
) -> dict[str, Any]:
    """Write an exact-text cache for the pages the feature cache holds.

    `capability_profile` defaults to `BOUND_CAPABILITY_PROFILE` -- every
    existing invocation keeps today's behaviour unchanged. DEC-024's large
    build and S2 pass `TYPED_WIDENED_V1_CAPABILITY_PROFILE` explicitly
    (`--capability-profile s2-typed-widened-v1`).

    The cohort is the source feature cache's pages, not the whole split: a
    page whose features are not cached would need a second extraction whose
    features could differ, so it is not attempted at all and the count of
    such feature rows is reported (`feature_rows_unmatched`).

    `limit` is `None` -- the whole cohort -- by default; see
    :func:`resolve_limit`. Whatever it resolves to is printed by
    :func:`cohort_intent` before the first page is read, and repeated in
    the summary as `cohort`/`attempt_target`/`limit`.

    `bindings_path` is the file `bindings` was read from: the freshness
    check (:func:`verify_bindings_freshness`) runs before the first page is
    read, and one of its seven keys can only be evidenced by a provenance
    sibling next to that file.

    `only_bundles` restricts the cohort to exactly those
    `document_bundle_id`s -- pins the cohort for an A/B across two store
    revisions, since the RFC003 split allocator is RNG-free but greedy and
    one changed document can scramble val/test membership. A file whose ids
    produce no page in this cohort is refused (:class:`OnlyBundlesInvalid`);
    ids absent from the split are not an error but are counted and printed
    before the loop.

    `release_lineage_proof_path`, when given (repeatable on the CLI),
    admits a source feature cache whose `release_root_hash` names an
    earlier release, iff one of the proofs proved every one of that
    source's keys unchanged into THIS release (T11, design section 8;
    first accepting proof wins). Absent, behaviour is unchanged: a
    source must name this build's own release directly.

    `workers` is the number of processes that materialize pages in
    parallel. With `workers > 1` each worker writes its page's row file
    directly to the writer's staging area and returns only a small record
    (the row never crosses IPC); a separate assembly pool saves and hashes
    full shard windows in parallel, and the parent commits shards in
    cohort order. The page sequence the writer sees is cohort order either
    way, so the cache content (every row, the identity, the summaries) is
    **identical** for any `workers` (see `TestBuildWorkers`). Shard file
    bytes are not a stable cross-process property of `torch.save`
    (per-process `.data/serialization_id`, pickle memoization), and are
    therefore not part of the claim. `workers=1` (the default) runs
    in-process and never touches a pool.

    `target_policy` (`v16_target_policy`, DEC-055) selects the v16 target
    projection: it enters `training_target_materializer_hash`, adds the
    `V16_SKIP_COUNTS` refusals and a `target_policy` manifest summary.
    `None` (the default) leaves every output exactly as before.

    `visual_input_policy` (DEC-056 addendum A): a page-image build --
    `feature_cache_dir` must be `None` (and no supplement, lineage proof or
    legacy-grammar flag), the cohort is the release's pages, rows carry
    the PNG reference instead of features (`PageImageSource`) and the
    identity binds the policy. `None` (the default): a feature build,
    exactly as before.

    `reuse_page_image_cache` opt-in replays a verified complete 7.0.0
    parent under monotonic named exclusions. Explicit parent TSVs reproduce
    its policy; a manifest-bound launcher/source attestation proves the
    historical un-hashed text budget. All shards and cohort positions are
    verified before output creation, and changed exclusions are materialized
    normally. The child uses the ordinary writer and records its derivation.
    """
    global _REUSE_PARENT
    _REUSE_PARENT = None  # A preceding reuse build must never affect a fresh build.
    if (
        not isinstance(checkpoint_every_shards, int)
        or isinstance(checkpoint_every_shards, bool)
        or checkpoint_every_shards < 1
    ):
        raise ValueError("checkpoint_every_shards must be a positive integer")
    validate_min_text_coverage(min_text_coverage)
    if reuse_page_image_cache is not None and (
        visual_input_policy is None or limit is not None or target_policy is None
    ):
        raise ExactTextCacheIncompatible(
            "reuse requires page-image mode, target policy and no finite limit"
        )
    if reuse_page_image_cache is None and (
        reuse_parent_page_exclusions or reuse_parent_build_config
    ):
        raise ExactTextCacheIncompatible(
            "parent reuse configuration requires reuse_page_image_cache"
        )
    page_image_mode = visual_input_policy is not None
    if page_image_mode and (
        feature_cache_dir is not None
        or supplement_feature_cache_dirs
        or release_lineage_proof_path is not None
        or legacy_feature_grammar
    ):
        raise FeatureCacheMismatch(
            "a page-image build (visual_input_policy) reads no feature cache: "
            "feature_cache_dir, supplements, --release-lineage-proof and "
            "--legacy-feature-grammar must all be absent"
        )
    if not page_image_mode and feature_cache_dir is None:
        raise FeatureCacheMismatch(
            "feature_cache_dir is required unless visual_input_policy selects a page-image build"
        )
    if workers < 1:
        raise ValueError(
            f"workers must be >= 1 (1 = in-process, no pool), got {workers}"
        )
    release_dir, store_dir = Path(release_dir), Path(store_dir)
    admission = "dev_only" if allow_dev_only else "strict"
    release_root_hash = _release_root_hash(release_dir)
    feature_cache_dirs = (
        [] if page_image_mode else [feature_cache_dir, *(supplement_feature_cache_dirs or [])]
    )

    source_is_exact_text, mismatch = (
        (False, None)
        if page_image_mode
        else _check_all_sources_same_type(
            Path(feature_cache_dir),  # type: ignore[arg-type]
            [Path(d) for d in (supplement_feature_cache_dirs or [])],
        )
    )
    if mismatch is not None:
        primary_type = "exact_text_cache" if source_is_exact_text else "structural_feature_cache"
        raise MixedCacheTypeMismatch(
            f"primary cache type ({primary_type}) does not match "
            f"supplement at {mismatch} -- all sources must have the same "
            f"cache_family (either both structural, or both 'exact_text_v1')"
        )

    grammar_records: tuple[dict[str, Any], ...] = ()
    source: Any = None
    if page_image_mode:
        # The source is built once the cohort is known (below): it is the
        # cohort's own PNG references, nothing to open here.
        grammar_records = (page_image_source_record(visual_input_policy),)  # type: ignore[arg-type]
    elif source_is_exact_text:
        if release_lineage_proof_path is not None:
            raise FeatureCacheMismatch(
                "--release-lineage-proof applies only to structural feature "
                "caches; an exact-text feature source carries its own identity"
            )
        if legacy_feature_grammar:
            raise FeatureCacheMismatch(
                "--legacy-feature-grammar applies only to structural feature "
                "caches; an exact-text feature source carries its own identity"
            )
        gradient_bearing = admission != "dev_only"
        source = etfs.open_exact_text_feature_sources(
            feature_cache_dir,
            supplement_feature_cache_dirs or [],
            gradient_bearing=gradient_bearing,
        )
        for source_dir, reader in zip(feature_cache_dirs, source.readers, strict=True):
            source_split = reader.identity.get("split")
            if source_split != split:
                raise etfs.ExactTextFeatureSourceMismatch(
                    f"{source_dir}: identity.split {source_split!r} != {split!r}"
                )
            source_task_kind = reader.identity.get("task_kind")
            if source_task_kind != task_kind:
                raise etfs.ExactTextFeatureSourceMismatch(
                    f"{source_dir}: identity.task_kind {source_task_kind!r} != "
                    f"{task_kind!r}"
                )
        all_source_dirs = [feature_cache_dir, *(supplement_feature_cache_dirs or [])]
        grammar_records = tuple(
            _grammar_record_from_exact_text_manifest(
                Path(all_source_dirs[i]),
                reader.manifest,
            )
            for i, reader in enumerate(source.readers)
        )
    else:
        if release_lineage_proof_path is None:
            proof_pairs: list[tuple[Any, Mapping[str, Any]]] = []
        else:
            paths = (
                list(release_lineage_proof_path)
                if isinstance(release_lineage_proof_path, (list, tuple))
                else [release_lineage_proof_path]
            )
            proof_pairs = [
                (path, json.loads(Path(path).read_text())) for path in paths
            ]
        _structural_source = open_feature_sources(
            feature_cache_dir,
            supplement_feature_cache_dirs or [],
            release_root_hash=release_root_hash,
            split=split,
            task_kind=task_kind,
            admission=admission,
            capability_profile=capability_profile,
            legacy_feature_grammar=legacy_feature_grammar,
            release_lineage_proof=proof_pairs,
            release_lineage_proof_path=release_lineage_proof_path,
        )
        source = _structural_source
    binding, tokenizer, width = _real_binding(native_vocab_size)
    bound = build_materializer_bundle(
        binding=binding,
        tokenizer=tokenizer,
        bindings=bindings,
        text_encoding_budget_ops=text_encoding_budget_ops,
        capability_profile=capability_profile,
        target_policy=target_policy,
    )
    verify_bindings_freshness(
        bindings=bindings,
        bindings_path=bindings_path,
        controller=bound.controller,
        capability_profile=capability_profile,
    )
    if not source_is_exact_text and not page_image_mode:
        grammar_records = source.grammar_records

    # Enumerated, not streamed, so the run can state its own scope before
    # it starts: an operator cannot fix "this is not the corpus you think"
    # from a progress line. One extra pass over the release manifests, no
    # store reads -- the page artifacts are still read lazily below.
    split_samples = 0
    single_page_samples = 0
    cohort: list[Any] = []
    split_bundle_ids: set[str] = set()
    for sample in iter_page_samples(
        release_dir, store_dir, split, task_kind,
        require_privacy_review=not allow_dev_only,
    ):
        split_samples += 1
        split_bundle_ids.add(sample.document_bundle_id)
        if only_bundles is not None and sample.document_bundle_id not in only_bundles:
            continue
        if single_page_only and sample.page_count != 1:
            continue
        single_page_samples += 1
        cohort.append(sample)

    cohort_keys = [
        (sample.document_bundle_id, sample.page_index) for sample in cohort
    ]
    if page_image_mode:
        source = PageImageSource(
            policy=visual_input_policy,  # type: ignore[arg-type]
            store_dir=store_dir,
            index={
                (sample.document_bundle_id, sample.page_index): dict(sample.image)
                for sample in cohort
            },
            grammar_records=grammar_records,
        )
    missing_feature_keys = (
        etfs.missing_page_keys(source, cohort_keys)
        if source_is_exact_text
        else sorted(key for key in cohort_keys if key not in source.index)
    )

    if only_bundles is not None:
        only_bundles_matched = len(only_bundles & split_bundle_ids)
        only_bundles_missing = len(only_bundles - split_bundle_ids)
        if not cohort:
            raise OnlyBundlesInvalid(
                f"--only-bundles: 0 of {len(only_bundles)} requested bundle id(s) "
                f"from {only_bundles_path or '<file>'} produced any page in split "
                f"{split!r}/{task_kind!r} with cached features -- refusing to build "
                "an empty cache"
            )
        print(
            f"[exact-text-cache] --only-bundles: restricting {split!r}/{task_kind!r} "
            f"to {len(only_bundles)} requested bundle id(s) ({only_bundles_matched} "
            f"matched, {only_bundles_missing} not found in the split)",
            flush=True,
        )

    attempt_target, reason = resolve_limit(limit, cohort_size=len(cohort))
    print(
        cohort_intent(
            split=split,
            task_kind=task_kind,
            split_samples=split_samples,
            single_page_samples=single_page_samples,
            cohort_size=len(cohort),
            attempt_target=attempt_target,
            reason=reason,
            min_text_coverage=min_text_coverage,
        ),
        flush=True,
    )

    only_bundles_extra = (
        only_bundles_manifest_fields(only_bundles) if only_bundles is not None else None
    )
    reuse_schema = None
    if reuse_page_image_cache is not None:
        parent = Path(reuse_page_image_cache)
        if Path(out_dir).resolve().is_relative_to(parent.resolve()):
            raise ExactTextCacheIncompatible("reuse output must differ from immutable parent")
        parent_config = reuse.verify_build_config(
            reuse_parent_build_config, parent, text_encoding_budget_ops
        )
        parent_manifest = json.loads((parent / "manifest.json").read_text())
        parent_identity = parent_manifest.get("identity") or {}
        reuse_schema = SchemaVersions(
            parent_identity["target_bundle_schema_version"],
            parent_identity["fragment_schema_version"],
            parent_identity["mask_projection_schema_version"],
        )
        old_exclusions = (
            load_page_exclusions(reuse_parent_page_exclusions)
            if reuse_parent_page_exclusions
            else None
        )
        parent_policy = replace(target_policy, page_exclusions=old_exclusions)
        expected_parent_identity = build_identity(
            feature_manifest={},
            visual_input_policy=visual_input_policy,
            binding=binding,
            bindings=bindings,
            release_root_hash=release_root_hash,
            split=split,
            task_kind=task_kind,
            admission=admission,
            schema_versions=reuse_schema,
            capability_profile=capability_profile,
            target_policy=parent_policy,
        )
        _REUSE_PARENT = reuse.verify_parent(
            parent=parent,
            cohort=cohort,
            expected_identity=asdict(expected_parent_identity),
            scope={
                "selection_kind": _selection_kind(
                    only_bundles=only_bundles, single_page_only=single_page_only, limit=limit
                ),
                "single_page_only": single_page_only,
                "limit": limit,
                "only_bundles_sha256": (only_bundles_extra or {}).get("only_bundles_sha256"),
                "only_bundles_count": (only_bundles_extra or {}).get("only_bundles_count"),
            },
            min_coverage=min_text_coverage,
            old_entries=old_exclusions.entries if old_exclusions else {},
            new_entries=target_policy.page_exclusions.entries
            if target_policy.page_exclusions
            else {},
            config=parent_config,
            skip_reasons=SKIP_COUNTS + V16_SKIP_COUNTS,
            feature_sources=list(grammar_records),
            workers=workers,
            parent_policy=parent_policy.descriptor(),
            shard_size=shard_size,
        )
        _REUSE_PARENT.provenance["child_policy"] = target_policy.descriptor()
    #: `V16_SKIP_COUNTS` exist only under a target policy, so a default
    #: build's summary keeps its exact key set.
    skip_reasons = SKIP_COUNTS + (V16_SKIP_COUNTS if target_policy is not None else ())
    counts = dict.fromkeys(skip_reasons, 0)
    written = 0
    attempted = 0
    matched: set[tuple[str, int]] = set()
    schema_versions: SchemaVersions | None = None
    writer: ExactTextCacheWriter | None = None
    identity: ExactTextCacheIdentity | None = None
    text_coverage_rows: list[dict[str, Any]] = []
    typed_channel_rows: list[dict[str, Any]] = []
    #: One row per page whose region channel was derived, mirroring
    #: `typed_channel_rows` (T1).
    region_channel_rows: list[dict[str, Any]] = []
    #: One `{document_bundle_id, page_index, reason}` row per cohort page
    #: that was NOT written, in cohort order -- the per-page counterpart of
    #: `counts`. `SKIP_COUNTS` says how many pages a reason cost; only this
    #: says WHICH. The S2 exit-gate cohort needs that: a slice of a split
    #: (DEC-033 §12) cannot check its own attributions against split-wide
    #: bucket totals, and two pages that differ only in why they were
    #: skipped are otherwise indistinguishable to it (both have a feature
    #: row and no written row). Cheap: one row per SKIPPED page.
    page_skip_rows: list[dict[str, Any]] = []
    resume_skip_records: list[dict[str, Any]] = []
    #: DEC-033 §5/§6/§10 (T6 wiring): only ever populated under
    #: `STANDALONE_V1_PROFILE_ID` -- `row_for_page` returns `ledger=None`
    #: for every other profile, and `_apply`/`_process_record` only ever
    #: append a non-`None` ledger.
    derivation_ledger_rows: list[dict[str, Any]] = []
    resume_position = 0

    def _record_skip(
        document_bundle_id: str,
        page_index: int,
        skip: str,
        continuation: bool,
        *,
        position: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """The only place a skip is accounted: the bucket, the T-9 subset
        and the per-page row, so a caller cannot record one without the
        others."""
        counts[skip] += 1
        # T-9 (design §4.1): `skipped_grammar_continuation` is a SUBSET of
        # `skipped_grammar` -- add it on top of, never instead of, the
        # parent so the RUNBOOK's `skipped_grammar` baseline still counts
        # this page. It is not a second row.
        if continuation:
            counts["skipped_grammar_continuation"] += 1
        page_skip_rows.append(
            {
                "document_bundle_id": document_bundle_id,
                "page_index": page_index,
                "reason": skip,
            }
        )
        resume_skip_records.append(
            {
                "position": position,
                "document_bundle_id": document_bundle_id,
                "page_index": page_index,
                "reason": skip,
                "continuation": continuation,
                "metadata": metadata,
            }
        )

    def _sync_build_progress(next_position: int, skip_count: int | None = None) -> None:
        assert writer is not None
        writer.record_summary(
            "build_progress",
            {
                "next_position": next_position,
                "skip_records": resume_skip_records
                if skip_count is None else resume_skip_records[:skip_count],
            },
        )

    def _restore_summary_metadata(metadata: dict[str, Any], *, is_written: bool) -> None:
        bundle = metadata["document_bundle_id"]
        page = metadata["page_index"]
        if metadata["ledger"] is not None:
            derivation_ledger_rows.append({**metadata["ledger"], "written": is_written})
        if metadata["typed"] is not None:
            typed_channel_rows.append(
                {
                    "document_bundle_id": bundle,
                    "page_index": page,
                    "flipped": metadata["typed"]["flipped"],
                    "typed_targets_node_not_opened_here": metadata["typed"][
                        "node_not_opened_here"
                    ],
                    "written": is_written,
                }
            )
        if metadata["region"] is not None:
            region_channel_rows.append(
                {
                    "document_bundle_id": bundle,
                    "page_index": page,
                    "region_targets_node_not_opened_here": metadata["region"][
                        "node_not_opened_here"
                    ],
                    "written": is_written,
                }
            )
        if metadata["coverage"] is not None:
            text_coverage_rows.append(
                {
                    "document_bundle_id": bundle,
                    "page_index": page,
                    "granted_chars": metadata["coverage"]["granted_chars"],
                    "store_chars": metadata["coverage"]["store_chars"],
                    "written": is_written,
                    "node_shares": metadata["node_shares"],
                }
            )

    def _apply(position: int, sample: Any, result: tuple[Any, ...]) -> None:
        """One materialized page into the accumulators and the writer, in
        cohort order -- the serial loop's per-page tail, shared by the
        in-process and the pool path so neither can drift from the other."""
        nonlocal written
        row, skip, coverage, typed, ledger, node_shares, continuation, region = result
        assert writer is not None
        if ledger is not None:
            derivation_ledger_rows.append({**ledger, "written": row is not None})
        if typed is not None:
            typed_channel_rows.append(
                {
                    "document_bundle_id": sample.document_bundle_id,
                    "page_index": sample.page_index,
                    "flipped": typed.flipped,
                    "typed_targets_node_not_opened_here": typed.node_not_opened_here,
                    "written": row is not None,
                }
            )
        if region is not None:
            region_channel_rows.append(
                {
                    "document_bundle_id": sample.document_bundle_id,
                    "page_index": sample.page_index,
                    "region_targets_node_not_opened_here": region.node_not_opened_here,
                    "written": row is not None,
                }
            )
        if coverage is not None:
            text_coverage_rows.append(
                {
                    "document_bundle_id": sample.document_bundle_id,
                    "page_index": sample.page_index,
                    "granted_chars": coverage.granted_chars,
                    "store_chars": coverage.store_chars,
                    "written": row is not None,
                    # P14 part 2: the per-node breakdown `text_coverage_summary`
                    # needs to separate a split-run denominator effect from a
                    # real misalignment. Cheap -- see `page_node_shares`. Stored
                    # as plain dicts (not `PageNodeShare` dataclass instances):
                    # this row is JSON-serialized into the manifest as-is
                    # (`writer.record_summary("text_coverage_rows", ...)`).
                    "node_shares": node_shares,
                }
            )
        if row is None:
            metadata = _build_metadata(
                position=position,
                document_bundle_id=sample.document_bundle_id,
                page_index=sample.page_index,
                coverage=coverage,
                typed=typed,
                ledger=ledger,
                node_shares=node_shares,
                region=region,
            )
            _record_skip(
                sample.document_bundle_id,
                sample.page_index,
                skip,
                continuation,
                position=position,
                metadata=metadata,
            )
            return
        _sync_build_progress(position + 1)
        writer.add(
            row,
            build_metadata=_build_metadata(
                position=position,
                document_bundle_id=sample.document_bundle_id,
                page_index=sample.page_index,
                coverage=coverage,
                typed=typed,
                ledger=ledger,
                node_shares=node_shares,
                region=region,
            ),
        )
        written += 1
        if progress_every and written % progress_every == 0:
            print(f"[exact-text-cache] {written}/{attempt_target} page(s)", flush=True)

    def _pin_writer(candidate: SchemaVersions) -> None:
        """Phase A's pin, both paths: the first page that declares all
        three schema versions fixes what this cache claims; build the
        identity, create the writer, and pin the worker state. With
        `workers > 1` the writer's `rows/` staging directory is created
        here too, because the materialize workers fork only AFTER this
        point and inherit it.
        """
        global _WORKER_STATE
        nonlocal schema_versions, identity, writer, written, attempted, resume_position
        nonlocal _shard_ordinal
        schema_versions = candidate
        identity = build_identity(
            feature_manifest={} if page_image_mode else source.readers[0].manifest,
            visual_input_policy=visual_input_policy,
            binding=binding,
            bindings=bindings,
            release_root_hash=release_root_hash,
            split=split,
            task_kind=task_kind,
            admission=admission,
            schema_versions=schema_versions,
            capability_profile=capability_profile,
            target_policy=target_policy,
        )
        writer = ExactTextCacheWriter(
            out_dir,
            identity=identity,
            # DEC-MP-4: declared, never inferred by a reader. The
            # `only_bundles_*` fields stay in `extra_manifest` too --
            # removing them would break every existing reader of that
            # key for nothing -- but the scope block is what a reader
            # is now refused without.
            selection_kind=_selection_kind(
                only_bundles=only_bundles,
                single_page_only=single_page_only,
                limit=limit,
            ),
            only_bundles_sha256=(
                only_bundles_extra["only_bundles_sha256"]
                if only_bundles_extra
                else None
            ),
            only_bundles_count=(
                only_bundles_extra["only_bundles_count"]
                if only_bundles_extra
                else None
            ),
            single_page_only=single_page_only,
            limit=limit,
            shard_size=shard_size,
            checkpoint_every_shards=checkpoint_every_shards,
            extra_manifest={
                **(only_bundles_extra or {}),
                "min_text_coverage": min_text_coverage,
                "feature_sources": list(grammar_records),
                **(
                    {"cache_derivation": _REUSE_PARENT.provenance}
                    if _REUSE_PARENT is not None
                    else {}
                ),
            },
        )
        if writer.resumed:
            progress = writer.extra_manifest.get("build_progress") or {}
            resume_position = int(progress.get("next_position", 0))
            counts.update(dict.fromkeys(skip_reasons, 0))
            text_coverage_rows.clear()
            typed_channel_rows.clear()
            region_channel_rows.clear()
            page_skip_rows.clear()
            derivation_ledger_rows.clear()
            resume_skip_records.clear()
            restored_metadata: list[tuple[dict[str, Any], bool]] = []
            for record in progress.get("skip_records", []):
                _record_skip(
                    record["document_bundle_id"],
                    record["page_index"],
                    record["reason"],
                    record["continuation"],
                    position=record["position"],
                    metadata=record.get("metadata"),
                )
                if record.get("metadata") is not None:
                    restored_metadata.append((record["metadata"], False))
            for metadata in writer.committed_metadata:
                restored_metadata.append((metadata, True))
            for metadata, is_written in sorted(
                restored_metadata, key=lambda item: item[0]["position"]
            ):
                _restore_summary_metadata(metadata, is_written=is_written)
            written = writer.committed_entries
            _shard_ordinal = writer.committed_shards
            attempted = resume_position
            matched.clear()
            matched.update(
                (sample.document_bundle_id, sample.page_index)
                for sample in cohort[:resume_position]
            )
        if workers > 1:
            rows_dir = writer.staging / "rows"
            # A previous interrupted build may have left row files here;
            # the cohort positions are stable across restarts, and a
            # leftover file would only ever be re-read by its own page.
            # Clear anyway: this build's records are the only authority.
            shutil.rmtree(rows_dir, ignore_errors=True)
            rows_dir.mkdir()
            _WORKER_STATE = (
                bound,
                source,
                store_dir,
                schema_versions,
                min_text_coverage,
                rows_dir,
            )
        else:
            _WORKER_STATE = (
                bound,
                source,
                store_dir,
                schema_versions,
                min_text_coverage,
            )

    def _skip_record(position: int, sample: Any, skip: str) -> tuple[Any, ...]:
        """The record a Phase A skip page produces without materializing:
        the same shape the pool workers return, so one record path serves
        both."""
        return (
            position,
            False,
            sample.document_bundle_id,
            sample.page_index,
            skip,
            None,
            None,
            None,
            None,
            False,
            None,
        )

    def _materialize_page(position: int, sample: Any) -> tuple[Any, ...]:
        """Phase A's in-process materialization (the pinning page,
        `workers > 1`): the same record contract as
        `_materialize_worker_payload`, so the pinning page flows through
        the same record path as every pooled page."""
        assert _WORKER_STATE is not None, "build() did not pin the worker state"
        bound, source, store_dir, schema_versions, min_text_coverage, rows_dir = (
            _WORKER_STATE
        )
        return _materialize_core(
            bound=bound,
            source=source,
            store_dir=store_dir,
            schema_versions=schema_versions,
            min_text_coverage=min_text_coverage,
            rows_dir=rows_dir,
            position=position,
            sample=sample,
        )

    #: `workers > 1` only: the shard window -- written pages awaiting an
    #: assembly job, in cohort order -- and the assembly futures in
    #: submission (shard) order.
    shard_window: list[tuple[int, tuple[str, int]]] = []
    assembly_pool: ProcessPoolExecutor | None = None
    #: Submitted assembly jobs awaiting commit, in submission (shard)
    #: order. Drained FIFO, so the manifest's shard order is the cohort
    #: order no matter how many assemblies run at once.
    assembly_pending: list[tuple[Any, list[tuple[str, int]], int, int]] = []
    _shard_ordinal = 0

    def _submit_shard_window() -> None:
        nonlocal _shard_ordinal, assembly_pool
        assert writer is not None
        if assembly_pool is None:
            # Lazy: a build that never fills a window never forks the
            # assembly pool at all.
            assembly_pool = ProcessPoolExecutor(
                max_workers=_assembly_pool_size(workers),
                mp_context=mp.get_context("fork"),
                initializer=_materialize_worker_init,
            )
        job = (
            _shard_ordinal,
            [position for position, _ in shard_window],
            writer.staging,
        )
        keys = [key for _, key in shard_window]
        # Submit and KEEP GOING. `.result()` here was a pool of six used
        # one job at a time: the main loop blocked on each ~1.3 GB shard
        # save+hash while every other core idled. Measured mid-build
        # 2026-09-23: 335 pages/min with the box 97.6% idle, a 4-hour
        # projection for 90,786 pages against the 1h05 the previous build
        # took. Order is untouched -- futures are drained FIFO below, so
        # `commit_shard` still runs in shard (cohort) order.
        assembly_pending.append((
            assembly_pool.submit(_assembly_job, job), keys,
            shard_window[-1][0] + 1, len(resume_skip_records),
        ))
        shard_window.clear()
        _shard_ordinal += 1
        while len(assembly_pending) > _assembly_pool_size(workers):
            _commit_oldest_shard()

    def _commit_oldest_shard() -> None:
        """Commit the oldest submitted shard, blocking only on it."""
        assert writer is not None
        future, keys, next_position, skip_count = assembly_pending.pop(0)
        name, count, digest = future.result()
        # Resume must advance only past this committed shard, never past
        # newer submissions whose rows have not reached the manifest.
        _sync_build_progress(next_position, skip_count)
        writer.commit_shard(name, count=count, sha256=digest, keys=keys)

    def _process_record(record: tuple[Any, ...]) -> None:
        """One page's record, in cohort order: the accumulators and the
        shard window -- the `workers > 1` counterpart of `_apply`, same
        fields, same order, so the manifest is `workers`-independent."""
        nonlocal written, attempted
        (
            _position,
            is_written,
            bundle,
            page,
            skip,
            coverage,
            typed,
            ledger,
            node_shares,
            continuation,
            region,
        ) = record
        attempted += 1
        matched.add((bundle, page))
        if ledger is not None:
            derivation_ledger_rows.append({**ledger, "written": is_written})
        if typed is not None:
            typed_channel_rows.append(
                {
                    "document_bundle_id": bundle,
                    "page_index": page,
                    "flipped": typed.flipped,
                    "typed_targets_node_not_opened_here": typed.node_not_opened_here,
                    "written": is_written,
                }
            )
        if region is not None:
            region_channel_rows.append(
                {
                    "document_bundle_id": bundle,
                    "page_index": page,
                    "region_targets_node_not_opened_here": region.node_not_opened_here,
                    "written": is_written,
                }
            )
        if coverage is not None:
            text_coverage_rows.append(
                {
                    "document_bundle_id": bundle,
                    "page_index": page,
                    "granted_chars": coverage.granted_chars,
                    "store_chars": coverage.store_chars,
                    "written": is_written,
                    # P14 part 2: per-node breakdown, JSON-safe dicts --
                    # see `_apply`.
                    "node_shares": node_shares,
                }
            )
        if not is_written:
            _record_skip(
                bundle,
                page,
                skip,
                continuation,
                position=_position,
                metadata=_build_metadata(
                    position=_position,
                    document_bundle_id=bundle,
                    page_index=page,
                    coverage=coverage,
                    typed=typed,
                    ledger=ledger,
                    node_shares=node_shares,
                    region=region,
                ),
            )
            return
        written += 1
        shard_window.append((_position, (bundle, page)))
        assert writer is not None
        if len(shard_window) >= writer.shard_size:
            _submit_shard_window()
        if progress_every and written % progress_every == 0:
            print(f"[exact-text-cache] {written}/{attempt_target} page(s)", flush=True)

    # Phase A -- serial, until the first page fixes the schema versions
    # and creates the writer. A page whose mask projection has no hash
    # cannot fix anything; it is counted and the search continues,
    # exactly as the pre-parallel loop did.
    index = 0
    if reuse_schema is not None:
        _pin_writer(reuse_schema)
    while writer is None and index < len(cohort):
        if written >= attempt_target:
            break
        sample = cohort[index]
        index += 1
        if (sample.document_bundle_id, sample.page_index) not in source.index:
            if workers == 1:
                attempted += 1
                matched.add((sample.document_bundle_id, sample.page_index))
                _record_skip(
                    sample.document_bundle_id,
                    sample.page_index,
                    "skipped_no_cached_features",
                    False,
                    position=index - 1,
                    metadata=_build_metadata(
                        position=index - 1,
                        document_bundle_id=sample.document_bundle_id,
                        page_index=sample.page_index,
                        coverage=None,
                        typed=None,
                        ledger=None,
                        node_shares=None,
                        region=None,
                    ),
                )
                continue
            _process_record(
                _skip_record(index - 1, sample, "skipped_no_cached_features")
            )
            continue
        item = build_item(store_dir, sample, with_geometry=True)
        mask_projection = item.mask_projection or {}
        mask_projection_hash = (sample.mask_projection or {}).get("sha256")
        if mask_projection_hash is None:
            # No mask projection artifact means no hash to bind. Nothing
            # here may substitute one: every LossUnit carries it.
            if workers == 1:
                attempted += 1
                matched.add((sample.document_bundle_id, sample.page_index))
                _record_skip(
                    sample.document_bundle_id,
                    sample.page_index,
                    "skipped_unencodable",
                    False,
                    position=index - 1,
                    metadata=_build_metadata(
                        position=index - 1,
                        document_bundle_id=sample.document_bundle_id,
                        page_index=sample.page_index,
                        coverage=None,
                        typed=None,
                        ledger=None,
                        node_shares=None,
                        region=None,
                    ),
                )
                continue
            _process_record(_skip_record(index - 1, sample, "skipped_unencodable"))
            continue
        if schema_versions is None:
            # The first page that declares all three versions fixes what
            # this cache claims; every later page is checked against it and
            # skipped (never restamped) if it differs.
            candidate = page_schema_versions(item.targets, item.fragment, mask_projection)
            if candidate is None:
                if workers == 1:
                    attempted += 1
                    matched.add((sample.document_bundle_id, sample.page_index))
                    _record_skip(
                        sample.document_bundle_id,
                        sample.page_index,
                        "skipped_unencodable",
                        False,
                        position=index - 1,
                        metadata=_build_metadata(
                            position=index - 1,
                            document_bundle_id=sample.document_bundle_id,
                            page_index=sample.page_index,
                            coverage=None,
                            typed=None,
                            ledger=None,
                            node_shares=None,
                            region=None,
                        ),
                    )
                    continue
                _process_record(_skip_record(index - 1, sample, "skipped_unencodable"))
                continue
            _pin_writer(candidate)
            if resume_position:
                index = resume_position
                break
        if workers == 1:
            attempted += 1
            matched.add((sample.document_bundle_id, sample.page_index))
            # The fixing page itself is materialized through the shared
            # worker function, like any later page (which re-reads its
            # item -- one page, once per run; a cheap store read).
            _apply(index - 1, sample, _materialize_worker(sample))
        else:
            _process_record(_materialize_page(index - 1, sample))

    if writer is None:
        raise FeatureCacheMismatch(
            f"no page of {feature_cache_dirs} was found in {release_dir} split "
            f"{split!r} / {task_kind!r}: nothing to build, and an empty cache "
            "would claim a cohort that was never attempted"
        )

    # Phase B -- the rest of the cohort, in cohort order. With
    # `workers > 1` the per-page compute fans out over a fork pool and
    # records are consumed in submission (cohort) order, so the writer's
    # page sequence -- and therefore the identity, rows and summary
    # fields -- is independent of `workers`.
    if resume_position:
        index = resume_position
    remaining = cohort[index:]
    if workers == 1:
        for position, sample in enumerate(remaining, start=index):
            if written >= attempt_target:
                break
            attempted += 1
            matched.add((sample.document_bundle_id, sample.page_index))
            _apply(position, sample, _materialize_worker(sample))
    else:
        materialize_pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("fork"),
            initializer=_materialize_worker_init,
        )
        try:
            payload = [(index + i, sample) for i, sample in enumerate(remaining)]
            for record in materialize_pool.map(
                _materialize_worker_payload,
                payload,
                chunksize=_materialize_chunksize(
                    len(payload), workers, page_image_mode=page_image_mode
                ),
            ):
                _process_record(record)
                if written >= attempt_target:
                    break
        except BaseException:
            # A page that dies mid-build must still leave every SHARD
            # that was already assembled committed, or the resume
            # checkpoint loses work the build actually did. Shards are
            # submitted only once their window is full, so every pending
            # one precedes the failure. Best effort: a pending assembly
            # that itself failed cannot be committed, and the original
            # exception is what propagates either way.
            while assembly_pending:
                try:
                    _commit_oldest_shard()
                except Exception:
                    break
            raise
        finally:
            materialize_pool.shutdown(wait=False, cancel_futures=True)
        # The last (partial) window, then drain the assembly pool in
        # submission order and commit each shard as its record arrives --
        # the manifest's shard order is the cohort order.
        if shard_window:
            _sync_build_progress(attempted)
            _submit_shard_window()
        while assembly_pending:
            _commit_oldest_shard()
        if assembly_pool is not None:
            assembly_pool.shutdown(wait=True)

    aggregate_coverage = text_coverage_summary(
        text_coverage_rows, min_text_coverage=min_text_coverage
    )
    finalize_text_coverage_rows(text_coverage_rows, min_text_coverage=min_text_coverage)
    writer.record_summary("min_text_coverage", min_text_coverage)
    writer.record_summary("text_coverage_rows", text_coverage_rows)
    writer.record_summary("text_coverage", aggregate_coverage)
    # Per row, never aggregated away: `flipped` is the diff that goes to
    # zero if the withdrawal is removed, and a per-page record is what
    # lets a reader of the manifest see WHICH page it went to zero on.
    writer.record_summary("typed_channel_rows", typed_channel_rows)
    # region_channel_rows (T1): one row per page whose region channel was
    # derived, mirroring typed_channel_rows -- see that field's own comment
    # for why this lives per-page rather than aggregated away.
    writer.record_summary("region_channel_rows", region_channel_rows)
    # In the MANIFEST, next to `text_coverage_rows`: the S2 gate cohort
    # already reads the manifest (`exact_text_cache.cache_manifest`) and no
    # shard, and a build summary JSON is a separate file a reader may not
    # have. One row per skipped page keeps it small (535 of 10 718 pages on
    # release-v13 val_iid).
    writer.record_summary("page_skip_rows", page_skip_rows)
    # DEC-049 bod 7: no cache identity covers the page-piece withdrawal table
    # the typed targets were masked under, so the manifest names its version.
    from unflatten.model.piece_withdrawal import PIECE_WITHDRAWAL_VERSION

    writer.record_summary("piece_withdrawal_version", PIECE_WITHDRAWAL_VERSION)
    # DEC-033 §5/§6/§10 (T6 wiring): `derivation_ledger_rows` is never
    # `None`-but-empty for the standalone profile and never present at
    # all for any other -- old-profile manifests stay byte-identical, no
    # new keys (`derivation_ledger_rows` is empty here in that case, and
    # this guard is the reason it is never written).
    is_standalone_profile = bound.capability_profile_id in STANDALONE_PAGE_PROFILE_IDS
    aggregate_derivation_ledger = (
        derivation_ledger_summary(derivation_ledger_rows) if is_standalone_profile else None
    )
    if is_standalone_profile:
        writer.record_summary("derivation_ledger_rows", derivation_ledger_rows)
        writer.record_summary("derivation_ledger", aggregate_derivation_ledger)
    writer.record_summary("feature_sources", list(grammar_records))
    target_policy_summary = (
        _target_policy_summary(target_policy, cohort, derivation_ledger_rows)
        if target_policy is not None
        else None
    )
    if target_policy_summary is not None:
        writer.record_summary("target_policy", target_policy_summary)
    _sync_build_progress(attempted)
    writer.close()

    return {
        "written": written,
        "attempted": attempted,
        "min_text_coverage": min_text_coverage,
        "text_coverage_rows": text_coverage_rows,
        "text_coverage": aggregate_coverage,
        "typed_channel_rows": typed_channel_rows,
        "region_channel_rows": region_channel_rows,
        "page_skip_rows": page_skip_rows,
        "piece_withdrawal_version": PIECE_WITHDRAWAL_VERSION,
        **(
            {
                "derivation_ledger_rows": derivation_ledger_rows,
                "derivation_ledger": aggregate_derivation_ledger,
            }
            if is_standalone_profile
            else {}
        ),
        # The scope this run actually ran with, in the artifact and not
        # only on a terminal nobody kept: `cohort` is what was available,
        # `attempt_target` what this run allowed itself, `limit` what the
        # operator asked for (`null` = the whole cohort).
        "split_samples": split_samples,
        "single_page_samples": single_page_samples,
        "cohort": len(cohort),
        "attempt_target": attempt_target,
        "limit": limit,
        **counts,
        "feature_cache_dirs": [str(d) for d in feature_cache_dirs],
        "feature_sources": list(grammar_records),
        "feature_cache_entries": len(source.index),
        "feature_rows_unmatched": len(set(source.index) - matched),
        "feature_rows_missing": len(missing_feature_keys),
        "native_vocab_size": width,
        "workers": workers,
        "out_dir": str(out_dir),
        "identity": asdict(identity),
        # Recorded in the artifact, not only in a report: two identity
        # fields the candidate track genuinely has no value for (module
        # docstring).
        "identity_gaps": {
            "feature_tap": f"source cache records null; stored as {NO_FEATURE_TAP!r}",
            "max_pixels": f"source cache records null; stored as {NO_MAX_PIXELS}",
        },
        **(only_bundles_extra or {}),
        **({"target_policy": target_policy_summary} if target_policy_summary is not None else {}),
    }


def main(argv: list[str] | None = None) -> int:
    def positive_integer(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return parsed

    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.build_exact_text_cache",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("store_dir", type=Path)
    parser.add_argument(
        "feature_cache_dir",
        help="the source feature cache, or '-' for a page-image build "
        "(--page-image-policy): no feature source at all",
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--supplement-feature-cache",
        type=Path,
        action="append",
        default=None,
        help="an additional feature cache directory whose pages are not "
        "in feature_cache_dir (the primary). Repeatable. Every supplement "
        "must match the primary's backbone, backbone_revision, processor, "
        "feature_tap, max_pixels, multiscale, merged_grid and "
        "storage_dtype, and no page key may appear in more than one "
        "source.",
    )
    parser.add_argument(
        "--bindings",
        type=Path,
        required=True,
        help="JSON file supplying the identity hashes this build does not "
        f"compute for itself: {', '.join(REQUIRED_BINDING_HASHES)}. Required "
        "with no default -- a missing, unknown or malformed entry is refused. "
        "Every one of the seven now has a producer (module docstring), "
        "with head_registry_hash owner-invoked via the bindings script "
        "(property_head_registry.head_registry_hash, never called in-repo). "
        "stage_policy_hash and conditioning_policy_hash additionally need "
        "owner-chosen content -- the λ weights and the conditioning mode -- "
        "which their producers refuse to default, so a cache built with "
        "placeholders for those two is not gradient-admissible",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--task-kind", default="structure_v0")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many pages have been WRITTEN. Absent (the "
        "default) builds the whole cohort -- there is no numeric default, "
        "because one would make 'build everything' unsayable and would stop "
        "a full build at a number nobody chose while reporting a complete "
        "cache. Whatever this resolves to is printed before the first page.",
    )
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument(
        "--checkpoint-every-shards", type=positive_integer, default=1,
        help="checkpoint the first full shard and then every N full shards; "
        "close always commits all rows. Interrupted builds replay rows after the checkpoint.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes to materialize pages with (default: 1 = in-process). "
        "The per-page work fans out over a fork pool but the writer stays "
        "sequential, so the cache content is identical for any count.",
    )
    parser.add_argument(
        "--text-encoding-budget-ops",
        type=int,
        default=DEFAULT_TEXT_ENCODING_BUDGET_OPS,
        help="tokenizer probes one text range may cost before the page is "
        f"skipped (default: {DEFAULT_TEXT_ENCODING_BUDGET_OPS})",
    )
    parser.add_argument(
        "--allow-dev-only",
        action="store_true",
        help="admit records whose privacy review is still open. RFC005 §9.2 "
        "blocks those from training; the manifest then says 'dev_only' and a "
        "gradient-bearing read of this cache is refused",
    )
    parser.add_argument(
        "--all-page-counts",
        action="store_true",
        help="do not restrict the cohort to single-page documents (DEC-007's "
        "S1 cohort is single-page; this is off by default)",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--min-text-coverage",
        type=float,
        default=0.0,
        help="skip a page whose fragment-owned text characters are fewer than "
        "this fraction of the store's text characters (0.0-1.0). Default 0.0 "
        "(no floor): every page reaching this build's authority stage is "
        "written regardless of coverage. D2 (2026-09-07): 101/1,176 real "
        "cache rows had granted/store < 0.5.",
    )
    parser.add_argument(
        "--only-bundles",
        type=Path,
        default=None,
        help="file of document_bundle_id (one per line, '#' comments and "
        "blank lines ignored) restricting the cohort to exactly those ids "
        "-- pins a cohort across two store revisions for an A/B, since the "
        "RFC003 split allocator can silently reshuffle val/test membership "
        "when one document changes.",
    )
    parser.add_argument(
        "--capability-profile",
        default=None,
        help="capability_profile.CapabilityProfile.profile_id to build under "
        "(e.g. 's2-typed-widened-v1', DEC-024). Absent (the default) uses "
        f"{BOUND_CAPABILITY_PROFILE.profile_id!r} -- today's behaviour, "
        "unchanged.",
    )
    parser.add_argument(
        "--release-lineage-proof",
        type=Path,
        action="append",
        default=None,
        help="a proof file from release_lineage_proof.py admitting a source "
        "feature cache (primary or supplement) whose release_root_hash names "
        "an earlier release, iff the proof's 'to' is this build's own "
        "release, its 'split' is --split, its proof_hash recomputes, and "
        "its keys_checked equals that source's own key count. Repeatable: "
        "each source is matched against the first proof that accepts it, "
        "so sources of different provenance name their own earlier release.",
    )
    parser.add_argument(
        "--legacy-feature-grammar",
        dest="legacy_feature_grammar",
        nargs=2,
        metavar=("DIR", "REASON"),
        action="append",
        default=None,
        help="admit a source feature cache (primary or supplement) whose "
        "grammar_profile/capability_profile_hash do not match this build's "
        "--capability-profile, e.g. every pre-DEC-024 cache (both null). "
        "Repeatable. DIR is matched by resolved path, never by substring, "
        "and a DIR that already conforms is refused -- the flag would cover "
        "nothing and its reason would be stale.",
    )
    parser.add_argument(
        "--target-policy",
        default=None,
        help="v16 target projection (DEC-055): 'v16-targets.v1' grants whitespace "
        "runs framed by owned words (V1a), collapses chains of withdrawn runs "
        "(V1b), measures visible text without an owner (V1c, floor is a "
        "placeholder), and refuses the audit's 'spill' signature. Enters "
        "training_target_materializer_hash. Absent (the default): today's "
        "targets, byte-identical. Requires a standalone-page --capability-profile.",
    )
    parser.add_argument(
        "--page-exclusions",
        type=Path,
        action="append",
        default=None,
        help="audit exclusion TSV (source_sha256 + page_index, e.g. "
        "docs/runs/loop-data-audit-2026-09-29.exclusions.tsv). Repeatable. Every "
        "listed page in the cohort is refused as skipped_defect_listed; the "
        "list's digest is bound into the target policy. Requires --target-policy.",
    )
    parser.add_argument(
        "--page-image-policy",
        type=Path,
        default=None,
        help="DEC-056 addendum A: write a page-image (7.0.0) cache -- rows carry "
        "the page PNG reference instead of features, the frozen ViT computes "
        "them during training. PATH is a VisualInputPolicy JSON "
        "(`python scripts/write_visual_input_policy.py`). Requires "
        "feature_cache_dir '-'.",
    )
    parser.add_argument(
        "--reuse-page-image-cache",
        type=Path,
        default=None,
        help="Verified complete page-image parent for exclusion-only replay.",
    )
    parser.add_argument(
        "--reuse-parent-page-exclusions",
        type=Path,
        action="append",
        default=None,
        help="Repeatable TSV reproducing the parent exclusion policy hash.",
    )
    parser.add_argument(
        "--reuse-parent-build-config",
        type=Path,
        default=None,
        help="Parent manifest/budget and hash-verified launcher/source attestation JSON.",
    )
    args = parser.parse_args(argv)
    visual_input_policy = None
    if args.page_image_policy is not None:
        if args.feature_cache_dir != "-":
            parser.error("--page-image-policy requires feature_cache_dir '-'")
        visual_input_policy = onfly_visual.VisualInputPolicy.from_payload(
            json.loads(args.page_image_policy.read_text())
        )
    elif args.feature_cache_dir == "-":
        parser.error("feature_cache_dir '-' is only valid with --page-image-policy")

    target_policy = resolve_target_policy(
        args.target_policy,
        page_exclusions=(
            load_page_exclusions(args.page_exclusions) if args.page_exclusions else None
        ),
    )
    capability_profile = (
        BOUND_CAPABILITY_PROFILE
        if args.capability_profile is None
        else resolve_capability_profile_by_id(args.capability_profile)
    )
    legacy_feature_grammar = {
        Path(dir_): reason for dir_, reason in (args.legacy_feature_grammar or [])
    }
    only_bundles = (
        load_only_bundles(args.only_bundles) if args.only_bundles is not None else None
    )
    summary = build(
        release_dir=args.release_dir,
        store_dir=args.store_dir,
        feature_cache_dir=(
            None if visual_input_policy is not None else Path(args.feature_cache_dir)
        ),
        supplement_feature_cache_dirs=args.supplement_feature_cache,
        out_dir=args.out_dir,
        bindings=load_bindings(args.bindings),
        bindings_path=args.bindings,
        split=args.split,
        task_kind=args.task_kind,
        limit=args.limit,
        shard_size=args.shard_size,
        checkpoint_every_shards=args.checkpoint_every_shards,
        allow_dev_only=args.allow_dev_only,
        single_page_only=not args.all_page_counts,
        text_encoding_budget_ops=args.text_encoding_budget_ops,
        progress_every=args.progress_every,
        only_bundles=only_bundles,
        only_bundles_path=args.only_bundles,
        min_text_coverage=args.min_text_coverage,
        capability_profile=capability_profile,
        legacy_feature_grammar=legacy_feature_grammar,
        release_lineage_proof_path=args.release_lineage_proof,
        workers=args.workers,
        target_policy=target_policy,
        visual_input_policy=visual_input_policy,
        reuse_page_image_cache=args.reuse_page_image_cache,
        reuse_parent_page_exclusions=args.reuse_parent_page_exclusions,
        reuse_parent_build_config=args.reuse_parent_build_config,
    )
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
