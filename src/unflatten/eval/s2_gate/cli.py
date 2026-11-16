"""S2 exit-gate CLI: `decode | score | report` (design §6, plan task 6).

Three subcommands, one per phase/module already built:

- `decode` -- phase A (`decode.py`, GPU, H200-only per design §12/§8):
  cohort manifest -> sealed artefact JSON.
- `score` -- phase B (`score.py`, CPU-only, design §8): one sealed
  artefact -> a `score.CohortScore` JSON.
- `report` -- the six criteria (`report.py`, CPU-only): the trained and
  frozen-init-baseline sealed artefacts -> the `S2ExitGateReport` JSON.
  `report` re-derives both arms' `score.CohortScore` itself (rather than
  reading `score`'s own output files) so it needs no second serialisation
  format for `CohortScore` across the phase boundary -- `score` remains
  useful as a standalone, single-arm inspection artefact.

**The `decode` subcommand's `functools.partial` wiring** (GOAL, deliberately
left unwired by the modules themselves): `decode.decode_cohort` calls its
`build_decode_kwargs` argument positionally, one page `entry` at a time
(`build_decode_kwargs(entry)`), but `s1_runner.build_gate13_decode_kwargs`
takes `entry` keyword-only alongside four other keyword arguments this CLI
already knows for the whole run (`model`/`vocabulary`/`image_token_id`/
`device`/`max_steps`). `_build_decode_kwargs_partial` below binds those five
with `functools.partial` over a one-line adapter that accepts `entry`
positionally and forwards it as the keyword `build_gate13_decode_kwargs`
requires -- so `decode_cohort` never has to know its shape changed.

**No default that decides anything** (GOAL): `--max-steps` has no default
(a decode's own cost ceiling is not this CLI's decision to make silently),
and no cohort-size argument exists at all -- cohort membership is bound
entirely by the cohort manifest file's own sha256 (`decode.load_cohort_manifest`),
never by a CLI flag. Refusals inconsistent argument combinations cannot
express structurally (e.g. `--arm trained` with no `--checkpoint`) are
`parser.error`, each with its own test.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.eval.s2_gate import decode as decode_mod
from unflatten.eval.s2_gate import report as report_mod
from unflatten.eval.s2_gate import score as score_mod
from unflatten.eval.s2_gate import sealed
from unflatten.model import checkpoint_ancestry as ca
from unflatten.model import component_lock as component_lock_mod
from unflatten.model import qwen_text_allowlist as qtal
from unflatten.model import s1_runner
from unflatten.model.capability_profile import CapabilityProfile, resolve_capability_profile_by_hash

# ---------------------------------------------------------------------------
# Checkpoint / vocabulary provenance refusals (should_fix, code review
# 2026-09-07) -- pure helpers so they are testable without a GPU or a real
# tokenizer snapshot.
# ---------------------------------------------------------------------------


class CheckpointProvenanceRefused(ValueError):
    """Refused: the loaded `--checkpoint` state does not carry evidence
    that it is what `--arm trained` claims. Without this, a
    `--random-features` null-model checkpoint would seal as `arm="trained"`
    indistinguishably from a real one (should_fix, code review
    2026-09-07)."""


# ---------------------------------------------------------------------------
# --reference-partial-heads (RFC005-ERRATA-005 T2): decode a checkpoint
# trained on an older model shape whose typed/table_plan/node_visual_aligner
# heads today's QwenDenseCandidateModel has and it does not -- e.g. the real
# `s4-discovery-v14-r4/checkpoint.pt` (measured: 702 of 991 non-backbone
# keys missing, 10 unexpected under the old dense-head shape). Never used
# for an ordinary decode: `_validate_decode_args` refuses it together with
# `--typed-decode`/`--region-decode`, since those heads are untrained on a
# partial-heads-loaded model and free-running token generation
# (`free_running_decode.py`) never reads any of the missing heads before
# termination in the first place (module docstring ~1139-1148).
# ---------------------------------------------------------------------------

#: `model.state_dict()` key prefixes `--reference-partial-heads` accepts as
#: MISSING from the checkpoint -- MEASURED against today's
#: `QwenDenseCandidateModel` for `s4-discovery-v14-r4/checkpoint.pt` (702
#: missing keys total): `expert_adapter_dense_heads.` (428),
#: `table_plan.` (236), `node_visual_aligner.` (14), plus every
#: `typed_heads_*` child (`typed_heads_row_cell.` 12, `typed_heads_table.`
#: 12 measured; any other `typed_heads_*` family is accepted on the same
#: ground -- a typed head is, by definition, one this checkpoint's older
#: model never trained). A missing key under any OTHER prefix is refused,
#: never silently absorbed.
REFERENCE_PARTIAL_HEADS_MISSING_PREFIXES: tuple[str, ...] = (
    "expert_adapter_dense_heads.",
    "table_plan.",
    "node_visual_aligner.",
    "typed_heads_",
)

#: `model.state_dict()` key prefix `--reference-partial-heads` accepts as
#: UNEXPECTED (present in the checkpoint, absent from today's model) --
#: MEASURED: 10 keys, all the OLD shape of the dense property heads (e.g.
#: `columns_widths_mpt.regression_head.*`, `layout_tabs_*`), under the same
#: child the missing-key allowance above covers. An unexpected key under
#: any other prefix is refused, never silently ignored.
#: Old-shape heads a reference checkpoint may carry: dense heads and the table
#: plan (the v14-r4 S4 checkpoint has `table_plan.columns_width_mpt.regression_head`);
#: neither is read while decoding tokens.
REFERENCE_PARTIAL_HEADS_UNEXPECTED_PREFIX = ("expert_adapter_dense_heads.", "table_plan.")


class ReferencePartialHeadsRefused(ValueError):
    """`--reference-partial-heads` could not load the checkpoint: a
    missing or unexpected `model.state_dict()` key falls outside
    `REFERENCE_PARTIAL_HEADS_MISSING_PREFIXES`/
    `REFERENCE_PARTIAL_HEADS_UNEXPECTED_PREFIX` -- a shape difference this
    reference mode does not recognise, never silently absorbed."""


def load_reference_partial_heads(
    model: Any, model_state_dict: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`model.load_state_dict(model_state_dict, strict=False)`, then refuse
    unless every reported `missing`/`unexpected` key falls under the
    allowed head prefixes above. Returns `(missing, unexpected)` (sorted)
    for the caller to record on the sealed artefact -- never silently
    swallowed, RFC005-ERRATA-005 T2's own "report exactly which checks were
    relaxed" instruction applies to the load itself too."""
    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    bad_missing = [
        key for key in missing if not key.startswith(REFERENCE_PARTIAL_HEADS_MISSING_PREFIXES)
    ]
    bad_unexpected = [
        key
        for key in unexpected
        if not key.startswith(REFERENCE_PARTIAL_HEADS_UNEXPECTED_PREFIX)
    ]
    if bad_missing or bad_unexpected:
        raise ReferencePartialHeadsRefused(
            "--reference-partial-heads: model.load_state_dict reports keys outside the "
            f"allowed head prefixes -- missing={sorted(bad_missing)!r} "
            f"unexpected={sorted(bad_unexpected)!r} (allowed missing prefixes "
            f"{REFERENCE_PARTIAL_HEADS_MISSING_PREFIXES}, allowed unexpected prefixes "
            f"{REFERENCE_PARTIAL_HEADS_UNEXPECTED_PREFIX})"
        )
    return tuple(sorted(missing)), tuple(sorted(unexpected))


def refuse_unless_checkpoint_provenance_ok(
    state: Mapping[str, Any], *, expect_s1_run_manifest_hash: str
) -> None:
    """Mirrors the two checks of `s1_train.load_checkpoint` this CLI can
    make without constructing a live optimizer: `feature_source` must be
    `"cache"` (never a null-model run) and `s1_run_manifest_hash` must
    match the run this decode is being sealed under, not a foreign one."""
    feature_source = state.get("feature_source")
    if feature_source != "cache":
        raise CheckpointProvenanceRefused(
            f"checkpoint feature_source={feature_source!r} != 'cache' -- a "
            "null-model (--random-features) checkpoint cannot seal as "
            "arm='trained'"
        )
    manifest_hash = state.get("s1_run_manifest_hash")
    if manifest_hash != expect_s1_run_manifest_hash:
        raise CheckpointProvenanceRefused(
            f"checkpoint s1_run_manifest_hash={manifest_hash!r} != expected "
            f"{expect_s1_run_manifest_hash!r} -- a checkpoint from a foreign "
            "run cannot seal as this run's trained arm"
        )


def refuse_unless_checkpoint_ancestry_verified(
    *, checkpoint_path: Path, checkpoint_sha256: str, embedded_ancestry_hash: str | None
) -> str:
    """The third of `s1_train.load_checkpoint`'s refusals this CLI can
    make without a live model/optimizer (coordinator follow-up, code
    review 2026-09-07): recompute `ancestry_hash` from the checkpoint's
    sidecar (`checkpoint_ancestry.sidecar_path`) and refuse unless it
    matches the `ancestry_hash` embedded in the checkpoint state.

    A recorded-but-unverified `ancestry_hash` is exactly the
    label-not-evidence pattern this package spent today removing
    elsewhere (the retired `no_grammar_bypass = True` literal, a
    self-agreeing `expected_accounting`, an unread `arm` field) -- so this
    function does not merely record the hash, it recomputes it and
    returns the DERIVED `checkpoint_id` (`compute_checkpoint_id`) for the
    caller to record instead of the bare hash.

    A `--arm trained` checkpoint with no sidecar is refused outright, not
    silently treated as verified: `s1_train.save_checkpoint` only omits
    the sidecar for a diagnostic run (`ancestry_base is None`), and a
    diagnostic run's weights have no ratified ancestry to seal as the
    trained arm.
    """
    path = ca.sidecar_path(checkpoint_path)
    if not path.exists():
        raise CheckpointProvenanceRefused(
            f"checkpoint ancestry sidecar {path} does not exist -- a "
            "'trained' arm checkpoint with no sidecar is a diagnostic run "
            "(s1_train.save_checkpoint writes no sidecar when "
            "ancestry_base is None) and cannot seal as this run's trained arm"
        )
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise CheckpointProvenanceRefused(
            f"checkpoint ancestry sidecar {path} could not be read: {exc}"
        ) from exc
    try:
        recomputed_ancestry_hash = ca.compute_ancestry_hash(doc)
    except ca.CheckpointAncestryInvalid as exc:
        raise CheckpointProvenanceRefused(
            f"checkpoint ancestry sidecar {path} is malformed: {exc}"
        ) from exc
    if recomputed_ancestry_hash != embedded_ancestry_hash:
        raise CheckpointProvenanceRefused(
            f"sidecar ancestry_hash (recomputed {recomputed_ancestry_hash!r}) != "
            f"checkpoint state's embedded ancestry_hash {embedded_ancestry_hash!r} "
            "-- the checkpoint and its sidecar disagree about the ancestry "
            "that authorized it"
        )
    return ca.compute_checkpoint_id(recomputed_ancestry_hash, checkpoint_sha256)


def checkpoint_provenance_fields(
    state: Mapping[str, Any], *, checkpoint_sha256: str, checkpoint_id: str
) -> dict[str, Any]:
    """The fields to bind into `eval_cache_identity` (and therefore onto
    the sealed artefact and its `input_parity_proof_hash`) so phase B can
    tell which checkpoint actually produced the `"trained"` arm.

    `checkpoint_id` is the DERIVED value from
    `refuse_unless_checkpoint_ancestry_verified` -- never the bare,
    unverified `ancestry_hash` off the checkpoint state (coordinator
    follow-up, code review 2026-09-07)."""
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_feature_source": state.get("feature_source"),
        "checkpoint_s1_run_manifest_hash": state.get("s1_run_manifest_hash"),
        "checkpoint_id": checkpoint_id,
    }


class VocabularyIdentityMismatch(ValueError):
    """Refused: the constructed `CompositeVocabulary`'s `artifact_hash`
    does not match the exact-text cache identity's own
    `composite_vocabulary_hash` (should_fix, code review 2026-09-07)."""


class ComponentLockMismatch(ValueError):
    """Refused: the `--component-lock` document's own computed hash does
    not match `run_manifest_bindings["component_lock_hash"]` -- a lock
    that disagrees with the bindings is a different lock than the run
    was bound to (must_fix, bug-eval-cli-component-lock-hash)."""


def refuse_unless_component_lock_matches_bindings(
    component_lock_hash: str, run_manifest_bindings: Mapping[str, Any]
) -> None:
    expected = run_manifest_bindings["component_lock_hash"]
    if component_lock_hash != expected:
        raise ComponentLockMismatch(
            f"component lock hash {component_lock_hash!r} != run manifest "
            f"bindings component_lock_hash {expected!r} -- this lock is not "
            "the one the run was bound to"
        )


def refuse_unless_vocabulary_hash_matches(
    vocabulary_artifact_hash: str, identity: Mapping[str, Any]
) -> None:
    expected = identity["composite_vocabulary_hash"]
    if vocabulary_artifact_hash != expected:
        raise VocabularyIdentityMismatch(
            f"vocabulary.artifact_hash={vocabulary_artifact_hash!r} != exact-text "
            f"cache identity composite_vocabulary_hash={expected!r} -- the "
            "vocabulary built for this run is not the one the cache was built with"
        )


def refuse_unless_allowlist_vocab_size_matches(
    allowlist_payload: Mapping[str, Any], candidate_vocab_size: int
) -> None:
    """The same A3 refusal `s1_runner.build_context_from_loaded` performs
    (`s1_runner.py:2674-2685`), added on the decode path too (should_fix,
    code review 2026-09-07): an allowlist built at another width puts
    byte-control/byte-value ids at offsets the head does not share."""
    native_vocab_size = allowlist_payload["native_vocab_size"]
    if native_vocab_size != candidate_vocab_size:
        raise qtal.TextTokenAllowlistUnacceptable(
            f"A3: allowlist native_vocab_size {native_vocab_size} != candidate "
            f"vocab_size {candidate_vocab_size} -- the composite vocabulary's "
            "packed ranges would disagree with the head's (§4.6)"
        )


# ---------------------------------------------------------------------------
# Shared I/O helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2))


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeDependencies:
    """Everything `run_decode` needs beyond the manifest/bindings/lock
    files it reads itself -- GPU-resident and model-shaped, so tests
    inject a fake one instead of constructing this for real (design §12:
    free-running decode does not fit the RTX 4070; only the H200 runs
    this for real)."""

    model: Any
    vocabulary: Any
    image_token_id: int
    device: Any
    eval_cache_identity: Mapping[str, Any]
    feature_lookup: Callable[[str], Any]
    decode_fn: Callable[..., Any]
    feature_presence: Callable[[str], bool] | None = None


def eval_cache_identity_from_exact_text_cache(cache_dir: Path) -> dict[str, Any]:
    """The `eval_cache_identity` that `decode.refuse_unless_eval_cache_identity_ok`
    must compare against the run manifest bindings is the **exact-text**
    cache's identity, not the feature cache's (must_fix 3, code review
    2026-09-07): the feature cache's manifest carries none of the seven
    binding keys, so feeding it in refuses every real run on all seven.
    The exact-text cache carries all seven and differs from the training
    cache by exactly `split` -- the field `refuse_unless_eval_cache_identity_ok`
    already excludes."""
    from unflatten.model.exact_text_cache import ExactTextCacheReader

    reader = ExactTextCacheReader(cache_dir, gradient_bearing=False)
    return dataclasses.asdict(reader.identity)


def resolve_eval_capability_profile(eval_cache_identity: Mapping[str, Any]) -> CapabilityProfile:
    """The capability profile decode/score/report must bind gate 13's
    decode and gate 7's replay to (2026-09-10 review item B): resolved
    from `eval_cache_identity`'s own `capability_profile_hash`, fail-closed
    -- never `PRIMARY_S1_CAPABILITY_PROFILE` hardcoded independently of
    what the cache under evaluation actually names. DEC-024 moves new S2
    caches to `s2-typed-widened-v1`; a cache built under any other known
    profile (`PRIMARY_S1`, `WIDENED_V1`, `TYPED_FROZEN_V1`) resolves to
    that profile instead -- this module does not assume DEC-024 profile
    even though it is the one it exists to support."""
    return resolve_capability_profile_by_hash(eval_cache_identity.get("capability_profile_hash"))


def feature_lookups_from_cache(
    feature_cache_dir: Path,
) -> tuple[Callable[[str], Any], Callable[[str], bool]]:
    """`(feature_lookup, feature_presence)` over a feature cache, keyed
    `(document_bundle_id, 0)`.

    The key -> shard index comes from the manifest's per-shard `rows` key
    lists (a JSON read, validated by `validate_manifest`); a shard is
    loaded -- and sha256-verified by `FeatureCacheReader` -- only when a
    cohort page in it is looked up. The previous version materialised
    every entry just to learn the keys, which on the DEC-033 v13 val cache
    (10 718 pages, ~42 MB each) was OOM-killed at 390 GB RSS
    (2026-09-18). `cache_shards=1`: cohort pages are looked up one at a
    time, so holding more than the current shard resident buys nothing."""
    from unflatten.model.build_exact_text_cache import FeatureSource, _is_exact_text_cache_dir
    from unflatten.model.feature_cache import FeatureCacheReader

    if _is_exact_text_cache_dir(feature_cache_dir):
        return _feature_lookups_from_exact_text_cache(feature_cache_dir)
    reader = FeatureCacheReader(feature_cache_dir, cache_shards=1)
    index: dict[tuple[str, int], tuple[int, int]] = {}
    for shard_ordinal, shard in enumerate(reader.manifest["shard_index"]):
        if "rows" not in shard:
            raise ValueError(
                f"{feature_cache_dir}: cache_version "
                f"{reader.manifest.get('cache_version')!r} records no per-shard "
                "'rows' key list; the page index cannot be built without "
                "reading every entry, which is refused"
            )
        for bundle, page in shard["rows"]:
            index[(bundle, int(page))] = (0, shard_ordinal)
    source = FeatureSource(readers=(reader,), index=index)

    def feature_lookup(document_bundle_id: str) -> Any:
        return source.entry(document_bundle_id, 0)

    def feature_presence(document_bundle_id: str) -> bool:
        return (document_bundle_id, 0) in index

    return feature_lookup, feature_presence


@dataclasses.dataclass(frozen=True)
class _ExactTextFeatureEntry:
    """The two fields `build_gate13_decode_kwargs` reads off a feature entry,
    plus the page's DeepStack levels on the page-image path (`None`
    otherwise; `_build_decode_kwargs_partial` forwards them only if set)."""

    document_bundle_id: str
    page_index: int
    features: Any
    grid_thw: Any
    deepstack_visual_embeds: Any = None
    #: DEC-061 (PROPOSED): a `StoryPattern` supplied by an out-of-src wrapper
    #: (the diagnostic oracle); read only for `--page-start-pattern-source entry`.
    page_start_pattern: Any = None


def _feature_lookups_from_exact_text_cache(
    cache_dir: Path,
) -> tuple[Callable[[str], Any], Callable[[str], bool]]:
    """`(feature_lookup, feature_presence)` over an exact-text cache, whose rows
    carry `features`/`grid_thw` since 5.0.0 -- the same auto-detection the cache
    builder uses for its feature source (`_is_exact_text_cache_dir`). v15b's val
    features exist only there; the structural val feature cache is gone. A
    page-image (7.0.0) cache holds no features and is refused here: it decodes
    through `feature_lookups_from_page_images` (`--visual-input-policy`)."""
    from unflatten.model.exact_text_cache import ExactTextCacheReader, is_page_image_identity

    reader = ExactTextCacheReader(cache_dir, gradient_bearing=False, cache_shards=1)
    if is_page_image_identity(dataclasses.asdict(reader.identity)):
        raise ValueError(
            f"{cache_dir} is a page-image (7.0.0) cache: its rows hold no features -- "
            "decode it with --visual-input-policy and --page-image-store"
        )
    index = {key: position for position, key in enumerate(reader.page_keys)}

    def feature_lookup(document_bundle_id: str) -> Any:
        position = index.get((document_bundle_id, 0))
        if position is None:
            return None
        row = reader[position]
        return _ExactTextFeatureEntry(
            document_bundle_id=row.document_bundle_id,
            page_index=row.page_index,
            features=row.features,
            grid_thw=row.grid_thw,
        )

    def feature_presence(document_bundle_id: str) -> bool:
        return (document_bundle_id, 0) in index

    return feature_lookup, feature_presence


def feature_lookups_from_page_images(
    cache_dir: Path,
    *,
    store_dir: Path,
    cache_policy: Any,
    encoder: Any,
    processor: Any,
) -> tuple[Callable[[str], Any], Callable[[str], bool]]:
    """`(feature_lookup, feature_presence)` over a page-image (7.0.0) cache
    (fresh S4, DEC-056 A): per cohort page the row's PNG is read from
    `store_dir` (sha256 verified), run through `processor` (grid checked
    against the row) and the frozen tower (`encoder`, an
    `onfly_visual.OnTheFlyVisualEncoder`). The entry carries the pooler
    output as `features` and, for a DeepStack encoder, the levels.

    Refused when the cache was not built under `cache_policy`, or when the
    encoder's pixel side (processor, budget, geometry, ViT revision)
    differs from it -- only the DeepStack taps may differ (a v15b
    checkpoint reads the same pixels pooler-only)."""
    from unflatten.model import onfly_visual as ov
    from unflatten.model.exact_text_cache import ExactTextCacheReader

    reader = ExactTextCacheReader(cache_dir, gradient_bearing=False, cache_shards=1)
    if reader.identity.feature_tap != cache_policy.feature_tap():
        raise ValueError(
            f"{cache_dir} was built under feature_tap {reader.identity.feature_tap!r}, not "
            f"this visual input policy's {cache_policy.feature_tap()!r}"
        )
    if ov.pooler_only_policy(encoder.policy) != ov.pooler_only_policy(cache_policy):
        raise ValueError(
            "the decode encoder's pixel side (processor, pixel budget, patch geometry, "
            "ViT revision, dtype) differs from the cache's visual input policy -- only "
            "the DeepStack taps may differ"
        )
    index = {key: position for position, key in enumerate(reader.page_keys)}

    def feature_lookup(document_bundle_id: str) -> Any:
        position = index.get((document_bundle_id, 0))
        if position is None:
            return None
        row = reader[position]
        png = ov.read_page_png(store_dir, row.page_image)
        pixels = ov.page_pixels(png, processor, encoder.policy)
        if not torch_equal(pixels.grid_thw, row.grid_thw):
            raise ValueError(
                f"{document_bundle_id}: processor grid {pixels.grid_thw.tolist()} != the "
                f"row's grid {row.grid_thw.tolist()}"
            )
        pooled, levels = encoder.encode(pixels.pixel_values, pixels.grid_thw)
        return _ExactTextFeatureEntry(
            document_bundle_id=row.document_bundle_id,
            page_index=row.page_index,
            features=pooled,
            grid_thw=row.grid_thw,
            deepstack_visual_embeds=tuple(levels) if levels else None,
        )

    def feature_presence(document_bundle_id: str) -> bool:
        return (document_bundle_id, 0) in index

    return feature_lookup, feature_presence


def _visual_policy_from_args(args: argparse.Namespace) -> Any:
    """`--visual-input-policy` + `--page-image-store` (together or neither)
    -> the loaded `VisualInputPolicy`, or `None` for the feature path."""
    policy_path = getattr(args, "visual_input_policy", None)
    store = getattr(args, "page_image_store", None)
    if (policy_path is None) != (store is None):
        raise ValueError("--visual-input-policy and --page-image-store go together")
    if policy_path is None:
        return None
    from unflatten.model import onfly_visual as ov

    return ov.VisualInputPolicy.from_payload(_read_json(policy_path))


def torch_equal(a: Any, b: Any) -> bool:
    import torch

    return bool(torch.equal(a.to(torch.int64), b.to(torch.int64)))


def decode_visual_policy_for_checkpoint(policy: Any, state: Mapping[str, Any]) -> Any:
    """Which visual policy decodes `state` (a loaded checkpoint dict).

    `policy is None` (the feature path, v15b): `None`, and a checkpoint
    trained with DeepStack is refused -- it loads strictly into the plain
    model (DeepStack adds no parameter) and would decode without its levels.
    On the page-image path the checkpoint's own `visual_feature_tap`
    decides: the policy's taps -> `policy`; no tap (v15b) -> the same
    pixels pooler-only; any other tap is refused."""
    from unflatten.model import onfly_visual as ov

    tap = ov.checkpoint_visual_feature_tap(state)
    if policy is None:
        ov.require_checkpoint_visual_feature_tap(
            state, expected=ov.POOLER_OUTPUT_TAP, what="--checkpoint"
        )
        return None
    if tap == ov.POOLER_OUTPUT_TAP:
        return ov.pooler_only_policy(policy)
    ov.require_checkpoint_visual_feature_tap(
        state, expected=policy.visual_feature_tap(), what="--checkpoint"
    )
    return policy


def build_decode_dependencies(args: argparse.Namespace) -> DecodeDependencies:
    """The real, GPU-resident construction path -- not exercised by the
    default test suite (mirrors `decode.py`'s own module docstring: a
    forward over real visual features does not fit the local GPU)."""
    import torch

    from unflatten.model import composite_vocab as cv
    from unflatten.model.candidate_loader import load_qwen_candidate
    from unflatten.model.free_running_decode import decode_free_running

    eval_cache_identity = eval_cache_identity_from_exact_text_cache(args.exact_text_cache_dir)

    device = torch.device(args.device)
    allowlist_payload = _read_json(args.allowlist)
    vocabulary = cv.make_composite_vocabulary(
        tokenizer_artifact_set_hash=args.tokenizer_artifact_set_hash,
        allowlist_payload=allowlist_payload,
    )
    refuse_unless_vocabulary_hash_matches(vocabulary.artifact_hash, eval_cache_identity)
    sdpa_candidate = load_qwen_candidate(device=str(device))
    refuse_unless_allowlist_vocab_size_matches(allowlist_payload, sdpa_candidate.vocab_size)
    # Fresh S4 page-image path (rebuild gap G5): the checkpoint's own tap picks
    # the levels the model is built with, before the model exists.
    cache_policy = _visual_policy_from_args(args)
    decode_policy = None
    if cache_policy is not None:
        from unflatten.model import onfly_visual as ov

        if args.arm == sealed.DecodeArm.TRAINED:
            peek = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
            decode_policy = decode_visual_policy_for_checkpoint(cache_policy, peek)
            del peek
        else:
            decode_policy = ov.pooler_only_policy(cache_policy)
    model = decode_mod.build_frozen_init_model(
        sdpa_candidate=sdpa_candidate,
        device=device,
        **({"deepstack": True} if decode_policy is not None and decode_policy.deepstack else {}),
    )
    if args.arm == sealed.DecodeArm.TRAINED:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_sha256 = ca.file_sha256(checkpoint_path)
        state = torch.load(args.checkpoint, weights_only=False, map_location=device)
        if cache_policy is None:
            decode_visual_policy_for_checkpoint(None, state)  # a DeepStack checkpoint is refused
        reference_partial_heads = getattr(args, "reference_partial_heads", False)
        identity_mismatches: dict[str, str] = {}

        # RFC005-ERRATA-005 T2: every identity check the ordinary path
        # makes against the checkpoint still RUNS in reference mode --
        # never silently skipped -- but a refusal is caught and RECORDED
        # (never hidden) instead of raised, since the reference may be a
        # checkpoint written before one of these checks' own provenance
        # fields existed (e.g. a legacy ancestry sidecar shape).
        try:
            refuse_unless_checkpoint_provenance_ok(
                state, expect_s1_run_manifest_hash=args.expect_s1_run_manifest_hash
            )
        except CheckpointProvenanceRefused as exc:
            if not reference_partial_heads:
                raise
            identity_mismatches["checkpoint_provenance"] = str(exc)

        try:
            checkpoint_id = refuse_unless_checkpoint_ancestry_verified(
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                embedded_ancestry_hash=state.get("ancestry_hash"),
            )
        except CheckpointProvenanceRefused as exc:
            if not reference_partial_heads:
                raise
            identity_mismatches["checkpoint_ancestry"] = str(exc)
            checkpoint_id = f"reference-partial-heads:{checkpoint_sha256}"

        if reference_partial_heads:
            missing, unexpected = load_reference_partial_heads(
                model, state["model_state_dict"]
            )
        else:
            model.load_state_dict(state["model_state_dict"])
            missing, unexpected = (), ()
        model.eval()
        eval_cache_identity = {
            **eval_cache_identity,
            **checkpoint_provenance_fields(
                state, checkpoint_sha256=checkpoint_sha256, checkpoint_id=checkpoint_id
            ),
        }
        if reference_partial_heads:
            eval_cache_identity["reference_partial_heads"] = True
            eval_cache_identity["reference_partial_heads_missing_keys"] = list(missing)
            eval_cache_identity["reference_partial_heads_unexpected_keys"] = list(unexpected)
            eval_cache_identity["reference_partial_heads_identity_mismatches"] = (
                identity_mismatches
            )

    if cache_policy is None:
        feature_lookup, feature_presence = feature_lookups_from_cache(args.feature_cache_dir)
    else:
        from transformers import AutoImageProcessor

        from unflatten.model import onfly_visual as ov
        from unflatten.model.candidate_loader import (
            QWEN_CANDIDATE_MODEL_ID,
            QWEN_CANDIDATE_REVISION,
        )

        feature_lookup, feature_presence = feature_lookups_from_page_images(
            args.exact_text_cache_dir,
            store_dir=args.page_image_store,
            cache_policy=cache_policy,
            encoder=ov.OnTheFlyVisualEncoder(sdpa_candidate, policy=decode_policy, device=device),
            processor=AutoImageProcessor.from_pretrained(
                QWEN_CANDIDATE_MODEL_ID, revision=QWEN_CANDIDATE_REVISION
            ),
        )

    return DecodeDependencies(
        model=model,
        vocabulary=vocabulary,
        image_token_id=sdpa_candidate.image_token_id,
        device=device,
        eval_cache_identity=eval_cache_identity,
        feature_lookup=feature_lookup,
        decode_fn=decode_free_running,
        feature_presence=feature_presence,
    )


def _build_decode_kwargs_partial(
    *,
    model: Any,
    vocabulary: Any,
    image_token_id: int,
    device: Any,
    max_steps: int,
    capability_profile: CapabilityProfile,
    loop_guard: str | None = None,
    typed_decode: bool = False,
    region_decode: bool = False,
    prefill_chunk_tokens: int | None = None,
    page_start_policy: str | None = None,
    page_start_pattern_fn: Callable[[Any], Any] | None = None,
    story_length_cap: str | None = None,
    story_length_cap_policy_path: str | None = None,
    story_retry: str | None = None,
    story_retry_policy_path: str | None = None,
) -> Callable[[Any], Mapping[str, Any]]:
    """`functools.partial` wiring over `s1_runner.build_gate13_decode_kwargs`
    (module docstring). `build_gate13_decode_kwargs` takes `entry`
    keyword-only; `decode_cohort` calls its `build_decode_kwargs` argument
    positionally with one `entry` per page. `_adapter` bridges the two so
    `functools.partial` can bind everything else once, for the whole
    cohort.

    `capability_profile` is `run_decode`'s resolution of the cache under
    evaluation's own bound profile (2026-09-10 review item B) -- every
    page in one cohort decodes under it. `build_gate13_decode_kwargs`
    requires this profile explicitly.

    `typed_decode` (AST Builder T1, default `False`): the same rebuild for
    `DecodeConfig.typed_decode` -- on, every decoded node's typed heads are
    read at its anchor and the sealed artefact carries the predictions;
    off, `typed_predictions` stays empty and the decode is byte-identical
    to before the flag existed.

    `loop_guard` (DEC-031 task 2, default `None`): `s1_runner.
    build_gate13_decode_kwargs` builds its own `DecodeConfig` with no
    `loop_guard` field -- `s1_runner.py` is out of scope for this task --
    so `_adapter` rebuilds that one field with `dataclasses.replace`
    rather than editing `s1_runner.py` to know about it. `None` leaves
    the built `config` untouched (byte-identical to before this task)."""

    def _adapter(entry: Any, **kwargs: Any) -> Mapping[str, Any]:
        built = dict(s1_runner.build_gate13_decode_kwargs(entry=entry, **kwargs))
        # Fresh S4 page-image path: a DeepStack entry's levels reach
        # `decode_free_running`; every other entry builds the same kwargs as before.
        levels = getattr(entry, "deepstack_visual_embeds", None)
        if levels is not None:
            built["deepstack_visual_embeds"] = tuple(
                level.to(kwargs["device"]) for level in levels
            )
        if loop_guard is not None:
            built["config"] = dataclasses.replace(built["config"], loop_guard=loop_guard)
        if typed_decode:
            built["config"] = dataclasses.replace(built["config"], typed_decode=True)
        if region_decode:
            built["config"] = dataclasses.replace(built["config"], region_decode=True)
        if prefill_chunk_tokens is not None:
            built["config"] = dataclasses.replace(
                built["config"], prefill_chunk_tokens=prefill_chunk_tokens
            )
        if page_start_policy is not None:
            built["config"] = dataclasses.replace(
                built["config"], page_start_policy=page_start_policy
            )
            pattern = page_start_pattern_fn(entry) if page_start_pattern_fn is not None else None
            if pattern is not None:
                built["page_start_pattern"] = pattern
        if story_length_cap is not None:
            built["config"] = dataclasses.replace(
                built["config"],
                story_length_cap=story_length_cap,
                story_length_cap_policy_path=story_length_cap_policy_path,
            )
        if story_retry is not None:
            built["config"] = dataclasses.replace(
                built["config"],
                story_retry=story_retry,
                story_retry_policy_path=story_retry_policy_path,
            )
        return built

    return functools.partial(
        _adapter,
        model=model,
        vocabulary=vocabulary,
        image_token_id=image_token_id,
        device=device,
        max_steps=max_steps,
        capability_profile=capability_profile,
    )


def _resolve_story_retry(args: argparse.Namespace) -> str | None:
    value = getattr(args, "story_retry", "none")
    return None if value == "none" else value


def _resolve_story_retry_policy_path(args: argparse.Namespace) -> str | None:
    path = getattr(args, "story_retry_policy", None)
    return None if path is None else str(path)


def _resolve_story_length_cap(args: argparse.Namespace) -> str | None:
    value = getattr(args, "story_length_cap", "none")
    return None if value == "none" else value


def _resolve_story_length_cap_policy_path(args: argparse.Namespace) -> str | None:
    path = getattr(args, "story_length_cap_policy", None)
    return None if path is None else str(path)


def _resolve_page_start_policy(args: argparse.Namespace) -> str | None:
    value = getattr(args, "page_start_policy", "none")
    return None if value == "none" else value


#: The tap the page-start head's moments are taken at (the pooled ViT output the decoder
#: receives as `entry.features`); equals `FEATURE_TAP` in `scripts/page_start_moments.py`.
PAGE_START_FEATURE_TAP = "pooler_output"


def _resolve_page_start_pattern_fn(
    args: argparse.Namespace, *, device: Any = "cpu"
) -> Callable[[Any], Any] | None:
    """DEC-061 (PROPOSED): the per-entry pattern source for `--page-start-policy`.
    `None` when the policy is off. `entry` reads the optional
    `_ExactTextFeatureEntry.page_start_pattern`; `head` loads the trained head bundle
    once, bound to this run's feature identity (committed policy + merge size +
    pooler-only visual policy hash + feature tap), and predicts from the entry's own
    pooled features. The `page_start_head` package is imported only on that branch."""
    if getattr(args, "page_start_policy", "none") == "none":
        return None
    if getattr(args, "page_start_pattern_source", "entry") == "entry":
        return lambda entry: getattr(entry, "page_start_pattern", None)

    from unflatten.model import onfly_visual as ov
    from unflatten.model.page_start_head.bundle import load_head_bundle
    from unflatten.model.page_start_policy import load_page_start_policy

    visual_policy = _visual_policy_from_args(args)
    if visual_policy is None:
        raise ValueError(
            "--page-start-pattern-source head needs --visual-input-policy: the head is bound to "
            "a pooler-only visual input policy hash, which the feature-cache path does not carry"
        )
    pooler_only = ov.pooler_only_policy(visual_policy)
    visual_hash = pooler_only.policy_hash()
    identity = load_page_start_policy().feature_identity_sha256(
        merge_size=int(pooler_only.merge_size),
        visual_input_policy_hash=visual_hash,
        feature_tap=PAGE_START_FEATURE_TAP,
    )
    bundle = load_head_bundle(
        args.page_start_head,
        device=str(device),
        expect_feature_identity_sha256=identity,
        expect_visual_policy_hash=visual_hash,
    )

    def _head_pattern(entry: Any) -> Any:
        features = getattr(entry, "features", None)
        grid_thw = getattr(entry, "grid_thw", None)
        if features is None or grid_thw is None:
            raise ValueError(
                f"page-start head source: entry {getattr(entry, 'document_bundle_id', '?')!r} "
                "carries no pooled features / grid_thw to predict from"
            )
        return bundle.predict(features, grid_thw)

    return _head_pattern


def _resolve_loop_guard(value: str) -> str | None:
    """`--loop-guard none` (the default) means `DecodeConfig.loop_guard =
    None` -- today's byte-identical decode; `v1` (DEC-031 task 2) and `v2`
    (DEC-036) pass the string straight through."""
    return None if value == "none" else value


def refuse_unless_loop_guard_v2_caps_bind(exact_text_cache_dir: Path) -> None:
    """DEC-036 design §3: a v2 decode's caps must have been measured on
    the train cache of the same target identity as the cache under
    evaluation -- `LoopGuardCapsStale` otherwise. Reads only the cache
    manifest and the caps file, so it runs before any model load."""
    from unflatten.model.loop_guard_v2_caps import (
        load_loop_guard_v2_caps,
        refuse_unless_caps_bind,
    )

    refuse_unless_caps_bind(
        load_loop_guard_v2_caps(), eval_cache_identity_from_exact_text_cache(exact_text_cache_dir)
    )


def run_decode(
    args: argparse.Namespace,
    *,
    build_dependencies: Callable[
        [argparse.Namespace], DecodeDependencies
    ] = build_decode_dependencies,
) -> int:
    raw_manifest = _read_json(args.cohort_manifest)
    manifest = decode_mod.load_cohort_manifest(raw_manifest)
    run_manifest_bindings = _read_json(args.run_manifest_bindings)
    component_lock = _read_json(args.component_lock)
    component_lock_hash = component_lock_mod.document_sha256(component_lock)
    refuse_unless_component_lock_matches_bindings(component_lock_hash, run_manifest_bindings)
    loop_guard = _resolve_loop_guard(getattr(args, "loop_guard", "none"))
    if loop_guard == "v2":
        refuse_unless_loop_guard_v2_caps_bind(args.exact_text_cache_dir)

    deps = build_dependencies(args)
    capability_profile = resolve_eval_capability_profile(deps.eval_cache_identity)

    build_decode_kwargs = _build_decode_kwargs_partial(
        model=deps.model,
        vocabulary=deps.vocabulary,
        image_token_id=deps.image_token_id,
        device=deps.device,
        max_steps=args.max_steps,
        capability_profile=capability_profile,
        loop_guard=loop_guard,
        typed_decode=getattr(args, "typed_decode", False),
        region_decode=getattr(args, "region_decode", False),
        prefill_chunk_tokens=getattr(args, "prefill_chunk_tokens", None),
        page_start_policy=_resolve_page_start_policy(args),
        page_start_pattern_fn=_resolve_page_start_pattern_fn(args, device=deps.device),
        story_length_cap=_resolve_story_length_cap(args),
        story_length_cap_policy_path=_resolve_story_length_cap_policy_path(args),
        story_retry=_resolve_story_retry(args),
        story_retry_policy_path=_resolve_story_retry_policy_path(args),
    )

    artefact = decode_mod.decode_cohort(
        manifest=manifest,
        cohort_manifest_sha256_value=raw_manifest["sha256"],
        arm=args.arm,
        eval_cache_identity=deps.eval_cache_identity,
        run_manifest_bindings=run_manifest_bindings,
        component_lock_hash=component_lock_hash,
        feature_lookup=deps.feature_lookup,
        build_decode_kwargs=build_decode_kwargs,
        decode_fn=deps.decode_fn,
        feature_presence=deps.feature_presence,
    )
    _write_json(args.output, sealed.artefact_to_dict(artefact))
    return 0


# ---------------------------------------------------------------------------
# Vocabulary/tokenizer -- shared by `score` and `report`
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VocabularyAndTokenizer:
    vocabulary: Any
    tokenizer: Any


def build_vocabulary_and_tokenizer(args: argparse.Namespace) -> VocabularyAndTokenizer:
    """The real construction path: `cv.make_composite_vocabulary` over a
    supplied allowlist payload, and the DEC-007 D3-pinned tokenizer
    (`qwen_tokenizer_artifacts.load_pinned_tokenizer`)."""
    from unflatten.model import composite_vocab as cv
    from unflatten.model.qwen_tokenizer_artifacts import load_pinned_tokenizer

    allowlist_payload = _read_json(args.allowlist)
    vocabulary = cv.make_composite_vocabulary(
        tokenizer_artifact_set_hash=args.tokenizer_artifact_set_hash,
        allowlist_payload=allowlist_payload,
    )
    identity = eval_cache_identity_from_exact_text_cache(args.exact_text_cache_dir)
    refuse_unless_vocabulary_hash_matches(vocabulary.artifact_hash, identity)
    tokenizer, _snapshot_dir = load_pinned_tokenizer()
    return VocabularyAndTokenizer(vocabulary=vocabulary, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# Shared: reading the exact-text cache and the cohort manifest's lineage map
# ---------------------------------------------------------------------------


def _read_exact_text_rows(cache_dir: Path, bundle_ids: Collection[str]) -> dict[str, Any]:
    """The exact-text rows of `bundle_ids` only, keyed by bundle id (the
    last row of a bundle in cache order wins, as when every row was read).

    Positions come from `reader.page_keys` (the manifest's key lists), so
    only the shards holding a wanted bundle are loaded and verified.
    `features` is dropped from each returned row: scoring reads the
    packed token ids and text ranges, never the visual features, which
    are ~40 MB of every ~41 MB row -- the v13 val cache holds 10 183 of
    them (~420 GB)."""
    import torch

    from unflatten.model.exact_text_cache import ExactTextCacheReader

    reader = ExactTextCacheReader(cache_dir, gradient_bearing=False, cache_shards=1)
    wanted = set(bundle_ids)
    positions = {
        bundle: position
        for position, (bundle, _page) in enumerate(reader.page_keys)
        if bundle in wanted
    }
    return {
        bundle: dataclasses.replace(reader[position], features=torch.empty(0))
        for bundle, position in sorted(positions.items(), key=lambda item: item[1])
    }


def _scored_bundle_ids(
    manifest: decode_mod.CohortManifest, *artefacts: sealed.SealedCohortArtefact
) -> set[str]:
    """Every bundle `_score_artefact` can look up a row for: the cohort's
    pages and each artefact's pages."""
    bundles = {page.document_bundle_id for page in manifest.pages}
    for artefact in artefacts:
        bundles.update(page.document_bundle_id for page in artefact.pages)
    return bundles


def _lineage_by_bundle(manifest: decode_mod.CohortManifest) -> dict[str, str]:
    return {page.document_bundle_id: page.lineage_root_id for page in manifest.pages}


def _score_artefact(
    *,
    artefact: sealed.SealedCohortArtefact,
    rows_by_key: Mapping[str, Any],
    lineage_by_bundle: Mapping[str, str],
    vt: VocabularyAndTokenizer,
) -> score_mod.CohortScore:
    """Score only the pages `score.py` can actually see (design/module
    docstring: pages with a matching exact-text row) -- the typed
    pre-decode error classes never had one and are folded back in by
    `report.reconcile_denominator`, one layer up, not here.

    Resolves `capability_profile` from THIS `artefact`'s own
    `eval_cache_identity.capability_profile_hash` (2026-09-10 review item
    B) -- `run_report` scores a trained and a baseline artefact
    separately, and each is scored under the profile ITS OWN cache
    identity names, never a profile shared or hardcoded across the two."""
    capability_profile = resolve_eval_capability_profile(artefact.eval_cache_identity)
    scoreable_pages = tuple(p for p in artefact.pages if p.document_bundle_id in rows_by_key)
    scoreable_rows = {k: v for k, v in rows_by_key.items() if k in lineage_by_bundle}
    return score_mod.score_cohort(
        scoreable_pages,
        scoreable_rows,
        lineage_by_bundle=lineage_by_bundle,
        vocabulary=vt.vocabulary,
        tokenizer=vt.tokenizer,
        capability_profile=capability_profile,
    )


def run_score(
    args: argparse.Namespace,
    *,
    build_vocabulary_and_tokenizer: Callable[
        [argparse.Namespace], VocabularyAndTokenizer
    ] = build_vocabulary_and_tokenizer,
) -> int:
    raw_manifest = _read_json(args.cohort_manifest)
    manifest = decode_mod.load_cohort_manifest(raw_manifest)
    artefact = sealed.artefact_from_json(Path(args.sealed_artefact).read_text())
    rows_by_key = _read_exact_text_rows(
        args.exact_text_cache_dir, _scored_bundle_ids(manifest, artefact)
    )
    vt = build_vocabulary_and_tokenizer(args)

    cohort_score = _score_artefact(
        artefact=artefact,
        rows_by_key=rows_by_key,
        lineage_by_bundle=_lineage_by_bundle(manifest),
        vt=vt,
    )
    _write_json(args.output, dataclasses.asdict(cohort_score))
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def run_report(
    args: argparse.Namespace,
    *,
    build_vocabulary_and_tokenizer: Callable[
        [argparse.Namespace], VocabularyAndTokenizer
    ] = build_vocabulary_and_tokenizer,
    run_checkpoint_resume_tests: Callable[
        ..., report_mod.CheckpointResumeEvidence
    ] = report_mod.run_checkpoint_resume_tests,
) -> int:
    raw_manifest = _read_json(args.cohort_manifest)
    manifest = decode_mod.load_cohort_manifest(raw_manifest)
    lineage_by_bundle = _lineage_by_bundle(manifest)
    trained_artefact = sealed.artefact_from_json(Path(args.trained_sealed_artefact).read_text())
    baseline_artefact = sealed.artefact_from_json(Path(args.baseline_sealed_artefact).read_text())
    rows_by_key = _read_exact_text_rows(
        args.exact_text_cache_dir,
        _scored_bundle_ids(manifest, trained_artefact, baseline_artefact),
    )
    vt = build_vocabulary_and_tokenizer(args)

    trained_score = _score_artefact(
        artefact=trained_artefact,
        rows_by_key=rows_by_key,
        lineage_by_bundle=lineage_by_bundle,
        vt=vt,
    )
    baseline_score = _score_artefact(
        artefact=baseline_artefact,
        rows_by_key=rows_by_key,
        lineage_by_bundle=lineage_by_bundle,
        vt=vt,
    )

    pyramid_tier_by_bundle = (
        _read_json(args.pyramid_tier_map) if args.pyramid_tier_map is not None else None
    )

    criterion_1 = report_mod.criterion_1_structural_validity(
        trained_score,
        manifest=manifest,
        cohort_accounting=trained_artefact.accounting,
        cohort_pages=trained_artefact.pages,
    )
    criterion_2 = report_mod.criterion_2_exact_text_vs_baseline(
        trained_score,
        baseline_score,
        manifest=manifest,
        cohort_manifest_sha256=raw_manifest["sha256"],
        trained_artefact=trained_artefact,
        baseline_artefact=baseline_artefact,
    )
    criterion_3 = report_mod.criterion_3_zero_silent_truncation_and_invalid_success(
        trained_score,
        manifest=manifest,
        cohort_accounting=trained_artefact.accounting,
        cohort_pages=trained_artefact.pages,
    )
    criterion_4 = report_mod.criterion_4_slices(
        trained_score, manifest=manifest, cohort_accounting=trained_artefact.accounting
    )
    evidence = run_checkpoint_resume_tests(test_file=args.checkpoint_resume_test_file)
    criterion_5 = report_mod.criterion_5_checkpoint_resume(evidence)
    criterion_6 = report_mod.criterion_6_pyramid_tiers(pyramid_tier_by_bundle)

    report = report_mod.build_report(
        criterion_1=criterion_1,
        criterion_2=criterion_2,
        criterion_3=criterion_3,
        criterion_4=criterion_4,
        criterion_5=criterion_5,
        criterion_6=criterion_6,
    )
    _write_json(args.output, report.as_dict())
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s2-gate", description="S2 exit-gate measurement: decode | score | report"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    decode_parser = subparsers.add_parser("decode", help="phase A: cohort -> sealed artefact")
    decode_parser.add_argument("--cohort-manifest", type=Path, required=True)
    decode_parser.add_argument("--feature-cache-dir", type=Path, required=True)
    decode_parser.add_argument("--exact-text-cache-dir", type=Path, required=True)
    decode_parser.add_argument(
        "--visual-input-policy", type=Path, default=None,
        help="fresh S4 (DEC-056 A): decode a page-image (7.0.0) --exact-text-cache-dir by "
        "running the frozen ViT per page under this policy; the checkpoint's "
        "visual_feature_tap picks DeepStack or pooler-only. Needs --page-image-store; "
        "--feature-cache-dir is then ignored.",
    )
    decode_parser.add_argument(
        "--page-image-store", type=Path, default=None,
        help="the object store the page-image cache's PNG references resolve in",
    )
    decode_parser.add_argument("--run-manifest-bindings", type=Path, required=True)
    decode_parser.add_argument("--component-lock", type=Path, required=True)
    decode_parser.add_argument("--allowlist", type=Path, required=True)
    decode_parser.add_argument("--tokenizer-artifact-set-hash", required=True)
    decode_parser.add_argument("--arm", required=True, choices=sorted(sealed.VALID_ARMS))
    decode_parser.add_argument("--checkpoint", type=Path, default=None)
    decode_parser.add_argument(
        "--expect-s1-run-manifest-hash",
        default=None,
        help="required with --arm trained: refuses a checkpoint from a foreign run",
    )
    decode_parser.add_argument("--device", default="cuda")
    decode_parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
        help="no default: a decode's own cost ceiling is not this CLI's decision to make",
    )
    decode_parser.add_argument(
        "--typed-decode",
        action="store_true",
        help="read the typed property heads at every decoded node anchor (AST Builder "
        "T1) and seal the predictions; off by default, which keeps the decode "
        "byte-identical to before the flag existed",
    )
    decode_parser.add_argument(
        "--region-decode",
        action="store_true",
        help="read NodeVisualAligner at decoded node anchors and seal closed region "
        "predictions; off by default",
    )
    decode_parser.add_argument(
        "--loop-guard",
        choices=("none", "v1", "v2"),
        default="none",
        help="DEC-031 structural loop guard (v1) or DEC-036 loop guard v2 (v1 plus "
        "text-run, periodic-unit and close-recovery rules; standalone profile only); "
        "'none' (default) keeps today's byte-identical decode",
    )
    decode_parser.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=None,
        help="W0-11: feed the KV-cache prefix pass in chunks of this many tokens instead "
        "of one call (bounds peak VRAM on a small GPU); unset (default) keeps today's "
        "byte-identical single-call prefix",
    )
    decode_parser.add_argument(
        "--page-start-policy",
        choices=("none", "page-start-story.v0"),
        default="none",
        help="DEC-061 (PROPOSED): narrow the first section's header/footer story decisions to a "
        "per-page story pattern; 'none' (default) keeps today's byte-identical decode",
    )
    decode_parser.add_argument(
        "--story-length-cap",
        choices=("none", "header-length-cap.v0"),
        default="none",
        help="header-length-cap.v0 (PROPOSED): close a header/footer still open after the "
        "policy's cap of accepted tokens; 'none' (default) keeps today's byte-identical decode",
    )
    decode_parser.add_argument(
        "--story-retry",
        choices=("none", "story-retry.v0"),
        default="none",
        help="story-retry.v0 (PROPOSED): decode a page again with an over-long header/footer kind "
        "banned; 'none' (default) keeps today's byte-identical decode",
    )
    decode_parser.add_argument(
        "--story-retry-policy",
        type=Path,
        default=None,
        help="policy file overriding the packaged story_retry_policy.json (measurement arms); "
        "requires --story-retry story-retry.v0",
    )
    decode_parser.add_argument(
        "--story-length-cap-policy",
        type=Path,
        default=None,
        help="policy file overriding the packaged header_length_cap_policy.json (measurement "
        "arms); requires --story-length-cap header-length-cap.v0",
    )
    decode_parser.add_argument(
        "--page-start-pattern-source",
        choices=("entry", "head"),
        default="entry",
        help="where the story pattern comes from: the feature entry (diagnostic oracle wrapper) "
        "or a trained page-start head (--page-start-head)",
    )
    decode_parser.add_argument(
        "--page-start-head",
        type=Path,
        default=None,
        help="page-start head bundle directory; required with --page-start-pattern-source head",
    )
    decode_parser.add_argument(
        "--reference-partial-heads",
        action="store_true",
        help="RFC005-ERRATA-005 T2: load --checkpoint allowing missing/unexpected keys "
        "only under the typed/table_plan/node_visual_aligner/expert_adapter_dense_heads "
        "prefixes (a checkpoint trained on an older model shape, e.g. "
        "s4-discovery-v14-r4/checkpoint.pt) -- refused together with --typed-decode/"
        "--region-decode, and only meaningful with --arm trained",
    )
    decode_parser.add_argument("--output", type=Path, required=True)

    score_parser = subparsers.add_parser("score", help="phase B: one sealed artefact -> a score")
    score_parser.add_argument("--sealed-artefact", type=Path, required=True)
    score_parser.add_argument("--exact-text-cache-dir", type=Path, required=True)
    score_parser.add_argument("--cohort-manifest", type=Path, required=True)
    score_parser.add_argument("--allowlist", type=Path, required=True)
    score_parser.add_argument("--tokenizer-artifact-set-hash", required=True)
    score_parser.add_argument("--output", type=Path, required=True)

    report_parser = subparsers.add_parser("report", help="the six S2 exit-gate criteria")
    report_parser.add_argument("--trained-sealed-artefact", type=Path, required=True)
    report_parser.add_argument("--baseline-sealed-artefact", type=Path, required=True)
    report_parser.add_argument("--exact-text-cache-dir", type=Path, required=True)
    report_parser.add_argument("--cohort-manifest", type=Path, required=True)
    report_parser.add_argument("--allowlist", type=Path, required=True)
    report_parser.add_argument("--tokenizer-artifact-set-hash", required=True)
    report_parser.add_argument("--pyramid-tier-map", type=Path, default=None)
    report_parser.add_argument(
        "--checkpoint-resume-test-file", default=report_mod.CHECKPOINT_RESUME_TEST_FILE
    )
    report_parser.add_argument("--output", type=Path, required=True)

    return parser


def _validate_decode_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.arm == sealed.DecodeArm.TRAINED and args.checkpoint is None:
        parser.error("--checkpoint is required when --arm trained")
    if args.arm == sealed.DecodeArm.FROZEN_INIT_BASELINE and args.checkpoint is not None:
        parser.error("--checkpoint must not be supplied when --arm frozen_init_baseline")
    if args.arm == sealed.DecodeArm.TRAINED and args.expect_s1_run_manifest_hash is None:
        parser.error("--expect-s1-run-manifest-hash is required when --arm trained")
    if (
        args.arm == sealed.DecodeArm.FROZEN_INIT_BASELINE
        and args.expect_s1_run_manifest_hash is not None
    ):
        parser.error(
            "--expect-s1-run-manifest-hash must not be supplied when "
            "--arm frozen_init_baseline"
        )
    if args.reference_partial_heads and (args.typed_decode or args.region_decode):
        parser.error(
            "--reference-partial-heads refuses to combine with --typed-decode/"
            "--region-decode -- those heads are untrained on a partial-heads-loaded "
            "checkpoint"
        )
    if args.reference_partial_heads and args.arm != sealed.DecodeArm.TRAINED:
        parser.error("--reference-partial-heads requires --arm trained")
    if args.page_start_policy == "none" and (
        args.page_start_pattern_source != "entry" or args.page_start_head is not None
    ):
        parser.error(
            "--page-start-pattern-source/--page-start-head require "
            "--page-start-policy page-start-story.v0"
        )
    if (
        getattr(args, "story_length_cap", "none") == "none"
        and getattr(args, "story_length_cap_policy", None) is not None
    ):
        parser.error("--story-length-cap-policy requires --story-length-cap header-length-cap.v0")
    if (
        getattr(args, "story_retry", "none") == "none"
        and getattr(args, "story_retry_policy", None) is not None
    ):
        parser.error("--story-retry-policy requires --story-retry story-retry.v0")
    if args.page_start_pattern_source == "head" and args.page_start_head is None:
        parser.error("--page-start-head is required when --page-start-pattern-source head")
    if args.page_start_pattern_source == "entry" and args.page_start_head is not None:
        parser.error(
            "--page-start-head must not be supplied when --page-start-pattern-source entry"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "decode":
        _validate_decode_args(parser, args)
        return run_decode(args)
    if args.command == "score":
        return run_score(args)
    if args.command == "report":
        return run_report(args)
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover - argparse prevents this
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
