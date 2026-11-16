"""Inputs the S1 runner CLI needs but has never built: a trained checkpoint,
and the wiring for gates 4, 5, 7 and 14 (plan `spec-runner-inputs-c5b`).

`s1_runner.py` declares the fields `S1RunnerContext` needs for these gates
but the CLI never fills them -- `--checkpoint` exists only to refuse. This
module is the missing bridge: it never runs a training loop, never adds a
new hash domain, and never changes a frozen artefact. It imports
`s1_runner` and `s1_train`; neither of those imports this module (`s1_train`
imports `s1_runner` only lazily inside `main`), so there is no cycle.

**Checkpoint loading goes exclusively through `s1_train.load_checkpoint`**
(all five of its expectations, verbatim) -- this module adds only the
refusals `load_checkpoint` cannot itself make: that a run manifest and a
checkpoint ancestry base were even supplied (never the
`expect_s1_run_manifest_hash=None` diagnostic path), that the checkpoint's
declared feature source really is `"cache"` (a null/shuffled-arm checkpoint
is not S1 evidence), and that the checkpoint's own recorded optimizer
algorithm matches the one the caller is about to resume with.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from unflatten.dataset.items import build_item
from unflatten.dataset.samples import iter_page_samples
from unflatten.model import checkpoint_ancestry as ca
from unflatten.model import composite_vocab as cv
from unflatten.model import exact_text_cache as etc
from unflatten.model import freeze_s1_run_manifest as fsrm
from unflatten.model import run_binding as rb
from unflatten.model import s1_runner, s1_train
from unflatten.model import training_materializer as tm
from unflatten.model.candidate_model import QwenDenseCandidateModel
from unflatten.model.capability_profile import CapabilityProfile
from unflatten.model.encode_fragment import encode_fragment_with_origins
from unflatten.model.prediction_ir import RawDocumentHypothesis
from unflatten.model.text_authority import (
    TextAuthorityAlignmentError,
    build_text_span_authorities,
)


class S1RunnerInputsRefused(RuntimeError):
    """This module refuses to proceed: missing/mismatched evidence for a
    checkpoint load, or a gate-input build that would otherwise silently
    fabricate or misalign evidence. Always a refusal, never a downgrade to
    a weaker claim."""


# ---------------------------------------------------------------------------
# Task 1: checkpoint loading and weight snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluatedCheckpoint:
    """What `load_trained_checkpoint` proved about the checkpoint it
    loaded -- `checkpoint_sha256`/`checkpoint_id` are always DERIVED from
    the file actually on disk and the ancestry the load verified, never
    the caller's declaration."""

    path: Path
    checkpoint_sha256: str
    ancestry_hash: str | None
    checkpoint_id: str
    step: int
    feature_source: str
    s1_run_manifest_hash: str | None
    optimizer_name: str
    restored_rng: bool


def load_trained_checkpoint(
    *,
    model: QwenDenseCandidateModel,
    optimizer: torch.optim.Optimizer,
    path: Path | str,
    run_manifest: Mapping[str, Any] | None,
    ancestry: Mapping[str, Any] | None,
    cache_identity: Mapping[str, Any] | None,
    feature_source: str,
    declared_optimizer: str,
) -> EvaluatedCheckpoint:
    """Load `path` onto `model`/`optimizer` and return what was proved.

    Refuses BEFORE ever touching `path` if `run_manifest` or `ancestry` is
    `None` (this function never constructs the `expect_s1_run_manifest_hash
    =None` diagnostic path -- that is `s1_train`'s own supported case for a
    diagnostic resume, not a claim gates 1-3/6/12/13 may run against as S1
    evidence) or if `feature_source != "cache"` (a null/shuffled-arm
    checkpoint is not S1 evidence, whatever gates it happens to pass).

    The actual load is `s1_train.load_checkpoint`, called with all five of
    its expectations, verbatim -- this function adds no alternate load
    path. After it succeeds, the checkpoint's OWN recorded optimizer name
    is checked against `declared_optimizer`: an AdamW checkpoint resumed
    into a caller who believes it is SGD would run gates 1-3 with the wrong
    algorithm's real dynamics while every artefact still named the
    declared one.

    `checkpoint_sha256`/`ancestry_hash`/`checkpoint_id` are derived from
    the loaded state and the file's own bytes -- never accepted as a
    caller-supplied value.
    """
    if run_manifest is None or ancestry is None:
        raise S1RunnerInputsRefused(
            "load_trained_checkpoint refuses without both run_manifest and "
            "ancestry -- constructing the expect_s1_run_manifest_hash=None "
            "diagnostic path here would let a checkpoint that carries no "
            "gradient-bearing claim stand in as S1 evidence"
        )
    if feature_source != s1_train.FEATURE_SOURCE_CACHE:
        raise S1RunnerInputsRefused(
            f"load_trained_checkpoint refuses feature_source={feature_source!r}: "
            "only a real-cache checkpoint is S1 evidence; a null/shuffled-arm "
            "checkpoint proves something about the control arm, not the run"
        )
    path = Path(path)
    progress, state = s1_train.load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expect_cache_identity=cache_identity,
        expect_feature_source=feature_source,
        expect_s1_run_manifest_hash=run_manifest["s1_run_manifest_hash"],
        expect_base_ancestry_hash=ancestry["ancestry_hash"],
        restore_rng=True,
    )
    if state.get("optimizer") != declared_optimizer:
        raise S1RunnerInputsRefused(
            f"checkpoint was written by optimizer {state.get('optimizer')!r}, "
            f"not the declared {declared_optimizer!r} -- resuming it into the "
            "wrong algorithm would run gates 1-3 with dynamics the run never "
            "claimed"
        )
    checkpoint_sha256 = ca.file_sha256(path)
    ancestry_hash = state.get("ancestry_hash")
    checkpoint_id = ca.compute_checkpoint_id(ancestry_hash, checkpoint_sha256)
    return EvaluatedCheckpoint(
        path=path,
        checkpoint_sha256=checkpoint_sha256,
        ancestry_hash=ancestry_hash,
        checkpoint_id=checkpoint_id,
        step=progress.step,
        feature_source=feature_source,
        s1_run_manifest_hash=run_manifest["s1_run_manifest_hash"],
        optimizer_name=declared_optimizer,
        restored_rng=True,
    )


def snapshot_trainable_state(model: QwenDenseCandidateModel) -> dict[str, torch.Tensor]:
    """Detached CPU clones of `model`'s trainable parameters only -- never
    the frozen Qwen weights `composite_head` holds as `persistent=False`
    buffers (`qwen_embedding_weight`/`qwen_lm_head_weight`), which
    `named_parameters()` never reports at all.

    Gates 1-3 run `ctx.gate123_steps` real optimizer steps against
    `ctx.model` as part of their own proof (design finding 1); gate 13's
    free-running decode must see the checkpoint's OWN trained weights, not
    "checkpoint plus five probe steps" -- this is the snapshot taken before
    the probe, `restore_trainable_state` undoes it before gate 13 runs.
    """
    return {name: p.detach().clone().cpu() for name, p in model.named_parameters()}


def restore_trainable_state(
    model: QwenDenseCandidateModel, snapshot: Mapping[str, torch.Tensor]
) -> None:
    """Restore exactly the parameters `snapshot_trainable_state` captured.

    Strict both ways: a name in `model.named_parameters()` missing from
    `snapshot`, or a `snapshot` name the model no longer has, is refused
    rather than silently restoring a partial state -- the same guarantee
    `nn.Module.load_state_dict(strict=True)` gives for a full state dict,
    applied here to the trainable-only subset this module snapshots."""
    current = dict(model.named_parameters())
    missing = sorted(set(current) - set(snapshot))
    extra = sorted(set(snapshot) - set(current))
    if missing or extra:
        raise S1RunnerInputsRefused(
            f"restore_trainable_state: snapshot does not match model parameters "
            f"(missing={missing}, extra={extra})"
        )
    with torch.no_grad():
        for name, p in current.items():
            p.copy_(snapshot[name].to(device=p.device, dtype=p.dtype))


def evaluated_checkpoint_payload(ec: EvaluatedCheckpoint) -> dict[str, Any]:
    """The JSON-serializable projection of `ec`, for the runner payload's
    additive `evaluated_checkpoint` key."""
    return {
        "path": str(ec.path),
        "checkpoint_sha256": ec.checkpoint_sha256,
        "ancestry_hash": ec.ancestry_hash,
        "checkpoint_id": ec.checkpoint_id,
        "step": ec.step,
        "feature_source": ec.feature_source,
        "s1_run_manifest_hash": ec.s1_run_manifest_hash,
        "optimizer_name": ec.optimizer_name,
        "restored_rng": ec.restored_rng,
    }


# ---------------------------------------------------------------------------
# Task 2: gate 14 inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Gate14Inputs:
    component_lock: dict[str, Any]
    architecture_manifest: dict[str, Any]
    run_manifest: dict[str, Any]
    cache_identity: dict[str, Any]
    allowlist: dict[str, Any]
    #: The run dir's cohort pin (`load_cache_expectation`); `None` for a
    #: diagnostic manifest, which pins none.
    expectation: etc.CacheExpectation | None = None


def load_cache_expectation(
    run_dir: Path | str, run_manifest: Mapping[str, Any]
) -> etc.CacheExpectation | None:
    """The cohort pin (DEC-MP-4) a gradient-bearing run manifest was frozen
    with, read from `run_dir` and checked against the manifest's
    `exact_text_cache_expectation_hash` before it is trusted.

    `None` for a non-gradient-bearing manifest, which pins no cohort.
    A gradient-bearing manifest without its pin -- or with one that does
    not hash to what the manifest binds -- is refused: without it
    `ExactTextTrainingDataset` refuses a gradient-bearing read and gate 14
    cannot present the cohort `RunManifestBindingVerifier` requires
    (2026-09-14: the S1 canary's gate run hit exactly that).
    """
    if not run_manifest.get("gradient_bearing"):
        return None
    path = Path(run_dir) / fsrm.EXACT_TEXT_CACHE_EXPECTATION_FILENAME
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise S1RunnerInputsRefused(
            f"{path} does not exist -- a gradient-bearing run manifest is read "
            "only under the cohort expectation it was frozen with (DEC-MP-4)"
        ) from exc
    actual = rb.compute_exact_text_cache_expectation_hash(payload)
    pinned = run_manifest.get("exact_text_cache_expectation_hash")
    if actual != pinned:
        raise S1RunnerInputsRefused(
            f"{path} hashes to {actual!r}, but the run manifest pins "
            f"exact_text_cache_expectation_hash {pinned!r}"
        )
    return etc.CacheExpectation.from_payload(payload)


def build_gate_14_inputs(
    *,
    exact_text_cache_dir: Path | str,
    run_dir: Path | str,
    component_lock_path: Path | str,
    architecture_manifest_path: Path | str,
) -> Gate14Inputs:
    """Read gate 14's four artefacts off disk.

    `cache_identity` comes from `exact_text_cache.cache_manifest(...)
    ["identity"]` -- the EXACT-TEXT cache's own identity, never
    `FeatureCacheReader.identity` (the candidate feature cache's). The two
    are easy to confuse (both are "the cache's identity") but only the
    exact-text one carries `component_lock_hash`; a `FeatureCacheReader`
    identity handed here is refused by name, not merely "field missing",
    because that mistake is the likely one a caller makes.
    """
    manifest = etc.cache_manifest(exact_text_cache_dir)
    cache_identity = manifest.get("identity")
    if not isinstance(cache_identity, dict) or "component_lock_hash" not in cache_identity:
        raise S1RunnerInputsRefused(
            "gate 14 needs the EXACT-TEXT cache's own identity "
            "(exact_text_cache.cache_manifest(...)['identity']), which carries "
            "component_lock_hash -- this identity does not. The likely mistake "
            "is passing FeatureCacheReader.identity (the candidate feature "
            "cache's identity) instead; that cache has no Component Lock "
            "binding at all"
        )
    run_dir = Path(run_dir)
    run_manifest = json.loads((run_dir / fsrm.S1_RUN_MANIFEST_FILENAME).read_text())
    allowlist = json.loads((run_dir / fsrm.QWEN_TEXT_ALLOWLIST_FILENAME).read_text())
    component_lock = json.loads(Path(component_lock_path).read_text())
    architecture_manifest = json.loads(Path(architecture_manifest_path).read_text())
    return Gate14Inputs(
        component_lock=component_lock,
        architecture_manifest=architecture_manifest,
        run_manifest=run_manifest,
        cache_identity=cache_identity,
        allowlist=allowlist,
        expectation=load_cache_expectation(run_dir, run_manifest),
    )


def with_gate_inputs(
    ctx: s1_runner.S1RunnerContext,
    *,
    gate14: Gate14Inputs | None = None,
    gate45: Gate45Inputs | None = None,
) -> s1_runner.S1RunnerContext:
    """`ctx` with gate 14/4-5 fields filled in, via `dataclasses.replace` --
    `build_context_from_loaded`'s own contract (every OTHER field it sets)
    is left untouched."""
    updates: dict[str, Any] = {}
    if gate14 is not None:
        updates.update(
            component_lock=gate14.component_lock,
            architecture_manifest=gate14.architecture_manifest,
            run_manifest=gate14.run_manifest,
            cache_identity=gate14.cache_identity,
            allowlist=gate14.allowlist,
            cache_expectation=gate14.expectation,
        )
    if gate45 is not None:
        updates.update(
            logits=gate45.logits,
            target_ids=gate45.target_ids,
            expected_missing_mask=gate45.expected_missing_mask,
        )
    if not updates:
        return ctx
    return replace(ctx, **updates)


# ---------------------------------------------------------------------------
# Task 3: gates 4/5 inputs
# ---------------------------------------------------------------------------


#: Gate 5's `blocked` reason when the run was not asked to read the object
#: store -- named, like `s1_runner.GATE_10_STORE_ABSENT_REASON`, so a
#: reader sees a deliberate scope choice rather than an oversight.
GATE_5_STORE_ABSENT_REASON = (
    "gate 5's expected_missing_mask needs the release/store's own "
    "target bundle and fragment to recompute withdrawal independently; "
    "this run was not given --text-corpus-from-store, so gate 5 stays blocked"
)


@dataclass(frozen=True, slots=True)
class Gate45Inputs:
    logits: torch.Tensor
    target_ids: torch.Tensor
    expected_missing_mask: torch.Tensor | None
    row_index: int
    document_bundle_id: str
    page_index: int
    gate5_blocked_reason: str | None


def resolve_source_authority(
    *,
    release_dir: Path | str,
    store_dir: Path | str,
    document_bundle_id: str,
    page_index: int,
    split: str,
    task_kind: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, tuple[tm.TextSpanAuthority, ...]]:
    """`(targets, fragment, text_spans)` for one real page, built the same
    way `build_exact_text_cache.row_for_page` builds them: find the page's
    own `PageSample` (`iter_page_samples`), read its artefacts
    (`build_item`), then reshape its real text spans
    (`build_text_span_authorities`, `ownership_projection_hash=
    targets["target_bundle_hash"]` -- the same convention `row_for_page`
    uses, no new hash invented here).

    `split`/`task_kind` select which cohort enumeration `iter_page_samples`
    walks; callers read them off the SAME `ExactTextCacheIdentity` the row
    being checked came from, so this reproduces the cohort the row was
    drawn from rather than picking a second, possibly disagreeing one.
    """
    match = None
    for sample in iter_page_samples(release_dir, store_dir, split, task_kind):
        if sample.document_bundle_id == document_bundle_id and sample.page_index == page_index:
            match = sample
            break
    if match is None:
        raise S1RunnerInputsRefused(
            f"no page sample for {document_bundle_id!r} page {page_index} in "
            f"split={split!r} task_kind={task_kind!r} -- cannot resolve source "
            "authority for gate 5"
        )
    item = build_item(store_dir, match)
    targets = item.targets
    if not targets or not targets.get("structural"):
        raise S1RunnerInputsRefused(
            f"{document_bundle_id!r} page {page_index}: no target bundle/"
            "structural events -- cannot resolve source authority for gate 5"
        )
    target_bundle_hash = targets["target_bundle_hash"]
    try:
        text_spans = build_text_span_authorities(
            document_bundle_id=document_bundle_id,
            target_bundle_hash=target_bundle_hash,
            text=targets.get("text") or {"spans": []},
            fragment=item.fragment or {},
            ownership_projection_hash=target_bundle_hash,
        )
    except TextAuthorityAlignmentError as exc:
        raise S1RunnerInputsRefused(
            f"{document_bundle_id!r} page {page_index}: text authority could "
            f"not be aligned: {exc}"
        ) from exc
    return targets, item.fragment, text_spans


def gate_5_from_inputs(gate45: Gate45Inputs | None) -> s1_runner.GateResult | None:
    """Gate 5, given `gate45`'s own inputs: if `gate45.gate5_blocked_reason`
    names why `expected_missing_mask` could not be recomputed (store
    absent, target-bundle-hash mismatch, structural-origin count
    mismatch, numel mismatch), that reason becomes gate 5's own `blocked`
    detail -- the same way `gate_10_text_roundtrip`'s `corpus_source`
    names gate 10's absent-corpus reason -- rather than
    `s1_runner.gate_5_missing_target_no_default`'s generic "no
    target_ids/expected_missing_mask supplied", which says nothing about
    WHY.

    Returns `None` when `gate45` is `None` (no `--checkpoint` at all, gate
    5 stays exactly as `evaluate_all_gates` reports it) or when it built a
    real `expected_missing_mask` (gate 5's ordinary pass/fail evaluation
    wins, unchanged)."""
    if gate45 is None or gate45.expected_missing_mask is not None:
        return None
    return s1_runner._blocked(5, "missing_target_no_default", gate45.gate5_blocked_reason)


def row_materialized_view(row: etc.ExactTextCacheRow) -> tm.MaterializedSequence:
    """`row`'s fields projected onto `MaterializedSequence`, for ALIGNMENT
    only (`s1_runner.composite_expected_missing_mask`'s own contract: it
    reads `loss_units`/`effective_loss_mask.shape` from this, never any of
    the effective-mask verdict fields). `ExactTextCacheRow` carries every
    field `MaterializedSequence` needs verbatim -- this is a projection,
    not a recomputation."""
    return tm.MaterializedSequence(
        packed_token_ids=row.packed_token_ids,
        packed_input_ids=row.packed_input_ids,
        effective_loss_mask=row.effective_loss_mask,
        loss_family_ids=row.loss_family_ids,
        controller_mode_ids=row.controller_mode_ids,
        structural_allowed_row_ptr=row.structural_allowed_row_ptr,
        structural_allowed_local_ids=row.structural_allowed_local_ids,
        loss_unit_index=row.loss_unit_index,
        loss_units=row.loss_units,
    )


def _gate5_mask_or_reason(
    row: etc.ExactTextCacheRow,
    *,
    target_ids: torch.Tensor,
    release_dir: Path | str,
    store_dir: Path | str | None,
    dataset: s1_train.ExactTextTrainingDataset,
    document_bundle_id: str,
    page_index: int,
) -> tuple[torch.Tensor | None, str | None]:
    """`(expected_missing_mask, gate5_blocked_reason)` -- exactly one of
    the two is non-`None`. Extracted from `build_gate_4_5_inputs`
    (2026-09-05 review suggestion) purely to give the store branch a name;
    every check and message is unchanged."""
    if store_dir is None:
        return None, GATE_5_STORE_ABSENT_REASON
    try:
        targets, fragment, text_spans = resolve_source_authority(
            release_dir=release_dir,
            store_dir=store_dir,
            document_bundle_id=document_bundle_id,
            page_index=page_index,
            split=dataset.identity.split,
            task_kind=dataset.identity.task_kind,
        )
    except S1RunnerInputsRefused as exc:
        return None, str(exc)
    if targets["target_bundle_hash"] != row.authority_target_hash:
        return None, (
            f"{document_bundle_id!r} page {page_index}: store target_bundle_hash "
            f"{targets['target_bundle_hash']!r} does not match the cache row's "
            f"authority_target_hash {row.authority_target_hash!r} -- the store and "
            "the cache row describe different documents"
        )
    structural = targets["structural"]
    if fragment is not None:
        _tokens, structural_origins = encode_fragment_with_origins(fragment, structural)
    else:
        _tokens, structural_origins = tm.document_level_tokens_with_origins(structural)
    structural_unit_count = sum(
        1 for unit in row.loss_units if unit.loss_family == "L_structure"
    )
    if len(structural_origins) != structural_unit_count:
        return None, (
            f"{document_bundle_id!r} page {page_index}: structural-origin "
            f"count ({len(structural_origins)}) does not match the row's "
            f"L_structure loss-unit count ({structural_unit_count}) -- "
            "recomputing withdrawal would index past the origin list, "
            "so gate 5 blocks instead of risking an IndexError"
        )
    view = row_materialized_view(row)
    candidate_mask = s1_runner.composite_expected_missing_mask(
        view, structural=structural, fragment=fragment, text_spans=text_spans
    )
    if candidate_mask.numel() != target_ids.numel():
        return None, (
            f"{document_bundle_id!r} page {page_index}: "
            f"expected_missing_mask has {candidate_mask.numel()} "
            f"position(s), target_ids has {target_ids.numel()} -- "
            "gate 5 needs 1:1 alignment"
        )
    return candidate_mask, None


def build_gate_4_5_inputs(
    *,
    model: QwenDenseCandidateModel,
    dataset: s1_train.ExactTextTrainingDataset,
    ranges: cv.PackedRanges,
    image_token_id: int,
    device: torch.device | str,
    release_dir: Path | str,
    store_dir: Path | str | None = None,
    max_rows: int = 4,
) -> Gate45Inputs:
    """Build gate 4/5's inputs from `dataset`'s first `max_rows` rows.

    Batch size 1 on purpose: gate 5's `expected_missing_mask` is built
    per-page (`resolve_source_authority`, one page's own targets/fragment),
    and a batch size of 1 is what makes its flattened positions align 1:1
    with `target_ids`'s flattened positions -- a padded multi-row batch
    would not.

    Scans for the first row carrying at least one `False` in
    `effective_loss_mask`; if none of the scanned rows has one, the LAST
    scanned row is used anyway and `gate_4_masks_zero_loss_contribution`
    itself reports `blocked` ("target_ids carries no IGNORE_INDEX position
    -- fixture proves nothing") -- that existing check is reused rather
    than duplicated here.

    `target_ids` applies `masked_fill(~loss_mask, IGNORE_INDEX)` here,
    because `collate_training_batch` deliberately does not (its own
    `S1TrainingBatch.target_ids` is unmasked `packed_token_ids`).

    `store_dir=None` leaves gate 5 blocked with `GATE_5_STORE_ABSENT_REASON`
    and `expected_missing_mask=None` -- gate 4/6-9/etc. still evidence over
    the same batch.
    """
    if len(dataset) == 0:
        raise S1RunnerInputsRefused("build_gate_4_5_inputs: dataset is empty")
    scan_rows = min(max_rows, len(dataset))
    chosen = scan_rows - 1
    for i in range(scan_rows):
        if not bool(dataset[i].effective_loss_mask.all()):
            chosen = i
            break

    sample = dataset[chosen]
    batch = s1_train.collate_training_batch(
        [sample], ranges=ranges, image_token_id=image_token_id, device=device
    )
    model.eval()
    with torch.no_grad():
        logits = s1_train.scored_logits(model, batch)
    target_ids = batch.target_ids.masked_fill(~batch.loss_mask, s1_runner.IGNORE_INDEX)

    row = dataset.reader[chosen]
    document_bundle_id = row.document_bundle_id
    page_index = row.page_index

    expected_missing_mask, gate5_blocked_reason = _gate5_mask_or_reason(
        row,
        target_ids=target_ids,
        release_dir=release_dir,
        store_dir=store_dir,
        dataset=dataset,
        document_bundle_id=document_bundle_id,
        page_index=page_index,
    )

    return Gate45Inputs(
        logits=logits,
        target_ids=target_ids,
        expected_missing_mask=expected_missing_mask,
        row_index=chosen,
        document_bundle_id=document_bundle_id,
        page_index=page_index,
        gate5_blocked_reason=gate5_blocked_reason,
    )


# ---------------------------------------------------------------------------
# Task 4: gate 7 inputs
# ---------------------------------------------------------------------------


class HypothesisCapture:
    """Callable wrapping `decode_free_running` (or any callee with the
    same contract): stores the `RawDocumentHypothesis` it returns on
    `self.hypothesis`, and RE-RAISES `DecodeMaxStepsExceeded`/
    `DecodeVocabularyExhausted` untouched.

    This exists so gate 13's own decode -- the free-running canary RFC005
    §14.4 requires -- is the SAME decode gate 7 replays, rather than gate 7
    running a second, separate decode. It must never become a swallowing
    wrapper: a caller that catches a typed decode failure here and returns
    a partial hypothesis instead would defeat gate 13's own mutation proof
    (`s1_runner.gate_13_free_running_no_silent_truncation`'s docstring).

    `__qualname__` is set explicitly (plain instances have none) so gate
    13's wiring evidence (`f"{decode_fn.__module__}.{decode_fn.__qualname__}"`)
    names this wrapper -- honest, since it IS the callee gate 13 invokes.

    Also stores a raised `DecodeMaxStepsExceeded`/`DecodeVocabularyExhausted`
    on `self.exception` before RE-RAISING it untouched (bug `max-steps-
    discards-the-evidence`) -- gate 13's own `except` clause only reads the
    exception's type name into `evidence["typed_failure"]`, so without this
    the `accepted_tokens`/`node_states` the exception carries would still be
    unreachable one level up in `decode.py`. Storing and re-raising is not
    swallowing: the exception still propagates exactly as before, nothing
    about control flow changes.
    """

    def __init__(self, decode_fn: Any = None) -> None:
        from unflatten.model.free_running_decode import (
            DecodeMaxStepsExceeded,
            DecodeVocabularyExhausted,
            decode_free_running,
        )

        self._decode_fn = decode_fn if decode_fn is not None else decode_free_running
        self._typed_failures = (DecodeMaxStepsExceeded, DecodeVocabularyExhausted)
        self.__module__ = type(self).__module__
        self.__qualname__ = type(self).__qualname__
        self.hypothesis: RawDocumentHypothesis | None = None
        self.exception: BaseException | None = None

    def __call__(self, **kwargs: Any) -> RawDocumentHypothesis:
        try:
            result = self._decode_fn(**kwargs)
        except self._typed_failures as exc:
            self.exception = exc
            raise
        self.hypothesis = result
        return result


def build_gate_7_inputs(
    hypothesis: RawDocumentHypothesis | None,
    *,
    vocabulary: cv.CompositeVocabulary,
    capability_profile: CapabilityProfile,
) -> tuple[Any, str, tuple[cv.TokenRef, ...]] | None:
    """`(controller, profile_id, tokens)` for
    `gate_7_grammar_constrained_valid_tree`, or `None` if `hypothesis` is
    absent or did not terminate.

    `hypothesis.accepted_tokens` are PACKED INTS (`prediction_ir.
    RawDocumentHypothesis.accepted_tokens: tuple[int, ...]`), not
    `TokenRef`s -- `vocabulary.unpack` recovers the typed tokens gate 7's
    replay needs.

    `capability_profile` is required keyword-only with no default. A caller
    that resolved the real bound profile passes it; S1 tests pass
    `PRIMARY_S1_CAPABILITY_PROFILE` explicitly. Gate 7 must replay under
    the SAME profile gate 13 decoded under, never a profile hardcoded
    independently of it (2026-09-10 review item 2)."""
    if hypothesis is None or not hypothesis.terminated:
        return None
    profile = capability_profile
    controller = s1_runner.build_controller_for_profile(vocabulary, profile)
    tokens = tuple(vocabulary.unpack(i) for i in hypothesis.accepted_tokens)
    return controller, profile.profile_id, tokens


def gate_7_from_hypothesis(
    hypothesis: RawDocumentHypothesis | None,
    *,
    vocabulary: cv.CompositeVocabulary,
    gate13_result: s1_runner.GateResult | None = None,
    capability_profile: CapabilityProfile,
) -> s1_runner.GateResult:
    """Gate 7, fed from a free-running hypothesis (gate 13's own decode,
    captured via `HypothesisCapture`) rather than a gold/teacher-forced
    stream -- a gold stream would only prove the grammar accepts the
    correct answer, never that the model's own free-running decode stays
    inside it (design finding, plan module docstring).

    `hypothesis is None` covers TWO different situations that used to
    share one (misleading) message: gate 13 genuinely never ran / its
    hypothesis was never captured (`gate13_result is None`, e.g.
    `--skip-real-decode`), and gate 13 DID run but its decode raised a
    typed failure (`DecodeMaxStepsExceeded`/`DecodeVocabularyExhausted`,
    `gate13_result.evidence["branch"] == "typed_failure"`) -- in the
    latter case gate 13 ran and even reported `pass`, but there is
    nothing for gate 7 to replay, which "did not run, or was not
    captured" flatly denies. `gate13_result` is optional so every
    existing caller that never had a `GateResult` to hand in keeps
    working, falling back to the "not run / not captured" phrasing.

    `capability_profile` is forwarded to `build_gate_7_inputs` verbatim --
    see its docstring for why a caller that resolved the real bound profile
    must pass it explicitly."""
    if hypothesis is None:
        if gate13_result is not None and gate13_result.evidence.get("branch") == "typed_failure":
            return s1_runner._blocked(
                7, "grammar_constrained_valid_tree",
                "no hypothesis supplied -- gate 13's decode raised a typed "
                "failure and did not terminate, so there is nothing to replay",
            )
        return s1_runner._blocked(
            7, "grammar_constrained_valid_tree",
            "no hypothesis supplied -- gate 13 did not run, or was not captured",
        )
    if not hypothesis.terminated:
        return s1_runner._blocked(
            7, "grammar_constrained_valid_tree",
            "hypothesis did not terminate -- not a valid free-running canary",
        )
    built = build_gate_7_inputs(
        hypothesis, vocabulary=vocabulary, capability_profile=capability_profile
    )
    assert built is not None  # terminated=True already checked above
    controller, profile_id, tokens = built
    return s1_runner.gate_7_grammar_constrained_valid_tree(controller, profile_id, tokens)
