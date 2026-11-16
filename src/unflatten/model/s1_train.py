"""The RFC005 S1 exact-text training loop: `ExactTextCacheReader` ->
`QwenDenseCandidateModel` -> masked, per-family loss -> optimizer step ->
resumable checkpoint.

This is the loop that did not exist. `train.py` reads the STRUCTURAL
feature cache and trains `StructureDecoder` (a `discovery`-class decoder
experiment); `trainability_smoke.py` says in its own docstring that it is
not an S1 gradient update; `s1_runner.py` evaluates gates and takes
gradient steps only inside a probe. None of them reads the exact-text
cache, which is the only artifact carrying real document text as targets.

**Building it authorizes nothing.** Every gradient-bearing entry point
here is fail-closed behind :func:`require_gradient_admissible`, which
demands a frozen `s1_run_manifest` verified by
`run_binding.RunManifestBindingVerifier` against the cache's own identity.
The cache that exists today (`cache-exact-text-s1-placeholder-bindings`)
carries six identity hashes with no producer, so no manifest can be frozen
truthfully against it and nothing here can run gradient-bearing over it.
That refusal is the point, not an obstacle to route around: a diagnostic
run (`gradient_bearing=False`) is how the loop is developed.

Two things this module composes that did NOT compose as written, each
resolved by reusing an existing mechanism rather than inventing one.
(A third -- `ExactTextCacheRow` carrying no `grid_thw`, which
`inject_visual` needs as the backbone's native mRoPE index and which
cannot be derived from the visual token count -- was closed at the source
on 2026-09-05 by cache version 1.1.0, which puts `grid_thw` on the row,
required and refused rather than defaulted. This module read it through a
second cache for one afternoon; it now reads the row. **A 1.0.0 cache is
refused, so `cache-exact-text-s1-placeholder-bindings` needs a rebuild
before this loop can read it.**)

1. **`packed_input_ids` carries no visual placeholder run.** The
   materializer emits the composite target stream only, and
   `inject_visual` enforces exactly one `image_token_id` placeholder per
   valid visual row. :func:`collate_training_batch` prepends that run.

2. **Batched pages have unequal visual lengths** (450/858/875 in the real
   cache), so the composite region would start at a different index per
   row. The visual run is therefore LEFT-padded to the batch maximum --
   the same convention Qwen's own batched generation uses, and the only
   one that keeps padding out of the middle of a sequence where
   `get_rope_index` would have to interpret it. Visual FEATURES stay
   right-padded with `visual_valid_lengths`, which is `VisualMemory`'s own
   convention; the two are independent because the scatter consumes valid
   visual rows row-major regardless of where the placeholders sit.

**Masking, the standing constraint.** A missing label is never a numeric
zero. The mask is not re-derived here: `effective_loss_mask` is the
five-way RFC003 conjunction `loss_unit.EffectiveLossMask` already
computed and hash-bound at materialization. This module only projects it
onto targets as `IGNORE_INDEX` and proves (in `test_s1_train.py`, by
deletion, the way gate 4 does) that a masked position contributes exactly
zero. Padded steps get `effective_loss_mask=False` -- note that this is
the OPPOSITE convention from `train.py`'s grammar mask, which pads
permissively so a padded row of `-inf` logits cannot softmax to NaN. A
loss mask and a grammar mask must not be padded alike; here nothing
consults a grammar, so `False` is right and permissive would be a leak.

**Per-family, never one number** (RFC005 R4-8). `family_loss_report`
returns `L_structure` and `L_text` separately, and its payload names its
regime `teacher_forced` with `free_running: None` -- an explicit "not
measured", not a zero and not a scalar standing in for both.

**No λ weighting is applied.** `stage_policy` and `conditioning_policy`
have no schema and no producer; the λ_f weights are the owner's content
decision (RFC005 §26.4) and cannot be derived. The backward loss is the
unweighted mean over scored tokens, and the report says so.

**Checkpointing is built for a runtime that vanishes.** S1 trains on a
Colab A100/L4, which disconnects without warning and takes `/content`
with it, so: the destination is always the caller's (`--out`, or the
notebook's mounted Drive path) and never derived from the working
directory; `train_steps` writes mid-run at `checkpoint_every` steps
through a caller-supplied callable, because saving only at the end loses
the whole run on a drop; each write goes to a `.tmp` sibling and
`os.replace`s into place, so a disconnect mid-write cannot truncate the
previous good checkpoint; and a checkpoint resumes from anywhere,
including another machine or another mount point.

**What a mid-run resume reproduces:** the model, the optimizer state, the
step counter, the global RNG state, AND the residual page order
(`TrainingProgress.pending_order`). The last is the one that looks
optional and is not -- page order comes from `torch.randperm` drawn one
permutation at a time and consumed `batch_size` indices per step, so a
run interrupted off a permutation boundary holds a residual list the RNG
state cannot reconstruct. With all five restored, resuming at step k and
running n more steps produces bit-identical parameters to running k+n
steps continuously (proved in `test_s1_train.py` at an odd step count,
which is the only place the residual list is non-empty). What it does NOT
reproduce is CUDA-side nondeterminism: this equality is asserted on CPU.

**The number to beat is not chance** (`--random-features`). The composite
grammar and the text prior are a strong baseline on their own -- the
structural family in particular leaves few legal continuations -- so a
falling `L_structure` is not by itself evidence that the decoder is
reading the page. The null arm replaces the cached features with noise and
is the thing a real run is scored against. For that comparison to mean
anything the two arms must differ only in INFORMATION, so the noise draws
from a per-page generator (never the global RNG, which `train_steps` reads
for its page order) and carries the page's own mean and sd (nothing
standardises features here, and raw standard-normal noise would arrive
quieter than real features and lose on loudness). Every log line and every
checkpoint records `feature_source`; a null-model checkpoint that reads as
a real one is the artefact this project keeps paying for.

**Memory.** Gradient checkpointing already exists across the frozen
decode call (`candidate_model.encode_hidden`); none is added here. One
real page is ~875 visual tokens plus up to ~2.6k composite steps, and the
teacher-forced forward alone measures ~10.26 GiB with 36 layers of
activations held for backward -- an RTX 4070 (11.59 GiB) cannot run this.
Target A100/L4. :func:`scored_logits` does buy back the largest avoidable
term: it runs the composite head over the SCORED positions only, not over
the visual placeholder run, which for a 875-visual/582-composite page is
60 % of a `(steps, 152k)` logit tensor.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from unflatten.model import checkpoint_ancestry as ca
from unflatten.model import composite_vocab as cv
from unflatten.model import run_binding as rb
from unflatten.model.candidate_model import QwenDenseCandidateModel, TeacherForcedBatch
from unflatten.model.capability_profile import (
    CapabilityProfileHashUnresolved,
    resolve_capability_profile_by_hash,
)
from unflatten.model.exact_text_cache import (
    DEFAULT_CACHE_SHARDS,
    CacheExpectation,
    ExactTextCacheIdentity,
    ExactTextCacheIncompatible,
    ExactTextCacheReader,
)
from unflatten.model.targets import IGNORE_INDEX
from unflatten.model.trainability_smoke import run_parameter_movement_check
from unflatten.model.training_materializer import LOSS_FAMILY_IDS

#: `loss_family_ids` is a uint8 tensor; this is the same mapping the
#: materializer packed it with, read from there rather than restated.
LOSS_FAMILY_NAME: dict[int, str] = {v: k for k, v in LOSS_FAMILY_IDS.items()}

#: What a run read as visual features. Every log line and every checkpoint
#: carries one of these, because after the fact the null model's numbers
#: and a real run's are otherwise indistinguishable.
FEATURE_SOURCE_CACHE = "cache"
FEATURE_SOURCE_NULL = "random_noise"
#: Real features from a DIFFERENT page. The sharper control of the two:
#: noise differs from real features in distribution as well as in content,
#: so "real lost to noise" has two explanations and cannot separate them.
#: Shuffled features ARE real features -- same cache, same distribution,
#: same channel correlations -- and the only thing broken is the binding
#: between this page's pixels and this page's text. A difference between
#: `cache` and `shuffled_pages` therefore has exactly one explanation.
FEATURE_SOURCE_SHUFFLED = "shuffled_pages"


class S1TrainingRefused(RuntimeError):
    """The run may not proceed: a cache/grid pairing that cannot be
    proved, a checkpoint bound to a different cache identity, or a
    gradient-bearing run without an admissible frozen `s1_run_manifest`.
    Always a refusal, never a downgrade to a weaker mode."""


class NonFiniteStepError(RuntimeError):
    """A training step produced a value that is not a finite number.

    Same policy as `train.py`: skip the step and stop. An optimizer step
    on a non-finite loss poisons every parameter it touches, and
    continuing past it turns a broken run into a run that merely looks
    bad.
    """


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExactTextTrainingSample:
    """One page, exactly as the exact-text cache row holds it."""

    document_bundle_id: str
    page_index: int
    features: torch.Tensor | None  # float16 [V, D]; None on a page-image row
    grid_thw: torch.Tensor  # int64 [3]
    packed_input_ids: torch.Tensor  # int64 [T]
    packed_token_ids: torch.Tensor  # int64 [T]
    effective_loss_mask: torch.Tensor  # bool [T]
    loss_family_ids: torch.Tensor  # uint8 [T]
    #: DEC-056 on-the-fly visual path, all `None` on a feature row (v15b):
    #: the row's PNG reference, then -- filled by `onfly_visual` in the data
    #: worker -- the processor's `pixel_values` and the merged visual-token
    #: count they produce. `collate_training_batch` defers such a batch's
    #: features to `onfly_visual.OnTheFlyVisualEncoder.materialize`.
    page_image: dict[str, Any] | None = None
    pixel_values: torch.Tensor | None = None
    visual_tokens: int | None = None


class ExactTextTrainingDataset(Dataset):
    """An exact-text cache as a torch dataset.

    `gradient_bearing` has no default here for the same reason
    `ExactTextCacheReader` gives it none: "forgot to pass it" must not
    silently become the weaker admission check.

    **One cache, not two** (2026-09-05). An earlier version of this class
    took a second `feature_cache_dir` and paired every row with the
    candidate feature cache by `(document_bundle_id, page_index)`, because
    `ExactTextCacheRow` carried no `grid_thw` and `inject_visual` needs one
    -- a workaround with a real cost, since it made an S1 run depend on
    shipping both caches. Cache version 1.1.0 puts `grid_thw` on the row
    itself, required and refused rather than defaulted, so the pairing is
    gone and the grid is read straight from the row. A 1.0.0 cache is
    refused on read by `ExactTextCacheReader`, which is correct and means
    the existing `cache-exact-text-s1-placeholder-bindings` needs a
    rebuild before this loop can read it.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        gradient_bearing: bool,
        expectation: CacheExpectation | None = None,
        cache_shards: int = DEFAULT_CACHE_SHARDS,
        randomize: bool = False,
        noise_seed: int = 0,
        shuffle: bool = False,
        shuffle_seed: int = 0,
    ) -> None:
        if gradient_bearing and expectation is None:
            raise S1TrainingRefused(
                "a gradient-bearing S1 run requires the frozen cohort "
                "expectation (DEC-MP-4): without the pinned expected page key "
                "set E and approved exclusions X, a cohort-restricted cache "
                "trains as though it were the full one and nothing in the "
                "cache itself can tell the two apart"
            )
        if randomize and shuffle:
            raise ValueError(
                "randomize and shuffle are two different controls and a run is "
                "one arm or the other; enabling both would produce noise and "
                "report it as shuffled features"
            )
        self.reader = ExactTextCacheReader(
            cache_dir,
            gradient_bearing=gradient_bearing,
            cache_shards=cache_shards,
            expectation=expectation,
        )
        #: The pinned cohort this dataset is a view over, or `None` for a
        #: diagnostic read of whatever the cache holds. When set, `len` and
        #: every index address `E \ X` and a page in `C \ E` is
        #: unreachable -- not filtered later, never indexed at all.
        self.expectation = expectation
        self.identity: ExactTextCacheIdentity = self.reader.identity
        self.gradient_bearing = gradient_bearing
        #: Replace features with noise to measure the grammar-only floor.
        self.randomize = randomize
        self.noise_seed = noise_seed
        #: Take each page's features from a different page.
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self._partners: tuple[int, ...] | None = None

    @property
    def feature_source(self) -> str:
        """Which arm this is, in the vocabulary the checkpoint records."""
        if self.randomize:
            return FEATURE_SOURCE_NULL
        if self.shuffle:
            return FEATURE_SOURCE_SHUFFLED
        return FEATURE_SOURCE_CACHE

    def _partner_of(self, index: int) -> int:
        """Which page lends its features to `index`.

        A **derangement**: no page may receive its own features, or that
        row silently becomes a real-features row inside the control arm
        and the control is diluted by however many fixed points a random
        permutation happens to have (about one, in expectation, at any
        length).

        Built from its own generator so it consumes no draw from the
        global RNG that `train_steps` uses for page order -- the same
        property `_noise` documents -- and it is a pure function of
        `shuffle_seed` and the dataset length, so a resumed run rebuilds
        the identical pairing.
        """
        if self._partners is None:
            total = len(self)
            if total < 2:
                raise ValueError(
                    f"cannot shuffle features across {total} page(s): a "
                    "derangement needs at least two"
                )
            generator = torch.Generator()
            generator.manual_seed(self.shuffle_seed)
            partners = torch.randperm(total, generator=generator).tolist()
            for position in range(total):
                if partners[position] == position:
                    other = (position + 1) % total
                    partners[position], partners[other] = (
                        partners[other],
                        partners[position],
                    )
            fixed = [i for i, j in enumerate(partners) if i == j]
            if fixed:
                raise RuntimeError(
                    f"shuffle left {len(fixed)} page(s) paired with themselves "
                    f"(first: {fixed[0]}); this arm would be part real features"
                )
            self._partners = tuple(partners)
        return self._partners[index]

    def _noise(self, features: torch.Tensor, index: int) -> torch.Tensor:
        """Noise standing in for one page's features.

        Two properties the obvious `torch.randn_like(features)` does not
        have, each of which would make the two arms differ in more than
        information:

        - **its own generator.** `randn_like` draws from the GLOBAL RNG,
          which `train_steps` also reads for `torch.randperm`; the null
          arm would consume draws the real arm does not and walk a
          different page order. Seeded per index, the noise is also a pure
          function of the page, so a resumed null run is the same run.
        - **the page's own first two moments.** Nothing standardises
          features in this loop (unlike `train.py`, where both arms are
          conditioned by `feature_channel_stats`), so raw standard-normal
          noise would arrive at the model far quieter than real features
          -- measured at 14x in the structural loop -- and the floor would
          be beaten on loudness rather than on information.
        """
        generator = torch.Generator(device=features.device)
        generator.manual_seed(self.noise_seed + index)
        noise = torch.randn(
            features.shape,
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        )
        real = features.float()
        return (noise.float() * real.std() + real.mean()).to(features.dtype)

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> ExactTextTrainingSample:
        row = self.reader[index]
        features, grid_thw = row.features, row.grid_thw
        if getattr(row, "page_image", None) is not None and (self.randomize or self.shuffle):
            raise S1TrainingRefused(
                "randomize/shuffle replace cached features; a page-image row has "
                "none (DEC-056 on-the-fly visual path) -- these control arms are "
                "not available on it"
            )
        if self.randomize:
            features = self._noise(row.features, index)
        elif self.shuffle:
            # `grid_thw` travels WITH the features, from the same partner
            # row. It is the backbone's own mRoPE index for that image and
            # its token count is the feature length: taking the features
            # from one page and the grid from another would hand
            # `inject_visual` a placeholder run of the wrong length. Not a
            # subtle mismatch -- a different arm than the one named.
            partner = self.reader[self._partner_of(index)]
            features, grid_thw = partner.features, partner.grid_thw
        return ExactTextTrainingSample(
            document_bundle_id=row.document_bundle_id,
            page_index=row.page_index,
            features=features,
            grid_thw=grid_thw,
            packed_input_ids=row.packed_input_ids,
            packed_token_ids=row.packed_token_ids,
            effective_loss_mask=row.effective_loss_mask,
            loss_family_ids=row.loss_family_ids,
            page_image=getattr(row, "page_image", None),
        )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class S1TrainingBatch:
    """A `TeacherForcedBatch` plus the supervision aligned to its SCORED
    positions only -- `target_ids`/`loss_mask`/`loss_family_ids` are
    `(batch, composite_steps)`, excluding the visual placeholder run and
    its left padding, which `scored_logits` also excludes."""

    model_batch: TeacherForcedBatch
    target_ids: torch.Tensor  # int64 [B, T]
    loss_mask: torch.Tensor  # bool [B, T]
    loss_family_ids: torch.Tensor  # uint8 [B, T]
    visual_steps: int
    #: DEC-056: `((pixel_values, grid_thw), ...)` per row of a page-image
    #: batch whose visual features are not computed yet -- `model_batch.
    #: visual_features` is then a `(B, V_max, 0)` placeholder and the batch
    #: must go through `onfly_visual.OnTheFlyVisualEncoder.materialize`
    #: before the model sees it. `None` on every feature batch.
    page_pixels: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None


def collate_training_batch(
    items: Iterable[ExactTextTrainingSample],
    *,
    ranges: cv.PackedRanges,
    image_token_id: int,
    device: torch.device | str = "cpu",
    visual_features_dtype: torch.dtype = torch.float32,
) -> S1TrainingBatch:
    """Pad a batch of pages and prepend each one's visual placeholder run.

    Layout per row, over `V_max + T_max` composite positions::

        [ pad * (V_max - v_i) ][ placeholder * v_i ][ composite (t_i) ][ pad ]
          attention 0            attention 1          attention 1       0

    The visual run is LEFT-padded so the composite region starts at the
    same index in every row; the composite tail is right-padded, and those
    steps carry `loss_mask=False`, so nothing scores them.

    Visual FEATURES are right-padded with `visual_valid_lengths`, which is
    `VisualMemory`'s own convention -- independent of the placeholder
    layout, because the scatter consumes valid visual rows row-major.
    """
    samples = list(items)
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch = len(samples)
    pending = samples[0].features is None
    if any((s.features is None) != pending for s in samples):
        raise ValueError("cannot collate feature rows and page-image rows into one batch")
    if pending and any(s.pixel_values is None or s.visual_tokens is None for s in samples):
        raise ValueError(
            "a page-image sample reached collate without pixel_values/visual_tokens -- "
            "it must come through onfly_visual's dataset"
        )
    counts = [
        int(s.visual_tokens) if pending else int(s.features.shape[0])  # type: ignore[union-attr]
        for s in samples
    ]
    visual_dim = 0 if pending else int(samples[0].features.shape[1])  # type: ignore[union-attr]
    v_max = max(counts)
    t_max = max(int(s.packed_token_ids.shape[0]) for s in samples)
    steps = v_max + t_max

    placeholder_id = cv.pack(cv.QwenRef(native_id=image_token_id), ranges)
    # Padding id: a structural id, so `embed_packed`'s range routing has a
    # legal branch for it. It is never scored and never a placeholder.
    pad_id = cv.pack(cv.StructuralRef(local_id=0), ranges)

    features = torch.zeros(batch, v_max, visual_dim, dtype=visual_features_dtype)
    valid_lengths = torch.zeros(batch, dtype=torch.long)
    composite_input_ids = torch.full((batch, steps), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch, steps, dtype=torch.long)
    target_ids = torch.zeros(batch, t_max, dtype=torch.long)
    loss_mask = torch.zeros(batch, t_max, dtype=torch.bool)
    family_ids = torch.zeros(batch, t_max, dtype=torch.uint8)
    grid_thw = torch.stack([s.grid_thw.to(torch.long) for s in samples])

    for row, sample in enumerate(samples):
        v = counts[row]
        t = int(sample.packed_token_ids.shape[0])
        if not pending:
            features[row, :v] = sample.features.to(visual_features_dtype)  # type: ignore[union-attr]
        valid_lengths[row] = v
        composite_input_ids[row, v_max - v : v_max] = placeholder_id
        composite_input_ids[row, v_max : v_max + t] = sample.packed_input_ids
        attention_mask[row, v_max - v :] = 1
        attention_mask[row, v_max + t :] = 0
        target_ids[row, :t] = sample.packed_token_ids
        loss_mask[row, :t] = sample.effective_loss_mask
        family_ids[row, :t] = sample.loss_family_ids

    model_batch = TeacherForcedBatch(
        visual_features=features.to(device),
        composite_input_ids=composite_input_ids.to(device),
        composite_attention_mask=attention_mask.to(device),
        image_grid_thw=grid_thw.to(device),
        visual_valid_lengths=valid_lengths.to(device),
    )
    return S1TrainingBatch(
        model_batch=model_batch,
        target_ids=target_ids.to(device),
        loss_mask=loss_mask.to(device),
        loss_family_ids=family_ids.to(device),
        visual_steps=v_max,
        page_pixels=(
            tuple((s.pixel_values, s.grid_thw) for s in samples) if pending else None  # type: ignore[misc]
        ),
    )


def scored_logits(model: QwenDenseCandidateModel, batch: S1TrainingBatch) -> torch.Tensor:
    """`(batch, composite_steps, ranges.total_count)` logits over the
    SCORED positions only.

    `model.forward` runs the composite head over every position including
    the visual placeholder run, whose logits are then thrown away. For a
    875-visual/582-composite page that is 60 % of a `(steps, ~152k)`
    tensor materialized for nothing, on a path whose memory is the binding
    constraint. This slices `encode_hidden`'s output first and runs the
    head on the remainder -- the same two-step composition
    `free_running_decode.py` uses, not a second forward path.
    """
    hidden, prefix_len = model.encode_hidden(batch.model_batch)
    composite_hidden = hidden[:, prefix_len + batch.visual_steps :, :]
    return model.composite_head(composite_hidden)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FamilyLoss:
    family: str
    loss_sum: float
    token_count: int

    @property
    def mean(self) -> float | None:
        """`None`, never `0.0`, when the family scored nothing in this
        batch -- a family with no supervised token has no loss, and a zero
        would read as a measurement."""
        return self.loss_sum / self.token_count if self.token_count else None


@dataclass(frozen=True, slots=True)
class TeacherForcedLossReport:
    """Per-family teacher-forced numbers for one batch or epoch.

    RFC005 R4-8: teacher-forced and free-running must be reported
    separately, and one number must not stand for both. This report names
    its regime and carries an explicit `free_running: None` -- "not
    measured", which is a different claim from a value.
    """

    families: tuple[FamilyLoss, ...]
    loss_sum: float
    scored_tokens: int

    @property
    def mean(self) -> float:
        return self.loss_sum / self.scored_tokens

    def payload(self) -> dict[str, Any]:
        return {
            "regime": "teacher_forced",
            "free_running": None,
            "scored_tokens": self.scored_tokens,
            "mean": self.mean,
            "families": {
                f.family: {"mean": f.mean, "token_count": f.token_count}
                for f in self.families
            },
            "stage_policy": None,
            "lambda_weighting": "none -- unweighted mean over scored tokens; the "
            "λ_f weights are the owner's content decision (RFC005 §26.4) and "
            "no stage_policy artifact exists to read them from",
        }


def masked_cross_entropy(
    logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """Per-position cross entropy, exactly `0.0` at every masked position.

    Masked targets are replaced by `IGNORE_INDEX` BEFORE the loss, so the
    masked position's target value is never read at all -- not read and
    then multiplied by zero, which would still fail on an out-of-range id
    and would still let a NaN target poison the sum.
    """
    scored = torch.where(loss_mask, target_ids, torch.full_like(target_ids, IGNORE_INDEX))
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        scored.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )


def family_loss_report(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_family_ids: torch.Tensor,
) -> TeacherForcedLossReport:
    """Per-family sums over the scored positions of one batch."""
    per_position = masked_cross_entropy(logits, target_ids, loss_mask).detach()
    flat_mask = loss_mask.reshape(-1)
    flat_families = loss_family_ids.reshape(-1)
    scored = int(flat_mask.sum())
    if scored == 0:
        raise ValueError(
            "no scored positions in this batch -- a zeroed loss reads as a "
            "measurement and there is nothing here to measure"
        )
    families = []
    for family_id, name in sorted(LOSS_FAMILY_NAME.items()):
        selected = flat_mask & (flat_families == family_id)
        families.append(
            FamilyLoss(
                family=name,
                loss_sum=float(per_position[selected].sum()),
                token_count=int(selected.sum()),
            )
        )
    return TeacherForcedLossReport(
        families=tuple(families),
        loss_sum=float(per_position.sum()),
        scored_tokens=scored,
    )


def batch_loss(logits: torch.Tensor, batch: S1TrainingBatch) -> torch.Tensor:
    """The scalar the optimizer minimizes: the mean over scored tokens.

    Unweighted on purpose -- see `TeacherForcedLossReport.payload`'s
    `lambda_weighting`.
    """
    per_position = masked_cross_entropy(logits, batch.target_ids, batch.loss_mask)
    scored = int(batch.loss_mask.sum())
    if scored == 0:
        raise ValueError("no scored positions in this batch; nothing to optimize")
    return per_position.sum() / scored


# ---------------------------------------------------------------------------
# Preflight: does the untrained model even read the visual signal?
# ---------------------------------------------------------------------------


def _mask_visual_run(batch: S1TrainingBatch) -> S1TrainingBatch:
    """Zero the composite attention mask over the visual placeholder run
    only (`[:, :batch.visual_steps]`), leaving the composite tail (its own
    padding included) exactly as collated.

    Both `TeacherForcedBatch` and `S1TrainingBatch` are frozen dataclasses,
    so this reconstructs them via `dataclasses.replace` rather than
    mutating in place.

    **Conservative, not a clean ablation.** `document_context_encoder`
    reads `memory_batch.memory` directly, before `inject_visual`'s scatter
    and before this mask is ever consulted -- masking the composite
    attention closes the path INTO the composite sequence, not the DCE
    prefix's own view of the visual memory. A real signal can therefore
    still leak through the DCE prefix even under this mask, which makes
    `passes=True` a conservative (harder to satisfy) result, not a lenient
    one; it does not make `passes=False` any less of a refusal.
    """
    model_batch = batch.model_batch
    masked_attention = model_batch.composite_attention_mask.clone()
    masked_attention[:, : batch.visual_steps] = 0
    masked_model_batch = replace(model_batch, composite_attention_mask=masked_attention)
    return replace(batch, model_batch=masked_model_batch)


def preflight_native_signal(
    model: QwenDenseCandidateModel,
    samples: Iterable[ExactTextTrainingSample],
    *,
    image_token_id: int,
    device: torch.device | str = "cpu",
    visual_encoder: Any | None = None,
) -> dict[str, Any]:
    """Does THIS untrained model read the cached visual features at all?

    For each sample, one row at a time under `torch.no_grad()`: `L_text`
    (a) with the real visual run attended to, and (b) with it masked out
    (`_mask_visual_run`, the conservative ablation described there). A
    model reading the page should score (a) lower than (b); `passes` is
    exactly that comparison over the mean across rows.

    This exists because a random `VisualMemory` projection at step 0 was
    measured to do the opposite -- see `visual_memory.py`'s module
    docstring and `scratchpad/audit-train/res96.json` -- and nothing
    before this caught it: seeds differed only in sign. This is a
    diagnostic over the CURRENT model weights, not a property of the
    architecture; it must be re-run whenever the weights change (e.g.
    after loading a checkpoint).

    Rows whose sample carries zero `L_text` tokens are skipped -- a page
    with no scored text says nothing about whether the model reads the
    image, and a bare `float("nan")`/`0.0` mean from an empty family would
    read as a measurement. Skipped rows are still named, in `skipped`, as
    `{document_bundle_id, page_index, reason: "no_text_tokens"}` -- silently
    dropping them would make a report over an all-skipped sample set look
    identical to one that measured nothing at all. When EVERY row is
    skipped there is no basis to compare real vs. masked, so `passes` is
    `False`, `mean_real`/`mean_masked` are `None`, and `reason` names why.

    `scored_logits` is looked up by its MODULE-LEVEL name at call time
    (not bound to a local), so a test can monkeypatch
    `unflatten.model.s1_train.scored_logits` and this function picks up
    the replacement.

    `visual_encoder` (DEC-056 on-the-fly path): the samples carry page
    pixels and each collated row is `visual_encoder.materialize`d before
    the two forwards; `None` reads cached features exactly as before.
    """
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    l_text_real: list[float] = []
    l_text_masked: list[float] = []
    with torch.no_grad():
        for sample in samples:
            batch = collate_training_batch(
                [sample],
                ranges=model.ranges,
                image_token_id=image_token_id,
                device=device,
            )
            if visual_encoder is not None:
                batch = visual_encoder.materialize(batch)
            text_family_id = LOSS_FAMILY_IDS["L_text"]
            text_tokens = int(
                (batch.loss_mask & (batch.loss_family_ids == text_family_id)).sum()
            )
            if text_tokens == 0:
                skipped.append(
                    {
                        "document_bundle_id": sample.document_bundle_id,
                        "page_index": sample.page_index,
                        "reason": "no_text_tokens",
                    }
                )
                continue
            masked_batch = _mask_visual_run(batch)

            real_logits = scored_logits(model, batch)
            real_report = family_loss_report(
                real_logits, batch.target_ids, batch.loss_mask, batch.loss_family_ids
            )
            masked_logits = scored_logits(model, masked_batch)
            masked_report = family_loss_report(
                masked_logits,
                masked_batch.target_ids,
                masked_batch.loss_mask,
                masked_batch.loss_family_ids,
            )
            real_mean = next(
                f.mean for f in real_report.families if f.family == "L_text"
            )
            masked_mean = next(
                f.mean for f in masked_report.families if f.family == "L_text"
            )
            rows.append(
                {
                    "document_bundle_id": sample.document_bundle_id,
                    "page_index": sample.page_index,
                    "l_text_real": real_mean,
                    "l_text_masked": masked_mean,
                }
            )
            l_text_real.append(real_mean)
            l_text_masked.append(masked_mean)

    mean_real = sum(l_text_real) / len(l_text_real) if l_text_real else None
    mean_masked = sum(l_text_masked) / len(l_text_masked) if l_text_masked else None
    passes = mean_real is not None and mean_masked is not None and mean_real < mean_masked
    rows_real_better = sum(
        1 for real, masked in zip(l_text_real, l_text_masked) if real < masked
    )
    margin_nats = (
        mean_masked - mean_real if mean_real is not None and mean_masked is not None else None
    )
    reason = "no row carried text tokens" if not rows else None
    return {
        "rows": rows,
        "skipped": skipped,
        "l_text_real": l_text_real,
        "l_text_masked": l_text_masked,
        "mean_real": mean_real,
        "mean_masked": mean_masked,
        "margin_nats": margin_nats,
        "rows_real_better": rows_real_better,
        "passes": passes,
        "reason": reason,
        "null_definition": (
            "masked = composite_attention_mask zeroed over the visual "
            "placeholder run only (_mask_visual_run); the "
            "document_context_encoder prefix still sees the unmasked "
            "visual memory directly, so this is a conservative ablation "
            "-- passes=True is a harder bar to clear than a clean one, "
            "not a weaker one."
        ),
    }


def require_preflight_passes(
    report: Mapping[str, Any], *, gradient_bearing: bool
) -> None:
    """Pure gate over `preflight_native_signal`'s report: a gradient-bearing
    run that did not pass is refused, message carrying both means so the
    refusal is legible without re-reading the JSON line printed alongside
    it. A diagnostic run (`gradient_bearing=False`) is never refused here --
    it only logs."""
    if not gradient_bearing:
        return
    if report.get("reason"):
        raise S1TrainingRefused(
            f"preflight refused a gradient-bearing run: {report['reason']} -- "
            "there is no basis to judge whether the untrained model reads "
            "its own visual features"
        )
    if not report["passes"]:
        raise S1TrainingRefused(
            "preflight refused a gradient-bearing run: masking the visual "
            f"run did not raise L_text (mean_real={report['mean_real']!r}, "
            f"mean_masked={report['mean_masked']!r}) -- the untrained model "
            "does not demonstrably read its own visual features; see "
            "visual_memory.py's module docstring"
        )


def preflight_row_indices(total: int, rows: int) -> list[int]:
    """Which dataset indices `--preflight-rows N` reads, in dataset order.

    `rows <= 0` is "off" -- an explicit empty selection, not a default
    that silently runs anyway with some other count."""
    if rows <= 0:
        return []
    return list(range(min(rows, total)))


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepEvidence:
    """One `train_step`'s per-family loss report plus gradient-norm
    evidence, both read BEFORE `clip_grad_norm_` rescales anything -- a
    post-clip norm reads back as `grad_clip` itself, which is a property of
    the caller's argument, not a measurement of what the backward pass
    produced."""

    report: TeacherForcedLossReport
    grad_norm_total: float
    grad_norm_groups: dict[str, float]


def _grad_norm(grads: list[torch.Tensor]) -> float:
    """L2 norm of `grads` treated as one concatenated vector -- the same
    reduction `torch.nn.utils.clip_grad_norm_` uses, so a group norm and
    the total are comparable numbers."""
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2))


def train_step(
    model: QwenDenseCandidateModel,
    batch: S1TrainingBatch,
    optimizer: torch.optim.Optimizer,
    *,
    grad_clip: float = 1.0,
) -> StepEvidence:
    """One forward/backward/step over one batch; returns its per-family
    loss numbers plus pre-clip gradient-norm evidence."""
    optimizer.zero_grad(set_to_none=True)
    logits = scored_logits(model, batch)
    loss = batch_loss(logits, batch)
    if not bool(torch.isfinite(loss)):
        raise NonFiniteStepError(f"loss is {float(loss.detach())}")
    report = family_loss_report(
        logits, batch.target_ids, batch.loss_mask, batch.loss_family_ids
    )
    loss.backward()

    # Grouped by the FIRST dotted component of the parameter's own name
    # (`visual_memory`, `document_context_encoder`, `dense_adapter`,
    # `composite_head`) -- computed here, before clipping touches any
    # `.grad` tensor.
    grouped_grads: dict[str, list[torch.Tensor]] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grouped_grads.setdefault(name.split(".")[0], []).append(param.grad)
    grad_norm_groups = {group: _grad_norm(grads) for group, grads in grouped_grads.items()}

    if grad_clip:
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        )
        grad_norm_total = float(norm)
        if not bool(torch.isfinite(norm)):
            optimizer.zero_grad(set_to_none=True)
            raise NonFiniteStepError(f"gradient norm is {float(norm)}")
    else:
        grad_norm_total = _grad_norm(
            [p.grad for p in model.parameters() if p.grad is not None]
        )
    optimizer.step()
    return StepEvidence(
        report=report, grad_norm_total=grad_norm_total, grad_norm_groups=grad_norm_groups
    )


@dataclass(frozen=True, slots=True)
class S1MovementEvidence:
    """`trainability_smoke.run_parameter_movement_check`'s report, plus the
    named tensors that never moved -- "K of N moved" is not the finding;
    WHICH ones did not is."""

    steps: int
    moved_count: int
    total_count: int
    never_moved: tuple[str, ...]


def parameter_movement(
    model: QwenDenseCandidateModel,
    batch: S1TrainingBatch,
    optimizer: torch.optim.Optimizer,
    *,
    steps: int = 3,
) -> S1MovementEvidence:
    """Per-tensor movement evidence over `steps` real steps of THIS loss,
    against the caller's own optimizer at its own lr.

    Delegates the comparison to `trainability_smoke`, which owns it, so
    there is one before/after rule in the repository, not two.
    """

    def step_fn() -> torch.Tensor:
        return batch_loss(scored_logits(model, batch), batch)

    report = run_parameter_movement_check(model, optimizer, step_fn, steps=steps)
    return S1MovementEvidence(
        steps=report.steps,
        moved_count=report.moved_count,
        total_count=report.total_count,
        never_moved=tuple(sorted(t.name for t in report.tensors if not t.moved)),
    )


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """Where a run is, to the granularity a resume must reproduce.

    `pending_order` is the load-bearing half and the easy one to omit.
    Page order comes from `torch.randperm(len(dataset))` drawn one
    permutation at a time and consumed `batch_size` indices per step, so a
    run interrupted anywhere other than a permutation boundary is holding
    a residual list that the RNG state alone cannot reconstruct --
    restoring the RNG and re-drawing would hand the resumed run a fresh
    permutation and silently re-show pages the interrupted run had already
    consumed. A fixture whose dataset size divides evenly by `batch_size`
    never lands mid-permutation and would report this as fine
    (`test_s1_train.py` uses an odd step count on purpose).
    """

    step: int
    pending_order: tuple[int, ...] = ()


def train_steps(
    model: QwenDenseCandidateModel,
    dataset: ExactTextTrainingDataset,
    optimizer: torch.optim.Optimizer,
    *,
    ranges: cv.PackedRanges,
    image_token_id: int,
    steps: int,
    progress: TrainingProgress | None = None,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
    grad_clip: float = 1.0,
    log_every: int = 0,
    checkpoint_every: int = 0,
    checkpoint: Callable[[TrainingProgress], None] | None = None,
) -> TrainingProgress:
    """Run `steps` optimizer steps over shuffled pages; returns the new
    :class:`TrainingProgress`.

    Shuffling reads `torch.randperm` from the global RNG, which is why
    `save_checkpoint` stores the RNG state -- without it a resumed run
    sees a different page order and is not the same run. The RNG is not
    sufficient on its own; see `TrainingProgress.pending_order`.

    **`checkpoint` is a callable taking one `TrainingProgress`, invoked
    every `checkpoint_every` steps, and the destination is entirely the
    caller's** (`main` wires it to :func:`save_checkpoint` at `--out`).
    This loop never chooses a path. The target for S1 is a Colab A100/L4,
    whose runtime disconnects without warning and takes `/content` with
    it: a mid-run checkpoint that lands on local disk is a checkpoint that
    does not exist after a drop, so the destination has to be a mounted
    Drive path the caller supplies, and mid-run cadence has to be
    available at all -- saving only at the end loses the whole run.
    """
    total = len(dataset)
    if total == 0:
        raise ValueError("cannot train over an empty dataset")
    if checkpoint_every and checkpoint is None:
        raise ValueError(
            f"checkpoint_every={checkpoint_every} with no `checkpoint` callable: "
            "a cadence that writes nowhere is not a cadence"
        )
    progress = progress or TrainingProgress(step=0)
    step = progress.step
    order: list[int] = list(progress.pending_order)
    for index in range(steps):
        while len(order) < batch_size:
            order.extend(torch.randperm(total).tolist())
        chosen, order = order[:batch_size], order[batch_size:]
        batch = collate_training_batch(
            [dataset[i] for i in chosen],
            ranges=ranges,
            image_token_id=image_token_id,
            device=device,
        )
        evidence = train_step(model, batch, optimizer, grad_clip=grad_clip)
        step += 1
        if log_every and (index + 1) % log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "feature_source": dataset.feature_source,
                        # Read off the optimizer itself rather than taken
                        # from a caller-supplied label: a label can
                        # disagree with the object, and then the log says
                        # what someone meant to run instead of what ran.
                        "optimizer": type(optimizer).__name__,
                        "lr": optimizer.param_groups[0]["lr"],
                        **evidence.report.payload(),
                        "grad_norm_total": evidence.grad_norm_total,
                        "grad_norm_groups": evidence.grad_norm_groups,
                    }
                ),
                flush=True,
            )
        # `checkpoint is not None` rather than leaning on the up-front
        # raise: that raise ties safety here to a CORRELATION between two
        # separate parameters, which is exactly the shape a type checker
        # cannot follow and a future edit can break silently.
        if checkpoint is not None and checkpoint_every and step % checkpoint_every == 0:
            checkpoint(TrainingProgress(step=step, pending_order=tuple(order)))
    return TrainingProgress(step=step, pending_order=tuple(order))


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _checkpoint_rng_states() -> dict[str, Any]:
    """Global generators only; never initialize an unused CUDA runtime."""
    numpy = sys.modules.get("numpy")
    numpy_state = None
    if numpy is not None:
        algorithm, keys, position, has_gauss, cached_gaussian = numpy.random.get_state()
        # NumPy arrays are not accepted by torch.load(weights_only=True).
        numpy_state = {
            "algorithm": algorithm,
            "keys": keys.tolist(),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": cached_gaussian,
        }
    return {
        "rng_state_version": 1,
        "rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": numpy_state,
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        ),
    }


def _validate_checkpoint_rng(state: Mapping[str, Any]) -> tuple[Any, Any]:
    """Check restore inputs using isolated generators before model mutation."""
    try:
        torch.Generator(device="cpu").set_state(state["rng_state"])
        if "rng_state_version" not in state:
            return None, None
        if state["rng_state_version"] != 1:
            raise ValueError("unsupported RNG state version")
        random.Random().setstate(state["python_rng_state"])
        numpy, numpy_state = None, None
        if state["numpy_rng_state"] is not None:
            # Optional dependency: required only when the checkpoint captured it.
            import numpy

            saved = state["numpy_rng_state"]
            numpy_state = (
                saved["algorithm"],
                numpy.asarray(saved["keys"], dtype=numpy.uint32),
                saved["position"], saved["has_gauss"], saved["cached_gaussian"],
            )
            numpy.random.RandomState().set_state(numpy_state)
        cuda_states = state["cuda_rng_states"]
        if cuda_states is not None:
            if (
                not isinstance(cuda_states, list)
                or not cuda_states
                or not torch.cuda.is_available()
                or torch.cuda.device_count() != len(cuda_states)
            ):
                raise ValueError("CUDA RNG requires the same available device count")
            for device_index, rng in enumerate(cuda_states):
                if (
                    not isinstance(rng, torch.Tensor)
                    or rng.device.type != "cpu"
                    or rng.dtype != torch.uint8
                    or rng.ndim != 1
                    or rng.numel() == 0
                ):
                    raise ValueError("invalid CUDA RNG state tensor")
                # Tensor shape alone cannot establish the backend's native
                # state format. Validate on a private generator for every
                # device before touching any global RNG or training state.
                torch.Generator(device=f"cuda:{device_index}").set_state(rng)
        return numpy, numpy_state
    except (ImportError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise S1TrainingRefused(f"checkpoint RNG restore refused: {exc}") from exc


def save_checkpoint(
    path: Path | str,
    *,
    model: QwenDenseCandidateModel,
    optimizer: torch.optim.Optimizer,
    progress: TrainingProgress,
    cache_identity: Mapping[str, Any],
    s1_run_manifest_hash: str | None,
    feature_source: str,
    ancestry_base: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> None:
    """Write a resumable checkpoint to `path`, wherever the caller says.

    **`path` is never derived from the working directory.** A Colab
    runtime disconnects without warning and `/content` does not survive
    it, so the caller (notebook cell, CLI `--out`) supplies a mounted
    Drive path and this function writes exactly there.

    Written through a `.tmp` sibling and `os.replace`, which is atomic on
    the same filesystem: a disconnect during a multi-hundred-MB write to
    Drive must not leave a truncated file where the previous good
    checkpoint was.

    `cache_identity` is stored so a resume onto a different cache is a
    refusal rather than a silently different run; `s1_run_manifest_hash`
    records which frozen manifest (if any) authorized the steps already
    taken -- `None` means the checkpoint came from a diagnostic run and
    carries no gradient-bearing claim.

    `feature_source` has no default on purpose: a null-model checkpoint
    that reads as a real one is exactly the artefact this records against,
    and "forgot to pass it" must not resolve to `cache`.

    `ancestry_base` is a verified `document_kind: "base"` checkpoint
    ancestry document (`checkpoint_ancestry.build_base_ancestry` /
    `verify_base`). When given, this checkpoint's generation-1 ancestry
    document is built (`extend_for_checkpoint`), its `ancestry_hash` is
    embedded in the checkpoint payload BEFORE the checkpoint's own bytes
    exist (design decision 5), and only once `os.replace` has landed the
    checkpoint at `path` is the document finalized (now that its
    `checkpoint_sha256` can be computed) and its sidecar written -- so a
    reader never sees a sidecar for a checkpoint that is not actually on
    disk yet. `ancestry_base=None` (a diagnostic run) writes no sidecar and
    embeds `ancestry_hash: None`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    doc = None
    ancestry_hash = None
    if ancestry_base is not None:
        doc = ca.extend_for_checkpoint(
            dict(ancestry_base),
            step=progress.step,
            run_id=run_id,
            s1_run_manifest_hash=s1_run_manifest_hash,
        )
        ancestry_hash = doc["ancestry_hash"]

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": progress.step,
            "pending_order": list(progress.pending_order),
            "cache_identity": dict(cache_identity),
            "s1_run_manifest_hash": s1_run_manifest_hash,
            "feature_source": feature_source,
            # Same reasoning as the log line: derived from the optimizer
            # object, so a checkpoint cannot claim an algorithm it was not
            # trained with. A resume that changes algorithm is a different
            # run and the reader can now see that it happened.
            "optimizer": type(optimizer).__name__,
            "lr": optimizer.param_groups[0]["lr"],
            **_checkpoint_rng_states(),
            "ancestry_hash": ancestry_hash,
        },
        temporary,
    )
    os.replace(temporary, path)

    if doc is not None:
        doc = ca.finalize_checkpoint_document(doc, path)
        ca.write_sidecar(doc, path)


def verify_checkpoint_state(
    path: Path | str,
    state: Mapping[str, Any],
    *,
    expect_s1_run_manifest_hash: str | None,
    expect_base_ancestry_hash: str | None,
    expect_cache_identity: Mapping[str, Any] | None = None,
    expect_feature_source: str | None = None,
) -> None:
    """Every refusal :func:`load_checkpoint` makes BEFORE touching the model
    (cache identity, feature source, manifest hash, ancestry sidecar), on an
    already-loaded `state` -- so an admission dry run can refuse a resume
    without building a model (S4 launcher review, 2026-09-24)."""
    if expect_cache_identity is not None and dict(expect_cache_identity) != state[
        "cache_identity"
    ]:
        raise S1TrainingRefused(
            "checkpoint was written against a different cache identity -- "
            "resuming onto another cache is a different run, not a continuation"
        )
    if (
        expect_feature_source is not None
        and state.get("feature_source") != expect_feature_source
    ):
        raise S1TrainingRefused(
            "checkpoint was written by a run with feature source "
            f"{state.get('feature_source')!r}, not {expect_feature_source!r}; "
            "continuing a null-model run as a real one (or the reverse) "
            "launders one arm's weights into the other's artefacts"
        )
    if state.get("s1_run_manifest_hash") != expect_s1_run_manifest_hash:
        raise S1TrainingRefused(
            "checkpoint was written under s1_run_manifest_hash="
            f"{state.get('s1_run_manifest_hash')!r}, not "
            f"{expect_s1_run_manifest_hash!r}; resuming across manifests, "
            "including diagnostic -> gradient-bearing, is a different run"
        )
    if expect_s1_run_manifest_hash is not None:
        try:
            ca.verify_sidecar(
                Path(path),
                expect_base_ancestry_hash=expect_base_ancestry_hash,
                expect_s1_run_manifest_hash=expect_s1_run_manifest_hash,
                embedded_ancestry_hash=state.get("ancestry_hash"),
            )
        except (ca.CheckpointAncestryInvalid, OSError) as exc:
            raise S1TrainingRefused(
                f"checkpoint ancestry sidecar refused: {exc}"
            ) from exc


def load_checkpoint(
    path: Path | str,
    *,
    model: QwenDenseCandidateModel,
    optimizer: torch.optim.Optimizer,
    expect_s1_run_manifest_hash: str | None,
    expect_base_ancestry_hash: str | None,
    expect_cache_identity: Mapping[str, Any] | None = None,
    expect_feature_source: str | None = None,
    restore_rng: bool = True,
) -> tuple[TrainingProgress, dict[str, Any]]:
    """Restore model, optimizer, RNG state and page-order position from a
    checkpoint written anywhere -- including a different machine, a
    different mount point, or a previous Colab runtime.

    Returns the `TrainingProgress` to hand back to :func:`train_steps`,
    plus the raw state for anything else a caller wants to read.

    `expect_feature_source` refuses a resume that would continue a
    null-model checkpoint as a real run (or the reverse): the weights
    would descend from noise while every artefact after the resume said
    `cache`.

    `expect_s1_run_manifest_hash` has no default on purpose: resuming
    across manifests, including diagnostic -> gradient-bearing, is a
    different run, so a caller cannot silently skip the check by omission.
    `None` means "diagnostic, no manifest claim" and is itself a value the
    checkpoint's own `s1_run_manifest_hash` must match -- a checkpoint
    written under one manifest (or none) resumed as the other is refused
    in both directions.

    New checkpoints include the global CPU/Python RNG, the NumPy global RNG
    when NumPy was loaded, and all CUDA device RNGs when CUDA was initialized.
    CUDA restoration requires the same available device count. This does not
    remove nondeterministic GPU operations or capture caller-owned generators.
    Legacy checkpoints restore only CPU torch RNG, recorded in the returned
    state as `rng_resume_scope="legacy_torch_cpu_only"`.

    `restore_rng=False` exists for the test that proves the RNG state is
    load-bearing (dropping it makes a resumed run diverge from the
    continuous one); a real resume must not use it.

    `expect_base_ancestry_hash` has no default, like
    `expect_s1_run_manifest_hash`, for the same reason: a caller cannot
    silently skip the ancestry check by omission. It is only consulted
    when `expect_s1_run_manifest_hash is not None` -- a gradient-bearing
    resume -- in which case the checkpoint's sidecar
    (`checkpoint_ancestry.verify_sidecar`) is verified BEFORE
    `load_state_dict`, and any `CheckpointAncestryInvalid` becomes a
    refusal. A diagnostic resume (`expect_s1_run_manifest_hash is None`)
    ignores the sidecar entirely.
    """
    state = torch.load(Path(path), weights_only=False)
    verify_checkpoint_state(
        path,
        state,
        expect_s1_run_manifest_hash=expect_s1_run_manifest_hash,
        expect_base_ancestry_hash=expect_base_ancestry_hash,
        expect_cache_identity=expect_cache_identity,
        expect_feature_source=expect_feature_source,
    )
    numpy, numpy_state = _validate_checkpoint_rng(state) if restore_rng else (None, None)
    try:
        model.load_state_dict(state["model_state_dict"])
    except RuntimeError as exc:
        if "visual_memory.norm." in str(exc):
            raise S1TrainingRefused(
                "load_checkpoint: model.load_state_dict refused -- this "
                "checkpoint predates the identity-init fix "
                "(bug-visual-memory-random-init) and carries stale "
                "visual_memory.norm.* keys the current model does not have; "
                "it cannot be resumed"
            ) from exc
        raise S1TrainingRefused(
            f"load_checkpoint: model.load_state_dict refused: {exc}"
        ) from exc
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if restore_rng:
        torch.set_rng_state(state["rng_state"])
        if "rng_state_version" in state:
            random.setstate(state["python_rng_state"])
            if numpy is not None:
                numpy.random.set_state(numpy_state)
            if state["cuda_rng_states"] is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    state["rng_resume_scope"] = (
        "not_restored"
        if not restore_rng
        else "global_rng_v1"
        if "rng_state_version" in state
        else "legacy_torch_cpu_only"
    )
    progress = TrainingProgress(
        step=state["step"], pending_order=tuple(state.get("pending_order", ()))
    )
    return progress, state


# ---------------------------------------------------------------------------
# Gradient admissibility
# ---------------------------------------------------------------------------


def require_gradient_admissible(
    *,
    cache_identity: Mapping[str, Any],
    run_manifest: Mapping[str, Any] | None,
    allowlist: Mapping[str, Any] | None,
    ancestry: Mapping[str, Any] | None = None,
    expectation: CacheExpectation | None = None,
    cli_lr: float | None = None,
    cli_algorithm: str | None = None,
    cli_batch_size: int | None = None,
    cli_seed: int | None = None,
    cli_steps: int | None = None,
    control_arm: bool = False,
) -> None:
    """Refuse a gradient-bearing run that is not bound to an admissible
    frozen `s1_run_manifest`.

    Every substantive check belongs to `run_binding.RunManifestBindingVerifier`
    and is delegated to it; this function owns only the two things it can
    know that the verifier cannot -- that a manifest was supplied at all,
    and that it actually declares `gradient_bearing`. Both are refusals,
    never a downgrade to a diagnostic run.

    `expectation` is the frozen cohort (DEC-MP-4) the run reads its cache
    under; the verifier refuses a gradient-bearing run whose manifest pins
    no expectation hash, or whose presented expectation is not the pinned
    one.

    `ancestry` is the checkpoint ancestry base document (design decision 1,
    DEC-009 condition 7): a gradient-bearing run refuses without one, and
    refuses one whose derived `checkpoint_rights_status` or
    `checkpoint_privacy_status` is not `approved` -- an unresolved rights
    or privacy decision must never authorize a gradient update.

    The `cli_*` values and `control_arm` are the CLI-vs-manifest parity
    inputs (1.2.0, #21 C): a gradient-bearing manifest's `run_config` is
    the frozen budget, and the CLI must reproduce it exactly -- a mismatch
    or a missing CLI value is a refusal, never a default, and a control
    arm is a diagnostic run that cannot be gradient-bearing. The budget
    itself comes from the manifest, which the verifier has just proved
    non-null for a gradient-bearing run.
    """
    if ancestry is None:
        raise S1TrainingRefused(
            "a gradient-bearing S1 run requires --ancestry: a verified "
            "checkpoint ancestry base document (DEC-009 condition 7)"
        )
    try:
        ca.verify_base(dict(ancestry))
    except ca.CheckpointAncestryInvalid as exc:
        raise S1TrainingRefused(f"checkpoint ancestry base refused: {exc}") from exc
    # Privacy is checked before rights: `derive_rights_status` already
    # requires every `dataset_exposures[i].decision` to be non-null (the
    # same condition `derive_privacy_status` checks), so whenever rights
    # would fail on that account privacy fails too -- checking rights
    # first would make the privacy refusal below unreachable. Checking
    # privacy first keeps both refusals independently reachable.
    if ca.derive_privacy_status(dict(ancestry)) != "approved":
        raise S1TrainingRefused(
            "checkpoint ancestry checkpoint_privacy_status is not 'approved' "
            "-- an unresolved privacy decision cannot authorize a gradient update"
        )
    if ca.derive_rights_status(dict(ancestry)) != "approved":
        raise S1TrainingRefused(
            "checkpoint ancestry checkpoint_rights_status is not 'approved' "
            "-- an unresolved rights decision cannot authorize a gradient update"
        )
    if run_manifest is None:
        raise S1TrainingRefused(
            "a gradient-bearing S1 run requires a frozen s1_run_manifest; none "
            "was supplied. Every identity hash now has a producer, but "
            "stage_policy_hash and conditioning_policy_hash still need "
            "owner-chosen content their producers refuse to default, so no "
            "manifest can be frozen truthfully against a placeholder cache "
            "yet -- that is the correct outcome, not an obstacle to route "
            "around"
        )
    if not run_manifest.get("gradient_bearing"):
        raise S1TrainingRefused(
            "s1_run_manifest declares gradient_bearing="
            f"{run_manifest.get('gradient_bearing')!r}; it cannot authorize a "
            "gradient update (design §8 rule 3a: declared, never inferred)"
        )
    # Resolved from the cache identity's own `capability_profile_hash`
    # (DEC-024 wiring), never an import-time global: the profile a run is
    # checked against must be the one its cache actually names, not
    # whichever profile happened to be bound when this module was written.
    try:
        capability_profile = resolve_capability_profile_by_hash(
            cache_identity.get("capability_profile_hash")
        )
    except CapabilityProfileHashUnresolved as exc:
        raise S1TrainingRefused(f"cache identity capability profile refused: {exc}") from exc
    try:
        rb.RunManifestBindingVerifier().verify(
            dict(run_manifest),
            dict(cache_identity),
            allowlist=dict(allowlist) if allowlist is not None else None,
            capability_profile=capability_profile,
            expectation=expectation,
        )
    except ExactTextCacheIncompatible as exc:
        raise S1TrainingRefused(f"s1_run_manifest binding refused: {exc}") from exc

    _check_run_config_parity(
        run_manifest,
        cli_lr=cli_lr,
        cli_algorithm=cli_algorithm,
        cli_batch_size=cli_batch_size,
        cli_seed=cli_seed,
        cli_steps=cli_steps,
        control_arm=control_arm,
    )


def _check_run_config_parity(
    run_manifest: Mapping[str, Any],
    *,
    cli_lr: float | None,
    cli_algorithm: str | None,
    cli_batch_size: int | None,
    cli_seed: int | None,
    cli_steps: int | None,
    control_arm: bool,
) -> None:
    """1.2.0 (owner decision 2026-09-13, #21 C): the run's budget is
    frozen in the manifest, and the CLI must reproduce it.

    A control arm is refused outright: the frozen budget is
    `augmentation: none`, and a run whose features are shuffled or
    randomized is a diagnostic arm, never a gradient-bearing result. Each
    missing `cli_*` value is itself a refusal, not a default: a silent
    fallback would let `main()` spend a budget the manifest never froze.
    Every comparison is exact -- lr through the scaled-int form the CBOR
    profile forces (`cli_lr * 1e6 == optimizer.lr_micro`), the rest by
    value.
    """
    if not run_manifest["gradient_bearing"]:
        return
    if control_arm:
        raise S1TrainingRefused(
            "control arm runs are diagnostic: a gradient-bearing run whose "
            "features are shuffled or randomized cannot train under its "
            "frozen run_config (augmentation: none)"
        )
    for name, value in (
        ("cli_lr", cli_lr),
        ("cli_algorithm", cli_algorithm),
        ("cli_batch_size", cli_batch_size),
        ("cli_seed", cli_seed),
        ("cli_steps", cli_steps),
    ):
        if value is None:
            raise S1TrainingRefused(
                f"{name} is required for a gradient-bearing run: the CLI "
                "must reproduce the frozen run_config exactly, and a missing "
                "CLI value is a refusal, never a default (1.2.0, #21 C)"
            )
    frozen = run_manifest["run_config"]
    lr_micro = int(round(cli_lr * 1e6))
    if lr_micro != frozen["optimizer"]["lr_micro"]:
        raise S1TrainingRefused(
            f"cli_lr {cli_lr!r} (lr_micro {lr_micro}) does not match the "
            f"frozen run_config optimizer.lr_micro "
            f"{frozen['optimizer']['lr_micro']} -- the run must spend the "
            "budget it froze, not the CLI's (1.2.0, #21 C)"
        )
    if cli_algorithm != frozen["optimizer"]["algorithm"]:
        raise S1TrainingRefused(
            f"cli_algorithm {cli_algorithm!r} does not match the frozen "
            f"run_config optimizer.algorithm "
            f"{frozen['optimizer']['algorithm']!r} (1.2.0, #21 C)"
        )
    if cli_batch_size != frozen["sampler"]["batch_size"]:
        raise S1TrainingRefused(
            f"cli_batch_size {cli_batch_size!r} does not match the frozen "
            f"run_config sampler.batch_size "
            f"{frozen['sampler']['batch_size']!r} (1.2.0, #21 C)"
        )
    if cli_seed != frozen["sampler"]["seed"]:
        raise S1TrainingRefused(
            f"cli_seed {cli_seed!r} does not match the frozen run_config "
            f"sampler.seed {frozen['sampler']['seed']!r} (1.2.0, #21 C)"
        )
    if cli_steps != frozen["steps"]:
        raise S1TrainingRefused(
            f"cli_steps {cli_steps!r} does not match the frozen run_config "
            f"steps {frozen['steps']!r} (1.2.0, #21 C)"
        )


def refuse_out_under_audited_roots(out: Path, ancestry: Mapping[str, Any]) -> None:
    """Refuse an `--out` checkpoint destination that sits under one of the
    S0 audit's own `legacy_ancestry.evidence.audited_roots` (design decision
    6): a checkpoint written there would make a FUTURE S0 legacy audit
    over those same roots find it and (correctly) classify it as an
    unaudited legacy checkpoint -- the exact finding `build_base_ancestry`
    requires be `False`."""
    resolved_out = out.resolve()
    audited_roots = ancestry.get("legacy_ancestry", {}).get("evidence", {}).get(
        "audited_roots", []
    )
    for root in audited_roots:
        resolved_root = Path(root).resolve()
        if resolved_out == resolved_root or resolved_root in resolved_out.parents:
            raise S1TrainingRefused(
                f"--out {out} resolves under audited root {root!r}: a future "
                "S0 legacy audit over that root would find this checkpoint "
                "and classify it as unaudited legacy ancestry. Write "
                "checkpoints under an unaudited path instead, e.g. "
                "~/unflatten-ckpt/<run>/"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin wiring
    """Run the S1 exact-text loop over a cache.

    `--gradient-bearing` requires `--run-manifest` (and `--allowlist`);
    without them the run is diagnostic and says so in every line it
    prints.
    """
    import argparse

    from unflatten.model.candidate_loader import load_qwen_candidate
    from unflatten.model.s1_runner import build_s1_optimizer

    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.s1_train", description=__doc__
    )
    parser.add_argument("cache", type=Path, help="exact-text cache directory")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--shuffle-features",
        action="store_true",
        help="control arm: give each page the features of a DIFFERENT page "
        "(a derangement, so no page gets its own). Sharper than "
        "--random-features, because shuffled features have the real "
        "distribution and the only broken thing is which page they describe: "
        "a gap between this arm and the real one has exactly one explanation",
    )
    parser.add_argument(
        "--optimizer",
        choices=("sgd", "adamw"),
        default="sgd",
        help="`sgd` (the default) keeps every earlier run comparable. Plain "
        "SGD at a small --lr barely moves a freshly initialised head, so a "
        "short run under it compares two nearly untrained models: measured "
        "2026-09-05, 200 steps at lr 1e-4 left L_structure flat at ~17.6 and "
        "the NOISE arm won on L_text. Use `adamw` for a run meant to actually "
        "train; the choice is recorded in the log and the checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="checkpoint path -- point it at a MOUNTED path (Drive) on Colab: "
             "the runtime disconnects without warning and /content does not "
             "survive it",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="write a mid-run checkpoint every N steps (0 = only at the end, "
             "which loses the whole run on a Colab disconnect)",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the global RNG, which decides model head init and page "
             "order. Required to compare a run against --random-features: "
             "without it the two arms differ in initialisation and page "
             "order as well as in features, and the comparison measures "
             "those instead",
    )
    parser.add_argument(
        "--random-features",
        action="store_true",
        help="NULL MODEL: replace the cached features with noise carrying "
             "each page's own scale. The grammar and the text prior are a "
             "strong baseline on their own, so a real run is scored against "
             "this arm, not against chance -- run both with the same seed, "
             "steps, batch size and lr, and compare the per-family curves",
    )
    parser.add_argument(
        "--preflight-rows",
        type=int,
        default=4,
        help="run `preflight_native_signal` over this many cache rows before "
        "training (0 = off). A gradient-bearing run that fails it is "
        "refused (`require_preflight_passes`); a diagnostic run only logs",
    )
    parser.add_argument("--gradient-bearing", action="store_true")
    parser.add_argument("--run-manifest", type=Path, default=None)
    parser.add_argument("--allowlist", type=Path, default=None)
    parser.add_argument(
        "--expectation",
        type=Path,
        default=None,
        help="frozen cohort expectation JSON (exact_text_cache_expectation.json "
        "from freeze_s1_run_manifest) -- required for --gradient-bearing "
        "(DEC-MP-4)",
    )
    parser.add_argument(
        "--ancestry",
        type=Path,
        default=None,
        help="verified checkpoint ancestry base document (JSON) -- required "
        "for --gradient-bearing (DEC-009 condition 7)",
    )
    args = parser.parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    run_manifest = (
        json.loads(args.run_manifest.read_text()) if args.run_manifest else None
    )
    allowlist = json.loads(args.allowlist.read_text()) if args.allowlist else None
    expectation = (
        CacheExpectation.from_payload(json.loads(args.expectation.read_text()))
        if args.expectation
        else None
    )
    ancestry = json.loads(args.ancestry.read_text()) if args.ancestry else None
    if ancestry is not None:
        ca.verify_base(ancestry)
        if args.out:
            refuse_out_under_audited_roots(args.out, ancestry)

    dataset = ExactTextTrainingDataset(
        args.cache,
        gradient_bearing=args.gradient_bearing,
        expectation=expectation,
        randomize=args.random_features,
        shuffle=args.shuffle_features,
        shuffle_seed=args.seed,
    )
    identity = asdict(dataset.identity)
    if args.shuffle_features:
        print(
            "[s1_train] SHUFFLED CONTROL: every page is given another page's "
            "features. Real distribution, wrong page. Nothing this run "
            "produces is a backbone result.",
            flush=True,
        )
    if args.random_features:
        print(
            "[s1_train] NULL MODEL: features are noise, grammar and text "
            "prior only. Nothing this run produces is a backbone result.",
            flush=True,
        )
    if args.gradient_bearing:
        require_gradient_admissible(
            cache_identity=identity,
            run_manifest=run_manifest,
            allowlist=allowlist,
            ancestry=ancestry,
            expectation=expectation,
            cli_lr=args.lr,
            cli_algorithm=args.optimizer,
            cli_batch_size=args.batch_size,
            cli_seed=args.seed,
            cli_steps=args.steps,
            control_arm=args.random_features or args.shuffle_features,
        )
    else:
        print(
            "[s1_train] DIAGNOSTIC run: no s1_run_manifest, no gradient-bearing "
            "claim. Nothing this run produces is S1 evidence.",
            flush=True,
        )

    candidate = load_qwen_candidate(device=args.device)
    model = QwenDenseCandidateModel(
        decode=candidate.decode,
        inject_visual=candidate.inject_visual,
        visual_dim=candidate.visual_hidden_size,
        d_model=candidate.hidden_size,
        ranges=cv.packed_ranges(candidate.vocab_size),
        qwen_embedding_weight=candidate.embed_tokens.weight,
        qwen_lm_head_weight=candidate.lm_head_weight,
    )
    model.to(args.device)
    optimizer = build_s1_optimizer(model, lr=args.lr, algorithm=args.optimizer)

    manifest_hash = run_manifest["s1_run_manifest_hash"] if run_manifest else None
    run_id = run_manifest["run_id"] if run_manifest else None
    base_ancestry_hash = ancestry["ancestry_hash"] if ancestry else None

    progress = None
    if args.resume:
        progress, _state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            expect_cache_identity=identity,
            expect_feature_source=dataset.feature_source,
            expect_s1_run_manifest_hash=manifest_hash,
            expect_base_ancestry_hash=base_ancestry_hash,
        )

    preflight_indices = preflight_row_indices(len(dataset), args.preflight_rows)
    if preflight_indices:
        preflight_report = preflight_native_signal(
            model,
            [dataset[i] for i in preflight_indices],
            image_token_id=candidate.image_token_id,
            device=args.device,
        )
        print(json.dumps({"preflight": preflight_report}), flush=True)
        require_preflight_passes(preflight_report, gradient_bearing=args.gradient_bearing)

    def write(at: TrainingProgress) -> None:
        save_checkpoint(
            args.out,
            model=model,
            optimizer=optimizer,
            progress=at,
            cache_identity=identity,
            s1_run_manifest_hash=manifest_hash,
            feature_source=dataset.feature_source,
            ancestry_base=ancestry,
            run_id=run_id,
        )
        print(f"[s1_train] checkpoint step {at.step} -> {args.out}", flush=True)

    progress = train_steps(
        model,
        dataset,
        optimizer,
        ranges=model.ranges,
        image_token_id=candidate.image_token_id,
        steps=args.steps,
        progress=progress,
        batch_size=args.batch_size,
        device=args.device,
        log_every=1,
        checkpoint_every=args.checkpoint_every if args.out else 0,
        checkpoint=write if args.out else None,
    )
    if args.out:
        save_checkpoint(
            args.out,
            model=model,
            optimizer=optimizer,
            progress=progress,
            cache_identity=identity,
            s1_run_manifest_hash=manifest_hash,
            feature_source=dataset.feature_source,
            ancestry_base=ancestry,
            run_id=run_id,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
