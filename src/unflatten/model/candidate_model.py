"""Qwen dense candidate — wires M2's four built pieces into one forward
pass (design §5, §8 M2 items 1-4; item 5's audit is in
`trainability_smoke.py`).

```text
visual_features (per-page, from a QwenCandidate's tower — caller's job)
  -> VisualMemory                          (adapter.projection, trainable)
  -> DocumentContextEncoder(visual memory) (document_context_encoder, trainable)
  -> valid visual rows, flattened          (drop padding before the scatter)
  -> QwenCandidate.inject_visual           (backbone.language, FROZEN --
                                             native placeholder scatter +
                                             native mRoPE position ids)
  -> [doc context ; injected composite sequence]   (context has no native
                                             Qwen equivalent, stays a prefix)
  -> QwenCandidate.decode                  (backbone.language, FROZEN)
  -> drop the doc-context prefix
  -> DenseExpertAdapter                    (expert_adapter.dense, trainable)
  -> CompositeDecodeHead                   (tree_text_decoder, trainable)
  -> packed composite-vocabulary logits
```

`document_context_encoder` was M3's carried-forward blocker, added here
(see `document_context_encoder.py`'s docstring for why its shape is an
ASSUMPTION, not a DEC-007 pin) rather than in M2, because M2's five-item
scope never listed it. `encode_hidden` exposes the un-sliced merged hidden
state for `free_running_decode.py`, which needs the last PREFIX position
before any composite token has been accepted (design §5.7: decode has no
gold prefix).

**Revision history, because the previous version of this docstring recorded
an assumption that turned out to be avoidable, not necessary.** M2's first
pass merged visual memory into the sequence as a plain prefix, attended to
via ordinary causal self-attention, reasoning that Qwen's native placeholder
scatter would tie this module to Qwen-specific config (`image_token_id`)
and cost backbone-neutrality. Review caught that this traded away more than
it protected: DEC-007 D1 took the *whole* Qwen candidate rather than just
the vision tower, and D4 chose the Qwen tokenizer over a standalone byte
tokenizer, specifically to keep Qwen's pretrained priors -- and a prefix the
backbone never saw positionally during pretraining spends a real part of
that prior for no measured reason (no test in this repo showed the native
path was unavailable; it was not tried). The fix is not "abandon
backbone-neutrality" -- it is that `BackboneAdapter` (RFC004 §13) already
exists as the seam that resolves the tension: `QwenCandidate.inject_visual`
does whatever the loaded backbone natively does (masked_scatter at Qwen's
own placeholder ids, its own mRoPE positional index), and THIS module calls
it without knowing which backbone it is. Backbone-neutral at this module's
boundary; on the backbone's own training distribution inside it. See
`candidate_loader.py`'s module docstring for the mechanism itself.

`document_context_encoder`'s summary tokens are NOT part of this native
merge, and that is not an inconsistency: Qwen has no placeholder id for
"document-level state" because it never saw that concept in pretraining --
there is no native convention to defer to, so a prefix (attended to via
ordinary causal self-attention, position ids `0..context_len-1`, with the
injected composite range's own native position ids uniformly shifted by
`context_len`) is the only available mechanism, not a shortcut avoided
elsewhere.

**Two remaining ASSUMPTIONS, neither pinned by DEC-007:**

1. **Pixel-to-visual-feature batching is out of scope here.** This module
   starts from an already-computed `(batch, tokens, visual_dim)` visual
   feature tensor, exactly `VisualMemory`'s own contract — it does not call
   `QwenCandidate.vision_forward` itself. Real Qwen3-VL batches images by
   concatenating variable per-image token counts and a `grid_thw` tensor,
   which is page/document batching policy (RFC003's territory), not model
   mechanics; folding it in here would blur that boundary for no M2 benefit.
2. **Non-Qwen composite positions get the native filler id `0`** in
   `native_id_sequence` (`_native_id_view` below) purely so
   `inject_visual`'s modality grouping has *some* native id to look at
   outside the placeholder run -- `get_rope_index` only branches on equality
   against `image_token_id`/`video_token_id`, so any id provably never equal
   to either is safe, and `0` is never one of them in either backend
   (`candidate_loader.build_stub_qwen_candidate` reserves its four sentinels
   at the TOP of the vocab; the real Qwen3-VL's `image_token_id`/
   `video_token_id` are 151655/151656). This is a documented, not computed,
   guarantee -- a future backbone whose id 0 IS a special token would need a
   different filler.

This module authorizes no gradient update: it declares `nn.Module`s and one
`forward`, nothing runs `.backward()` or builds an optimizer.

**The mixed-precision dtype rule (DEC-007 D3.1), stated once, for the next
boundary someone adds.** This pipeline crosses the frozen/trainable dtype
line three times -- `decode`'s `inputs_embeds` (candidate_loader.py),
`embed_packed`'s `out` buffer and the Qwen LM-head readout (composite_
head.py), and `dense_adapter`'s input (`encode_hidden` below) -- and the
same shaped bug was found and fixed at TWO of those three independently,
on the same day, in OPPOSITE directions (the first fix cast a trainable
buffer down to the frozen side's dtype; the second, caught by review a few
hours later, had a different buffer following the frozen side's dtype and
silently narrowing the trainable path). That the shape recurred means the
first fix was local, not general. The rule, so a fourth boundary does not
repeat it a third time:

- **The cast lives at the boundary, and it casts the FROZEN side to match
  the TRAINABLE side, never the other way -- FOR A SMALL TENSOR.** A
  trainable module's own storage dtype is an implementation decision
  (DEC-007 D3.1: the bf16 pin covers frozen base weights and compute, not
  trainable storage); reading a frozen bf16 tensor into an fp32 (or
  whatever) trainable computation is a lossless upcast, and for a
  small tensor it is the only direction that is always safe regardless of
  which dtype the trainable side happens to be configured with today. **It
  has a size boundary, and this rule does not hold past it** -- see the
  next bullet.
- **SIZE EXCEPTION: for a large frozen weight, cast the ACTIVATION down
  instead, and upcast the (small) result.** Found on a real GPU run,
  2026-09-04, not in any test: `composite_head.py`'s Qwen LM-head readout
  followed the rule above -- upcast the frozen `qwen_lm_head_weight` to the
  trainable `bridged`'s dtype -- and OOM'd materialising a full fp32 copy of
  the ~152k-row Qwen vocab (1.45 GiB) on every forward call. The general
  rule is O(1) per boundary; this one is O(vocab x d_model) per step,
  because the frozen tensor being upcast is vocab-sized, not the small
  per-token tensor the rule was written imagining. The fix: cast `bridged`
  (small: `(..., d_model)`) DOWN to the weight's dtype, matmul in that
  dtype, upcast only the (small) `(..., qwen_count)` result. This is also
  more faithful to DEC-007 D3, which pins bf16 as the COMPUTE dtype -- the
  size exception happens to point the same direction the pin already did.
  **Not bit-exact** with the fp32-matmul path it replaced (measured on a
  synthetic fixture, `test_composite_head.py`: max_abs=0.0368,
  mean_abs=0.00375 against a logit scale of 1.643 -- a few percent
  relative, consistent with one bf16 matmul's rounding, not a correctness
  bug); characterise a new boundary's numeric cost the same way rather
  than assuming it away.
- **Never assume elementwise promotion covers a matmul.** `x - buffer`
  between a bf16 tensor and an fp32 buffer silently promotes to fp32 (and
  `nn.LayerNorm` tolerates a dtype-mismatched weight); `nn.Linear`,
  `masked_scatter`, and indexed assignment (`out[mask] = ...`) do not --
  they raise. Verify each new boundary directly (a two-line REPL check)
  rather than extrapolating from a nearby operator that happened to be
  forgiving; two of today's three boundary bugs were exactly this
  extrapolation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from unflatten.model import typed_head_config
from unflatten.model.candidate_architecture_variant import CandidateArchitectureVariant
from unflatten.model.candidate_loader import DecoderLike, InjectedEmbeds, InjectVisualLike
from unflatten.model.composite_head import TREE_TEXT_DECODER_GROUP_ID, CompositeDecodeHead
from unflatten.model.composite_vocab import PackedRanges
from unflatten.model.document_context_encoder import (
    DOCUMENT_CONTEXT_ENCODER_GROUP_ID,
    DocumentContextEncoder,
)
from unflatten.model.expert_adapter import EXPERT_ADAPTER_DENSE_GROUP_ID, DenseExpertAdapter
from unflatten.model.level_router.grammar_state import GrammarTrace
from unflatten.model.level_router.module import (
    LEVEL_ROUTER_GROUP_ID,
    DocumentLevelRouter,
    LevelMemory,
    gather_level_features,
)
from unflatten.model.node_visual_aligner import NodeVisualAlignerHead
from unflatten.model.property_head_registry import property_head_registry
from unflatten.model.sparse_expert_adapter.adapter import SparseExpertAdapter
from unflatten.model.sparse_expert_adapter.reasoning_integration import (
    SparseExpertAdapterWithReasoning,
)
from unflatten.model.speculative.split_causal_attention import (
    ATTN_IMPLEMENTATION_NAME as SPLIT_CAUSAL_ATTN_IMPLEMENTATION_NAME,
)
from unflatten.model.typed_head_config import (
    CARDINALITY_HEADS,
    COLOR_PALETTE_V0,
    NULLABLE_COLOR_POINTERS,
    family_override,
    star_pointer_cap_attributes,
)
from unflatten.model.typed_property_heads import (
    BooleanPropertyHead,
    CardinalityHead,
    EnumPropertyHead,
    IntegerHeadConfig,
    IntegerPropertyHead,
    PaletteHead,
    SlotHead,
    scale_for_pointer,
)
from unflatten.model.typed_property_pointer_map import FAMILY, enum_domain
from unflatten.model.visual_compression import contracts as visual_compression_contracts
from unflatten.model.visual_compression.adaptive_v0 import AdaptiveVisualMemoryCompressor
from unflatten.model.visual_memory import VISUAL_MEMORY_GROUP_ID, VisualMemory

#: Native id given to non-Qwen (structural/byte-fallback, and document
#: context prefix) positions in `native_id_sequence` -- see the module
#: docstring's assumption 2.
_NATIVE_ID_FILLER = 0


def _native_id_view(composite_input_ids: torch.Tensor, ranges: PackedRanges) -> torch.Tensor:
    """`composite_input_ids` -> the native-id-shaped tensor `inject_visual`
    needs: a Qwen composite position (this includes visual placeholder
    positions, packed as ordinary `QwenRef`s) unpacks to its native id;
    every other position gets `_NATIVE_ID_FILLER`."""
    is_qwen = (composite_input_ids >= ranges.qwen_start) & (
        composite_input_ids < ranges.byte_control_start
    )
    native = torch.full_like(composite_input_ids, _NATIVE_ID_FILLER)
    return torch.where(is_qwen, composite_input_ids - ranges.qwen_start, native)


#: The head_id whose tensor is already wired as the model's `dense_adapter`
#: child (`DenseExpertAdapter`) -- every other registered head_id gets its
#: own child module built from `property_head_registry`.
_DENSE_HEAD_ID = "expert_adapter.dense"

#: The child attribute for the S3 typed-head group covering `expert_adapter
#: .dense`'s own pointer paths (design §4.5, DEC-028 D6). This is an
#: EXPLICIT mapping, not `_DENSE_HEAD_ID.replace(".", "_")` -- that would
#: yield `"expert_adapter_dense"`, one underscore away from the existing
#: `dense_adapter` child and confusable with it; the trailing `_heads`
#: keeps the two names unambiguous at a glance.
_DENSE_HEAD_GROUP_ATTR = "expert_adapter_dense_heads"

#: The trainable-group id owning the `node_visual_aligner` child's
#: parameters (DEC-027). The child is a real `nn.Module` (unlike the frozen
#: `QwenCandidate`), so its parameters appear in `model.named_parameters()`
#: and must be accounted for by `parameter_group_ids` -- but it is frozen
#: until S2, so it is never a trainable S1 group.
NODE_VISUAL_ALIGNER_GROUP_ID = "node_visual_aligner"


def typed_head_children() -> dict[str, str]:
    """`{child attribute name: head_id}` for every registered head_id,
    including `expert_adapter.dense` itself -- ordinary registered head_ids
    map `head_id` with `.` mapped to `_`; `expert_adapter.dense` maps
    explicitly to `expert_adapter_dense_heads` (not
    `_DENSE_HEAD_ID.replace(".", "_")`, see `_DENSE_HEAD_GROUP_ATTR`) and is
    always the LAST entry, because `QwenDenseCandidateModel.__init__` builds
    these children in this dict's iteration order and this group must be
    built after every other module to keep the S1 RNG draw order unchanged
    (`test_candidate_model.py`'s bit-identity tests). `expert_adapter.dense`
    itself is ALSO already wired as the model's `dense_adapter` child
    (`DenseExpertAdapter`, trainable in S1) -- this dict only adds its
    FROZEN typed-head group, a distinct module under a distinct attribute.
    Reused by gate 15's partition check
    (`s1_runner.gate_15_optimizer_membership_complete`), which must
    recognise these frozen-in-S1 children instead of reporting them as an
    unclaimed module."""
    children = {
        hid.replace(".", "_"): hid
        for hid in sorted(set(property_head_registry().values()) - {_DENSE_HEAD_ID})
    }
    children[_DENSE_HEAD_GROUP_ATTR] = _DENSE_HEAD_ID
    return children


def parameter_group_ids(model: QwenDenseCandidateModel) -> dict[str, str]:
    """`{model.named_parameters() name: trainable-group id}` for every
    parameter this model owns -- the same rule `s2_train._model_group_states`
    builds its own `group_id_of` map with (s2_train.py:241-255), kept HERE
    instead so `s2_train.py` stays byte-frozen while the S2 run is live
    (design §5 [governance R1]). `test_candidate_model.py` asserts the two
    maps agree on the stub candidate model, so they cannot drift silently;
    `s2_train` may adopt this function once the S2 run finalises (a
    separate, later change, not this one)."""
    group_id_of: dict[str, str] = {}
    for name, _ in model.visual_memory.named_parameters():
        group_id_of[f"visual_memory.{name}"] = VISUAL_MEMORY_GROUP_ID
    for name, _ in model.document_context_encoder.named_parameters():
        group_id_of[f"document_context_encoder.{name}"] = DOCUMENT_CONTEXT_ENCODER_GROUP_ID
    for name, _ in model.dense_adapter.named_parameters():
        group_id_of[f"dense_adapter.{name}"] = EXPERT_ADAPTER_DENSE_GROUP_ID
    for name, _ in model.composite_head.named_parameters():
        group_id_of[f"composite_head.{name}"] = TREE_TEXT_DECODER_GROUP_ID
    for name, _ in model.node_visual_aligner.named_parameters():
        group_id_of[f"node_visual_aligner.{name}"] = NODE_VISUAL_ALIGNER_GROUP_ID
    for child_name, head_id in typed_head_children().items():
        child = getattr(model, child_name, None)
        if child is None:
            continue
        for name, _ in child.named_parameters():
            group_id_of[f"{child_name}.{name}"] = head_id
    # RFC007 (DEC-047 dormant): only present when the variant enables it; its
    # group is in no v15b stage policy, so `build_s4_optimizer` refuses it by
    # name until a ratified v16 policy adds the row.
    if model.level_router is not None:
        for name, _ in model.level_router.named_parameters():
            group_id_of[f"level_router.{name}"] = LEVEL_ROUTER_GROUP_ID
    return group_id_of


@dataclass(frozen=True)
class WarmStartLoadReport:
    """What `warm_start_load_state_dict` actually did, per `model.state_dict()`
    key -- every key falls into exactly one of the four tuples below, so a
    caller can log the outcome without re-deriving it."""

    loaded_keys: tuple[str, ...]
    reinitialized_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def warm_start_load_state_dict(
    model: nn.Module, checkpoint_state_dict: Mapping[str, torch.Tensor]
) -> WarmStartLoadReport:
    """Load every `checkpoint_state_dict` entry whose shape matches
    `model`'s own parameter/buffer shape; a key present in both but with a
    MISMATCHED shape (e.g. an `explicit_zero_bin_pointers` head's
    `bin_logits_head` gaining two bins, RFC004-ERRATA-006 E1) is left at
    its freshly initialized value instead of being handed to
    `nn.Module.load_state_dict`, which raises a bare `RuntimeError: size
    mismatch` even under `strict=False` -- `strict` only governs
    missing/unexpected KEYS, never shape agreement on a shared key.

    Every affected key is named in the returned report -- a re-initialised
    head is never silently indistinguishable from a loaded one; a caller
    (e.g. an S6/S4 warm start) is expected to print or log it, never
    swallow it.
    """
    own_state = model.state_dict()
    loaded: list[str] = []
    reinitialized: list[str] = []
    missing: list[str] = []
    compatible: dict[str, torch.Tensor] = {}
    for key, own_tensor in own_state.items():
        if key not in checkpoint_state_dict:
            missing.append(key)
            continue
        checkpoint_tensor = checkpoint_state_dict[key]
        if tuple(checkpoint_tensor.shape) != tuple(own_tensor.shape):
            reinitialized.append(key)
            continue
        compatible[key] = checkpoint_tensor
        loaded.append(key)
    unexpected = tuple(sorted(set(checkpoint_state_dict) - set(own_state)))
    model.load_state_dict(compatible, strict=False)
    return WarmStartLoadReport(
        loaded_keys=tuple(sorted(loaded)),
        reinitialized_keys=tuple(sorted(reinitialized)),
        missing_keys=tuple(sorted(missing)),
        unexpected_keys=unexpected,
    )


def _pointer_attribute(pointer: str) -> str:
    """A legal Python attribute name for one pointer path, unique within a
    head group (verified by `test_candidate_model` against the registry)."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", pointer).strip("_")


#: RFC004-ERRATA-006 E1 (R1): the width pointers the erratum names as its
#: own bimodal-`{0, W}` examples ("tblW"/"tcW" in OOXML shorthand) -- the
#: exact list a run turns on is an owner decision (erratum's OPEN item 2),
#: never picked silently here. A caller opts a pointer in explicitly via
#: `explicit_zero_bin_pointers`; this constant is only a name for the
#: erratum's own examples, not a default (the default stays `frozenset()`,
#: byte-identical).
R1_EXPLICIT_ZERO_BIN_POINTERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("table", "/layout/width/value_mpt"),
        ("cell", "/layout/width_mpt"),
    }
)


def _leaf_head(
    family: str,
    node_type: str,
    pointer: str,
    d_model: int,
    integer_config: IntegerHeadConfig,
    explicit_zero_bin_pointers: frozenset[tuple[str, str]] = frozenset(),
) -> nn.Module:
    """One family instance for `(node_type, pointer)` -- the exact
    construction `_build_typed_head_group` always used, factored out so a
    `*`-pointer's `SlotHead` (T3, design §4.3/§10.1) can build `k_slots` of
    THIS SAME thing via `inner_factory`, never a parallel construction that
    could silently drift from the single-instance case.

    `explicit_zero_bin_pointers` (RFC004-ERRATA-006 E1 / DEC-049 point 2):
    default `frozenset()` is byte-identical to before this parameter
    existed. A `(node_type, pointer)` entry only takes effect on an
    integer, `bins_plus_residual` head -- `IntegerHeadConfig.__post_init__`
    already refuses `explicit_zero_bin=True` under `direct_regression`, so
    naming a pointer under the wrong strategy fails loudly here rather
    than being silently ignored.
    """
    if family == "enum":
        return EnumPropertyHead(
            d_model=d_model, num_classes=len(enum_domain(node_type, pointer))
        )
    if family == "boolean":
        return BooleanPropertyHead(d_model=d_model)
    # DEC-032: every integer head's scale is derived from its own pointer,
    # never a value shared verbatim from `integer_config` -- a page width
    # (`_mpt`) and a grid column count need different learning units.
    # `bins_plus_residual` already operates in raw units and refuses
    # `scale != 1`, so it is left as the caller configured it.
    pointer_config = (
        replace(integer_config, scale=scale_for_pointer(pointer))
        if integer_config.strategy == "direct_regression"
        else integer_config
    )
    if (node_type, pointer) in explicit_zero_bin_pointers:
        pointer_config = replace(pointer_config, explicit_zero_bin=True)
    return IntegerPropertyHead(d_model=d_model, config=pointer_config)


def _build_typed_head_group(
    head_id: str,
    d_model: int,
    integer_config: IntegerHeadConfig,
    explicit_zero_bin_pointers: frozenset[tuple[str, str]] = frozenset(),
) -> nn.Module:
    """One `nn.Module` child per registered pointer path owned by `head_id`.

    A pointer path shared by several node types inside one group (RFC004-
    ERRATA-001 E3/E4: one anchor head serves floating_image AND text_frame,
    one geometry head serves inline_image AND floating_image) contributes
    exactly ONE head -- the family of the first node type seen for that
    path, and a second node type mapping the same path to a different
    family is a registry bug that fails here, not a silently duplicated
    property family. Deferred (string/object) pointers get no head: they
    have no family in `FAMILY` and stay out of the parameter surface until
    their own stage.

    A `*`-pointer (`typed_head_config.STAR_POINTERS`, T3, design §4.3/
    §10.1) gets a `SlotHead` of `k_slots = getattr(typed_head_config,
    cap_attribute)` instances of the SAME family/class instead of one bare
    instance -- `s4_train.replaced_keys`'s warm-start transplant (T8) reads
    the resulting state_dict key shape purely by convention, never by
    importing `SlotHead`.
    """
    group = nn.Module()
    star_caps = star_pointer_cap_attributes()
    family_of: dict[str, str] = {}
    for (node_type, pointer), assigned in sorted(property_head_registry().items()):
        if assigned != head_id:
            continue
        family = FAMILY.get((node_type, pointer))
        # T4 (design §5.1/§10.1): a DEFERRED (string/object) pointer has no
        # `FAMILY` entry at all -- `family_override` is consulted BEFORE
        # skipping it, so the 21 colour and 4 border-object pointers still
        # get a head even though `FAMILY.get` returns `None` for them.
        override = family_override(node_type, pointer) if family is None else None
        if family is None and override is None:
            continue
        effective_family = family if family is not None else override
        if pointer in family_of:
            if family_of[pointer] != effective_family:
                raise ValueError(
                    f"{head_id!r}: pointer {pointer!r} is mapped to family "
                    f"{family_of[pointer]!r} on one node type and "
                    f"{effective_family!r} on another -- the registry must "
                    "agree, or the group would need two heads for one "
                    "pointer path"
                )
            continue
        family_of[pointer] = effective_family
        pointer_attr = _pointer_attribute(pointer)
        cap_attribute = star_caps.get(pointer_attr)
        if cap_attribute is not None:
            head: nn.Module = SlotHead(
                inner_factory=lambda nt=node_type, p=pointer, fam=family: _leaf_head(
                    fam, nt, p, d_model, integer_config, explicit_zero_bin_pointers
                ),
                k_slots=getattr(typed_head_config, cap_attribute),
            )
        elif override == "palette":
            head = PaletteHead(
                d_model=d_model,
                palette_size=len(COLOR_PALETTE_V0),
                nullable=(node_type, pointer) in NULLABLE_COLOR_POINTERS,
            )
        elif override == "borders_null":
            head = BooleanPropertyHead(d_model=d_model)
            pointer_attr = f"{pointer_attr}_is_null"
        else:
            head = _leaf_head(
                family, node_type, pointer, d_model, integer_config, explicit_zero_bin_pointers
            )
        group.add_module(pointer_attr, head)
    if head_id == _DENSE_HEAD_ID:
        # Cardinality heads (design §4.1/§4.5, `spec-typed-heads-missing-
        # families`): a repeated container's element COUNT is a tensor
        # identity, not an RFC001 pointer -- `CARDINALITY_HEADS` is a fixed
        # list in `typed_head_config`, not a `property_head_registry` scan,
        # precisely so this head never moves `head_registry_hash`.
        for _node_type, container_pointer, cap_attribute in CARDINALITY_HEADS:
            k_max = getattr(typed_head_config, cap_attribute)
            group.add_module(
                _pointer_attribute(container_pointer),
                CardinalityHead(d_model=d_model, k_max=k_max),
            )
    if len(family_of) == 0:
        raise ValueError(f"{head_id!r} owns no family pointer path in the registry")
    return group


@dataclass(frozen=True)
class TeacherForcedBatch:
    """One teacher-forced fixture batch for `QwenDenseCandidateModel.forward`.

    `visual_features` is `(batch, visual_tokens, visual_dim)`; `visual_valid_lengths`
    is `None` for an unpadded batch (declares every visual position real,
    matching `VisualMemory.forward`'s own convention). `composite_input_ids`
    is `(batch, steps)` of already-packed ids (`composite_vocab.pack`
    output) -- every position standing for a visual token MUST be a packed
    `QwenRef(native_id=image_token_id)` (repeated once per visual token in
    that sample, in order), matching `visual_features`'s per-sample
    valid-token count exactly, or `inject_visual` raises. `image_grid_thw`
    is passed straight through to `inject_visual`'s native positional index.
    `composite_attention_mask` is `(batch, steps)`, 1 == real, 0 == padding.

    `deepstack_visual_embeds` (DEC-056): `None` (every v15b caller) or one
    tensor per DeepStack level, each `(total_valid_visual_tokens,
    hidden_size)` -- the frozen tower's raw level output, rows in the SAME
    row-major (batch, token) order the placeholder scatter consumes. Only a
    model built with `deepstack=True` accepts it, and such a model refuses a
    batch without it -- except that the RFC007 level router (when enabled)
    reads the levels as separate query-side memories without native injection.
    """

    visual_features: torch.Tensor
    composite_input_ids: torch.Tensor
    composite_attention_mask: torch.Tensor
    image_grid_thw: torch.Tensor
    visual_valid_lengths: torch.Tensor | None = None
    deepstack_visual_embeds: tuple[torch.Tensor, ...] | None = None


@dataclass(frozen=True)
class IncrementalDecodeState:
    """Opaque state chaining `QwenDenseCandidateModel.encode_prefix` into
    repeated `encode_step` calls -- design §20's KV-cache free-running
    decode path. Nothing outside `candidate_model.py`/`free_running_decode.py`
    should construct or inspect one of these; it exists so the two methods
    can hand a caller exactly what the NEXT call needs, without that caller
    (`free_running_decode`) having to know anything about `past_key_values`,
    mRoPE deltas, or cache bookkeeping.

    `composite_len` is the count of composite-range positions already fed
    to the frozen decoder (visual placeholder block, then one more per
    accepted token) -- `encode_step`'s own `cache_position`, in Qwen's
    generation vocabulary. `rope_deltas` is `None` exactly when the
    backbone has no native positional convention (mirrors `InjectedEmbeds.
    position_ids is None`, the stub's case). `context_embeds` is the
    document-context prefix embeddings (`merged_embeds[:, :context_len, :]`,
    computed once by `encode_prefix`/`encode_prefix_with_visual_hidden`) --
    carried here so `encode_step` can pass the same `visual_state` a
    sparse+reasoning `_apply_expert_adapter` call needs, without
    recomputing the prefix on every step."""

    past_key_values: Any
    context_len: int
    composite_len: int
    rope_deltas: torch.Tensor | None
    device: torch.device
    context_embeds: torch.Tensor
    # RFC007 (DEC-047 dormant): `None` unless the model has a level router.
    # The per-page level memory is built once in the prefix pass; the
    # grammar trace advances one token per `encode_step` and is cropped
    # together with the KV cache (`crop`), so `level_router_trace.length ==
    # composite_len` always holds.
    level_router_memory: LevelMemory | None = None
    level_router_trace: GrammarTrace | None = None

    def snapshot(self) -> KVCacheSnapshot:
        """RFC004 §37.4-5 (DEC-047, dormant/opt-in): an immutable handle a
        later `rollback_to` can return to. Records only `composite_len` --
        see `KVCacheSnapshot`'s docstring for why that is everything a
        rollback needs, for either cache backend this module supports.
        Nothing in the production decode path (`free_running_decode.py`)
        calls this."""
        return KVCacheSnapshot(composite_len=self.composite_len)

    def crop(self, n_tokens: int) -> IncrementalDecodeState:
        """Drop the LAST `n_tokens` composite-range positions from
        `past_key_values`, returning a NEW state (DEC-047: additive,
        opt-in, no production call site). `n_tokens=0` returns `self`
        unchanged. Raises `ValueError` for a negative count or a count
        exceeding `composite_len` (nothing before the visual/structural
        prefix this state started from can ever be cropped)."""
        if n_tokens < 0:
            raise ValueError(f"n_tokens must be >= 0, got {n_tokens}")
        if n_tokens > self.composite_len:
            raise ValueError(
                f"cannot crop {n_tokens} tokens from a state with only "
                f"{self.composite_len} composite-range positions"
            )
        if n_tokens == 0:
            return self
        return replace(
            self,
            past_key_values=_crop_kv_cache(self.past_key_values, n_tokens),
            composite_len=self.composite_len - n_tokens,
            level_router_trace=(
                self.level_router_trace.back(n_tokens)
                if self.level_router_trace is not None
                else None
            ),
        )

    def rollback_to(self, snapshot: KVCacheSnapshot) -> IncrementalDecodeState:
        """Undo every `encode_step` call made since `snapshot` was taken --
        the §37.1 "rollback" half of the draft/verify/commit loop. Just
        `self.crop` of the token count `snapshot` is behind by; `crop`
        already refuses a `snapshot` that is not actually behind `self`
        (e.g. one taken on a different, already-superseded state)."""
        return self.crop(self.composite_len - snapshot.composite_len)


@dataclass(frozen=True, slots=True)
class KVCacheSnapshot:
    """Opaque handle from `IncrementalDecodeState.snapshot()`, restorable via
    `IncrementalDecodeState.rollback_to()` (RFC004 §37.4-5, DEC-047: dormant,
    default-off, additive -- the production decode path never constructs
    one). Records only `composite_len`, not a copy of `past_key_values`:
    the real HF `DynamicCache` this wraps in production is one mutable
    object shared and appended-to by every `encode_step` call (`.crop`
    mutates it in place), so "restoring" an earlier point means cropping
    that shared cache's tail back to it, never swapping in a stored copy --
    a stored copy would also defeat the point of a cache."""

    composite_len: int


def _crop_kv_cache(cache: Any, n_tokens: int) -> Any:
    """Remove the last `n_tokens` positions from one decode-state KV cache,
    for either backend `IncrementalDecodeState.past_key_values` holds today:
    an HF `Cache` (duck-typed by its `.crop` method -- verified against the
    installed `transformers` version, not assumed: `DynamicLayer.crop`'s
    negative-argument branch removes exactly `abs(tokens_to_remove)` trailing
    positions, in place, per layer) mutated and returned as the SAME object;
    or `candidate_loader._StubLanguageModel`'s plain
    `tuple[tuple[key, value], ...]` (each tensor `(batch, heads, seq,
    head_dim)`, seq on dim -2), which is never mutated elsewhere in this
    module -- sliced into a NEW tuple instead, since a plain tensor has no
    in-place crop."""
    if hasattr(cache, "crop"):
        cache.crop(-n_tokens)
        return cache
    return tuple((key[..., :-n_tokens, :], value[..., :-n_tokens, :]) for key, value in cache)


def _embedding_rms(weight: torch.Tensor) -> float:
    """RMS of a token-embedding table, in float32 without a float copy of it."""
    norm = torch.linalg.vector_norm(weight.detach(), dtype=torch.float32)
    return float(norm) / weight.numel() ** 0.5


class QwenDenseCandidateModel(nn.Module):
    """The trainable surface of the M2 dense candidate.

    Holds the four trainable submodules (`adapter.projection`,
    `document_context_encoder`, `expert_adapter.dense`, `tree_text_decoder`)
    as real `nn.Module` children, so `.parameters()`/`.named_parameters()`
    enumerate what the registry says this model trains. In addition it
    holds ONE `nn.Module` child per remaining `property_head_registry`
    head_id (`typed_heads.row_cell`, `typed_heads.anchor`,
    `typed_heads.table`, `typed_heads.inline_image`, `table_plan`) -- the
    S3 typed property heads, frozen in S1 (`requires_grad=False`) and
    outside the S1 forward path, so an optimizer built from
    `model.parameters()` cannot reach them before S3, and the S1 logits
    stay bit-identical with or without them (proven in
    `test_candidate_model.py`). It also holds the `node_visual_aligner`
    child (`NodeVisualAlignerHead`, DEC-027): the S2 region/visual aligner,
    frozen in S1 (`requires_grad=False`) and outside the S1 forward path,
    constructed after every S1 module and before the typed-head children so
    its RNG draws cannot shift the S1 draw order. The `expert_adapter.dense`
    head_id is the existing `dense_adapter` child, not a separate module. The
    frozen `QwenCandidate` is held as a plain attribute, not a child module: its
    weights must never appear in this model's own `.parameters()`, or an
    optimizer built from `model.parameters()` would see (and, if
    misconfigured, update) frozen Qwen tensors that belong to
    `backbone.language`, not to any group this model owns.
    """

    def __init__(
        self,
        *,
        decode: DecoderLike,
        inject_visual: InjectVisualLike,
        visual_dim: int,
        d_model: int,
        ranges: PackedRanges,
        qwen_embedding_weight: torch.Tensor,
        qwen_lm_head_weight: torch.Tensor,
        dense_adapter_layers: int = 1,
        document_context_tokens: int = 4,
        typed_integer_config: IntegerHeadConfig = IntegerHeadConfig(strategy="direct_regression"),
        explicit_zero_bin_pointers: frozenset[tuple[str, str]] = frozenset(),
        variant: CandidateArchitectureVariant | None = None,
        deepstack: bool = False,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self._decode = decode
        self._inject_visual = inject_visual
        self.d_model = d_model
        self.ranges = ranges
        # DEC-056: a plain attribute, not a module -- no parameter, no RNG
        # draw, no state_dict key, so a `deepstack=False` model is exactly
        # the v15b model. `True` makes every decoder call over the visual
        # prefix add the batch's DeepStack levels at the placeholder
        # positions (`_deepstack_decode_kwargs`); which checkpoint may run
        # which way is enforced at load (`onfly_visual.require_checkpoint_
        # visual_feature_tap`, `candidate_runtime.require_runtime_deepstack_matches`).
        self.deepstack = bool(deepstack)
        self.image_token_id = image_token_id
        if self.deepstack and image_token_id is None:
            raise ValueError(
                "deepstack=True needs image_token_id to locate the visual positions"
            )
        # DEC-047 dormant switch (W6-1): `None` and `CandidateArchitectureVariant()`
        # are interchangeable and select today's dense/bypass model exactly --
        # every branch keyed off this below constructs NOTHING extra (no new
        # `nn.Module`, no RNG draw) on that default path, so a default-variant
        # model's module tree, RNG draw order and state_dict keys are exactly
        # what they were before this parameter existed.
        self.architecture_variant = (
            variant if variant is not None else CandidateArchitectureVariant()
        )
        self.visual_memory = VisualMemory(visual_dim=visual_dim, d_model=d_model)
        self.document_context_encoder = DocumentContextEncoder(
            d_model=d_model,
            context_tokens=document_context_tokens,
            output_scale=_embedding_rms(qwen_embedding_weight),
        )
        self.dense_adapter = DenseExpertAdapter(d_model=d_model, layers=dense_adapter_layers)
        self.composite_head = CompositeDecodeHead(
            d_model=d_model,
            ranges=ranges,
            qwen_embedding_weight=qwen_embedding_weight,
            qwen_lm_head_weight=qwen_lm_head_weight,
        )
        self.typed_integer_config = typed_integer_config
        # RFC004-ERRATA-006 E1 (R1) / DEC-049 point 2: default `frozenset()`
        # builds every integer head exactly as before this parameter
        # existed (`_build_typed_head_group` below sees an empty set and
        # never overrides `explicit_zero_bin`). Kept as a plain attribute
        # (not re-derived) so a caller (e.g. `s6_train.build_arms`) can
        # assert two models were built with the SAME setting.
        self.explicit_zero_bin_pointers = explicit_zero_bin_pointers
        # The node visual aligner (DEC-027) is constructed after every S1
        # module and before the typed-head children, so an identically-
        # seeded build without it still initialises the S1 modules from the
        # same RNG stream; frozen here -- S2 makes it trainable (the stage
        # state policy's S2 override), the S1 forward never reads it.
        self.node_visual_aligner = NodeVisualAlignerHead(d_model=d_model)
        self.node_visual_aligner.requires_grad_(False)
        # The typed property heads are constructed LAST, after every S1
        # module, so an identically-seeded build without them initialises
        # the S1 modules from the same RNG stream (test_candidate_model
        # proves the S1 surface bit-identical). They are frozen here: the
        # registry puts their groups in `frozen_base` until S3, and the
        # S1 forward never reads them.
        for child_name, head_id in typed_head_children().items():
            group = _build_typed_head_group(
                head_id, d_model, typed_integer_config, explicit_zero_bin_pointers
            )
            group.requires_grad_(False)
            setattr(self, child_name, group)

        # Sparse Word Expert Adapter (RFC004 §22/§23, DEC-047 dormant) and the
        # post-encoder VisualCompressor (RFC004-ERRATA-005 E5, DEC-047
        # dormant) are constructed LAST, after every S1 module AND the typed
        # heads, for the same reason those are constructed after S1 (see the
        # comment above `node_visual_aligner`): an identically-seeded build
        # under the default variant draws NO extra randomness here (both
        # `if`s below are skipped entirely), so every earlier module's
        # weights -- including `dense_adapter`'s, which `from_dense` below
        # copies bit-for-bit into the sparse shared expert -- are identical
        # between a default-variant model and a same-seeded sparse/adaptive
        # variant model. `None` on the default variant -- see
        # `_apply_expert_adapter` / `_apply_post_encoder_compressor`.
        self.sparse_adapter: SparseExpertAdapter | None = None
        self.sparse_reasoning_adapter: SparseExpertAdapterWithReasoning | None = None
        if self.architecture_variant.expert_adapter.mode == "sparse":
            self.sparse_adapter = SparseExpertAdapter.from_dense(
                self.dense_adapter, self.architecture_variant.expert_adapter.sparse_manifest
            )
            if self.architecture_variant.reasoning.enabled:
                self.sparse_reasoning_adapter = SparseExpertAdapterWithReasoning(
                    self.sparse_adapter,
                    spec_id=self.architecture_variant.reasoning.spec_id,
                    attach_point=self.architecture_variant.reasoning.attach_point,
                )
        self.post_encoder_compressor_module: AdaptiveVisualMemoryCompressor | None = None
        if self.deepstack and (
            self.architecture_variant.post_encoder_compressor.profile != "bypass"
        ):
            raise ValueError(
                "deepstack=True is not supported together with a post-encoder "
                "compressor: a merge would leave the DeepStack rows without a "
                "defined position"
            )
        if self.architecture_variant.post_encoder_compressor.profile == "adaptive_v0":
            self.post_encoder_compressor_module = AdaptiveVisualMemoryCompressor(
                d_model, policy=self.architecture_variant.post_encoder_compressor.policy
            )
        # RFC007 variant B (DEC-047 dormant) -- constructed after everything
        # above for the same RNG-order reason; `None` on the default variant,
        # which then draws no randomness and adds no state_dict key.
        self.level_router: DocumentLevelRouter | None = None
        if self.architecture_variant.level_router.enabled:
            self.level_router = DocumentLevelRouter(
                d_model=d_model,
                visual_dim=visual_dim,
                policy=self.architecture_variant.level_router.policy,
            )

    def _apply_post_encoder_compressor(
        self, memory_batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """RFC004-ERRATA-005 E5 post-encoder VisualCompressor slot (DEC-047
        dormant, default off). The default ``"bypass"`` profile
        (`post_encoder_compressor_module is None`) returns `memory_batch`'s
        own tensors completely unchanged -- `_merge_inputs` is byte-identical
        to before this slot existed. ``"adaptive_v0"`` wraps them in the
        `visual_compression.contracts.VisualMemory` the compressor expects
        and returns its `coarse_memory` tensors instead: at step 0
        (`RiskAssignmentHead` zero-init) every token starts its own
        "protected" group, so `coarse_memory` is itself bit-identical to
        `memory_batch` (E5 rule 1; `test_adaptive_v0.py
        ::test_step_zero_matches_bypass_identity`).

        The third element is the compressor's `retention_map` (original token
        -> compressed group, RFC004 §15.2), `None` on bypass;
        `_inject_compressed_visual` reads it to give every merged token its
        anchor's mRoPE position.
        """
        if self.post_encoder_compressor_module is None:
            return memory_batch.memory, memory_batch.key_padding_mask, None
        batch_size = memory_batch.memory.shape[0]
        page_ordinals = torch.arange(batch_size, device=memory_batch.memory.device)
        compressed = self.post_encoder_compressor_module.compress(
            visual_compression_contracts.VisualMemory(
                tokens=memory_batch.memory,
                key_padding_mask=memory_batch.key_padding_mask,
                page_ordinals=page_ordinals,
            )
        )
        return (
            compressed.coarse_memory.tokens,
            compressed.coarse_memory.key_padding_mask,
            compressed.retention_map,
        )

    def _inject_compressed_visual(
        self,
        batch: TeacherForcedBatch,
        original_visual: torch.Tensor,
        compressed_visual: torch.Tensor,
        retention_map: torch.Tensor,
    ) -> tuple[InjectedEmbeds, torch.Tensor, torch.Tensor]:
        """Native injection for a MERGING post-encoder compressor.

        `batch.composite_input_ids` carries one placeholder per ORIGINAL
        visual token (built before the compressor ran -- `candidate_model.py`
        module docstring, assumption 1), and `batch.image_grid_thw` is the
        ORIGINAL grid. Positions therefore come from ONE native
        `inject_visual` call over exactly that original sequence and grid;
        the merged block then replaces the placeholder block:

        - compressed token `g` takes the mRoPE id of its ANCHOR, the first
          original token of its group (`min n: retention_map[n] == g`), so a
          merged token sits where its group starts in the page geometry the
          frozen decoder was trained on (VC-1 rehearsal rule: anchor ids, no
          compaction, no rescaled grid);
        - every position after the block keeps its uncompressed id;
        - `rope_deltas` grows by the number of removed tokens, so
          `encode_step`'s `rope_deltas + composite_len` continues exactly
          where the uncompressed sequence would.

        This replaces the old `(1, 1, M)` grid rewrite: Qwen3-VL's
        `get_rope_index` divides H and W by `spatial_merge_size` (2), so that
        grid produced `1 // 2 == 0` rows -- zero positions for M tokens -- and
        failed on the real backbone; the stub, which has no positions, never
        showed it (`test_candidate_model_post_encoder_merge_positions.py`).

        Any group structure works (regular grid merge or `adaptive_v0`'s
        irregular same-class runs): only `retention_map` is read. Single page
        only (`batch == 1`, the compressor's own restriction), and the
        placeholder block must be one contiguous run at the front (checked).
        Returns `(injected, composite_input_ids, composite_attention_mask)`
        of the compressed composite range.
        """
        ids = batch.composite_input_ids
        n_visual = batch.visual_features.shape[1]
        n_kept = compressed_visual.shape[0]
        if ids.shape[0] != 1:
            raise ValueError(f"a merging compressor supports batch == 1, got {ids.shape[0]}")
        placeholder = ids[0, 0]
        if not bool((ids[:, :n_visual] == placeholder).all()) or bool(
            (ids[:, n_visual:] == placeholder).any()
        ):
            raise ValueError("visual placeholders must be one contiguous block at the front")
        retention_map = retention_map.to(ids.device)
        if retention_map.shape != (n_visual,) or bool((retention_map < 0).any()) or bool(
            (retention_map >= n_kept).any()
        ):
            raise ValueError(
                f"retention_map of shape {tuple(retention_map.shape)} does not map "
                f"{n_visual} original tokens onto {n_kept} compressed tokens"
            )
        anchors = torch.full((n_kept,), n_visual, dtype=torch.long, device=ids.device)
        anchors.scatter_reduce_(
            0, retention_map, torch.arange(n_visual, device=ids.device), reduce="amin"
        )
        if bool((anchors == n_visual).any()):
            raise ValueError("retention_map leaves a compressed token without any original token")

        full_embeds = self.composite_head.embed_packed(ids)
        full = self._inject_visual(
            inputs_embeds=full_embeds,
            visual_embeds=original_visual,
            native_id_sequence=_native_id_view(ids, self.ranges),
            image_grid_thw=batch.image_grid_thw,
            attention_mask=batch.composite_attention_mask,
        )
        inputs_embeds = torch.cat(
            [
                compressed_visual.unsqueeze(0).to(full.inputs_embeds.dtype),
                full.inputs_embeds[:, n_visual:],
            ],
            dim=1,
        )
        composite_input_ids = torch.cat([ids[:, :1].expand(-1, n_kept), ids[:, n_visual:]], dim=1)
        mask = batch.composite_attention_mask
        composite_attention_mask = torch.cat(
            [mask[:, :1].expand(-1, n_kept), mask[:, n_visual:]], dim=1
        )
        position_ids = None
        rope_deltas = None
        if full.position_ids is not None:
            position_ids = torch.cat(
                [full.position_ids[:, :, anchors], full.position_ids[:, :, n_visual:]], dim=-1
            )
            if full.rope_deltas is not None:
                rope_deltas = full.rope_deltas + (n_visual - n_kept)
        injected = InjectedEmbeds(
            inputs_embeds=inputs_embeds, position_ids=position_ids, rope_deltas=rope_deltas
        )
        return injected, composite_input_ids, composite_attention_mask

    def _apply_expert_adapter(
        self, hidden: torch.Tensor, *, visual_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """RFC004 §22 Word Expert Adapter slot (DEC-047 dormant, default off).

        ``"dense"`` (default) calls `self.dense_adapter` exactly as before --
        byte-identical to this method not existing. ``"sparse"`` calls the
        `SparseExpertAdapter` built from that SAME dense adapter via
        `from_dense`, whose step-0 output equals the dense adapter's output
        exactly (`sparse_expert_adapter/adapter.py` module docstring,
        condition 2). ``"sparse"`` with reasoning enabled additionally runs
        `SparseExpertAdapterWithReasoning`, whose specialization ends in a
        zero-initialized output projection (RFC004-ERRATA-005 E1 rule 4), so
        reasoning on/off agree bit-for-bit at step 0 too. `eligibility_mask`
        is all-ones here: this dormant switch carries no grammar-confirmed
        context at the model boundary yet, so nothing is masked ineligible.

        Wired into every hidden-state readout this model has: the
        full-recompute training path (`encode_hidden`) AND the KV-cache
        incremental decode paths (`encode_prefix`, `encode_prefix_with_
        visual_hidden`, `encode_step`) -- all four call sites pass the SAME
        `visual_state` (the document-context prefix embeddings,
        `merged_embeds[:, :context_len, :]`, carried across `encode_step`
        calls via `IncrementalDecodeState.context_embeds` since that prefix
        is only ever computed once, in `encode_prefix`/`encode_prefix_with_
        visual_hidden`) and `attention_mask` shape. This is causal-safe by
        construction, not by convention: `SparseExpertAdapter` is per-token
        and batch-independent (`sparse_expert_adapter/adapter.py`'s module
        docstring -- router, top-k and softmax all reduce only over the
        expert axis, never across positions), and `SparseExpertAdapterWith
        Reasoning`'s gate (`reasoning/gate.py`, one `nn.Linear` per position)
        and reasoning module both read only the CURRENT position's hidden
        state plus `visual_state` -- a fixed prefix that never depends on
        which composite tokens have been decoded yet. So a position `t`'s
        output depends only on positions `<= t`, whether computed inside one
        `encode_hidden` call over the whole sequence or one `encode_step`
        call per position against a KV cache (`test_candidate_model_
        architecture_variant.py`'s incremental-decode-equivalence tests).
        """
        if self.architecture_variant.expert_adapter.mode == "dense":
            return self.dense_adapter(hidden)
        assert self.sparse_adapter is not None
        if self.sparse_reasoning_adapter is None:
            return self.sparse_adapter(hidden)
        eligibility_mask = torch.ones(hidden.shape[:-1], dtype=torch.float32, device=hidden.device)
        result = self.sparse_reasoning_adapter(
            hidden,
            visual_state=visual_state,
            attention_mask=attention_mask,
            eligibility_mask=eligibility_mask,
        )
        return result.hidden

    def _level_router_memory(self, batch: TeacherForcedBatch) -> LevelMemory | None:
        """RFC007: the per-page level memory, or `None` without a router."""
        if self.level_router is None:
            return None
        levels, key_padding_mask = gather_level_features(
            self.level_router.policy,
            visual_features=batch.visual_features,
            visual_valid_lengths=batch.visual_valid_lengths,
            deepstack_visual_embeds=batch.deepstack_visual_embeds,
        )
        return self.level_router.build_memory(levels, key_padding_mask)

    def _level_router_trace(self, composite_input_ids: torch.Tensor) -> GrammarTrace | None:
        """RFC007: the grammar trace after the prefix's composite tokens
        (single-page incremental decode only, like `encode_step`)."""
        if self.level_router is None:
            return None
        if composite_input_ids.shape[0] != 1:
            raise ValueError("the level router's incremental decode path is single-page (batch 1)")
        trace = GrammarTrace.start()
        for packed_id in composite_input_ids[0].tolist():
            trace = trace.advance(packed_id, self.ranges)
        return trace

    def _level_router_state_ids(
        self, composite_input_ids: torch.Tensor, context_len: int
    ) -> torch.Tensor:
        """`(batch, context_len + steps)` grammar-state ids; the document-
        context prefix positions get the empty state (0)."""
        assert self.level_router is not None
        policy = self.level_router.policy
        rows = []
        for row in composite_input_ids.tolist():
            trace = GrammarTrace.start()
            ids = [0] * context_len
            for packed_id in row:
                trace = trace.advance(packed_id, self.ranges)
                ids.append(trace.state.state_id(policy))
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long, device=composite_input_ids.device)

    def _trace_state_ids(self, *traces: GrammarTrace) -> torch.Tensor:
        """`(1, len(traces))` state ids for the incremental paths."""
        assert self.level_router is not None
        policy = self.level_router.policy
        device = self.level_router.output.weight.device
        return torch.tensor(
            [[trace.state.state_id(policy) for trace in traces]], dtype=torch.long, device=device
        )

    def _apply_level_router(
        self, hidden: torch.Tensor, state_ids: torch.Tensor, memory: LevelMemory | None
    ) -> torch.Tensor:
        """RFC007 variant B slot, right after `_apply_expert_adapter`. Every
        call site is behind a `level_router is not None` check, so the default
        path never reaches this method and stays byte-identical."""
        assert self.level_router is not None and memory is not None
        return hidden + self.level_router(hidden, state_ids, memory).delta.to(hidden.dtype)

    def _merge_inputs(
        self, batch: TeacherForcedBatch
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, int, torch.Tensor | None, int
    ]:
        """The embedding-construction pipeline `encode_hidden` and
        `encode_prefix` (design §20) share exactly: visual memory, document
        context, composite embed, native scatter, then the context-length
        position-id shift. Returns `(merged_embeds, merged_attention_mask,
        position_ids, context_len, rope_deltas, composite_visual_token_count)``
        -- everything the frozen `decode` call needs, one level BEFORE that
        call, so the two callers differ only in how they call `decode`
        (full recompute vs. cache-building), never in how they build its
        input. `composite_visual_token_count` is the per-sample number of
        visual placeholder positions actually scattered into the composite
        range (a single scalar shared by every row, exactly like
        `context_len` -- one composite range length per batch tensor) --
        equal to `batch.visual_features.shape[1]` unless a merging
        post-encoder compressor shrank it (`_inject_compressed_visual`);
        `encode_prefix_with_visual_hidden` needs the real count to slice out
        the visual hidden-state span for the Node Visual Aligner.
        """
        memory_batch = self.visual_memory(
            batch.visual_features, valid_lengths=batch.visual_valid_lengths
        )
        visual_tokens, visual_key_padding_mask, retention_map = (
            self._apply_post_encoder_compressor(memory_batch)
        )
        context = self.document_context_encoder(
            visual_tokens, key_padding_mask=visual_key_padding_mask
        )
        # Only the VALID visual rows, flattened in (batch, token) row-major
        # order -- exactly the order `inject_visual`'s scatter expects. A
        # padded visual row must never occupy a real placeholder slot.
        valid_visual_embeds = visual_tokens[~visual_key_padding_mask]

        # Per-sample token count, NOT `valid_visual_embeds.shape[0]` (that is
        # flattened across the whole batch, so comparing it against
        # `batch.visual_features.shape[1]` -- itself per-sample -- would
        # wrongly read an unpadded `batch_size > 1` bypass batch as a merge).
        # A merging compressor only ever produces `batch == 1`
        # (`AdaptiveVisualMemoryCompressor.compress`'s own restriction). No
        # merge (bypass, step-0 zero-init risk head, every ratio 1) keeps the
        # original path byte-identical.
        if retention_map is None or visual_tokens.shape[1] == batch.visual_features.shape[1]:
            composite_input_ids = batch.composite_input_ids
            composite_attention_mask = batch.composite_attention_mask
            composite_embeds = self.composite_head.embed_packed(composite_input_ids)
            native_id_sequence = _native_id_view(composite_input_ids, self.ranges)
            injected = self._inject_visual(
                inputs_embeds=composite_embeds,
                visual_embeds=valid_visual_embeds,
                native_id_sequence=native_id_sequence,
                image_grid_thw=batch.image_grid_thw,
                attention_mask=composite_attention_mask,
            )
        else:
            injected, composite_input_ids, composite_attention_mask = (
                self._inject_compressed_visual(
                    batch,
                    memory_batch.memory[~memory_batch.key_padding_mask],
                    valid_visual_embeds,
                    retention_map,
                )
            )

        context_len = context.shape[1]
        merged_embeds = torch.cat([context, injected.inputs_embeds], dim=1)
        context_attention = torch.ones(
            context.shape[:2], dtype=composite_attention_mask.dtype, device=context.device
        )
        merged_attention_mask = torch.cat(
            [context_attention, composite_attention_mask], dim=1
        )

        position_ids = None
        if injected.position_ids is not None:
            batch_size = composite_input_ids.shape[0]
            context_positions = (
                torch.arange(context_len, device=merged_embeds.device)
                .view(1, 1, -1)
                .expand(3, batch_size, -1)
            )
            # `inject_visual`'s position ids assume the composite range
            # starts at absolute position 0; shifting the whole (3, batch,
            # steps) tensor by a constant `context_len` is valid for mRoPE
            # exactly the way Qwen's own incremental-generation `delta`
            # shift already is (candidate_loader.py's `get_rope_index`).
            shifted_composite_positions = injected.position_ids + context_len
            position_ids = torch.cat([context_positions, shifted_composite_positions], dim=-1)

        return (
            merged_embeds,
            merged_attention_mask,
            position_ids,
            context_len,
            injected.rope_deltas,
            visual_tokens.shape[1],
        )

    def _deepstack_decode_kwargs(
        self, batch: TeacherForcedBatch, context_len: int
    ) -> dict[str, Any]:
        """DEC-056: the `visual_pos_masks`/`deepstack_visual_embeds` pair for
        a decoder call over the WHOLE merged sequence, or `{}` for a
        non-DeepStack model -- so a v15b model's decoder calls keep exactly
        their old keyword set.

        The mask is `False` over the `context_len` document-context prefix
        and `True` at every composite placeholder (`image_token_id`), i.e.
        exactly the rows `inject_visual` scattered the visual tokens into;
        HF's `_deepstack_process` reads `hidden[mask]` row-major, the same
        order the levels' rows are in. A mismatch in either direction
        (DeepStack model without levels, or levels for a model that never
        trained with them) is refused, never silently dropped or added."""
        levels = batch.deepstack_visual_embeds
        if levels is None:
            if self.deepstack:
                raise ValueError(
                    "this model was built with deepstack=True but the batch carries "
                    "no DeepStack levels -- a DeepStack checkpoint never runs without them"
                )
            return {}
        if not self.deepstack:
            if self.level_router is not None:
                # RFC007 pilot arm "router on, native DeepStack off": the level
                # router consumes the levels as query-side memories
                # (`_level_router_memory`); nothing is injected natively.
                return {}
            raise ValueError(
                "the batch carries DeepStack levels but this model was built with "
                "deepstack=False -- a checkpoint trained without DeepStack never runs with it"
            )
        image_mask = _native_id_view(batch.composite_input_ids, self.ranges) == int(
            self.image_token_id  # type: ignore[arg-type]
        )
        n_visual = int(image_mask.sum())
        for index, level in enumerate(levels):
            if level.dim() != 2 or level.shape[0] != n_visual:
                raise ValueError(
                    f"DeepStack level {index} has shape {tuple(level.shape)}, expected "
                    f"({n_visual}, hidden) -- one row per visual placeholder"
                )
        prefix = torch.zeros(
            image_mask.shape[0], context_len, dtype=torch.bool, device=image_mask.device
        )
        return {
            "visual_pos_masks": torch.cat([prefix, image_mask], dim=1),
            "deepstack_visual_embeds": tuple(level.to(image_mask.device) for level in levels),
        }

    def encode_hidden(self, batch: TeacherForcedBatch) -> tuple[torch.Tensor, int]:
        """`(batch, prefix_len + steps, d_model)` dense-adapted decoder hidden
        state over the WHOLE merged sequence -- `document_context_encoder`'s
        summary positions, then the composite-token positions (with visual
        content natively scattered INTO that range by `inject_visual`, not
        occupying a separate span) -- plus `prefix_len`, the number of
        non-composite positions at the front.

        `forward` slices this down to the composite range for training
        logits (unchanged shape/behaviour). This is the FULL-RECOMPUTE
        path: every call rebuilds and re-decodes the whole sequence from
        scratch, still used by training (needs gradients through every
        position) and by `free_running_decode`'s own non-cache fallback
        (design §20's equivalence test runs this arm as the reference).
        `encode_prefix`/`encode_step` are the KV-cache alternative for
        inference-only decode.
        """
        merged_embeds, merged_attention_mask, position_ids, context_len, _rope_deltas, _n_visual = (
            self._merge_inputs(batch)
        )

        # Gradient checkpointing across the FROZEN decode call (2026-09-04
        # finding): gradient must still flow BACK through `self._decode`
        # to reach the trainable `composite_head` embeddings that feed
        # into it, even though `_decode`'s own weights get no grad -- so
        # autograd has to retain every one of its 36 layers' activations
        # for the backward pass regardless of freezing. Measured: one real
        # page (875 visual tokens, `grid_thw=(1,70,50)`, a typical page at
        # this resolution, not an outlier) OOMs an RTX 4070 (12 GB) doing
        # this with activations retained, on top of the resident 8.3 GB
        # bf16 candidate. Checkpointing recomputes `_decode`'s forward
        # during backward instead of storing its activations -- textbook
        # for fitting 4B-class training on a consumer GPU, and it changes
        # no numerical result (`use_reentrant=False` supports the
        # `DecoderLike` object return; see `test_candidate_model.py`'s
        # before/after comparison on a case that fits without it). This is
        # a training-time backward-pass strategy, not a model or decode
        # config value -- it touches no field the RFC004 §39/RFC005 §14.4
        # Component Lock's `decode_config` kind binds.
        decoded = _grad_checkpoint(
            self._decode,
            use_reentrant=False,
            inputs_embeds=merged_embeds,
            attention_mask=merged_attention_mask,
            position_ids=position_ids,
            **self._deepstack_decode_kwargs(batch, context_len),
        )
        # `decode`'s frozen backend returns its own compute dtype (bf16
        # under DEC-007 D3), and `dense_adapter` (`nn.Linear`, unlike
        # `nn.LayerNorm`) has no implicit dtype promotion -- it raises
        # rather than silently upcasting. DEC-007 D3.1: the trainable
        # groups' storage dtype is an implementation decision, not part of
        # the pin, so this cast reads `dense_adapter`'s OWN dtype rather
        # than hardcoding fp32 -- whatever precision the trainable side is
        # actually configured with is what the frozen output gets cast to.
        trainable_dtype = self.dense_adapter.layers[0].norm.weight.dtype
        hidden = decoded.last_hidden_state.to(trainable_dtype)
        adapted = self._apply_expert_adapter(
            hidden,
            visual_state=merged_embeds[:, :context_len, :],
            attention_mask=merged_attention_mask,
        )
        if self.level_router is not None:
            adapted = self._apply_level_router(
                adapted,
                self._level_router_state_ids(batch.composite_input_ids, context_len),
                self._level_router_memory(batch),
            )
        return adapted, context_len

    def encode_prefix(
        self,
        batch: TeacherForcedBatch,
        *,
        prefill_chunk_tokens: int | None = None,
        initial_cache: Any | None = None,
    ) -> tuple[IncrementalDecodeState, torch.Tensor]:
        """KV-cache PREFIX pass (design §20). Builds the SAME merged
        embeddings `encode_hidden` would (`_merge_inputs`, shared), but
        forwards them through the frozen decoder ONCE with `use_cache=True`
        instead of every step -- `free_running_decode`'s incremental path
        calls this exactly once, with `batch.composite_input_ids` equal to
        the fixed visual placeholder block (no accepted tokens yet, matching
        the free-running loop's own step-0 input).

        Returns the opaque `IncrementalDecodeState` `encode_step` continues
        from, and the dense-adapted hidden state at the LAST prefix
        position -- exactly what `encode_hidden`'s caller used to read out
        of `hidden[:, -1, :]` for its first decode decision.

        Inference-only: no `_grad_checkpoint` here (unlike `encode_hidden`)
        -- `free_running_decode` wraps its whole loop in
        `torch.inference_mode()`, so there is no backward pass whose
        activations checkpointing would be protecting; adding it would only
        cost a wasted recomputation this path has no use for.

        `prefill_chunk_tokens` (W0-11, opt-in): `None` (default) is exactly
        today's behaviour -- one `self._decode` call over the whole merged
        sequence, byte-identical to before this parameter existed. A
        positive int instead feeds the same merged embeddings through
        `self._decode` in successive chunks of that many positions,
        chaining `past_key_values` between chunks (`_prefill_chunked`) --
        the sole reason a ~8,580-token 300 DPI prefix OOMs a 12 GiB GPU is
        that a single `use_cache=True` forward materialises attention state
        for the WHOLE prefix at once; chunking bounds that peak at the cost
        of feeding the prefix in more than one decoder call. The visual
        scatter (`inject_visual`) already ran inside `_merge_inputs` before
        this ever sees the embeddings, so a chunk boundary can fall anywhere
        in the merged sequence without splitting anything the frozen decoder
        treats as one unit.

        `initial_cache` (owner requirement 2026-09-27, quantized KV cache):
        `None` (default) is exactly today's behaviour -- `self._decode`
        builds its own cache internally. Passing a `Float8KVCache()`
        instance instead seeds the prefix pass with it, so the WHOLE decode
        (prefix + every `encode_step`) accumulates K/V in that cache object
        rather than a full-precision `DynamicCache`. Refused together with
        `prefill_chunk_tokens` -- `_prefill_chunked` chains its own
        `past_key_values` across chunk calls and has never been taught to
        seed from a caller-supplied cache, so silently ignoring
        `initial_cache` there would be a silent behaviour change."""
        if initial_cache is not None and prefill_chunk_tokens is not None:
            raise ValueError(
                "initial_cache is not supported together with prefill_chunk_tokens"
            )
        merged_embeds, merged_attention_mask, position_ids, context_len, rope_deltas, _n_visual = (
            self._merge_inputs(batch)
        )
        deepstack_kwargs = self._deepstack_decode_kwargs(batch, context_len)
        if prefill_chunk_tokens is None:
            decoded = self._decode(
                inputs_embeds=merged_embeds,
                attention_mask=merged_attention_mask,
                position_ids=position_ids,
                past_key_values=initial_cache,
                use_cache=True,
                **deepstack_kwargs,
            )
            last_hidden_state = decoded.last_hidden_state
            past_key_values = decoded.past_key_values
        else:
            last_hidden_state, past_key_values = self._prefill_chunked(
                merged_embeds,
                merged_attention_mask,
                position_ids,
                chunk_tokens=prefill_chunk_tokens,
                deepstack_kwargs=deepstack_kwargs,
            )
        trainable_dtype = self.dense_adapter.layers[0].norm.weight.dtype
        context_embeds = merged_embeds[:, :context_len, :]
        last_hidden = self._apply_expert_adapter(
            last_hidden_state[:, -1:, :].to(trainable_dtype),
            visual_state=context_embeds,
            attention_mask=merged_attention_mask,
        )
        router_memory = self._level_router_memory(batch)
        router_trace = self._level_router_trace(batch.composite_input_ids)
        if router_trace is not None:
            last_hidden = self._apply_level_router(
                last_hidden, self._trace_state_ids(router_trace), router_memory
            )
        last_hidden = last_hidden[:, 0, :]
        state = IncrementalDecodeState(
            past_key_values=past_key_values,
            context_len=context_len,
            # NOT `batch.composite_input_ids.shape[1]` -- a merging
            # post-encoder compressor shrinks the composite range built by
            # `_merge_inputs` below what the caller's own batch tensor
            # started with (`_inject_compressed_visual`); reading
            # the length back off `merged_attention_mask` (minus the
            # context prefix) reports what was actually fed to `_decode`.
            composite_len=merged_attention_mask.shape[1] - context_len,
            rope_deltas=rope_deltas,
            device=merged_embeds.device,
            context_embeds=context_embeds,
            level_router_memory=router_memory,
            level_router_trace=router_trace,
        )
        return state, last_hidden

    def encode_prefix_with_visual_hidden(
        self,
        batch: TeacherForcedBatch,
        *,
        prefill_chunk_tokens: int | None = None,
        initial_cache: Any | None = None,
    ) -> tuple[IncrementalDecodeState, torch.Tensor, torch.Tensor]:
        """Run the KV prefix once and retain its dense visual hidden states.

        Region decoding is opt-in, so the long-standing two-value
        ``encode_prefix`` fake-model API remains untouched.  This method uses
        the same one decoder invocation as the ordinary prefix path.

        `prefill_chunk_tokens`: same opt-in chunked prefill as
        `encode_prefix` (see its docstring); `None` keeps this byte-identical
        to before the parameter existed.

        `initial_cache` (2026-09-28, owner requirement: region-decode pages
        must be able to use an alternate KV cache too, e.g. quantized or
        preallocated): same contract as `encode_prefix`'s own
        `initial_cache` -- `None` (default) is exactly today's behaviour,
        `self._decode` builds its own cache. Refused together with
        `prefill_chunk_tokens` for the same reason `encode_prefix` refuses
        it (`_prefill_chunked` never learned to seed from a caller-supplied
        cache)."""
        if initial_cache is not None and prefill_chunk_tokens is not None:
            raise ValueError(
                "initial_cache is not supported together with prefill_chunk_tokens"
            )
        merged_embeds, merged_attention_mask, position_ids, context_len, rope_deltas, n_visual = (
            self._merge_inputs(batch)
        )
        deepstack_kwargs = self._deepstack_decode_kwargs(batch, context_len)
        if prefill_chunk_tokens is None:
            decoded = self._decode(
                inputs_embeds=merged_embeds,
                attention_mask=merged_attention_mask,
                position_ids=position_ids,
                past_key_values=initial_cache,
                use_cache=True,
                **deepstack_kwargs,
            )
            last_hidden_state = decoded.last_hidden_state
            past_key_values = decoded.past_key_values
        else:
            last_hidden_state, past_key_values = self._prefill_chunked(
                merged_embeds,
                merged_attention_mask,
                position_ids,
                chunk_tokens=prefill_chunk_tokens,
                deepstack_kwargs=deepstack_kwargs,
            )
        trainable_dtype = self.dense_adapter.layers[0].norm.weight.dtype
        context_embeds = merged_embeds[:, :context_len, :]
        dense_hidden = self._apply_expert_adapter(
            last_hidden_state.to(trainable_dtype),
            visual_state=context_embeds,
            attention_mask=merged_attention_mask,
        )
        router_memory = self._level_router_memory(batch)
        router_trace = self._level_router_trace(batch.composite_input_ids)
        if router_trace is not None:
            dense_hidden = self._apply_level_router(
                dense_hidden,
                self._level_router_state_ids(batch.composite_input_ids, context_len),
                router_memory,
            )
        state = IncrementalDecodeState(
            past_key_values=past_key_values,
            context_len=context_len,
            # See `encode_prefix`'s identical comment: NOT
            # `batch.composite_input_ids.shape[1]`.
            composite_len=merged_attention_mask.shape[1] - context_len,
            rope_deltas=rope_deltas,
            device=merged_embeds.device,
            context_embeds=context_embeds,
            level_router_memory=router_memory,
            level_router_trace=router_trace,
        )
        # NOT `batch.visual_features.shape[1]` -- a merging post-encoder
        # compressor shrinks the visual placeholder span actually scattered
        # into the composite range (`_merge_inputs`'s `n_visual`); reading
        # the ORIGINAL token count here would slice the wrong span out of
        # `dense_hidden` and hand the Node Visual Aligner state that no
        # longer lines up with `retention_map`'s own indexing.
        visual_hidden = dense_hidden[:, context_len : context_len + n_visual, :]
        return state, dense_hidden[:, -1, :], visual_hidden

    def _prefill_chunked(
        self,
        merged_embeds: torch.Tensor,
        merged_attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        *,
        chunk_tokens: int,
        deepstack_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """W0-11: feed `merged_embeds` through `self._decode` in successive
        chunks of `chunk_tokens` positions instead of one call, chaining
        `past_key_values` exactly the way `encode_step`'s own single-token
        continuation already does -- `attention_mask` sliced to the full
        history-so-far, `position_ids`/`inputs_embeds` sliced to only the
        NEW chunk, matching the `DecoderLike` contract every other
        incremental call in this module already relies on. Returns the
        per-chunk `last_hidden_state`s concatenated back into one
        `(batch, total_steps, hidden)` tensor (so callers that need the
        FULL sequence, e.g. `encode_prefix_with_visual_hidden`, see exactly
        what a single un-chunked call would have produced) and the final
        `past_key_values`.

        `deepstack_kwargs` (DEC-056): the whole-sequence pair from
        `_deepstack_decode_kwargs`; each chunk gets its own slice of the mask
        and exactly the level rows whose positions fall inside it (a chunk
        with no visual position gets none), so a chunked DeepStack prefill
        adds the same residual at the same positions as an unchunked one."""
        if chunk_tokens <= 0:
            raise ValueError(f"prefill_chunk_tokens must be positive, got {chunk_tokens}")
        total = merged_embeds.shape[1]
        past_key_values: Any = None
        hidden_chunks: list[torch.Tensor] = []
        row_of: torch.Tensor | None = None
        if deepstack_kwargs:
            full_mask = deepstack_kwargs["visual_pos_masks"]
            row_of = (full_mask.reshape(-1).long().cumsum(0) - 1).reshape(full_mask.shape)
        for start in range(0, total, chunk_tokens):
            end = min(start + chunk_tokens, total)
            chunk_position_ids = (
                position_ids[:, :, start:end] if position_ids is not None else None
            )
            chunk_deepstack: dict[str, Any] = {}
            if row_of is not None and deepstack_kwargs:
                chunk_mask = deepstack_kwargs["visual_pos_masks"][:, start:end]
                if bool(chunk_mask.any()):
                    rows = row_of[:, start:end][chunk_mask]
                    chunk_deepstack = {
                        "visual_pos_masks": chunk_mask,
                        "deepstack_visual_embeds": tuple(
                            level[rows] for level in deepstack_kwargs["deepstack_visual_embeds"]
                        ),
                    }
            decoded = self._decode(
                inputs_embeds=merged_embeds[:, start:end, :],
                attention_mask=merged_attention_mask[:, :end],
                position_ids=chunk_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                **chunk_deepstack,
            )
            past_key_values = decoded.past_key_values
            hidden_chunks.append(decoded.last_hidden_state)
        return torch.cat(hidden_chunks, dim=1), past_key_values

    def encode_step(
        self,
        state: IncrementalDecodeState,
        new_token_id: int,
        *,
        hidden_taps: tuple[int, ...] | None = None,
    ) -> (
        tuple[IncrementalDecodeState, torch.Tensor]
        | tuple[IncrementalDecodeState, torch.Tensor, tuple[torch.Tensor, ...]]
    ):
        """KV-cache STEP pass (design §20). Forwards exactly the ONE new
        composite token against `state.past_key_values`, extending it --
        never rebuilds the visual scatter or re-embeds anything already in
        the cache.

        Position id continuation is design §20's named hazard:
        `inject_visual`'s native mRoPE ids are computed once, in
        `encode_prefix`, for the composite range starting at local index 0.
        `state.composite_len` is exactly the count of composite-range
        positions already fed to the decoder (`cache_position`, in Qwen's
        own generation vocabulary), so `state.rope_deltas + state.
        composite_len` -- broadcast identically across all 3 mRoPE axes,
        valid for pure-text continuation past the visual block (the
        `get_rope_index` formula this mirrors: `mrope_position_deltas =
        llm_positions.max() + 1 - len(input_ids)`, then continuation
        position = `cache_position + delta`) -- is exactly the id
        `inject_visual` would have assigned this token had the WHOLE
        sequence been run through it at once. Shifted by `state.
        context_len` to land in the same absolute frame `encode_hidden`/
        `encode_prefix` already use. `None` when the backbone has no
        native positional convention (the stub), matching every other
        `position_ids=None` fallback in this module.

        `hidden_taps` (DSpark multi-layer conditioning, 2026-09-27; additive
        opt-in): `None` (default, every existing caller) is byte-identical
        to before this parameter existed -- `self._decode` is called exactly
        as before and this returns the same 2-tuple. A tuple of layer
        indices instead asks `self._decode` for its full per-layer stack
        (`output_hidden_states=True`) and additionally returns a 3rd tuple
        element: the FROZEN (pre-`dense_adapter`) hidden state at each
        requested layer, at this one new position, in the order given --
        `decoded.hidden_states[i]` is HF's own convention (entry 0 is the
        embedding output, entry i>0 is decoder layer i's output)."""
        composite_embeds = self.composite_head.embed_packed(
            torch.tensor([[new_token_id]], dtype=torch.long, device=state.device)
        )
        attention_mask = torch.ones(
            1, state.context_len + state.composite_len + 1, dtype=torch.long, device=state.device
        )
        position_ids = None
        if state.rope_deltas is not None:
            local_position = state.rope_deltas + state.composite_len  # (batch, 1)
            position_ids = (
                local_position.view(1, 1, -1).expand(3, local_position.shape[0], -1)
                + state.context_len
            )
        decoded = self._decode(
            inputs_embeds=composite_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=state.past_key_values,
            use_cache=True,
            output_hidden_states=hidden_taps is not None,
        )
        trainable_dtype = self.dense_adapter.layers[0].norm.weight.dtype
        hidden = self._apply_expert_adapter(
            decoded.last_hidden_state.to(trainable_dtype),
            visual_state=state.context_embeds,
            attention_mask=attention_mask,
        )
        router_trace = None
        if state.level_router_trace is not None:
            router_trace = state.level_router_trace.advance(new_token_id, self.ranges)
            hidden = self._apply_level_router(
                hidden, self._trace_state_ids(router_trace), state.level_router_memory
            )
        hidden = hidden[:, 0, :]
        new_state = IncrementalDecodeState(
            past_key_values=decoded.past_key_values,
            context_len=state.context_len,
            composite_len=state.composite_len + 1,
            rope_deltas=state.rope_deltas,
            device=state.device,
            context_embeds=state.context_embeds,
            level_router_memory=state.level_router_memory,
            level_router_trace=router_trace,
        )
        if hidden_taps is None:
            return new_state, hidden
        taps = tuple(decoded.hidden_states[i][:, 0, :] for i in hidden_taps)
        return new_state, hidden, taps

    def encode_steps(
        self,
        state: IncrementalDecodeState,
        new_token_ids: Sequence[int],
        *,
        hidden_taps: tuple[int, ...] | None = None,
    ) -> (
        tuple[torch.Tensor, IncrementalDecodeState]
        | tuple[torch.Tensor, IncrementalDecodeState, tuple[torch.Tensor, ...]]
    ):
        """RFC004 §37.1/§37.4 (DEC-047 dormant): append `len(new_token_ids)`
        new composite tokens to `state.past_key_values` in ONE decoder call
        -- the mechanism a speculative block VERIFY needs to ever be faster
        than `len(new_token_ids)` successive `encode_step` calls (today's
        one-token-per-forward-pass `encode_step`, which makes MTP/DFlash
        unable to ever be faster). Reuses the same chunked-prefill mechanism
        `_prefill_chunked`/`encode_step` already rely on: `attention_mask`
        covers the full history-so-far plus this new chunk, `position_ids`/
        `inputs_embeds` cover only the new chunk, continuing `state.
        rope_deltas` the same way `encode_step` already does (see its own
        docstring) but for `len(new_token_ids)` consecutive local offsets
        at once instead of one.

        Returns `(hidden_per_position, new_state)`: `hidden_per_position` is
        `(batch, len(new_token_ids), d_model)`, the dense-adapted hidden
        state at EACH new position, in order -- position `i` is exactly what
        `len(new_token_ids)` successive `encode_step` calls would have
        produced as the i-th call's own return value (`test_candidate_
        model_encode_steps.py`'s `allclose` equivalence). `new_state` is
        chained past all `len(new_token_ids)` tokens, so a caller can
        `.crop()` it back to fewer if only a prefix of the block is
        eventually accepted (RFC004 §37.1's rollback).

        `new_token_ids` must be non-empty -- a zero-token call has no
        defined hidden-state shape to return.

        `hidden_taps` (DSpark multi-layer conditioning, 2026-09-27;
        additive opt-in, same contract as `encode_step`'s parameter of the
        same name): `None` (default) is byte-identical to before this
        parameter existed. A tuple of layer indices returns a 3rd tuple
        element, `(len(hidden_taps), batch, k, d_model)` stacked FROZEN
        (pre-`dense_adapter`) hidden states -- position `i` along the `k`
        axis is exactly what `encode_step`'s own `hidden_taps` return would
        have produced as the i-th call's tap.

        DSpark rehearsal 2026-09-28 (`docs/runs/dspark-rehearsal-
        2026-09-27.md` §16): `self._decode` is called with `attention_mask=
        None`, not the all-ones tensor below -- this decode is always
        single-page with no padding, so that tensor is always fully valid,
        and `transformers/integrations/sdpa_attention.py` only takes the
        fast flash-decode kernel for k>1 when `attn_mask is None`
        (`is_causal=True` otherwise falls back to the ~10x slower memory-
        efficient attention kernel, measured on the real backbone). The
        all-ones tensor itself is kept for `_apply_expert_adapter`'s typed
        (non-`Optional`) `attention_mask` parameter -- that dormant sparse+
        reasoning path is unaffected.

        DSpark rehearsal 2026-09-28 §19-§20: passing `attention_mask=None`
        above turned out to be a no-op on the real backbone -- transformers
        5.16.1's `create_causal_mask` rebuilds an explicit mask for this
        exact shape (`q_length>1` over a non-empty cache) regardless of what
        `self._decode` was handed, which still forces the slow memory-
        efficient SDPA kernel (confirmed by profiled kernel name, not by
        the argument this module passes). For `k > 1` on CUDA with a
        non-empty cache, this now additionally requests `unflatten.model.
        speculative.split_causal_attention`'s HF attention implementation
        for this one call, which bypasses `create_causal_mask` structurally
        instead of via the mask argument. `k == 1` (`encode_step`) never
        requests it -- that shape already gets the fast kernel."""
        if len(new_token_ids) == 0:
            raise ValueError("encode_steps requires at least one token")
        k = len(new_token_ids)
        composite_embeds = self.composite_head.embed_packed(
            torch.tensor([list(new_token_ids)], dtype=torch.long, device=state.device)
        )
        attention_mask = torch.ones(
            1, state.context_len + state.composite_len + k, dtype=torch.long, device=state.device
        )
        position_ids = None
        if state.rope_deltas is not None:
            offsets = torch.arange(k, device=state.device).view(1, -1)
            local_positions = state.rope_deltas + state.composite_len + offsets  # (batch, k)
            position_ids = (
                local_positions.view(1, local_positions.shape[0], -1).expand(3, -1, -1)
                + state.context_len
            )
        has_cached_prefix = (state.context_len + state.composite_len) > 0
        attn_implementation_override = (
            SPLIT_CAUSAL_ATTN_IMPLEMENTATION_NAME
            if k > 1 and has_cached_prefix and state.device.type == "cuda"
            else None
        )
        decoded = self._decode(
            inputs_embeds=composite_embeds,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=state.past_key_values,
            use_cache=True,
            output_hidden_states=hidden_taps is not None,
            attn_implementation_override=attn_implementation_override,
        )
        trainable_dtype = self.dense_adapter.layers[0].norm.weight.dtype
        hidden = self._apply_expert_adapter(
            decoded.last_hidden_state.to(trainable_dtype),
            visual_state=state.context_embeds,
            attention_mask=attention_mask,
        )
        router_trace = state.level_router_trace
        if router_trace is not None:
            step_traces = []
            for token_id in new_token_ids:
                router_trace = router_trace.advance(token_id, self.ranges)
                step_traces.append(router_trace)
            hidden = self._apply_level_router(
                hidden, self._trace_state_ids(*step_traces), state.level_router_memory
            )
        new_state = IncrementalDecodeState(
            past_key_values=decoded.past_key_values,
            context_len=state.context_len,
            composite_len=state.composite_len + k,
            rope_deltas=state.rope_deltas,
            device=state.device,
            context_embeds=state.context_embeds,
            level_router_memory=state.level_router_memory,
            level_router_trace=router_trace,
        )
        if hidden_taps is None:
            return hidden, new_state
        taps = torch.stack([decoded.hidden_states[i] for i in hidden_taps])
        return hidden, new_state, taps

    def forward(self, batch: TeacherForcedBatch) -> torch.Tensor:
        """`(batch, steps, ranges.total_count)` packed composite logits."""
        hidden, prefix_len = self.encode_hidden(batch)
        composite_hidden = hidden[:, prefix_len:, :]
        return self.composite_head(composite_hidden)
