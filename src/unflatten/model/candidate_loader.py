"""Qwen3-VL candidate loader — M2 item 1, DEC-007 D1/D3.

`docs/plans/dense-end-to-end-model-v0-design` (doc) §5.1/§8 M2 requires "a
candidate loader that does NOT discard the language half". The pre-S0
backbone probe (`unflatten.probe.backbones`) is a different thing on
purpose: `_drop_language_half` there is deliberate, because a visual-feature
probe has no use for a decoder. This module is the opposite: it is the
*decoder* candidate track (DEC-007 D1), so the language model, its input
embedding and its output (LM) head are the entire reason the checkpoint is
loaded at all -- discarding them here would repeat the probe's shortcut in
exactly the place the design says not to (design §5.1: "a loader that drops
the language half throws away the thing being bought").

Pin (DEC-007 D3, Component-Lock material): ``Qwen/Qwen3-VL-4B-Instruct`` at
revision ``ebb281ec70b05090aa6165b016eac8ec08e71b17``, bf16, SDPA attention.
Base weights are FROZEN in S1 — this module only loads and freezes; it
authorizes no gradient update and builds no optimizer.

**Group tagging is an ASSUMPTION, not a DEC-007 pin.** The trainable-group
registry (`trainable_group_registry.py`) names three separate frozen
backbone groups: ``backbone.visual``, ``backbone.language``,
``backbone.projector``. Qwen3-VL's own module tree does not offer a clean
seam matching that split: `Qwen3VLVisionModel` bundles its patch
embedding/transformer blocks *and* its patch-merger projection (the thing
that maps vision width to the language model's hidden size) inside one
`.visual` submodule, with no externally-callable boundary between them
short of reaching into `visual.merger` / `visual.deepstack_merger_list` by
name. This module does exactly that split for bookkeeping
(`group_id_of_qwen_parameter`), but it does not change the audit's actual
claim: every one of these three groups is ``frozen`` in S1, so the only
behavioural fact under test — zero gradient — is identical whichever side
of that internal seam a given tensor lands on. If DEC-007 is amended to
require a real, separately swappable projector module, this split needs to
become a real module boundary, not just a name.

**Native visual injection (`inject_visual`) is the `BackboneAdapter` seam.**
RFC004 §13's `BackboneAdapter` exists precisely so the Word reconstruction
architecture can stay ignorant of which base model it is talking to;
`candidate_model.py` calls `QwenCandidate.inject_visual` without knowing
whether that means Qwen's own placeholder-token `masked_scatter` + `mrope`
positional index (this module, real backend) or a prefix-concatenation
fallback (this module, if the native path were ever unavailable — it is
not, for Qwen3-VL, so no fallback is implemented here). Keeping the merge
*inside* the candidate, behind one method, is what lets the model stay
backbone-neutral while the backbone itself stays on-distribution: a
prefix-concatenation merge done in `candidate_model.py` would have been
positionally something Qwen never saw in pretraining, which spends exactly
the prior DEC-007 D1/D4 took the whole candidate (not just the vision
tower) and the Qwen tokenizer (not a standalone byte tokenizer) to keep.

`inject_visual` implements two of Qwen3-VL's own mechanisms, reached via
its public `Qwen3VLModel` methods rather than reimplemented by hand:

- **placeholder scatter** (`get_placeholder_mask` / `masked_scatter`):
  wherever `native_id_sequence` equals `image_token_id`, the corresponding
  row of `inputs_embeds` is replaced by a row of `visual_embeds`, in order.
- **native positional handling** (`get_rope_index`, Qwen3-VL's 3-axis
  temporal/height/width mRoPE): position ids for the image region are
  derived from `image_grid_thw`, not a plain `arange`, exactly as Qwen's
  own top-level `forward` computes them for a genuine multimodal input.

**DeepStack (DEC-056) is opt-in, never implicit.** Qwen3-VL also feeds the
outputs of `visual.deepstack_merger_list` (ViT blocks
`deepstack_visual_indexes`, [5, 11, 17] for the 4B pin) into the language
model: `Qwen3VLTextModel.forward` adds level `k` to the hidden states at the
visual positions AFTER decoder layer `k` (`_deepstack_process`). Every run
through v15b trained without it (`vision_forward` returns `pooler_output`
only). `vision_forward_with_deepstack` returns the pooler output AND the
levels from the SAME tower call, and `decode` passes `visual_pos_masks`/
`deepstack_visual_embeds` straight through to HF's own text model -- only
when a caller supplies them, so every existing call is byte-identical.
Whether a model uses DeepStack is a property of the checkpoint
(`QwenDenseCandidateModel(deepstack=...)`), never of this loader.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

#: DEC-007 D3 pin. A revision change is a decision, not maintenance (mirrors
#: the LibreOffice precedent, DEC-004) — bump this only with a new decision.
QWEN_CANDIDATE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_CANDIDATE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


@dataclass(frozen=True)
class InjectedEmbeds:
    """`inject_visual`'s result: merged embeddings plus whatever position
    ids the backbone's native positional convention requires (`None` when
    the backbone has none — `decode` must then fall back to its own
    default, exactly as `nn.TransformerDecoder`/HF causal LMs already do
    when `position_ids` is omitted).

    `rope_deltas` (design §20) is Qwen's own `get_rope_index` continuation
    delta, `None` exactly when `position_ids` is `None` (same "no native
    convention" condition). `encode_hidden`/`encode_prefix` never use it —
    it exists so `encode_step`'s incremental KV-cache path can assign the
    NEXT composite token's mRoPE position id (`rope_deltas + how many
    composite-range positions have already been processed`) without
    re-deriving `inject_visual`'s own numbering by hand, which is exactly
    where design §20 warns the position arithmetic is easy to get subtly
    wrong."""

    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor | None
    rope_deltas: torch.Tensor | None = None


class DecoderLike(Protocol):
    """Minimal duck-typed surface `candidate_model.QwenDenseCandidateModel`
    needs from the frozen causal decoder half.

    Deliberately narrow, mirroring `exact_text.QwenTokenizerLike`'s reasoning:
    keeping this Protocol minimal lets the default (fast, offline) test path
    use a tiny stub instead of the real 8.3 GB checkpoint.

    `past_key_values`/`use_cache` (design §20, the KV-cache free-running
    decode path) are optional, additive pass-through parameters -- every
    existing caller that omits them gets EXACTLY today's behaviour (no
    cache built, no cache consumed). Widening this frozen boundary with
    optional parameters changes no weight and no compute the frozen decoder
    was already going to do; that is what keeps it inside DEC-007 D3's pin
    (see `free_running_decode.py`'s module docstring for the fuller
    argument).

    `output_hidden_states` (DSpark multi-layer taps, 2026-09-27): another
    optional, additive pass-through, default `False` -- every existing
    caller that omits it keeps EXACTLY today's behaviour (no per-layer
    states retained). `True` asks the decoder for its FULL per-layer
    hidden-state stack (HF's own convention: entry 0 is the embedding
    output, entry i>0 is the output of decoder layer i), returned as
    `.hidden_states`, so a caller can read a fixed set of intermediate
    layers instead of only the final one.

    `attn_implementation_override` (2026-09-28, `docs/runs/dspark-
    rehearsal-2026-09-27.md` §20): another optional, additive pass-through,
    default `None` -- every existing caller that omits it keeps EXACTLY
    today's behaviour (DEC-007 D3's pinned attention implementation,
    untouched). A name from `unflatten.model.speculative.
    split_causal_attention` instead asks this ONE call to run under that
    HF attention implementation instead of the pin -- see that module's
    docstring for why a q_len>1 speculative-decode verify block needs this.

    `visual_pos_masks`/`deepstack_visual_embeds` (DEC-056, 2026-09-29):
    optional, additive, default `None` -- Qwen3-VL's own DeepStack inputs,
    passed through to the text model unchanged. `visual_pos_masks` is
    `(batch, steps)` over THIS call's positions; level `k` holds one row per
    `True` position, in row-major order, and is added after decoder layer `k`.
    Only a caller that supplies them gets DeepStack.
    """

    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        attn_implementation_override: str | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: Sequence[torch.Tensor] | None = None,
    ) -> Any: ...  # returns an object with `.last_hidden_state`, and, when
    # `use_cache=True`, a `.past_key_values` the caller must feed back into
    # the NEXT call to extend the same cache; when `output_hidden_states=True`,
    # also a `.hidden_states` tuple (embeddings, then one entry per layer).


class VisionTowerLike(Protocol):
    """Minimal duck-typed frozen vision tower surface.

    Returns embeddings already projected to `hidden_size` — for Qwen3-VL
    that projection (the `merger`) lives inside the tower itself (module
    docstring); a different candidate's tower may or may not bundle its own
    projector the same way, which is exactly why this is a Protocol call
    rather than a fixed attribute path.
    """

    def __call__(
        self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor
    ) -> torch.Tensor: ...


class InjectVisualLike(Protocol):
    """The `BackboneAdapter` seam for native visual injection. See the
    module docstring's "Native visual injection" section for why this
    exists as its own method rather than being folded into `decode`."""

    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor,
        visual_embeds: torch.Tensor,
        native_id_sequence: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> InjectedEmbeds: ...


@dataclass(frozen=True)
class QwenCandidate:
    """A loaded (real or stub), fully-frozen Qwen3-VL candidate.

    `hidden_size` is the language model's width and doubles as the visual
    embedding width this candidate's tower emits (§5.1: the tower's own
    projector already targets it). `visual_hidden_size` is kept as a
    separate field even though it equals `hidden_size` for this candidate,
    because `VisualMemory.visual_dim` must never assume backbone == Qwen
    (CLAUDE.md: "No hardcoded hidden dimensions anywhere").

    `image_token_id`/`vision_start_token_id`/`vision_end_token_id` are the
    native ids a caller must place in `native_id_sequence` (and, packed via
    `composite_vocab.QwenRef`, in the composite token stream itself) to mark
    where visual content goes -- see `inject_visual`.

    `attn_implementation` records what this specific instance was actually
    loaded with -- `"sdpa"` for the DEC-007 D3 production pin, `"eager"` for
    RFC005 §14.4 gate 12's reference mode (see `load_qwen_candidate`'s
    docstring), `"stub"` for `build_stub_qwen_candidate` (its toy decoder has
    no real attention-kernel choice to report; claiming `"eager"` or `"sdpa"`
    for it would assert a kernel identity it does not have). A caller that
    declares one mode but was actually handed another must be able to check
    this field rather than debug a confusing numerical failure against a
    model that was silently the wrong one all along.
    """

    model_id: str
    hidden_size: int
    visual_hidden_size: int
    vocab_size: int
    attn_implementation: str
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    embed_tokens: nn.Embedding
    lm_head_weight: torch.Tensor  # (vocab_size, hidden_size), frozen
    vision_forward: VisionTowerLike
    decode: DecoderLike
    inject_visual: InjectVisualLike
    #: Every named parameter across the whole candidate, source-tagged by
    #: `group_id_of_qwen_parameter` for the trainability audit. Not a
    #: `nn.Module` itself (the real and stub backends have different module
    #: trees) — a flat mapping is the one shape both can produce.
    named_parameters: dict[str, nn.Parameter]
    #: DEC-056: `(pixel_values, grid_thw=) -> (pooler_output, levels)`, the
    #: SAME tower call as `vision_forward` plus the DeepStack levels, one
    #: `(tokens, hidden_size)` tensor per `deepstack_visual_indexes` entry.
    #: `None` for a candidate that has no DeepStack path.
    vision_forward_with_deepstack: (
        Callable[..., tuple[torch.Tensor, tuple[torch.Tensor, ...]]] | None
    ) = None
    #: The ViT blocks the levels are tapped from (`[5, 11, 17]` for the 4B
    #: pin); empty when there is no DeepStack path. Part of the visual-input
    #: identity (`onfly_visual.VisualInputPolicy`).
    deepstack_visual_indexes: tuple[int, ...] = ()
    #: Pre-merge patches per merged visual token along each spatial axis:
    #: merged tokens = t * h * w / spatial_merge_size**2 of `grid_thw`. The
    #: real tower merges 2x2; the stub tower does not merge (1).
    spatial_merge_size: int = 1


def group_id_of_qwen_parameter(name: str) -> str:
    """Map one of `QwenCandidate.named_parameters`' keys to its frozen
    trainable-group id. See the module docstring for why this split is an
    assumption, not a DEC-007 pin.
    """
    if "language_model" in name or name.startswith(("embed_tokens", "lm_head")):
        return "backbone.language"
    if "merger" in name:
        return "backbone.projector"
    return "backbone.visual"


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def load_qwen_candidate(
    *,
    model_id: str = QWEN_CANDIDATE_MODEL_ID,
    revision: str = QWEN_CANDIDATE_REVISION,
    dtype: torch.dtype = torch.bfloat16,
    device: str | None = None,
    attn_implementation: str = "sdpa",
) -> QwenCandidate:
    """Load the real, full Qwen3-VL candidate — vision tower AND language
    half, both frozen. Heavy (network + ~8 GB VRAM); callers exercising this
    in tests must mark the test `slow`.

    `device` moves every loaded weight there (`model.to(device)`, AFTER
    freezing -- `requires_grad=False` survives a `.to()` call, so the order
    only matters for how long the CPU copy exists). `None` (the default)
    leaves the model wherever `from_pretrained` put it -- CPU, since this
    loader never passes `device_map=` -- rather than silently assuming
    CUDA is available: this loader's own docstring claiming "~8 GB VRAM"
    was true of the WEIGHTS, not of where they ended up, until a real
    forward pass on real pixels (candidate_backbone.py's end-to-end proof)
    surfaced the gap. When `device` IS given, every parameter's actual
    device is checked against it and a straggler raises `RuntimeError`
    immediately -- a caller-supplied `device` that silently only PARTLY
    moved the model used to fail nowhere and simply run at CPU speed; this
    is the runtime guard the 2026-09-04 incident (this docstring, above)
    argued a slow test alone does not give a caller who never runs it.

    `attn_implementation` defaults to DEC-007 D3's pinned `"sdpa"` --
    production inference/training must never silently drift off that pin.
    The one caller allowed to override it is RFC005 §14.4 gate 12's own
    "reference režim" clause ("rovnaký sample s iným batch paddingom má
    ekvivalentný výstup v deklarovanej tolerancii/reference režime"):
    `"eager"` (explicit `softmax(QK^T + mask)V`, no fused kernel) is that
    reference mode, not a workaround invented outside the RFC. Measured
    2026-09-04: SDPA's bf16 attention kernel is NOT bitwise reproducible
    across different total sequence lengths (a real, unrelated-to-masking
    property of its kernel dispatch/blocking) -- a padded-vs-unpadded
    gate-12-shaped comparison showed max diff ~0.59 / mean ~0.13 on logits
    (scale ~16.75) under `"sdpa"`, drift appearing only past ~8 total
    tokens (exactly 0.0 at 4->6), and EXACTLY 0.0 at `torch.allclose`'s
    fp32-tight defaults under `"eager"` for the identical inputs -- proving
    the drift is a kernel property, not a masking or `inject_visual` leak
    (position ids, rotary cos/sin and the causal mask were independently
    confirmed bit-identical at every real position in both modes first).
    A future reader must not "simplify" gate 12 back onto the production
    SDPA pin because eager looks redundant with `load_qwen_candidate`'s
    default -- that redundancy is the point: gate 12 needs the mode that
    is actually reproducible, and production needs the mode that is
    actually pinned, and they are different for a real, load-bearing reason.
    """
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, dtype=dtype, attn_implementation=attn_implementation
    )
    model.eval()
    _freeze(model)
    if device is not None:
        model = model.to(device)
        want = torch.device(device).type
        stale = [name for name, param in model.named_parameters() if param.device.type != want]
        if stale:
            raise RuntimeError(
                f"model.to({device!r}) did not move every parameter -- "
                f"{len(stale)} still elsewhere (first: {stale[0]!r}); this is the "
                "runtime check for the exact bug that once made a GPU build "
                "silently run at CPU speed instead of failing (2026-09-04)"
            )
    return candidate_from_hf_model(
        model, model_id=model_id, dtype=dtype, attn_implementation=attn_implementation
    )


def candidate_from_hf_model(
    model: Any, *, model_id: str, dtype: torch.dtype, attn_implementation: str
) -> QwenCandidate:
    """Wrap an already-loaded, frozen, `.eval()` Qwen3-VL
    `*ForConditionalGeneration` as a `QwenCandidate`. `load_qwen_candidate`
    is this plus `from_pretrained` at the DEC-007 D3 pin; split out so a test
    can drive the SAME closures over a tiny randomly initialised Qwen3-VL
    (DEC-056 DeepStack reference tests) instead of the 4B checkpoint.
    `dtype` is the frozen compute dtype `decode` casts its inputs to."""
    from unflatten.model.decode_graph import graphed_decode_step_or_none

    inner = model.model
    visual = inner.visual
    # Same rewrite `probe/features.py` already needed for this exact tower
    # (module docstring there): some cuDNN builds fail patch_embed's
    # `conv3d` outright (`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`) or, on
    # builds where it merely runs, take ~23 s/page against 0.2 ms for the
    # mathematically identical matmul form. Must run AFTER the device move
    # (`optimize_patch_embed`'s own docstring): the rewrite closes over the
    # weight tensor, so doing it first would close over a CPU copy.
    from unflatten.probe.backbones import optimize_patch_embed

    optimize_patch_embed(visual)
    language_model = inner.language_model
    embed_tokens = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    image_token_id = model.config.image_token_id
    video_token_id = model.config.video_token_id
    vision_start_token_id = model.config.vision_start_token_id
    vision_end_token_id = model.config.vision_end_token_id

    def vision_forward(pixel_values: torch.Tensor, *, grid_thw: torch.Tensor) -> torch.Tensor:
        # `Qwen3VLVisionModel.forward` (installed transformers 5.16.1)
        # returns `BaseModelOutputWithDeepstackFeatures`, not the plain
        # 2-tuple this used to unpack positionally -- `output.last_hidden_state`
        # is the PRE-merge patch grid, and the merged (post-`spatial_merge_size`)
        # tokens this module's docstring promises live in `.pooler_output`.
        # A positional `a, b = output` broke here because `ModelOutput.__iter__`
        # yields every populated field (>2 when `deepstack_features` is also
        # set), which a real end-to-end forward exercises and a
        # `.decode`-only smoke test never did -- attribute access is not
        # sensitive to how many fields the object happens to populate.
        output = visual(pixel_values, grid_thw=grid_thw)
        return output.pooler_output

    def vision_forward_with_deepstack(
        pixel_values: torch.Tensor, *, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        # DEC-056: the same single tower call, both fields read off it --
        # `deepstack_features` is populated by `Qwen3VLVisionModel.forward`
        # on every call; `vision_forward` above simply never read it.
        output = visual(pixel_values, grid_thw=grid_thw)
        return output.pooler_output, tuple(output.deepstack_features)

    def decode(
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        attn_implementation_override: str | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: Sequence[torch.Tensor] | None = None,
    ) -> Any:
        # DEC-056: DeepStack inputs are forwarded ONLY when given, so every
        # existing call below keeps exactly its old keyword set.
        deepstack_kwargs: dict[str, Any] = {}
        if deepstack_visual_embeds is not None:
            if visual_pos_masks is None:
                raise ValueError("deepstack_visual_embeds given without visual_pos_masks")
            deepstack_kwargs = {
                "visual_pos_masks": visual_pos_masks,
                "deepstack_visual_embeds": list(deepstack_visual_embeds),
            }
        # DEC-007 D3.1: the bf16 pin covers frozen base weights and compute,
        # NOT the trainable groups' storage dtype (an fp32 master copy is
        # allowed and, per the review that forced this clarification, may be
        # required -- bf16's 8 mantissa bits can round `p + lr*grad` back to
        # exactly `p`). `candidate_model.py`'s trainable submodules may
        # therefore hand this frozen decoder fp32 (or any other dtype)
        # `inputs_embeds`; casting to this candidate's own compute dtype HERE,
        # once, at the frozen boundary, is what makes that safe -- the
        # trainable side never needs to know what dtype the frozen decoder
        # runs in, and the frozen decoder never sees a dtype it wasn't loaded
        # in.
        #
        # `past_key_values`/`use_cache` (design §20) are pass-through only --
        # this closure never builds a cache itself, HF's own Qwen3-VL text
        # model does (`transformers.cache_utils.DynamicCache` when
        # `use_cache=True` and no cache was supplied). Every existing caller
        # that omits both keeps EXACTLY today's behaviour.
        #
        # `attn_implementation_override` (dspark-rehearsal §20): `None`
        # (every existing caller) leaves `language_model.config.
        # _attn_implementation` at the DEC-007 D3 pin. A name given for this
        # ONE call temporarily swaps it, restoring the pin in `finally` even
        # if `language_model` raises -- every attention layer reads
        # `self.config._attn_implementation` fresh on each forward (the
        # same shared `config` object), so this is safe to toggle per call.
        # `kv_cache_mode="static_graph"` (owner speed item 1, 2026-09-29):
        # a plain q=1 step over a `GraphedStaticKVCache` replays piecewise
        # CUDA graphs instead; every other call returns `None` here and
        # runs exactly as before (`decode_graph`'s module docstring).
        graphed = (
            None
            if deepstack_kwargs
            else graphed_decode_step_or_none(
                language_model,
                inputs_embeds=inputs_embeds.to(dtype),
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                attn_implementation_override=attn_implementation_override,
            )
        )
        if graphed is not None:
            return graphed
        if attn_implementation_override is None:
            return language_model(
                inputs_embeds=inputs_embeds.to(dtype),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                **deepstack_kwargs,
            )
        pinned_attn_implementation = language_model.config._attn_implementation
        language_model.config._attn_implementation = attn_implementation_override
        try:
            return language_model(
                inputs_embeds=inputs_embeds.to(dtype),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                **deepstack_kwargs,
            )
        finally:
            language_model.config._attn_implementation = pinned_attn_implementation

    inject_visual = make_native_inject_visual(
        get_rope_index=inner.get_rope_index,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
    )

    named_parameters = {
        f"visual.{name}": param for name, param in visual.named_parameters()
    }
    named_parameters.update(
        {f"language_model.{name}": param for name, param in language_model.named_parameters()}
    )
    named_parameters["embed_tokens.weight"] = embed_tokens.weight
    if lm_head is not None and lm_head.weight is not embed_tokens.weight:
        named_parameters["lm_head.weight"] = lm_head.weight

    hidden_size = model.config.text_config.hidden_size
    return QwenCandidate(
        model_id=model_id,
        hidden_size=hidden_size,
        visual_hidden_size=hidden_size,
        vocab_size=embed_tokens.weight.shape[0],
        attn_implementation=attn_implementation,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        vision_end_token_id=vision_end_token_id,
        embed_tokens=embed_tokens,
        lm_head_weight=lm_head.weight if lm_head is not None else embed_tokens.weight,
        vision_forward=vision_forward,
        decode=decode,
        inject_visual=inject_visual,
        named_parameters=named_parameters,
        vision_forward_with_deepstack=vision_forward_with_deepstack,
        deepstack_visual_indexes=tuple(int(i) for i in visual.deepstack_visual_indexes),
        spatial_merge_size=int(visual.spatial_merge_size),
    )


def make_native_inject_visual(
    *, get_rope_index: Any, image_token_id: int, video_token_id: int
) -> InjectVisualLike:
    """The real backbone's `inject_visual`: Qwen's placeholder scatter plus its
    own `get_rope_index` (3-axis mRoPE). A module-level factory (was a closure
    inside `load_qwen_candidate`) so a test can drive the SAME code with a tiny
    randomly initialised `Qwen3VLModel.get_rope_index` instead of the 4B
    checkpoint (`test_candidate_model_post_encoder_merge_positions.py`)."""

    def inject_visual(
        *,
        inputs_embeds: torch.Tensor,
        visual_embeds: torch.Tensor,
        native_id_sequence: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> InjectedEmbeds:
        # Qwen's own `get_placeholder_mask`/`masked_scatter` pattern
        # (module docstring): every position whose native id is
        # `image_token_id` gets replaced, in order, by one row of
        # `visual_embeds`. A count mismatch is a caller bug (wrong number
        # of placeholder ids for the visual tokens actually produced) --
        # `masked_scatter` would otherwise silently consume/leave rows.
        image_mask = native_id_sequence == image_token_id
        n_placeholders = int(image_mask.sum())
        flat_visual = visual_embeds.reshape(-1, visual_embeds.shape[-1])
        if flat_visual.shape[0] != n_placeholders:
            raise ValueError(
                f"{flat_visual.shape[0]} visual embeddings but {n_placeholders} "
                f"{image_token_id!r} placeholder positions in native_id_sequence"
            )
        expanded_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        merged = inputs_embeds.masked_scatter(
            expanded_mask, flat_visual.to(inputs_embeds.dtype)
        )
        # `get_rope_index` wants `mm_token_type_ids`: 0 text / 1 image /
        # 2 video per position (its own docstring). Ordinarily the Qwen
        # image processor returns this alongside `input_ids`; here it is a
        # pure, deterministic function of the two placeholder ids this
        # method already has, so there is nothing for a processor to add.
        mm_token_type_ids = torch.zeros_like(native_id_sequence)
        mm_token_type_ids = torch.where(image_mask, 1, mm_token_type_ids)
        mm_token_type_ids = torch.where(
            native_id_sequence == video_token_id, 2, mm_token_type_ids
        )
        position_ids, rope_deltas = get_rope_index(
            native_id_sequence,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        # `rope_deltas` used to be discarded here. Design §20's incremental
        # KV-cache path needs it (`InjectedEmbeds`'s own docstring) to keep
        # assigning mRoPE ids to new tokens after the prefix pass without
        # calling `get_rope_index` again on every step.
        return InjectedEmbeds(
            inputs_embeds=merged, position_ids=position_ids, rope_deltas=rope_deltas
        )

    return inject_visual


class _StubVisionTower(nn.Module):
    """Tiny frozen stand-in with a `visual.*` / `merger.*` name split
    matching `group_id_of_qwen_parameter`, so the stub exercises the same
    three-way frozen-group audit the real candidate does."""

    def __init__(self, visual_hidden_size: int, hidden_size: int) -> None:
        super().__init__()
        self.blocks = nn.Linear(visual_hidden_size, visual_hidden_size)
        self.merger = nn.Linear(visual_hidden_size, hidden_size)

    def forward(self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor) -> torch.Tensor:
        del grid_thw  # unused in the stub; the real tower derives shape from it
        return self.merger(torch.tanh(self.blocks(pixel_values)))


def _stub_layer_qkv(
    layer: nn.TransformerEncoderLayer, normed: torch.Tensor, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(query, key, value)`, each `(batch, heads, steps, head_dim)`, from
    `layer.self_attn`'s OWN `in_proj_weight`/`in_proj_bias` -- the exact
    projection `nn.MultiheadAttention.forward` applies internally, factored
    out so `_stub_cached_forward` (design §20) can cache the projected
    `key`/`value` per layer instead of only ever seeing them consumed
    inside one opaque `forward` call."""
    batch, steps, d_model = normed.shape
    head_dim = d_model // num_heads
    projected = nn.functional.linear(
        normed, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias
    )
    query, key, value = projected.chunk(3, dim=-1)

    def _split(t: torch.Tensor) -> torch.Tensor:
        return t.view(batch, steps, num_heads, head_dim).transpose(1, 2)

    return _split(query), _split(key), _split(value)


def _stub_layer_step(
    layer: nn.TransformerEncoderLayer,
    hidden: torch.Tensor,
    num_heads: int,
    past_kv: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """One `nn.TransformerEncoderLayer` (`norm_first=True`) forward pass,
    written out by hand (matching `TransformerEncoderLayer`'s own
    `_sa_block`/`_ff_block` exactly -- pre-norm residual attention, then
    pre-norm residual FFN, both no-op'd dropout since the frozen stub is
    always `.eval()`) so `hidden`'s NEW positions only ever need to attend
    against `past_kv`'s cached key/value plus their own, rather than
    reforming `nn.MultiheadAttention`'s internal call over the whole
    sequence again. Returns the layer's output for exactly `hidden`'s
    (new) positions, and this layer's (key, value) cache extended by them.
    """
    batch, new_len, d_model = hidden.shape
    head_dim = d_model // num_heads
    normed = layer.norm1(hidden)
    query, key, value = _stub_layer_qkv(layer, normed, num_heads)
    if past_kv is not None:
        past_key, past_value = past_kv
        key = torch.cat([past_key, key], dim=2)
        value = torch.cat([past_value, value], dim=2)
    new_cache = (key, value)

    scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
    # Causal mask over the FULL `(new_len, past_len + new_len)` score block --
    # query position `past_len + i` may attend to key position `j` iff
    # `j <= past_len + i`. This subsumes both calling shapes: the design §20
    # PREFIX pass (`past_kv is None`, `past_len == 0`, ordinary `new_len x
    # new_len` causal triu) and `encode_step`'s single-token continuation
    # (`new_len == 1`, every key already causally valid, mask is all-zero).
    # W0-11 (chunked prefill) added a THIRD shape this must also get right:
    # a later chunk's continuation call with `past_kv is not None` AND
    # `new_len > 1` -- the new chunk's own positions must still be masked
    # from each other, which the old two-branch version (unmasked whenever
    # `past_kv is not None`) assumed could never happen.
    past_len = past_kv[0].shape[2] if past_kv is not None else 0
    row = torch.arange(new_len, device=hidden.device).unsqueeze(1) + past_len
    col = torch.arange(past_len + new_len, device=hidden.device).unsqueeze(0)
    causal = torch.where(
        col <= row,
        torch.zeros((), dtype=scores.dtype, device=hidden.device),
        torch.full((), float("-inf"), dtype=scores.dtype, device=hidden.device),
    )
    scores = scores + causal
    weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(weights, value)  # (batch, heads, new_len, head_dim)
    context = context.transpose(1, 2).reshape(batch, new_len, d_model)
    attn_out = layer.self_attn.out_proj(context)
    hidden = hidden + attn_out

    normed2 = layer.norm2(hidden)
    ff = layer.linear2(layer.activation(layer.linear1(normed2)))
    hidden = hidden + ff
    return hidden, new_cache


class _StubLanguageModel(nn.Module):
    """Tiny frozen causal decoder stand-in with the real decoder's calling
    convention (`inputs_embeds`/`attention_mask` in, `.last_hidden_state`
    out) so `candidate_model` code never branches on stub-vs-real."""

    def __init__(self, hidden_size: int, layers: int = 2, heads: int = 2) -> None:
        super().__init__()
        self.heads = heads
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: Sequence[torch.Tensor] | None = None,
    ) -> Any:
        if deepstack_visual_embeds is not None:
            return self._forward_deepstack(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
        # The stub has no native positional convention (no rope, no learned
        # position table) -- `position_ids` is accepted, matching
        # `DecoderLike`, but genuinely unused; causal order alone still
        # governs it via the mask below. A real backbone with a positional
        # mechanism must NOT do this silently -- see `load_qwen_candidate`'s
        # `decode`, which always forwards `position_ids` through.
        del position_ids
        if past_key_values is None and not use_cache:
            # Unchanged from before design §20: every existing (non-cache)
            # caller keeps this exact code path, byte-for-byte.
            #
            # A causal decoder attends only to the past; the stub combines
            # the causal mask and the caller's padding mask into ONE
            # boolean mask before calling the encoder. Passing them
            # separately (`mask=` + `src_key_padding_mask=`) hits PyTorch's
            # fast/nested-tensor path, which silently produces different
            # numbers when the two masks' dtypes disagree (a bool causal
            # mask can't combine with a bool key-padding mask on that path
            # without a warning-flagged deprecation) -- exactly the kind of
            # "runs, no error, wrong numbers" failure this project keeps a
            # standing note about (CLAUDE.md's LibreOffice-profile
            # incident). Merging by hand avoids depending on that fast path
            # at all.
            batch, steps = attention_mask.shape
            causal = torch.triu(
                torch.ones(steps, steps, dtype=torch.bool, device=inputs_embeds.device),
                diagonal=1,
            )
            key_padding = ~attention_mask.bool()  # (batch, steps), True == padding
            combined = causal.unsqueeze(0) | key_padding.unsqueeze(1)  # (batch, steps, steps)
            combined = combined.repeat_interleave(self.heads, dim=0)  # (batch*heads, steps, steps)
            if output_hidden_states:
                # DSpark multi-layer taps: no cheap per-layer hook on
                # `nn.TransformerEncoder`'s whole-stack call, so this branch
                # loops its own layers manually -- same layers, same mask,
                # only to also retain each one's output. Nothing calls this
                # branch with `output_hidden_states=True` today (`encode_step`/
                # `encode_steps` always take the KV-cache branch below); it
                # exists so the Protocol's contract holds here too rather
                # than only in the branch tests currently exercise.
                states = [inputs_embeds]
                hidden = inputs_embeds
                for layer in self.encoder.layers:
                    hidden = layer(hidden, src_mask=combined)
                    states.append(hidden)
                return _HiddenStateOutput(hidden, hidden_states=tuple(states))
            hidden = self.encoder(inputs_embeds, mask=combined)
            return _HiddenStateOutput(hidden)

        # KV-cache path (design §20). `free_running_decode`'s usage is
        # always batch=1, no padding -- the manual per-layer attention below
        # does not implement a key-padding mask, so a padded batch here
        # would silently attend into pad positions; that shape is not a
        # cached-decode input anywhere in this repo (`decode_free_running`'s
        # own `visual_features.shape[0] != 1` guard), so this is checked
        # rather than assumed.
        batch = inputs_embeds.shape[0]
        if batch != 1:
            raise ValueError(
                f"_StubLanguageModel's KV-cache path is batch=1 only (no padding mask "
                f"implemented for it), got batch={batch}"
            )
        layers = list(self.encoder.layers)
        layer_pasts = past_key_values if past_key_values is not None else [None] * len(layers)
        hidden = inputs_embeds
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        states = [inputs_embeds] if output_hidden_states else None
        for layer, layer_past in zip(layers, layer_pasts, strict=True):
            hidden, layer_cache = _stub_layer_step(layer, hidden, self.heads, layer_past)
            new_cache.append(layer_cache)
            if states is not None:
                states.append(hidden)
        return _HiddenStateOutput(
            hidden,
            past_key_values=tuple(new_cache) if use_cache else None,
            hidden_states=tuple(states) if states is not None else None,
        )

    def _forward_deepstack(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        use_cache: bool,
        visual_pos_masks: torch.Tensor | None,
        deepstack_visual_embeds: Sequence[torch.Tensor],
    ) -> Any:
        """DEC-056 stub counterpart of `Qwen3VLTextModel`'s DeepStack loop:
        level `k` is added at the `visual_pos_masks` positions AFTER layer
        `k`. Always the manual per-layer path, with the same combined causal
        + padding mask as the non-cache branch; a cached call is batch=1."""
        if visual_pos_masks is None:
            raise ValueError("deepstack_visual_embeds given without visual_pos_masks")
        layers = list(self.encoder.layers)
        if len(deepstack_visual_embeds) > len(layers):
            raise ValueError(
                f"{len(deepstack_visual_embeds)} DeepStack levels for a {len(layers)}-layer stub"
            )

        def _add(hidden: torch.Tensor, level: int) -> torch.Tensor:
            if level >= len(deepstack_visual_embeds):
                return hidden
            hidden = hidden.clone()
            rows = hidden[visual_pos_masks] + deepstack_visual_embeds[level].to(hidden.dtype)
            hidden[visual_pos_masks] = rows
            return hidden

        hidden = inputs_embeds
        if past_key_values is None and not use_cache:
            batch, steps = attention_mask.shape
            causal = torch.triu(
                torch.ones(steps, steps, dtype=torch.bool, device=inputs_embeds.device),
                diagonal=1,
            )
            key_padding = ~attention_mask.bool()
            combined = causal.unsqueeze(0) | key_padding.unsqueeze(1)
            combined = combined.repeat_interleave(self.heads, dim=0)
            for level, layer in enumerate(layers):
                hidden = _add(layer(hidden, src_mask=combined), level)
            return _HiddenStateOutput(hidden)
        if inputs_embeds.shape[0] != 1:
            raise ValueError("_StubLanguageModel's KV-cache path is batch=1 only")
        layer_pasts = past_key_values if past_key_values is not None else [None] * len(layers)
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for level, (layer, layer_past) in enumerate(zip(layers, layer_pasts, strict=True)):
            hidden, layer_cache = _stub_layer_step(layer, hidden, self.heads, layer_past)
            hidden = _add(hidden, level)
            new_cache.append(layer_cache)
        return _HiddenStateOutput(
            hidden, past_key_values=tuple(new_cache) if use_cache else None
        )


@dataclass(frozen=True)
class _HiddenStateOutput:
    last_hidden_state: torch.Tensor
    past_key_values: Any | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


def build_stub_qwen_candidate(
    *,
    visual_hidden_size: int = 6,
    hidden_size: int = 8,
    vocab_size: int = 37,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    deepstack_levels: int = 0,
) -> QwenCandidate:
    """A tiny, deterministic, fully-offline stand-in with `QwenCandidate`'s
    exact interface. Default path for every non-`slow` test: RTX 4070 SUPER
    headroom is thin (~11.3 GB free, DEC-007 D3), and the real candidate
    also needs a network fetch on a cold cache — neither belongs in the
    default suite.

    `dtype` is the FROZEN stub's own storage/compute dtype, mirroring
    `load_qwen_candidate`'s `dtype` -- default fp32 keeps every existing
    caller's numbers unchanged, but a test proving the DEC-007 D3.1 mixed-
    precision boundary (frozen bf16, trainable fp32 master copy) can pass
    `torch.bfloat16` here without needing the real 8.3 GB checkpoint.

    `deepstack_levels` (DEC-056): `0` (default) builds exactly the stub it
    always was -- no extra module, no extra RNG draw. `k > 0` adds `k` stub
    DeepStack mergers (drawn AFTER every existing parameter) and a
    `vision_forward_with_deepstack`; the stub decoder adds level `i` after
    its layer `i`, so `k` may not exceed its 2 layers.
    """
    if vocab_size < 4:
        raise ValueError(
            f"vocab_size must be >= 4 (4 reserved for image/video/vision-start/vision-end "
            f"sentinel ids, matching the real candidate's convention), got {vocab_size}"
        )
    torch.manual_seed(seed)
    tower = _StubVisionTower(visual_hidden_size, hidden_size).to(dtype)
    language_model = _StubLanguageModel(hidden_size).to(dtype)
    embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=dtype)
    lm_head_weight = nn.Parameter(torch.randn(vocab_size, hidden_size, dtype=dtype))
    tower.eval()
    language_model.eval()
    _freeze(tower)
    _freeze(language_model)
    embed_tokens.weight.requires_grad_(False)
    lm_head_weight.requires_grad_(False)
    deepstack_mergers = nn.ModuleList(
        nn.Linear(visual_hidden_size, hidden_size) for _ in range(deepstack_levels)
    ).to(dtype)
    deepstack_mergers.eval()
    _freeze(deepstack_mergers)

    # Reserved sentinel ids at the top of the vocab, mirroring the real
    # candidate's `config.{image,video,vision_start,vision_end}_token_id`
    # convention: real, fixed vocab entries, not out-of-range magic values.
    image_token_id = vocab_size - 1
    video_token_id = vocab_size - 2
    vision_start_token_id = vocab_size - 3
    vision_end_token_id = vocab_size - 4

    def vision_forward(pixel_values: torch.Tensor, *, grid_thw: torch.Tensor) -> torch.Tensor:
        return tower(pixel_values, grid_thw=grid_thw)

    def vision_forward_with_deepstack(
        pixel_values: torch.Tensor, *, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        pooled = tower(pixel_values, grid_thw=grid_thw)
        inner = torch.tanh(tower.blocks(pixel_values))
        return pooled, tuple(merger(inner) for merger in deepstack_mergers)

    def decode(
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        attn_implementation_override: str | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: Sequence[torch.Tensor] | None = None,
    ) -> Any:
        if deepstack_visual_embeds is not None:
            del attn_implementation_override
            return language_model(
                inputs_embeds=inputs_embeds.to(dtype),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
        # Mirrors `load_qwen_candidate`'s `decode`: cast to this (frozen)
        # candidate's own compute dtype at the boundary, so a caller whose
        # trainable modules run in a different dtype (DEC-007 D3.1's fp32
        # master copy) never has to know or care what dtype the frozen half
        # of the stub uses.
        #
        # `past_key_values`/`use_cache` (design §20) are forwarded straight
        # through to `_StubLanguageModel`, which implements the cache
        # itself -- a stub that silently ignored these would keep every
        # existing test green while the real candidate's cache path
        # diverged (design §20's own warning), so the stub must actually
        # cache, not just accept the parameters.
        #
        # `attn_implementation_override` (dspark-rehearsal §20) is accepted
        # and ignored -- the toy decoder has no real attention-kernel choice
        # to report (`QwenCandidate.attn_implementation`'s own docstring
        # makes the same point for `"stub"`), so there is nothing here to
        # switch.
        del attn_implementation_override
        return language_model(
            inputs_embeds=inputs_embeds.to(dtype),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

    def inject_visual(
        *,
        inputs_embeds: torch.Tensor,
        visual_embeds: torch.Tensor,
        native_id_sequence: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> InjectedEmbeds:
        # Same placeholder-scatter contract as the real candidate's
        # `inject_visual` (module docstring); the stub has no mRoPE, so it
        # returns `position_ids=None` -- a real backbone with a positional
        # mechanism must not do this, `load_qwen_candidate`'s version never
        # does.
        del image_grid_thw, attention_mask
        image_mask = native_id_sequence == image_token_id
        n_placeholders = int(image_mask.sum())
        flat_visual = visual_embeds.reshape(-1, visual_embeds.shape[-1])
        if flat_visual.shape[0] != n_placeholders:
            raise ValueError(
                f"{flat_visual.shape[0]} visual embeddings but {n_placeholders} "
                f"{image_token_id!r} placeholder positions in native_id_sequence"
            )
        expanded_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        merged = inputs_embeds.masked_scatter(expanded_mask, flat_visual.to(inputs_embeds.dtype))
        return InjectedEmbeds(inputs_embeds=merged, position_ids=None)

    named_parameters: dict[str, nn.Parameter] = {
        f"visual.{name}": param for name, param in tower.named_parameters()
    }
    named_parameters.update(
        {
            f"language_model.{name}": param
            for name, param in language_model.named_parameters()
        }
    )
    named_parameters["embed_tokens.weight"] = embed_tokens.weight
    named_parameters["lm_head.weight"] = lm_head_weight
    named_parameters.update(
        {
            f"visual.deepstack_merger_list.{name}": param
            for name, param in deepstack_mergers.named_parameters()
        }
    )

    return QwenCandidate(
        model_id="stub/qwen3-vl-tiny",
        hidden_size=hidden_size,
        visual_hidden_size=visual_hidden_size,
        vocab_size=vocab_size,
        attn_implementation="stub",
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        vision_end_token_id=vision_end_token_id,
        embed_tokens=embed_tokens,
        lm_head_weight=lm_head_weight,
        vision_forward=vision_forward,
        decode=decode,
        inject_visual=inject_visual,
        named_parameters=named_parameters,
        vision_forward_with_deepstack=(
            vision_forward_with_deepstack if deepstack_levels > 0 else None
        ),
        deepstack_visual_indexes=tuple(range(deepstack_levels)),
    )
