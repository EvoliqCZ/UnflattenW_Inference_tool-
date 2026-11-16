"""Resident diagnostic runtime over the candidate-native vision/decode path."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from unflatten.model import composite_vocab
from unflatten.model.candidate_backbone import CandidateVisionBackbone
from unflatten.model.candidate_loader import (
    QWEN_CANDIDATE_MODEL_ID,
    QWEN_CANDIDATE_REVISION,
    QwenCandidate,
    load_qwen_candidate,
)
from unflatten.model.candidate_model import QwenDenseCandidateModel
from unflatten.model.capability_profile import (
    CapabilityProfile,
    CapabilityProfileHashUnresolved,
    resolve_capability_profile_from_component_lock,
)
from unflatten.model.component_lock import text_tokenizer_identity
from unflatten.model.composite_grammar import CompositeGrammarController
from unflatten.model.decode_graph import GraphedStaticKVCache
from unflatten.model.free_running_decode import DecodeConfig, decode_free_running
from unflatten.model.input_contract import ModelSafeInput, to_payload_dict, validate_no_forbidden
from unflatten.model.level_router.policy import (
    LevelRouterPolicy,
    LevelRouterPolicyError,
    level_router_policy_from_payload,
)
from unflatten.model.level_router.variant_switch import (
    CHECKPOINT_POLICY_KEY as LEVEL_ROUTER_CHECKPOINT_KEY,
)
from unflatten.model.level_router.variant_switch import level_router_variant
from unflatten.model.loop_guard_v2_caps import (
    LoopGuardCapsStale,
    load_loop_guard_v2_caps,
    refuse_unless_caps_bind,
)
from unflatten.model.manifests import (
    component_lock_hash,
    validate_exact_text_s1_component_lock,
)
from unflatten.model.prediction_ir import RawDocumentHypothesis
from unflatten.model.quantization.kv_cache import Float8KVCache, Float8KVCacheChunked
from unflatten.model.quantization.quantize import (
    QuantizationError,
    decoder_module,
    quantize_candidate,
    share_sibling_input_quantization,
    vision_module,
)
from unflatten.model.quantization.spec import QuantizationSpec, bf16_spec
from unflatten.model.qwen_tokenizer_artifacts import (
    TokenizerArtifactSetInvalid,
    load_pinned_tokenizer,
)
from unflatten.model.qwen_tokenizer_artifacts import (
    tokenizer_artifact_set_hash as compute_tokenizer_artifact_set_hash,
)
from unflatten.model.speculative.static_kv_cache import StaticKVCache
from unflatten.model.typed_head_config import typed_head_config_hash
from unflatten.probe.backbones import optimize_patch_embed

logger = logging.getLogger(__name__)


class CandidateRuntimeIdentityMismatch(ValueError):
    """A loaded runtime component differs from its startup pin."""


class CandidateRuntimeInputInvalid(ValueError):
    """The input is outside the one-page diagnostic runtime contract."""


@dataclass(frozen=True, slots=True)
class RuntimeIdentityPin:
    model_id: str
    revision: str
    attention_implementation: str
    checkpoint_sha256: str
    component_lock_hash: str


@dataclass(frozen=True, slots=True)
class PreparedCandidateInput:
    visual_features: torch.Tensor
    image_grid_thw: torch.Tensor
    #: DEC-056: the page's DeepStack levels, only for a DeepStack runtime.
    deepstack_visual_embeds: tuple[torch.Tensor, ...] | None = None


@dataclass(frozen=True, slots=True)
class DecodePolicy:
    """The service's frozen decode-time policy block (design D5/D7: the
    model bundle's `decode_policy.json`, mirroring the ratified selection
    policy's `decode` block). Immutable so a running worker can never drift
    from the value it started with."""

    loop_guard: str | None
    max_steps: int
    typed_decode: bool
    region_decode: bool
    selection_policy_hash: str
    #: W0-11 opt-in (see `free_running_decode.DecodeConfig.prefill_chunk_
    #: tokens`): `None` (default) keeps the resident worker's decode exactly
    #: today's single-call prefix pass.
    prefill_chunk_tokens: int | None = None
    #: DEC-055 (PROPOSED): the loop guard v3 arm when `loop_guard == "v3"`
    #: (`None` = the v3 policy's default arm); only a worker `--loop-guard v3`
    #: override sets it, never a bundle.
    loop_guard_v3_arm: str | None = None


def verify_decode_bindings(
    state: Mapping[str, Any],
    decode_policy: DecodePolicy,
    *,
    caps_sha256_pin: str,
    decode_steps_hard_ceiling: int,
) -> None:
    """Refuse every decode-time binding defect before any Qwen candidate is
    loaded (design D5, D6, G8-T4). Pure over an already-loaded checkpoint
    `state` and the frozen `decode_policy` -- no I/O beyond the loop guard
    v2 caps file, itself read lazily only when `loop_guard == "v2"`."""
    if decode_policy.max_steps > decode_steps_hard_ceiling:
        raise CandidateRuntimeIdentityMismatch(
            f"decode_policy.max_steps {decode_policy.max_steps} exceeds "
            f"decode_steps_hard_ceiling {decode_steps_hard_ceiling}"
        )
    if "optimizer_state_dict" in state:
        raise CandidateRuntimeIdentityMismatch(
            "checkpoint carries optimizer_state_dict; only a weights-only "
            "checkpoint (D6) is accepted"
        )
    if decode_policy.loop_guard in ("v2", "v3"):
        # v3 wraps v2 with the same caps file, so it binds the same way.
        caps = load_loop_guard_v2_caps()
        if not hmac.compare_digest(caps.sha256, caps_sha256_pin):
            raise CandidateRuntimeIdentityMismatch(
                f"loop guard v2 caps sha256 {caps.sha256!r} does not match "
                f"its bundle pin {caps_sha256_pin!r}"
            )
        cache_identity = state.get("cache_identity")
        if not isinstance(cache_identity, Mapping):
            raise CandidateRuntimeIdentityMismatch(
                "checkpoint carries no cache_identity to bind loop guard v2 caps"
            )
        try:
            refuse_unless_caps_bind(caps, cache_identity)
        except LoopGuardCapsStale as exc:
            raise CandidateRuntimeIdentityMismatch(str(exc)) from exc
    if decode_policy.typed_decode:
        checkpoint_thc_hash = state.get("typed_head_config_hash")
        current_thc_hash = typed_head_config_hash()
        if checkpoint_thc_hash is None or checkpoint_thc_hash != current_thc_hash:
            raise CandidateRuntimeIdentityMismatch(
                f"checkpoint typed_head_config_hash {checkpoint_thc_hash!r} != "
                f"current {current_thc_hash!r}"
            )


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _load_verified_checkpoint(
    path: Path, *, expected_sha256: str, map_location: torch.device
) -> Mapping[str, Any]:
    """Load one regular, non-empty checkpoint from its already-verified bytes."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CandidateRuntimeIdentityMismatch("checkpoint cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CandidateRuntimeIdentityMismatch("checkpoint is not a regular file")
        if before.st_size <= 0:
            raise CandidateRuntimeIdentityMismatch("checkpoint is empty")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            checkpoint_bytes = stream.read()
            after = os.fstat(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if before.st_size != len(checkpoint_bytes) or after.st_size != before.st_size:
        raise CandidateRuntimeIdentityMismatch("checkpoint size changed while being read")
    if not hmac.compare_digest(_sha256(checkpoint_bytes), expected_sha256):
        raise CandidateRuntimeIdentityMismatch("checkpoint raw SHA-256 does not match its pin")
    state = torch.load(
        io.BytesIO(checkpoint_bytes), weights_only=True, map_location=map_location
    )
    if not isinstance(state, Mapping):
        raise CandidateRuntimeIdentityMismatch("checkpoint payload is not a mapping")
    return state


def verify_runtime_identity(
    *,
    candidate: QwenCandidate,
    loaded_revision: str,
    checkpoint_state: Mapping[str, Any],
    component_lock: dict[str, Any],
    pin: RuntimeIdentityPin,
    checkpoint_bytes: bytes | None = None,
    checkpoint_sha256: str | None = None,
) -> RuntimeIdentityPin:
    """Verify actual candidate, checkpoint bytes and lock; never trust labels alone."""
    lock_issues = validate_exact_text_s1_component_lock(component_lock)
    if lock_issues:
        raise CandidateRuntimeIdentityMismatch(f"component lock invalid: {lock_issues[0].message}")
    actual_checkpoint_hash = checkpoint_sha256
    if checkpoint_bytes is not None:
        actual_checkpoint_hash = _sha256(checkpoint_bytes)
    actual_lock_hash = component_lock_hash(component_lock)
    actual = (candidate.model_id, loaded_revision, candidate.attn_implementation)
    expected = (pin.model_id, pin.revision, pin.attention_implementation)
    if actual != expected:
        raise CandidateRuntimeIdentityMismatch(
            f"candidate identity {actual!r} does not match pinned identity {expected!r}"
        )
    if actual_checkpoint_hash != pin.checkpoint_sha256:
        raise CandidateRuntimeIdentityMismatch("checkpoint raw SHA-256 does not match its pin")
    if actual_lock_hash != pin.component_lock_hash:
        raise CandidateRuntimeIdentityMismatch("component lock hash does not match its pin")
    cache_identity = checkpoint_state.get("cache_identity")
    if not isinstance(cache_identity, Mapping) or (
        cache_identity.get("component_lock_hash") != actual_lock_hash
    ):
        raise CandidateRuntimeIdentityMismatch(
            "checkpoint cache identity is not bound to the verified component lock"
        )
    return pin


def _move_candidate_to_device(candidate: QwenCandidate, device: torch.device) -> None:
    """Move an already-quantized, CPU-resident candidate to `device`
    (owner requirement 2026-09-27: quantize on CPU, move afterward --
    quantizing a GPU-resident bf16 candidate in place OOMs on the first
    replaced linear, `docs/runs/quantization-rtx4070-2026-09-27.md`
    finding 3). `nn.Module.to()` moves parameters/buffers in place, so
    this mutates the exact modules `candidate.decode`/`vision_forward`
    already close over -- no new candidate object is built.

    `lm_head_weight` is a plain tensor field (not itself an `nn.Module`,
    and not always the same object as `embed_tokens.weight` --
    `candidate_loader.load_qwen_candidate` only ties them when the loaded
    checkpoint does), so it is moved explicitly via in-place `.data`
    reassignment rather than relying on it being reachable from
    `decoder_module`/`embed_tokens`.

    Real-hardware finding 2026-09-27: `load_qwen_candidate` always runs
    `optimize_patch_embed` (`probe.backbones`) once, unconditionally, right
    after its own (possibly no-op, when `device=None`) `.to(device)` call.
    That rewrite replaces the patch embedding's `.forward` with a closure
    over `conv.weight`/`conv.bias` AS THEY WERE AT THAT MOMENT -- when this
    function's caller loaded onto CPU first (`device=None`, to quantize),
    the closure captured CPU tensors, and `vision_module(candidate).to
    (device)` above moves the conv module's own parameters but cannot reach
    into that already-built closure. Re-running `optimize_patch_embed`
    AFTER the move rebuilds the closure over the now-GPU-resident tensors
    -- exactly the order its own docstring already documents, just applied
    a second time for the CPU-first quantized path."""
    decoder_module(candidate).to(device)
    vision = vision_module(candidate)
    vision.to(device)
    optimize_patch_embed(vision)
    candidate.embed_tokens.to(device)
    if candidate.lm_head_weight.device != device:
        candidate.lm_head_weight.data = candidate.lm_head_weight.data.to(device)


#: DEC-056 `visual_feature_tap` prefix of a checkpoint trained with the
#: native DeepStack levels (`onfly_visual.visual_feature_tap`).
_DEEPSTACK_TAP_PREFIX = "pooler_output+deepstack:"


def runtime_level_router_policy(checkpoint_state: Mapping[str, Any]) -> LevelRouterPolicy | None:
    """RFC007: the level-router policy a checkpoint was trained with, or
    `None` (no `level_router_policy` key -- every v15b checkpoint).

    Refused before any GPU load: a malformed payload; a policy still
    `PROPOSED` (no trainer can produce one without the owner's record); a
    checkpoint that does not declare the DeepStack levels the router reads
    as its memories (`visual_feature_tap`, DEC-056). The reverse case -- a
    router's weights under a stripped key -- is the `strict=True` load's
    unexpected-keys refusal."""
    payload = checkpoint_state.get(LEVEL_ROUTER_CHECKPOINT_KEY)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise CandidateRuntimeIdentityMismatch("checkpoint level_router_policy must be an object")
    try:
        policy = level_router_policy_from_payload(payload)
    except LevelRouterPolicyError as exc:
        raise CandidateRuntimeIdentityMismatch(
            f"checkpoint level_router_policy refused: {exc}"
        ) from exc
    if policy.status != "RATIFIED":
        raise CandidateRuntimeIdentityMismatch(
            f"checkpoint level_router_policy status is {policy.status!r} (PROPOSED "
            "placeholder); only a ratified RFC007 policy can have been trained"
        )
    from unflatten.model import onfly_visual as ov

    tap = ov.checkpoint_visual_feature_tap(checkpoint_state)
    if not tap.startswith(_DEEPSTACK_TAP_PREFIX):
        raise CandidateRuntimeIdentityMismatch(
            f"checkpoint carries a level router but visual_feature_tap is {tap!r}: the "
            "router reads the DeepStack levels, so the runtime only serves it together "
            f"with a '{_DEEPSTACK_TAP_PREFIX}...' checkpoint (DEC-056, RFC007 §5.2)"
        )
    return policy


def require_runtime_deepstack_matches(
    state: Mapping[str, Any],
    *,
    deepstack: bool,
    deepstack_visual_indexes: tuple[int, ...] | None = None,
) -> None:
    """DEC-056 point 2: a checkpoint trained without DeepStack never runs
    with it, and one trained with it never without. The checkpoint's
    stamped `visual_feature_tap` (absent == `pooler_output`, every v15b
    checkpoint) decides; with `deepstack_visual_indexes` the exact taps of
    the loaded tower must match too."""
    from unflatten.model import onfly_visual as ov

    tap = ov.checkpoint_visual_feature_tap(state)
    if deepstack != (tap != ov.POOLER_OUTPUT_TAP):
        raise CandidateRuntimeIdentityMismatch(
            f"checkpoint visual_feature_tap {tap!r} but the runtime was started with "
            f"deepstack={deepstack} -- a checkpoint trained without DeepStack never "
            "runs with it, and one trained with it never without"
        )
    if deepstack and deepstack_visual_indexes is not None:
        expected = ov.visual_feature_tap(deepstack_visual_indexes)
        if tap != expected:
            raise CandidateRuntimeIdentityMismatch(
                f"checkpoint visual_feature_tap {tap!r} != the loaded tower's {expected!r}"
            )


def build_candidate_model(
    candidate: QwenCandidate,
    device: torch.device,
    *,
    level_router_policy: LevelRouterPolicy | None = None,
    deepstack: bool = False,
) -> QwenDenseCandidateModel:
    """Assemble the existing dense candidate without adding another model path.

    `level_router_policy` (RFC007, from `runtime_level_router_policy`):
    `None` (default) is exactly today's model. `deepstack` (DEC-056, default
    off) builds the DeepStack variant of the same model -- no extra
    parameter, only the decoder-call behaviour."""
    ranges = composite_vocab.packed_ranges(candidate.vocab_size)
    model = QwenDenseCandidateModel(
        decode=candidate.decode,
        inject_visual=candidate.inject_visual,
        visual_dim=candidate.visual_hidden_size,
        d_model=candidate.hidden_size,
        ranges=ranges,
        qwen_embedding_weight=candidate.embed_tokens.weight,
        qwen_lm_head_weight=candidate.lm_head_weight,
        variant=level_router_variant(level_router_policy),
        **(
            {"deepstack": True, "image_token_id": candidate.image_token_id}
            if deepstack
            else {}
        ),
    )
    model.to(device)
    model.eval()
    return model


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateRuntimeIdentityMismatch(f"{label} JSON cannot be loaded") from exc
    if not isinstance(value, dict):
        raise CandidateRuntimeIdentityMismatch(f"{label} JSON must be an object")
    return value


def _build_locked_vocabulary(
    *,
    component_lock: Mapping[str, Any],
    allowlist: Mapping[str, Any],
    tokenizer_artifact_set_hash: str,
    native_vocab_size: int,
) -> composite_vocab.CompositeVocabulary:
    """Validate and bind all text identities before constructing vocabulary state."""
    try:
        composite_vocab.validate_qwen_text_allowlist(allowlist)
    except composite_vocab.TextTokenAllowlistInvalid as exc:
        raise CandidateRuntimeIdentityMismatch(f"allowlist invalid: {exc}") from exc
    if allowlist["native_vocab_size"] != native_vocab_size:
        raise CandidateRuntimeIdentityMismatch("allowlist native vocabulary size mismatch")
    if allowlist["tokenizer_artifact_set_hash"] != tokenizer_artifact_set_hash:
        raise CandidateRuntimeIdentityMismatch("allowlist tokenizer identity mismatch")

    components = {entry["kind"]: entry for entry in component_lock["components"]}
    try:
        lock_identity = text_tokenizer_identity(component_lock)
    except KeyError as exc:
        raise CandidateRuntimeIdentityMismatch(f"component lock tokenizer entry: {exc}") from exc
    if lock_identity != tokenizer_artifact_set_hash:
        raise CandidateRuntimeIdentityMismatch("component lock tokenizer identity mismatch")

    allowlist_identity = composite_vocab.qwen_text_allowlist_hash(allowlist)
    vocabulary_payload, ranges = composite_vocab.build_composite_vocabulary(
        tokenizer_artifact_set_hash=tokenizer_artifact_set_hash,
        text_allowlist_hash=allowlist_identity,
        native_vocab_size=native_vocab_size,
    )
    vocabulary_identity = composite_vocab.composite_vocabulary_hash(vocabulary_payload)
    if components["composite_vocabulary"].get("semantic_hash") != vocabulary_identity:
        raise CandidateRuntimeIdentityMismatch("component lock vocabulary identity mismatch")
    return composite_vocab.CompositeVocabulary(
        artifact_hash=vocabulary_identity,
        ranges=ranges,
        qwen_text_allowlist=frozenset(allowlist["allowed_native_ids"]),
        tokenizer_artifact_set_hash=tokenizer_artifact_set_hash,
    )


#: Every value `CandidateRuntime.kv_cache_mode` accepts.
KV_CACHE_MODES = ("bf16_dynamic", "static", "static_graph", "fp8_full", "fp8_chunked")


@dataclass(slots=True)
class CandidateRuntime:
    """One already-loaded candidate/checkpoint, reused for serialized requests."""

    candidate: QwenCandidate
    vision: CandidateVisionBackbone
    model: QwenDenseCandidateModel
    vocabulary: composite_vocab.CompositeVocabulary
    #: Frozen decode-time policy (design D5): builds every `DecodeConfig`
    #: this runtime ever issues -- never rebuilt ad hoc per call.
    decode_policy: DecodePolicy
    identity: RuntimeIdentityPin
    #: Resolved from the loaded component lock's `model_capability_profile`
    #: component (DEC-024 wiring), never an import-time global: decoding
    #: must use the profile this checkpoint's own lock names, not
    #: whichever profile happened to be bound when this module was
    #: written.
    capability_profile: CapabilityProfile
    #: The DEC-007 D3-pinned tokenizer this checkpoint's decode identity was
    #: verified against at load time (D6/§26); sealing/rendering read this,
    #: never a module-level default.
    tokenizer: Any
    #: `state["cache_identity"]` of the loaded checkpoint (D5/G4) -- the
    #: exact-text cache identity the loop guard v2 caps and the decode's own
    #: identity are bound to.
    eval_cache_identity: Mapping[str, Any]
    component_lock_hash: str
    selection_policy_hash: str
    decode_fn: Callable[..., RawDocumentHypothesis] = decode_free_running
    #: Owner requirement 2026-09-27: `"bf16"` (default) is exactly today's
    #: behaviour -- `decode_features` passes no `initial_cache_factory` at
    #: all. `"fp8"` seeds the decode's KV cache with `Float8KVCache`.
    kv_cache_dtype: str = "bf16"
    #: Owner requirement 2026-09-28 (`docs/runs/dspark-rehearsal-2026-09-27.md`
    #: §22-§23): the ONE switch for what actually backs the KV cache,
    #: replacing two earlier, narrower knobs (`allow_fp8_kv_cache_on_region_
    #: decode: bool`, and a `fp8_kv_cache_mode` string that only chose
    #: between two fp8 backends) -- neither is read anywhere outside this
    #: module and its own tests, so both were removed outright, no alias.
    #:
    #: `"bf16_dynamic"` (default) is exactly today's EFFECTIVE behaviour on
    #: every decode path, including region_decode: `decode_features` passes
    #: no `initial_cache_factory` at all, regardless of `kv_cache_dtype`'s
    #: label -- `decode_free_running`'s `region_decode` branch used to
    #: silently drop any `initial_cache_factory` (`encode_prefix_with_
    #: visual_hidden` had no `initial_cache` parameter at all), so a
    #: `kv_cache_dtype="fp8"` runtime's region_decode pages ALWAYS ran on a
    #: bf16 `DynamicCache` in practice; keeping this the default now that
    #: the plumbing fix lets the factory actually reach that path preserves
    #: that effective behaviour rather than silently doubling latency.
    #:
    #: Any other value IS the explicit opt-in -- no separate region_decode
    #: gate on top of it: `"static"` selects a preallocated `StaticKVCache`,
    #: `"static_graph"` the same cache with its plain q=1 decode steps
    #: replayed as piecewise CUDA graphs (`decode_graph`, owner speed item
    #: 1, 2026-09-29; tokens bit-identical to `"static"`),
    #: `"fp8_full"`/`"fp8_chunked"` select `Float8KVCache`/
    #: `Float8KVCacheChunked` (see `kv_cache_effective`'s docstring for why
    #: the fp8 choices still need `kv_cache_dtype == "fp8"` to take effect).
    kv_cache_mode: str = "bf16_dynamic"
    #: Owner speed item 5 (2026-09-29, `docs/runs/fp8-qkv-fusion-2026-09-29.
    #: md`): whether the fp8 decoder's q/k/v and gate/up siblings share one
    #: activation quantization (`quantize.share_sibling_input_quantization`).
    #: `False` (default) is exactly today's behaviour; set only by
    #: `load_candidate_runtime`, recorded here so a report can state it.
    fp8_shared_input_quant: bool = False

    def kv_cache_effective(self) -> str:
        """What KV cache backend the NEXT `decode_features` call will
        actually use: `"bf16_dynamic"`, `"static"`, `"static_graph"`,
        `"fp8_full"`, or
        `"fp8_chunked"`. Exists so a caller (logging, a future readiness/
        decode-report field) can report the true state instead of trusting
        `kv_cache_mode`'s label, which can still be downgraded below.

        An `"fp8_full"`/`"fp8_chunked"` request is downgraded to
        `"bf16_dynamic"` when `kv_cache_dtype != "fp8"` -- the bundle's own
        `QuantizationSpec` pin, not `kv_cache_mode`, says whether this
        checkpoint was ever quantized for an fp8 KV cache at all; asking
        for one on a bf16 bundle would build a cache the model's weights
        were never validated against, not a cheaper equivalent. `"static"`
        has no such precondition -- it stores full-precision K/V, just in a
        preallocated buffer, so it works under any `kv_cache_dtype`.

        `"static"` is downgraded to `"bf16_dynamic"` when the decode policy
        opts into chunked prefill (`prefill_chunk_tokens`): both prefix
        methods refuse `initial_cache` together with it, so a static cache
        there would fail every page rather than run on a slower cache.
        `"static_graph"` is the same cache, so the same rule applies."""
        if self.kv_cache_mode in ("fp8_full", "fp8_chunked") and self.kv_cache_dtype != "fp8":
            return "bf16_dynamic"
        if (
            self.kv_cache_mode in ("static", "static_graph")
            and self.decode_policy.prefill_chunk_tokens is not None
        ):
            return "bf16_dynamic"
        return self.kv_cache_mode

    def extract(self, model_input: ModelSafeInput) -> PreparedCandidateInput:
        """Validate pixels and perform only candidate-native visual extraction."""
        _validate_input(model_input)
        plane = model_input.planes[0]
        image = Image.frombytes("RGB", (plane.width_px, plane.height_px), plane.pixels)
        levels: tuple[torch.Tensor, ...] | None = None
        if getattr(self.model, "deepstack", False):
            extracted, raw_levels = self.vision.extract_with_deepstack(image)
        else:
            extracted = self.vision.extract(image)
        device = next(self.model.parameters()).device
        if getattr(self.model, "deepstack", False):
            levels = tuple(level.to(device) for level in raw_levels)
        features = extracted.visual_features.to(device)
        if features.dim() == 2:
            features = features.unsqueeze(0)
        grid = torch.tensor([extracted.grid_thw], dtype=torch.long, device=device)
        return PreparedCandidateInput(features, grid, deepstack_visual_embeds=levels)

    def decode_features(self, prepared: PreparedCandidateInput) -> RawDocumentHypothesis:
        """Run constrained free decoding over previously extracted visual features."""
        controller = CompositeGrammarController(
            vocabulary=self.vocabulary,
            capability_profile_id=self.capability_profile.profile_id,
            structural_profile=self.capability_profile.ast_profile,
            capability_excluded_local_ids=(
                self.capability_profile.excluded_structural_local_ids
            ),
        )
        effective = self.kv_cache_effective()
        if self.kv_cache_dtype == "fp8" and effective == "bf16_dynamic":
            logger.info(
                "kv_cache_effective=bf16_dynamic (kv_cache_dtype=fp8, but "
                "kv_cache_mode=%s)",
                self.kv_cache_mode,
            )
        elif self.kv_cache_mode in ("static", "static_graph") and effective == "bf16_dynamic":
            logger.info(
                "kv_cache_effective=bf16_dynamic (kv_cache_mode=%s, but "
                "prefill_chunk_tokens=%s)",
                self.kv_cache_mode,
                self.decode_policy.prefill_chunk_tokens,
            )
        if effective == "fp8_full":
            initial_cache_factory = Float8KVCache
        elif effective == "fp8_chunked":
            initial_cache_factory = Float8KVCacheChunked
        elif effective in ("static", "static_graph"):
            # No speculative rollback happens through this call (`decode_fn`
            # defaults to `decode_free_running`, never the speculative
            # variant, in every production caller) so the capacity only
            # needs to cover the whole KV prefix -- the document-context
            # summary tokens, the visual block and the BOS token after it
            # (`QwenDenseCandidateModel._merge_inputs`) -- plus one
            # `encode_step` for every step `decode_policy.max_steps` allows.
            # Leaving out the context tokens overflowed the buffer on the
            # last steps of a page that runs to `max_steps`, so it failed as
            # `StaticKVCacheCapacityExceeded` instead of
            # `DecodeMaxStepsExceeded`.
            capacity = (
                self.model.document_context_encoder.context_tokens
                + prepared.visual_features.shape[1]
                + 1
                + self.decode_policy.max_steps
            )

            cache_class = StaticKVCache if effective == "static" else GraphedStaticKVCache

            def initial_cache_factory() -> StaticKVCache:
                return cache_class(capacity=capacity)
        else:
            initial_cache_factory = None
        return self.decode_fn(
            model=self.model,
            controller=controller,
            vocabulary=self.vocabulary,
            visual_features=prepared.visual_features,
            image_grid_thw=prepared.image_grid_thw,
            image_token_id=self.candidate.image_token_id,
            initial_cache_factory=initial_cache_factory,
            **(
                {}
                if prepared.deepstack_visual_embeds is None
                else {"deepstack_visual_embeds": prepared.deepstack_visual_embeds}
            ),
            config=DecodeConfig(
                capability_profile_id=self.capability_profile.profile_id,
                max_steps=self.decode_policy.max_steps,
                loop_guard=self.decode_policy.loop_guard,
                loop_guard_v3_arm=self.decode_policy.loop_guard_v3_arm,
                typed_decode=self.decode_policy.typed_decode,
                region_decode=self.decode_policy.region_decode,
                prefill_chunk_tokens=self.decode_policy.prefill_chunk_tokens,
            ),
        )

    def decode(self, model_input: ModelSafeInput) -> RawDocumentHypothesis:
        """Preserve the existing combined adapter for non-worker callers."""
        return self.decode_features(self.extract(model_input))


def _validate_input(model_input: ModelSafeInput) -> None:
    forbidden = validate_no_forbidden(to_payload_dict(model_input))
    if forbidden:
        raise CandidateRuntimeInputInvalid(forbidden[0].message)
    if (
        model_input.input_mode != "single_page"
        or model_input.profile != "word-core-v0"
        or len(model_input.planes) != 1
        or model_input.planes[0].page_ordinal != 0
        or model_input.scale.status != "known"
        or model_input.scale.dpi != 300
    ):
        raise CandidateRuntimeInputInvalid(
            "runtime requires one ordinal-0 single_page, word-core-v0, known 300-DPI input"
        )


def load_candidate_runtime(
    *,
    checkpoint_path: Path,
    component_lock_path: Path,
    allowlist_path: Path,
    tokenizer_artifact_set_hash: str,
    pin: RuntimeIdentityPin,
    device: str,
    decode_policy: DecodePolicy,
    caps_sha256_pin: str,
    decode_steps_hard_ceiling: int,
    quantization: QuantizationSpec | None = None,
    kv_cache_mode: str = "bf16_dynamic",
    fp8_shared_input_quant: bool = False,
    deepstack: bool = False,
) -> CandidateRuntime:
    """Load and verify the sole resident diagnostic model at worker startup.

    The checkpoint is always read onto the CPU (D6) and every decode-time
    binding (`verify_decode_bindings`) is refused before `load_qwen_candidate`
    even runs, so a bad checkpoint never pays for a GPU candidate load.

    `quantization` (owner requirement 2026-09-27): `None` (default, treated
    as `bf16_spec()`) is exactly today's behaviour -- `load_qwen_candidate`
    loads straight onto `device`. A non-bf16 spec instead loads the
    candidate onto the CPU, quantizes it there
    (`quantization.quantize.quantize_candidate`), and only then moves the
    (now much smaller) decoder/vision/embedding tensors to `device`
    (`_move_candidate_to_device`) -- quantizing a GPU-resident candidate in
    place OOMs on the first replaced linear on a 12 GiB card
    (`docs/runs/quantization-rtx4070-2026-09-27.md` finding 3).

    `kv_cache_mode` (owner requirement 2026-09-28): `"bf16_dynamic"`
    (default) is exactly today's effective behaviour. Kept a plain runtime
    setting, not part of `QuantizationSpec`, so it never changes
    `QuantizationSpec.spec_hash` for any bundle. See
    `CandidateRuntime.kv_cache_mode`'s docstring for the other three values.

    `fp8_shared_input_quant` (owner speed item 5, 2026-09-29): `False`
    (default) is today's behaviour. `True` makes the fp8 decoder's sibling
    projections share one activation quantization -- bit-identical output,
    `docs/runs/fp8-qkv-fusion-2026-09-29.md`. A runtime setting like
    `kv_cache_mode`, never part of `QuantizationSpec.spec_hash`; refused
    before any load unless `quantization.decoder_precision == "fp8"`.

    `deepstack` (DEC-056, default off): run a checkpoint trained with
    DeepStack. The checkpoint's own `visual_feature_tap` must agree in both
    directions -- a v15b checkpoint (no tap) is refused under `deepstack`,
    a DeepStack checkpoint without it -- before the candidate is loaded, and
    the exact taps are checked against the loaded tower after.
    """
    from transformers import AutoImageProcessor

    quantization = quantization or bf16_spec()
    if fp8_shared_input_quant and quantization.decoder_precision != "fp8":
        raise QuantizationError(
            "fp8_shared_input_quant needs decoder_precision='fp8', got "
            f"{quantization.decoder_precision!r}"
        )
    torch_device = torch.device(device)
    state = _load_verified_checkpoint(
        checkpoint_path,
        expected_sha256=pin.checkpoint_sha256,
        map_location=torch.device("cpu"),
    )
    verify_decode_bindings(
        state,
        decode_policy,
        caps_sha256_pin=caps_sha256_pin,
        decode_steps_hard_ceiling=decode_steps_hard_ceiling,
    )
    level_router_policy = runtime_level_router_policy(state)
    require_runtime_deepstack_matches(state, deepstack=deepstack)
    # The pinned tokenizer is a startup identity gate, not an optional
    # renderer helper. Refuse a missing or substituted snapshot before the
    # expensive Qwen candidate (and GPU) is loaded.
    try:
        tokenizer, snapshot = load_pinned_tokenizer(
            model_id=QWEN_CANDIDATE_MODEL_ID, revision=QWEN_CANDIDATE_REVISION
        )
        loaded_tokenizer_hash = compute_tokenizer_artifact_set_hash(
            snapshot, model_id=QWEN_CANDIDATE_MODEL_ID, revision=QWEN_CANDIDATE_REVISION
        )
    except TokenizerArtifactSetInvalid as exc:
        raise CandidateRuntimeIdentityMismatch(f"tokenizer snapshot refused: {exc}") from exc
    if loaded_tokenizer_hash != tokenizer_artifact_set_hash:
        raise CandidateRuntimeIdentityMismatch(
            "tokenizer artifact-set hash does not match its pin"
        )
    candidate = load_qwen_candidate(
        model_id=QWEN_CANDIDATE_MODEL_ID,
        revision=QWEN_CANDIDATE_REVISION,
        # Quantized: load onto CPU first (`device=None`), quantize there,
        # then move the quantized candidate to `device` below -- never
        # quantize a GPU-resident candidate in place (module docstring).
        device=None if not quantization.is_bf16 else device,
        attn_implementation="sdpa",
    )
    if not quantization.is_bf16:
        quantize_candidate(candidate, quantization)
        if fp8_shared_input_quant:
            share_sibling_input_quantization(decoder_module(candidate))
        _move_candidate_to_device(candidate, torch_device)
    component_lock = _load_json_object(component_lock_path, "component lock")
    allowlist = _load_json_object(allowlist_path, "allowlist")
    verify_runtime_identity(
        candidate=candidate,
        loaded_revision=QWEN_CANDIDATE_REVISION,
        checkpoint_sha256=pin.checkpoint_sha256,
        checkpoint_state=state,
        component_lock=component_lock,
        pin=pin,
    )
    vocabulary = _build_locked_vocabulary(
        component_lock=component_lock,
        allowlist=allowlist,
        tokenizer_artifact_set_hash=tokenizer_artifact_set_hash,
        native_vocab_size=candidate.vocab_size,
    )
    if deepstack:
        require_runtime_deepstack_matches(
            state, deepstack=True, deepstack_visual_indexes=candidate.deepstack_visual_indexes
        )
    model = build_candidate_model(
        candidate,
        torch_device,
        level_router_policy=level_router_policy,
        deepstack=deepstack,
    )
    model.load_state_dict(state["model_state_dict"], strict=True)
    processor = AutoImageProcessor.from_pretrained(
        QWEN_CANDIDATE_MODEL_ID, revision=QWEN_CANDIDATE_REVISION
    )
    vision = CandidateVisionBackbone(candidate=candidate, processor=processor, device=device)
    try:
        capability_profile = resolve_capability_profile_from_component_lock(component_lock)
    except CapabilityProfileHashUnresolved as exc:
        raise CandidateRuntimeIdentityMismatch(
            f"component lock capability profile refused: {exc}"
        ) from exc
    actual_lock_hash = component_lock_hash(component_lock)
    return CandidateRuntime(
        candidate=candidate,
        vision=vision,
        model=model,
        vocabulary=vocabulary,
        decode_policy=decode_policy,
        identity=pin,
        capability_profile=capability_profile,
        tokenizer=tokenizer,
        eval_cache_identity=dict(state["cache_identity"]),
        component_lock_hash=actual_lock_hash,
        selection_policy_hash=decode_policy.selection_policy_hash,
        kv_cache_dtype=quantization.kv_cache_dtype,
        kv_cache_mode=kv_cache_mode,
        fp8_shared_input_quant=fp8_shared_input_quant,
    )
