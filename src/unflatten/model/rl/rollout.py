"""Stochastic constrained rollouts for S4.5 (RFC005-ERRATA-010 E3.3/E5.2; dormant).

A separate sampler: ``free_running_decode.decode_free_running`` and
``DecodeConfig`` are NOT touched, so inference stays the pinned greedy decode and
no sampling field leaks into a Model Release or decode config (E5.2). This module
composes the same grammar/model surface (``_legal_packed_ids``, ``encode_prefix`` /
``encode_step`` / ``composite_head``, ``LoopGuardV3``) around a seeded sampler.

Sampling order is fixed: grammar-legal ids -> logits restricted to that set ->
temperature -> top-p -> renormalise -> draw. An id outside the legal set therefore
has probability exactly zero at any temperature. Temperature and top-p come from the
policy as integer micro units; the per-rollout seed is a ``domain_hash`` of
``(run_seed, doc_id, rollout_index)``.

A trajectory carries tokens and decode evidence only -- no reward, no gold, no
log-probability. Old log-probs are recomputed later on the teacher-forced update
path under the same legal mask (the per-step legal-set digest is recorded so the
two can be compared).
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch

from unflatten.model import events_vocab
from unflatten.model.candidate_model import KVCacheSnapshot, TeacherForcedBatch
from unflatten.model.capability_profile import STANDALONE_V1_PROFILE_ID
from unflatten.model.composite_vocab import CompositeVocabulary, QwenRef, StructuralRef
from unflatten.model.free_running_decode import (
    DecodeMaxStepsExceeded,
    DecodeVocabularyExhausted,
    _legal_packed_ids,
)
from unflatten.model.hashing import domain_hash
from unflatten.model.rl.ids import ROLLOUT_SEED_DOMAIN
from unflatten.model.rl.policies import (
    INTEGER_HEAD_KINDS,
    SAMPLEABLE_HEAD_KINDS,
    verify_rollout_policy,
)
from unflatten.model.rl.scorers import MICRO

STATUS_TERMINATED: Final = "terminated"
STATUS_MAX_STEPS: Final = DecodeMaxStepsExceeded.__name__
STATUS_VOCAB_EXHAUSTED: Final = DecodeVocabularyExhausted.__name__
#: Code-level safety limit on ``group_size * max_steps`` of one call (not a policy field).
MAX_GROUP_STEPS: Final = 1_000_000
#: Substrings no trajectory field name may contain (E3.1: reward/gold never enter
#: tensors; E4: log-probs are recomputed on the update path).
FORBIDDEN_FIELD_PARTS: Final = ("reward", "gold", "logprob", "log_prob", "advantage", "bbox")


class RolloutRefused(ValueError):
    """A rollout request that the policy or the S4.5 contract forbids."""


@dataclass(frozen=True)
class RewindEvent:
    """One loop-guard rewind (policy ``loop_guard_rewind_recorded_in_trajectory``).
    ``step`` is the number of tokens accepted when the guard fired."""

    step: int
    to_position: int
    banned_packed_id: int
    rule: str
    period: int
    count: int


@dataclass(frozen=True)
class RolloutTrajectory:
    rollout_index: int
    seed: int
    policy_hash: str
    #: Accepted packed ids; on a typed failure, the partial stream.
    token_ids: tuple[int, ...]
    #: ``legal_set_sizes[i]`` / ``legal_set_digests[i]`` describe the grammar-legal
    #: set token ``i`` was drawn from (before any guard narrowing).
    legal_set_sizes: tuple[int, ...]
    legal_set_digests: tuple[str, ...]
    rewinds: tuple[RewindEvent, ...]
    #: ``"terminated"`` or the typed failure name (``DecodeMaxStepsExceeded`` /
    #: ``DecodeVocabularyExhausted``); a failure is data, never a raised crash.
    status: str
    steps: int
    #: ``LoopGuardV3.record()`` when the policy runs loop guards, else ``None``.
    guard_record: Mapping[str, Any] | None
    #: Indices into ``token_ids`` where the guard replaced the sampled id (veto
    #: substitution; ``force_content`` is not one). Truncated on rewind.
    substituted_positions: tuple[int, ...] = ()


def trajectory_field_violations(cls: type) -> list[str]:
    """Field names of dataclass ``cls`` that may not appear on a trajectory."""
    return [
        f.name
        for f in dataclasses.fields(cls)
        if any(part in f.name.lower() for part in FORBIDDEN_FIELD_PARTS)
    ]


def derive_rollout_seed(run_seed: int, doc_id: str, rollout_index: int) -> int:
    """63-bit seed ``H(run_seed, doc_id, rollout_index)``; integers and one string only."""
    for name, value in (("run_seed", run_seed), ("rollout_index", rollout_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RolloutRefused(f"{name} must be a non-negative int, got {value!r}")
    if not isinstance(doc_id, str) or not doc_id:
        raise RolloutRefused(f"doc_id must be a non-empty str, got {doc_id!r}")
    digest = domain_hash(
        ROLLOUT_SEED_DOMAIN,
        {"run_seed": run_seed, "doc_id": doc_id, "rollout_index": rollout_index},
    )
    return int(digest.removeprefix("sha256:")[:16], 16) & (2**63 - 1)


def _refuse_bad_policy(policy: Mapping[str, Any]) -> None:
    verify_rollout_policy(policy)
    integer = [k for k in policy["typed_head_sampling"] if k in INTEGER_HEAD_KINDS]
    if integer:  # defensive: build_rollout_policy and the schema already refuse this
        raise RolloutRefused(f"integer regression heads are argmax, never sampled: {integer}")


def _sample_index(
    logits: torch.Tensor, policy: Mapping[str, Any], generator: torch.Generator
) -> int:
    """Draw one index from ``logits`` (already restricted to the legal set):
    temperature, then top-p (ties broken by lower index), renormalise, multinomial."""
    scaled = logits.detach().to("cpu", torch.float64) * (MICRO / policy["temperature_micro"])
    probs = torch.softmax(scaled, dim=0)
    top_p_micro = policy["top_p_micro"]
    if top_p_micro < MICRO:
        order = torch.sort(probs, descending=True, stable=True).indices
        cumulative = torch.cumsum(probs[order], dim=0)
        keep = int(torch.searchsorted(cumulative, torch.tensor(top_p_micro / MICRO)).item()) + 1
        masked = torch.zeros_like(probs)
        masked[order[:keep]] = probs[order[:keep]]
        probs = masked / masked.sum()
    return int(torch.multinomial(probs, 1, generator=generator).item())


def sample_from_legal(
    logits_row: torch.Tensor,
    legal_ids: Sequence[int],
    policy: Mapping[str, Any],
    generator: torch.Generator,
) -> int:
    """Packed id drawn from ``legal_ids`` only: the mask comes first, temperature
    and top-p act on the already-masked set."""
    if not legal_ids:
        raise RolloutRefused("cannot sample from an empty legal set")
    ids = sorted(legal_ids)
    index = torch.from_numpy(np.asarray(ids, dtype=np.int64)).to(logits_row.device)
    return ids[_sample_index(logits_row.index_select(0, index), policy, generator)]


def select_head_value(
    kind: str, logits: torch.Tensor, policy: Mapping[str, Any], generator: torch.Generator
) -> int:
    """Class index for one typed head: sampled for the kinds the policy lists,
    argmax for every other categorical kind and for integer regression kinds
    (owner decision). ``boolean`` takes one logit ``l`` and is the two-class
    ``[0, l]``. Unknown kinds are refused."""
    if kind in INTEGER_HEAD_KINDS:
        return int(torch.argmax(logits.reshape(-1)).item())
    if kind not in SAMPLEABLE_HEAD_KINDS:
        raise RolloutRefused(f"unknown typed head kind {kind!r}")
    flat = logits.reshape(-1)
    if kind == "boolean":
        if flat.numel() != 1:
            raise RolloutRefused(f"boolean head takes one logit, got {flat.numel()}")
        flat = torch.cat([torch.zeros_like(flat), flat])
    if kind not in policy["typed_head_sampling"]:
        return int(torch.argmax(flat).item())
    return _sample_index(flat, policy, generator)


def _legal_digest(legal_ids: Sequence[int]) -> str:
    data = np.asarray(sorted(legal_ids), dtype=np.int64).tobytes()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _pin(row: torch.Tensor, legal_ids: Sequence[int], chosen: int) -> torch.Tensor:
    """``row`` with ``chosen`` made the strict legal maximum, so the guard's own
    greedy step lands on the sample; on a veto it falls to the next-best legal id."""
    pinned = row.clone()
    top = row.index_select(0, torch.tensor(sorted(legal_ids), device=row.device)).max()
    pinned[chosen] = top + 1
    return pinned


def _default_guard(vocabulary: CompositeVocabulary, arm: str | None) -> Any:
    from unflatten.model.loop_guard_v2_caps import load_loop_guard_v2_caps
    from unflatten.model.loop_guard_v3 import LoopGuardV3, load_loop_guard_v3_policy, resolve_arm

    policy = load_loop_guard_v3_policy()
    return LoopGuardV3(
        ranges=vocabulary.ranges,
        policy=policy,
        arm=resolve_arm(policy, arm),
        v2_caps=load_loop_guard_v2_caps(),
    )


def _build_guard(
    policy: Mapping[str, Any],
    vocabulary: CompositeVocabulary,
    capability_profile_id: str,
    arm: str | None,
    factory: Callable[[], Any] | None,
) -> Any:
    if not policy["loop_guards_in_policy"]:
        if factory is not None or arm is not None:
            raise RolloutRefused("loop guard supplied but policy.loop_guards_in_policy is false")
        return None
    if factory is not None:
        return factory()
    if capability_profile_id != STANDALONE_V1_PROFILE_ID:
        raise RolloutRefused(
            f"loop guard v3 applies only to {STANDALONE_V1_PROFILE_ID!r}, "
            f"got {capability_profile_id!r}"
        )
    return _default_guard(vocabulary, arm)


def _checked_cache(cache: Any) -> Any:
    """D9: only a plain ``DynamicCache`` (static/graphed/fp8 subclass it: exact type)."""
    from transformers import DynamicCache

    if type(cache) is not DynamicCache:
        raise RolloutRefused(f"initial_cache_factory must build a DynamicCache, got {type(cache)}")
    return cache


class _Walk:
    """Mutable state of one rollout: grammar state, cache, hidden, and the
    per-token ledgers a rewind truncates together."""

    def __init__(self, controller: Any, vocabulary: CompositeVocabulary, profile: str) -> None:
        self.controller, self.vocabulary = controller, vocabulary
        self.state = controller.fresh(profile)
        self.states = [self.state]
        self.tokens: list[int] = []
        self.sizes: list[int] = []
        self.digests: list[str] = []
        self.rewinds: list[RewindEvent] = []
        self.substituted: list[int] = []
        self.anchors: dict[int, torch.Tensor] = {}
        self.cache: Any = None
        self.hidden: torch.Tensor | None = None
        self.base_len = 0

    def accept(self, token_id: int, legal: Sequence[int]) -> None:
        self.sizes.append(len(legal))
        self.digests.append(_legal_digest(legal))
        ref = self.vocabulary.unpack(token_id)
        if isinstance(ref, StructuralRef):
            assert self.hidden is not None, "hidden state is set before the first accept"
            self.anchors[len(self.tokens)] = self.hidden[0].detach().clone()
        self.state = self.controller.step(self.state, ref)
        self.tokens.append(token_id)
        self.states.append(self.state)

    def rewind(self, decision: Any, step: int) -> None:
        position = decision.position
        if position not in self.anchors:
            raise RuntimeError(f"loop guard rewind to {position}: not a structural position")
        self.rewinds.append(
            RewindEvent(
                step=step,
                to_position=position,
                banned_packed_id=decision.banned,
                rule=decision.rule,
                period=decision.period,
                count=decision.count,
            )
        )
        self.hidden = self.anchors[position].unsqueeze(0)
        del self.tokens[position:], self.sizes[position:], self.digests[position:]
        del self.states[position + 1 :]
        self.substituted = [p for p in self.substituted if p < position]
        self.state = self.states[position]
        self.anchors = {i: a for i, a in self.anchors.items() if i < position}
        self.cache = self.cache.rollback_to(KVCacheSnapshot(composite_len=self.base_len + position))


def _rollout(
    *,
    model: Any,
    controller: Any,
    vocabulary: CompositeVocabulary,
    prefix_batch: TeacherForcedBatch,
    capability_profile_id: str,
    policy: Mapping[str, Any],
    guard: Any,
    seed: int,
    max_steps: int,
    rollout_index: int,
    initial_cache_factory: Callable[[], Any] | None = None,
    prefill_chunk_tokens: int | None = None,
) -> RolloutTrajectory:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    walk = _Walk(controller, vocabulary, capability_profile_id)
    prefix_kwargs: dict[str, Any] = {}
    if initial_cache_factory is not None:
        prefix_kwargs["initial_cache"] = _checked_cache(initial_cache_factory())
    if prefill_chunk_tokens is not None:
        prefix_kwargs["prefill_chunk_tokens"] = prefill_chunk_tokens
    walk.cache, walk.hidden = model.encode_prefix(prefix_batch, **prefix_kwargs)
    if guard is not None:
        walk.base_len = walk.cache.snapshot().composite_len
    status = STATUS_MAX_STEPS
    steps = 0
    while steps < max_steps:
        if walk.state.mode == "terminated":
            break
        legal = _legal_packed_ids(controller, walk.state, vocabulary)
        if not legal:
            status = STATUS_VOCAB_EXHAUSTED
            break
        steps += 1
        row = model.composite_head(walk.hidden)[0]
        chosen = sample_from_legal(row, legal, policy, generator)
        sampled = chosen
        if guard is not None:
            chosen = guard.decide(
                _pin(row, legal, chosen),
                legal,
                domain=controller.allowed(walk.state),
                state=walk.state,
                step=len(walk.tokens),
            )
            if not isinstance(chosen, int):
                guard.rewind_to(chosen)
                walk.rewind(chosen, len(walk.tokens))
                continue
        if chosen != sampled:
            walk.substituted.append(len(walk.tokens))
        walk.accept(chosen, legal)
        if guard is not None:
            guard.observe(chosen)
        if walk.state.mode != "terminated":
            walk.cache, walk.hidden = model.encode_step(walk.cache, chosen)
    if walk.state.mode == "terminated":
        status = STATUS_TERMINATED
    return RolloutTrajectory(
        rollout_index=rollout_index,
        seed=seed,
        policy_hash=policy["rollout_policy_hash"],
        token_ids=tuple(walk.tokens),
        legal_set_sizes=tuple(walk.sizes),
        legal_set_digests=tuple(walk.digests),
        rewinds=tuple(walk.rewinds),
        status=status,
        steps=steps,
        guard_record=guard.record() if guard is not None else None,
        substituted_positions=tuple(walk.substituted),
    )


def _resolve_chunk(
    policy: Mapping[str, Any], requested: int | None, cache_factory: Callable[[], Any] | None
) -> int | None:
    profile = policy["execution_profile"]
    chunk = profile["prefill_chunk_tokens"]
    if requested is not None and requested != chunk:
        raise RolloutRefused(f"prefill_chunk_tokens={requested} differs from policy's {chunk!r}")
    if chunk is not None and cache_factory is not None:
        raise RolloutRefused("initial_cache_factory is not supported together with chunked prefill")
    return chunk


def _resolve_arm(policy: Mapping[str, Any], requested: str | None) -> str | None:
    """Policy arm ``"none"`` = no explicit arm; any other policy arm is authoritative."""
    arm = policy["loop_guard_arm"]
    if arm == "none":
        return requested
    if not policy["loop_guards_in_policy"]:
        raise RolloutRefused(f"policy names loop_guard_arm {arm!r} but runs no loop guards")
    if requested is not None and requested != arm:
        raise RolloutRefused(f"loop_guard_arm {requested!r} differs from the policy's {arm!r}")
    return arm


def _checked_levels(
    model: Any, levels: Sequence[torch.Tensor] | None, n_visual: int
) -> tuple[torch.Tensor, ...] | None:
    if levels is None:
        if getattr(model, "deepstack", False):
            raise RolloutRefused("deepstack=True model: deepstack_visual_embeds is required")
        return None
    if getattr(model, "deepstack", None) is False and getattr(model, "level_router", None) is None:
        raise RolloutRefused("deepstack_visual_embeds given to a model built with deepstack=False")
    out = tuple(levels)
    for i, level in enumerate(out):
        if level.dim() != 2 or level.shape[0] != n_visual:
            raise RolloutRefused(
                f"DeepStack level {i} has shape {tuple(level.shape)}, expected ({n_visual}, hidden)"
            )
    return out


def sample_rollouts(
    *,
    model: Any,
    controller: Any,
    vocabulary: CompositeVocabulary,
    visual_features: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    policy: Mapping[str, Any],
    doc_id: str,
    run_seed: int,
    capability_profile_id: str,
    max_steps: int | None = None,
    loop_guard_arm: str | None = None,
    loop_guard_factory: Callable[[], Any] | None = None,
    deepstack_visual_embeds: Sequence[torch.Tensor] | None = None,
    initial_cache_factory: Callable[[], Any] | None = None,
    prefill_chunk_tokens: int | None = None,
) -> list[RolloutTrajectory]:
    """``policy["group_size"]`` independent seeded rollouts for one page.

    Single-page, batch 1, KV-cache path. ``rollout_budget`` is a cross-document
    allowance and is not enforced here (the caller owns that ledger). Loop guards
    (``LoopGuardV3``, a fresh instance per rollout) run only when the policy says
    so; their rewinds are recorded on each trajectory.

    The hashed policy is authoritative for ``max_steps`` (``None`` takes it; an explicit
    value is a legacy override), ``loop_guard_arm`` and ``prefill_chunk_tokens`` (a
    caller value must equal the policy's). ``deepstack_visual_embeds`` is required by a
    ``deepstack=True`` model and refused by one built without."""
    _refuse_bad_policy(policy)
    if visual_features.dim() != 3 or visual_features.shape[0] != 1:
        raise ValueError(
            f"sample_rollouts is single-page/batch=1, got {tuple(visual_features.shape)}"
        )
    limit: int = policy["max_steps"] if max_steps is None else max_steps
    if limit <= 0:
        raise ValueError(f"max_steps must be positive, got {limit}")
    if policy["group_size"] * limit > MAX_GROUP_STEPS:
        raise RolloutRefused(f"group_size * max_steps exceeds the safety limit {MAX_GROUP_STEPS}")
    prefill_chunk_tokens = _resolve_chunk(policy, prefill_chunk_tokens, initial_cache_factory)
    loop_guard_arm = _resolve_arm(policy, loop_guard_arm)
    levels = _checked_levels(model, deepstack_visual_embeds, visual_features.shape[1])
    device = visual_features.device

    def new_guard() -> Any:
        return _build_guard(
            policy, vocabulary, capability_profile_id, loop_guard_arm, loop_guard_factory
        )

    first_guard = new_guard()
    if first_guard is not None:
        from unflatten.model.loop_guard_v3 import ForceContentController, parse_arm

        if parse_arm(first_guard.record()["arm"]).force_content:
            controller = ForceContentController.wrap(controller)
    placeholder = vocabulary.pack(QwenRef(image_token_id))
    bos = vocabulary.pack(StructuralRef(local_id=events_vocab.TOKEN_IDS[events_vocab.END_DOCUMENT]))
    block = [placeholder] * visual_features.shape[1] + [bos]
    prefix = TeacherForcedBatch(
        visual_features=visual_features,
        composite_input_ids=torch.tensor([block], dtype=torch.long, device=device),
        composite_attention_mask=torch.ones(1, len(block), dtype=torch.long, device=device),
        image_grid_thw=image_grid_thw,
        deepstack_visual_embeds=levels,
    )
    out: list[RolloutTrajectory] = []
    with torch.inference_mode():
        for index in range(policy["group_size"]):
            out.append(
                _rollout(
                    model=model,
                    controller=controller,
                    vocabulary=vocabulary,
                    prefix_batch=prefix,
                    capability_profile_id=capability_profile_id,
                    policy=policy,
                    guard=first_guard if index == 0 else new_guard(),
                    seed=derive_rollout_seed(run_seed, doc_id, index),
                    max_steps=limit,
                    rollout_index=index,
                    initial_cache_factory=initial_cache_factory,
                    prefill_chunk_tokens=prefill_chunk_tokens,
                )
            )
    return out
