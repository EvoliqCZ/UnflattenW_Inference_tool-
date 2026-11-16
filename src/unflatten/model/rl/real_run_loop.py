"""One page -> one S4.5 micro-batch (design D1/D2/D6/D7/D9; PROPOSED, dormant).

Called by :mod:`unflatten.model.rl.real_run`; it opens nothing (every resource is
a port). For one train page:

1. ``sample_rollouts`` (the hashed rollout policy is authoritative for
   ``max_steps`` / loop-guard arm / prefill chunk; DeepStack follows the model);
2. per trajectory a teacher-forced pass of the policy under ``no_grad`` -> the
   typed walk (``walk_typed(mode="sample")``) and the OLD token log-probs, both
   recomputed on the update path (``legal_logprobs``), never read from the
   sampler;
3. the chain adapter + ``score_candidate`` -> ``(status, scalar_micro)`` of the
   sealed evaluation; ``None`` never becomes a number here;
4. ``collate_rollout_batch`` (closed envelope; reward/gold are not in it) with
   the legal mask narrowed to NOT-substituted steps (D2);
5. reference log-probs under ``no_grad`` from ``reference_forward_logits``;
6. a ``new_logprobs_fn`` closure that re-runs the teacher-forced forward under
   grad inside ``update_step``.

Typed heads. A sampled typed decision (a head of a group in
``selected_typed_heads`` whose kind is in the policy's ``typed_head_sampling``)
adds its log-prob to the token log-prob at the position of the node's anchor
event, so ``l_rl`` sees one log-prob per step of the joint action. Heads outside
``selected_typed_heads`` are argmax and carry no term; the state map admits only
``typed_heads.*`` groups, so the ``expert_adapter.dense`` cardinality heads are
never sampled (``walk_typed`` draws them only if that group id is selected).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from unflatten.model import events_vocab
from unflatten.model.candidate_model import TeacherForcedBatch
from unflatten.model.composite_vocab import CompositeVocabulary, QwenRef, StructuralRef
from unflatten.model.node_lifecycle import NodeLifecycleTracker
from unflatten.model.region_decode import decode_regions
from unflatten.model.rl.chain_adapter import sealed_regions
from unflatten.model.rl.collate import RolloutBatch, collate_rollout_batch
from unflatten.model.rl.exposure import rollout_seed_digest
from unflatten.model.rl.logprobs import legal_logprobs
from unflatten.model.rl.real_run_ports import LoopContext, PageRecord, PageVisual
from unflatten.model.rl.real_run_score import Scored, score_page_candidate
from unflatten.model.rl.reference_loader import reference_forward_logits
from unflatten.model.rl.rollout import (
    RolloutTrajectory,
    derive_rollout_seed,
    sample_rollouts,
)
from unflatten.model.rl.typed_sampling import walk_typed
from unflatten.model.rl.update import MicroBatch

PAD_ID = 0
_TOKEN_NAME_BY_ID = {i: n for n, i in events_vocab.TOKEN_IDS.items()}


@dataclass(frozen=True)
class _Row:
    """One trajectory's update-side data (tokens, digests, forced typed draws)."""

    tokens: tuple[int, ...]
    digests: tuple[str, ...]
    substituted: tuple[int, ...]
    batch: TeacherForcedBatch | None
    forced: Mapping[tuple[str, str], int]
    anchors: Mapping[str, int]


def teacher_forced_batch(
    visual: PageVisual,
    vocabulary: CompositeVocabulary,
    image_token_id: int,
    token_ids: Sequence[int],
) -> TeacherForcedBatch:
    """``[image placeholders][BOS, t0 .. t_{n-2}]`` with the page's DeepStack levels."""
    n_visual = visual.visual_features.shape[1]
    bos = vocabulary.pack(StructuralRef(local_id=events_vocab.TOKEN_IDS[events_vocab.END_DOCUMENT]))
    block = [vocabulary.pack(QwenRef(image_token_id))] * n_visual + [bos] + list(token_ids[:-1])
    device = visual.visual_features.device
    levels = visual.deepstack_visual_embeds
    return TeacherForcedBatch(
        visual_features=visual.visual_features,
        composite_input_ids=torch.tensor([block], dtype=torch.long, device=device),
        composite_attention_mask=torch.ones(1, len(block), dtype=torch.long, device=device),
        image_grid_thw=visual.image_grid_thw,
        deepstack_visual_embeds=tuple(levels) if levels is not None else None,
    )


def _anchor_events(token_ids: Sequence[int], vocabulary: CompositeVocabulary) -> dict[str, int]:
    """``{local_node_id: event index of its anchor}`` (open event, or the leaf event)."""
    tracker = NodeLifecycleTracker()
    for index, packed in enumerate(token_ids):
        ref = vocabulary.unpack(int(packed))
        if isinstance(ref, StructuralRef):
            tracker.apply_structural_token(_TOKEN_NAME_BY_ID[ref.local_id], event_index=index)
    out: dict[str, int] = {}
    for s in tracker.states:
        event = s.open_event_index if s.slot_kind == "opened_node" else s.leaf_event_index
        if event is not None:
            out[s.local_node_id] = event
    return out


def _typed_addend(
    logprobs: Sequence[Any], anchors: Mapping[str, int], steps: int, like: Tensor
) -> Tensor:
    """``(steps,)`` tensor with every sampled typed log-prob added at its anchor event."""
    if not logprobs:
        return torch.zeros(steps, dtype=like.dtype, device=like.device)
    index = torch.tensor(
        [anchors[h.local_node_id] for h in logprobs], dtype=torch.long, device=like.device
    )
    values = torch.stack([h.logprob.reshape(()) for h in logprobs]).to(like.dtype)
    return torch.zeros(steps, dtype=like.dtype, device=like.device).index_add(0, index, values)


def _rollout_controller(controller: Any, trajectories: Sequence[RolloutTrajectory]) -> Any:
    """The controller the sampler actually used: ``ForceContentController``-wrapped
    when the guard arm forces content (the replay must see the same legal sets)."""
    record = trajectories[0].guard_record if trajectories else None
    if record is None:
        return controller
    from unflatten.model.loop_guard_v3 import ForceContentController, parse_arm

    if parse_arm(record["arm"]).force_content:
        return ForceContentController.wrap(controller)
    return controller


def _token_logprobs(
    ctx: LoopContext, model: Any, controller: Any, row: _Row, n_visual: int
) -> tuple[Tensor, Tensor, Tensor, int]:
    """``(logprobs, legal_mask, hidden, prefix_len)`` of one trajectory under ``model``
    (``legal_logprobs`` over the teacher-forced composite-head rows). The caller
    owns grad mode."""
    assert row.batch is not None
    hidden, prefix_len = model.encode_hidden(row.batch)
    logits = model.composite_head(hidden[:, prefix_len + n_visual :, :])[0]
    lp, mask = legal_logprobs(
        logits=logits,
        token_ids=row.tokens,
        legal_set_digests=row.digests,
        controller=controller,
        vocabulary=ctx.opened.vocabulary,
        capability_profile_id=ctx.opened.capability_profile_id,
        policy=ctx.rollout_policy,
        substituted_positions=row.substituted,
    )
    return lp, mask, hidden, prefix_len


def _replay_typed(
    ctx: LoopContext,
    model: Any,
    hidden: Tensor,
    prefix_len: int,
    n_visual: int,
    row: _Row,
    like: Tensor,
) -> Tensor:
    """Typed log-probs of the forced draws under ``model``, added at the anchors."""
    walk = walk_typed(
        model,
        hidden,
        row.tokens,
        ctx.opened.vocabulary,
        prefix_len=prefix_len,
        visual_steps=n_visual,
        policy=ctx.rollout_policy,
        mode="replay",
        selected_typed_heads=ctx.heads,
        forced_indices=row.forced,
    )
    return _typed_addend(walk.logprobs, row.anchors, len(row.tokens), like)


def _candidate_regions(
    model: Any, walk: Any, hidden: Tensor, prefix_len: int, n_visual: int, visual: PageVisual
) -> tuple[Any, ...]:
    """Sealed region predictions of one candidate: ``decode_regions`` (argmax, no
    gradient) over the anchors of the teacher-forced pass and the visual span of the
    same hidden, as a free-running decode reads them. A policy without a
    ``node_visual_aligner`` yields none."""
    aligner = getattr(model, "node_visual_aligner", None)
    if aligner is None:
        return ()
    visual_hidden = hidden[0, prefix_len : prefix_len + n_visual, :]
    hypotheses = decode_regions(
        aligner,
        walk.node_states,
        visual_hidden=visual_hidden,
        image_grid_thw=visual.image_grid_thw.to(visual_hidden.device),
    )
    return sealed_regions(hypotheses)


def _pad(rows: Sequence[Tensor], width: int) -> Tensor:
    return torch.stack([F.pad(r, (0, width - r.shape[0])) for r in rows])


@dataclass(frozen=True)
class _Prepared:
    """One trajectory after the no-grad pass: update-side row, old/reference
    log-probs, legal mask and the typed predictions the chain is fed."""

    row: _Row
    old: Tensor
    ref: Tensor
    mask: Tensor
    predictions: Sequence[Any]
    regions: Sequence[Any] = ()


def _prepare(
    ctx: LoopContext,
    controller: Any,
    visual: PageVisual,
    traj: RolloutTrajectory,
    *,
    doc_id: str,
    index: int,
    n_visual: int,
) -> _Prepared:
    tokens = traj.token_ids
    if not tokens:  # nothing was sampled: no log-prob, masked out by padding
        empty = torch.zeros(0, dtype=torch.float64)
        return _Prepared(
            _Row((), (), (), None, {}, {}), empty, empty, torch.zeros(0, dtype=torch.bool), ()
        )
    opened = ctx.opened
    batch = teacher_forced_batch(visual, opened.vocabulary, opened.image_token_id, tokens)
    anchors = _anchor_events(tokens, opened.vocabulary)
    row = _Row(tokens, traj.legal_set_digests, traj.substituted_positions, batch, {}, anchors)
    generator = torch.Generator(device="cpu").manual_seed(
        derive_rollout_seed(ctx.run_seed, f"{doc_id}#typed", index)
    )
    with torch.no_grad():
        old, mask, hidden, prefix_len = _token_logprobs(
            ctx, opened.policy, controller, row, n_visual
        )
        walk = walk_typed(
            opened.policy,
            hidden,
            tokens,
            opened.vocabulary,
            prefix_len=prefix_len,
            visual_steps=n_visual,
            policy=ctx.rollout_policy,
            generator=generator,
            mode="sample",
            selected_typed_heads=ctx.heads,
        )
        row = replace(
            row, forced={(h.local_node_id, h.pointer): h.sampled_index for h in walk.logprobs}
        )
        regions = _candidate_regions(opened.policy, walk, hidden, prefix_len, n_visual, visual)
        old = old + _typed_addend(walk.logprobs, anchors, len(tokens), old)
        ref_logits = reference_forward_logits(ctx.reference, batch)[0, n_visual:, :]
        ref, _ = legal_logprobs(
            logits=ref_logits,
            token_ids=tokens,
            legal_set_digests=traj.legal_set_digests,
            controller=controller,
            vocabulary=opened.vocabulary,
            capability_profile_id=opened.capability_profile_id,
            policy=ctx.rollout_policy,
            substituted_positions=traj.substituted_positions,
        )
        if ctx.heads:
            ref_hidden, ref_prefix = ctx.reference.encode_hidden(batch)  # type: ignore[operator]
            ref = ref + _replay_typed(
                ctx, ctx.reference, ref_hidden, ref_prefix, n_visual, row, ref
            )
    return _Prepared(row, old, ref, mask, walk.predictions, regions)


@dataclass(frozen=True)
class RolloutGroup:
    """One page's sampled group after the no-grad pass and scoring."""

    trajectories: Sequence[RolloutTrajectory]
    prepared: Sequence[_Prepared]
    scored: Sequence[Scored]
    visual: PageVisual
    gold: Any
    doc_id: str
    page: Any


def rollout_group(ctx: LoopContext, page: Any, *, epoch: int) -> RolloutGroup:
    """Sample, prepare (old/ref log-probs, typed walk) and score ``page``'s group."""
    ports, opened = ctx.ports, ctx.opened
    assert ports.build_visual is not None and ports.build_gold is not None
    visual, gold = ports.build_visual(page), ports.build_gold(page)
    view_id = f"{page.page_key[0]}:{page.page_key[1]}"
    doc_id = f"{view_id}#e{epoch}"
    opened.policy.eval()
    trajectories = sample_rollouts(
        model=opened.policy,
        controller=opened.controller,
        vocabulary=opened.vocabulary,
        visual_features=visual.visual_features,
        image_grid_thw=visual.image_grid_thw,
        image_token_id=opened.image_token_id,
        policy=ctx.rollout_policy,
        doc_id=doc_id,
        run_seed=ctx.run_seed,
        capability_profile_id=opened.capability_profile_id,
        deepstack_visual_embeds=visual.deepstack_visual_embeds,
        loop_guard_factory=ports.loop_guard_factory,
    )
    controller = _rollout_controller(opened.controller, trajectories)
    n_visual = visual.visual_features.shape[1]
    prepared = [
        _prepare(ctx, controller, visual, t, doc_id=doc_id, index=i, n_visual=n_visual)
        for i, t in enumerate(trajectories)
    ]
    scored = [
        score_page_candidate(
            ctx, t, p.predictions, visual, gold, f"{gold.document_bundle_id}-r{i}", p.regions
        )
        for i, (t, p) in enumerate(zip(trajectories, prepared, strict=True))
    ]
    return RolloutGroup(trajectories, prepared, scored, visual, gold, doc_id, page)


def build_microbatch(ctx: LoopContext, page: Any, *, epoch: int) -> tuple[MicroBatch, PageRecord]:
    """The update-side micro-batch of ``page`` (one group of ``group_size`` rollouts)."""
    return microbatch_from_group(ctx, rollout_group(ctx, page, epoch=epoch))


def microbatch_from_group(ctx: LoopContext, group: RolloutGroup) -> tuple[MicroBatch, PageRecord]:
    """The micro-batch and page record of an already sampled and scored group."""
    ports, opened = ctx.ports, ctx.opened
    trajectories, prepared, scored = group.trajectories, group.prepared, group.scored
    visual, gold, doc_id = group.visual, group.gold, group.doc_id
    page = group.page
    view_id = f"{page.page_key[0]}:{page.page_key[1]}"
    n_visual = visual.visual_features.shape[1]
    controller = _rollout_controller(opened.controller, trajectories)
    closed = collate_rollout_batch(trajectories, group_ids=[0] * len(trajectories), pad_id=PAD_ID)
    width = closed.token_ids.shape[1]
    keep = _pad([p.mask.to(torch.long) for p in prepared], width).to(torch.bool)
    rollout: RolloutBatch = replace(
        closed,
        padding_mask=closed.padding_mask.to(keep.device),
        legal_mask=closed.legal_mask.to(keep.device) & keep,
    )

    def new_logprobs_fn() -> Tensor:
        out: list[Tensor] = []
        for p in prepared:
            if not p.row.tokens:
                out.append(torch.zeros(0, dtype=torch.float64))
                continue
            lp, _mask, hidden, prefix_len = _token_logprobs(
                ctx, opened.policy, controller, p.row, n_visual
            )
            if ctx.heads:
                lp = lp + _replay_typed(ctx, opened.policy, hidden, prefix_len, n_visual, p.row, lp)
            out.append(lp)
        return _pad(out, width)

    sft = None
    if ctx.supervised_weight_micro > 0 and ports.build_sft_batch is not None:
        sft = ports.build_sft_batch(page)
    micro = MicroBatch(
        new_logprobs_fn=new_logprobs_fn,
        old_logprobs=_pad([p.old for p in prepared], width),
        ref_logprobs=_pad([p.ref for p in prepared], width),
        rollout=rollout,
        reward_status=tuple(x.status for x in scored),
        scalar_micro=tuple(x.scalar_micro for x in scored),
        provenance=ctx.provenance,
        sft_batch=sft,
    )
    record = PageRecord(
        lineage_root_id=page.lineage_root_id,
        source=gold.source,
        cell=gold.cell,
        view_id=view_id,
        rollout_count=len(trajectories),
        guard_intervention_counts=tuple(
            None if t.guard_record is None else len(t.substituted_positions) for t in trajectories
        ),
        rollout_seed_digest=rollout_seed_digest(ctx.run_seed, doc_id, len(trajectories)),
        statuses=tuple(x.status for x in scored),
        scalar_micro=tuple(x.scalar_micro for x in scored),
        outcomes=tuple(x.outcome for x in scored),
        render_probe_statuses=tuple(x.probe_status for x in scored if x.probe_status is not None),
    )
    return micro, record
