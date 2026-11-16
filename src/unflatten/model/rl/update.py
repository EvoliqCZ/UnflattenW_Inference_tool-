"""One S4.5 optimizer step: guarded optimizer build + ``update_step`` (design D2/D3/D7; dormant).

Composes, never forks, the S4 pieces: ``candidate_model.parameter_group_ids`` for
the parameter -> group map, ``loss.l_rl`` for the objective, ``auxiliary`` for the
non-policy supervised term, ``lr_schedule`` for the rate. The outer loop,
checkpointing and data iteration are the composer's.

* ``build_rl_optimizer`` VERIFIES the freeze state against ``s45_allowed_states``
  and refuses; it never freezes anything itself. The model builds with
  ``adapter.projection`` and ``document_context_encoder`` trainable and
  ``node_visual_aligner`` FROZEN (``candidate_model.py``), so ``freeze_to_state_map``
  is a separate, explicit call. The aligner (auxiliary-only) is enabled explicitly
  (``real_run._enable_trainable_extras``) and only when ``supervised_weight_micro > 0``.
* ``update_step`` order: stage state -> policy checks (top_p, execution profile) ->
  ``zero_grad(set_to_none=True)`` -> per micro-batch teacher-forced forward under
  grad, ``l_rl`` against DETACHED old/reference log-probs (+ weighted auxiliary),
  loss / accumulation exactly once, backward -> ``clip_grad_norm_`` -> non-finite
  refusal BEFORE ``optimizer.step()`` -> LR from the schedule -> step.
* The rollout sampler's own log-probs are a diagnostic: ``logprob_drift`` over
  detached tensors after the loss is built, never an input to it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from unflatten.model import s1_train
from unflatten.model.candidate_model import parameter_group_ids
from unflatten.model.rl import state_map as sm
from unflatten.model.rl.auxiliary import SftAuxBatch, auxiliary_loss, combine_with_auxiliary
from unflatten.model.rl.collate import RolloutBatch
from unflatten.model.rl.logprobs import REQUIRED_TOP_P_MICRO, LogprobDrift, LogprobsRefused
from unflatten.model.rl.logprobs import logprob_drift as _drift
from unflatten.model.rl.loss import OLD_LOGPROBS_SOURCE, RlLossProvenance, l_rl
from unflatten.model.rl.lr_schedule import LR_UNIT, lr_at, lr_scale_micro
from unflatten.model.rl.policies import verify_rollout_policy
from unflatten.model.rl.scorers import MICRO
from unflatten.model.rl.training_policy import (
    check_execution_profile_matches,
    check_lr_groups,
    loss_policy_from,
    verify_rl_training_policy,
)


class UpdateRefused(ValueError):
    """A stage-state, optimizer or batch precondition of the update is violated."""


def _group_params(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """``{group id: parameters}`` of EVERY model parameter; an unmapped one that
    requires grad is refused (an unaccounted trainable tensor)."""
    group_of = parameter_group_ids(model)  # type: ignore[arg-type]
    out: dict[str, list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        group = group_of.get(name)
        if group is None:
            if param.requires_grad:
                raise UpdateRefused(
                    f"parameter {name!r} requires grad but maps to no trainable group"
                )
            continue
        out.setdefault(group, []).append(param)
    return out


def _states(run_class: str, heads: Collection[str]) -> dict[str, frozenset[str]]:
    return sm.s45_allowed_states(run_class, selected_typed_heads=tuple(heads))


def assert_stage_state(
    model: nn.Module,
    run_class: str,
    selected_typed_heads: Collection[str] = (),
    *,
    supervised_weight_micro: int,
) -> dict[str, list[nn.Parameter]]:
    """Refuse a model whose requires_grad groups differ from the S4.5 map;
    returns ``{trainable group: parameters}``."""
    states = _states(run_class, selected_typed_heads)
    trainable: dict[str, list[nn.Parameter]] = {}
    for group, params in _group_params(model).items():
        live = [p for p in params if p.requires_grad]
        if not live:
            continue
        if group not in states or "trainable" not in states[group]:
            allowed = sorted(states.get(group, ()))
            raise UpdateRefused(
                f"group {group!r} is trainable but S4.5 allows only {allowed or 'no state'} "
                "(freeze it explicitly with freeze_to_state_map)"
            )
        if group in sm.SUPERVISED_AUXILIARY_ONLY and supervised_weight_micro == 0:
            raise UpdateRefused(
                f"group {group!r} is auxiliary-only and trainable with supervised_weight_micro=0: "
                "nothing trains it"
            )
        trainable[group] = live
    required = sm.s45_trainable_groups(run_class, selected_typed_heads=tuple(selected_typed_heads))
    missing = sorted(required - set(trainable))
    if missing:
        raise UpdateRefused(f"required trainable group(s) {missing} have no trainable parameter")
    return trainable


def freeze_to_state_map(
    model: nn.Module,
    run_class: str,
    selected_typed_heads: Collection[str] = (),
    *,
    auxiliary_groups_trainable: bool = False,
) -> list[str]:
    """``requires_grad_(False)`` on every group the S4.5 map does not allow to
    train (and on auxiliary-only groups unless asked); returns the groups frozen."""
    states = _states(run_class, selected_typed_heads)
    frozen: list[str] = []
    for group, params in _group_params(model).items():
        if group not in states:
            raise UpdateRefused(f"group {group!r} is not in the S4.5 state map")
        keep = "trainable" in states[group] and (
            auxiliary_groups_trainable or group not in sm.SUPERVISED_AUXILIARY_ONLY
        )
        if not keep:
            for p in params:
                p.requires_grad_(False)
            frozen.append(group)
    return sorted(frozen)


def build_rl_optimizer(
    model: nn.Module,
    training_policy: Mapping[str, Any],
    run_class: str,
    selected_typed_heads: Collection[str] = (),
    *,
    reference: nn.Module | None = None,
) -> torch.optim.Optimizer:
    """AdamW, one param group per trainable group id, lr (schedule step 0) and
    weight decay from the policy. Refuses -- never fixes -- a freeze state that
    differs from ``s45_allowed_states``, and any parameter owned by ``reference``."""
    verify_rl_training_policy(training_policy)
    check_lr_groups(training_policy, run_class, selected_typed_heads)
    weight = training_policy["supervised_weight_micro"]
    decay = training_policy["weight_decay_micro"]
    if weight is None or decay is None:
        raise UpdateRefused("supervised_weight_micro / weight_decay_micro are not frozen")
    trainable = assert_stage_state(
        model, run_class, selected_typed_heads, supervised_weight_micro=weight
    )
    if reference is not None:
        shared = {id(p) for p in reference.parameters()}
        for group, params in trainable.items():
            if any(id(p) in shared for p in params):
                raise UpdateRefused(f"group {group!r} holds parameter(s) owned by the reference")
    groups: list[dict[str, Any]] = []
    for group, params in sorted(trainable.items()):
        if group not in training_policy["lr_micro"]:
            raise UpdateRefused(f"trainable group {group!r} has no learning rate in the policy")
        groups.append(
            {
                "params": params,
                "lr": lr_at(0, training_policy, group) * LR_UNIT,
                "weight_decay": decay / MICRO,
                "group_id": group,
            }
        )
    return torch.optim.AdamW(groups)


@dataclass(frozen=True)
class MicroBatch:
    """One accumulation slice. ``new_logprobs_fn`` runs the teacher-forced forward
    under grad (B1 ``legal_logprobs``) and returns ``(N, T)``; ``old_logprobs`` were
    recomputed on the update path under no_grad and ``ref_logprobs`` come from the
    frozen reference -- both detached. ``rollout_logprobs`` is diagnostic only."""

    new_logprobs_fn: Callable[[], Tensor]
    old_logprobs: Tensor
    ref_logprobs: Tensor
    rollout: RolloutBatch
    reward_status: Sequence[str]
    scalar_micro: Sequence[int | None]
    provenance: RlLossProvenance
    sft_batch: SftAuxBatch | None = None
    rollout_logprobs: Tensor | None = None


@dataclass(frozen=True)
class RlStepResult:
    step: int
    loss: float
    policy_loss: float
    kl_loss: float
    kl_mean: float
    grad_norm: float
    clipped: bool
    lr_scale_micro: int
    zero_advantage_rate_micro: int | None
    groups_total: int
    excluded_insufficient_evidence: int
    gate_failed_count: int
    auxiliary_reports: tuple[dict[str, Any], ...]
    logprob_drift: LogprobDrift | None


def _merge_drift(parts: list[LogprobDrift]) -> LogprobDrift | None:
    if not parts:
        return None
    n = sum(p.positions for p in parts)
    return LogprobDrift(
        max_abs=max(p.max_abs for p in parts),
        mean_abs=sum(p.mean_abs * p.positions for p in parts) / n,
        positions=n,
    )


def _check_policies(training_policy: Mapping[str, Any], rollout_policy: Mapping[str, Any]) -> None:
    verify_rl_training_policy(training_policy)
    verify_rollout_policy(rollout_policy)
    if rollout_policy["top_p_micro"] != REQUIRED_TOP_P_MICRO:
        raise LogprobsRefused(
            f"top_p_micro must be {REQUIRED_TOP_P_MICRO} on the update path, "
            f"got {rollout_policy['top_p_micro']!r}"
        )
    check_execution_profile_matches(training_policy, rollout_policy)


def update_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    microbatches: Sequence[MicroBatch],
    *,
    training_policy: Mapping[str, Any],
    rollout_policy: Mapping[str, Any],
    step: int,
    run_class: str,
    selected_typed_heads: Collection[str] = (),
) -> RlStepResult:
    """One optimizer step over ``accumulation_steps`` micro-batches (see module doc)."""
    _check_policies(training_policy, rollout_policy)
    accumulation = training_policy["accumulation_steps"]
    clip, weight = training_policy["grad_clip_micro"], training_policy["supervised_weight_micro"]
    if None in (accumulation, clip, weight):
        raise UpdateRefused("accumulation_steps / grad_clip_micro / supervised_weight_micro null")
    if len(microbatches) != accumulation:
        raise UpdateRefused(f"{len(microbatches)} micro-batches for accumulation {accumulation}")
    assert_stage_state(model, run_class, selected_typed_heads, supervised_weight_micro=weight)
    loss_policy = loss_policy_from(training_policy)
    heads = tuple(selected_typed_heads)

    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "policy": 0.0, "kl_loss": 0.0, "kl_mean": 0.0}
    zero_groups = groups = excluded = gate_failed = 0
    reports: list[dict[str, Any]] = []
    drifts: list[LogprobDrift] = []
    for mb in microbatches:
        rl = l_rl(
            new_logprobs=mb.new_logprobs_fn(),
            old_logprobs=mb.old_logprobs,
            ref_logprobs=mb.ref_logprobs,
            padding_mask=mb.rollout.padding_mask,
            legal_mask=mb.rollout.legal_mask,
            group_ids=mb.rollout.group_ids,
            reward_status=mb.reward_status,
            scalar_micro=mb.scalar_micro,
            policy=loss_policy,
            provenance=mb.provenance,
            old_logprobs_source=OLD_LOGPROBS_SOURCE,
        )
        aux = auxiliary_loss(
            model,
            mb.sft_batch,
            selected_typed_heads=heads,
            supervised_weight_micro=weight,
            rollout_batch=mb.rollout,
        )
        loss = combine_with_auxiliary(rl, aux, supervised_weight_micro=weight)
        if not bool(torch.isfinite(loss)):
            optimizer.zero_grad(set_to_none=True)
            raise s1_train.NonFiniteStepError(f"loss is {float(loss.detach())}")
        (loss / accumulation).backward()
        totals["loss"] += float(loss.detach()) / accumulation
        totals["policy"] += float(rl.policy_loss.detach()) / accumulation
        totals["kl_loss"] += float(rl.kl_loss.detach()) / accumulation
        totals["kl_mean"] += rl.kl_mean / accumulation
        zero_groups += rl.groups_zero_advantage
        groups += rl.groups_total
        excluded += rl.excluded_insufficient_evidence
        gate_failed += rl.gate_failed_count
        if aux is not None:
            reports.append(aux.report)
        if mb.rollout_logprobs is not None:
            for i in range(mb.old_logprobs.shape[0]):
                d = _drift(mb.old_logprobs[i], mb.rollout_logprobs[i], mb.rollout.legal_mask[i])
                if d is not None:
                    drifts.append(d)

    params = [p for g in optimizer.param_groups for p in g["params"]]
    max_norm = clip / MICRO
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if not bool(torch.isfinite(norm)):
        optimizer.zero_grad(set_to_none=True)
        raise s1_train.NonFiniteStepError(f"gradient norm is {float(norm)}")
    for g in optimizer.param_groups:
        g["lr"] = lr_at(step, training_policy, g["group_id"]) * LR_UNIT
    optimizer.step()
    return RlStepResult(
        step=step,
        loss=totals["loss"],
        policy_loss=totals["policy"],
        kl_loss=totals["kl_loss"],
        kl_mean=totals["kl_mean"],
        grad_norm=float(norm),
        clipped=float(norm) > max_norm,
        lr_scale_micro=lr_scale_micro(step, training_policy),
        zero_advantage_rate_micro=zero_groups * MICRO // groups if groups else None,
        groups_total=groups,
        excluded_insufficient_evidence=excluded,
        gate_failed_count=gate_failed,
        auxiliary_reports=tuple(reports),
        logprob_drift=_merge_drift(drifts),
    )
