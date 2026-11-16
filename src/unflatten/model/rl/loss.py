"""S4.5 ``L_rl``: GRPO policy-gradient term + ``L_kl`` (RFC005-ERRATA-010 E4; dormant).

Standalone, NOT a ``LossUnit`` family (precedent:
``visual_compression.losses.l_visual_compressor``). ``L_rl`` is in neither
``training_materializer.LOSS_FAMILY_IDS`` nor ``SUPPORTED_LOSS_FAMILIES`` nor
``loss_unit.schema.json``: the reward-weighted term's LossUnit class is
OPEN-005 (variant a, ``rollout_reward``) and nothing here pre-empts it. This
module only computes numbers; no optimizer, no gradient step.

Contract, in the order it bites:

* **Old log-probs come from the update path.** ``old_logprobs_source`` is a
  required argument that accepts exactly ``"update_path_recomputed"``: the
  caller re-ran the teacher-forced forward over the rollout's tokens under the
  same grammar-legal renormalisation. The rollout sampler never stores
  log-probs (``rollout.FORBIDDEN_FIELD_PARTS``). Old and reference log-probs
  must be detached -- an attached ``old`` would make the ratio identically 1
  with a silently cancelled gradient.
* **Missing reward is not zero.** ``insufficient_evidence`` candidates are
  excluded from advantage, policy gradient and KL and are counted;
  ``gate_failed:*`` candidates get the caller-supplied bounded penalty
  (``RlLossPolicy.gate_failed_reward_micro``).
* **Advantage is group-relative**, per-group mean/std over the included
  candidates' integer-micro rewards. A group with fewer than two included
  candidates, or whose included rewards are all equal (integer compare, no
  epsilon), has advantage exactly 0 and is counted in ``zero_advantage_rate``.
* **Masks.** A position counts only if ``padding_mask & legal_mask``; masked
  positions are replaced before ``exp`` so a NaN there cannot reach the loss
  or the gradient.

``L_kl`` is the k3 estimator ``exp(ref-new) - (ref-new) - 1`` (>= 0, == 0 when
policy == reference), averaged like the policy term.

Out of scope: wiring over real model forwards. This module neither calls the
model nor gathers node anchors; ``typed_categorical_logprobs`` takes anchor
logits that the caller already gathered (``s4_typed_loss.gather_anchor_hidden``
+ head forward). The sampler's top-p truncation is not part of the
log-probability: the ratio is taken under the temperature-scaled full
distribution on both the old and the new side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

import torch
from torch import Tensor

from unflatten.model.rl.policies import INTEGER_HEAD_KINDS, SAMPLEABLE_HEAD_KINDS
from unflatten.model.rl.scorers import MICRO

OLD_LOGPROBS_SOURCE: Final = "update_path_recomputed"
LENGTH_NORMALISATIONS: Final = ("token_mean", "sum")
LOSS_SCOPE: Final = "rollout_reward"
STATUS_SCORED: Final = "scored"
STATUS_INSUFFICIENT: Final = "insufficient_evidence"
STATUS_GATE_PREFIX: Final = "gate_failed:"


class RlLossRefused(ValueError):
    """An input the S4.5 loss contract forbids; never coerced."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class RlLossPolicy:
    """Clipping / KL / length / penalty policy. No defaults: values are frozen by
    the owner before the first gradient (RFC005-ERRATA-010 E7.4)."""

    clip_epsilon_micro: int
    kl_coefficient_micro: int
    length_normalisation: str
    gate_failed_reward_micro: int

    def __post_init__(self) -> None:
        for name in ("clip_epsilon_micro", "kl_coefficient_micro", "gate_failed_reward_micro"):
            if not _is_int(getattr(self, name)):
                raise RlLossRefused(
                    f"{name} must be an int micro unit, got {getattr(self, name)!r}"
                )
        if not 0 < self.clip_epsilon_micro < MICRO:
            raise RlLossRefused(
                f"clip_epsilon_micro must be in (0, {MICRO}), got {self.clip_epsilon_micro}"
            )
        if self.kl_coefficient_micro < 0:
            raise RlLossRefused(
                f"kl_coefficient_micro must be >= 0, got {self.kl_coefficient_micro}"
            )
        if self.length_normalisation not in LENGTH_NORMALISATIONS:
            raise RlLossRefused(
                f"length_normalisation {self.length_normalisation!r} not in {LENGTH_NORMALISATIONS}"
            )
        if not -MICRO <= self.gate_failed_reward_micro <= 0:
            raise RlLossRefused(
                f"gate_failed_reward_micro must be in [-{MICRO}, 0] so a failed gate never "
                f"outranks a scored candidate (RFC006-ERRATA-001 E1(b)4), "
                f"got {self.gate_failed_reward_micro}"
            )


@dataclass(frozen=True, slots=True)
class RlLossProvenance:
    """What the loss is bound to (precedent: ``SealedDerivedProvenance``)."""

    reward_policy_hash: str
    rollout_policy_hash: str
    reference_checkpoint_id: str
    reference_ancestry_hash: str
    typed_head_config_hash: str

    def __post_init__(self) -> None:
        for name in self.__slots__:  # type: ignore[attr-defined]
            if not getattr(self, name):
                raise RlLossRefused(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class RlLossResult:
    loss: Tensor
    policy_loss: Tensor
    kl_loss: Tensor
    kl_mean: float
    #: float view kept for existing callers and tests; the exact value is
    #: ``zero_advantage_rate_micro`` (None when there were no groups).
    zero_advantage_rate: float
    zero_advantage_rate_micro: int | None
    groups_total: int
    groups_zero_advantage: int
    excluded_insufficient_evidence: int
    gate_failed_count: int
    clipped_fraction: float
    advantages: Tensor
    provenance: RlLossProvenance
    loss_scope: str = LOSS_SCOPE


def _candidate_rewards(
    statuses: Any, scalars: Any, policy: RlLossPolicy
) -> tuple[list[int | None], list[bool], int, int]:
    """``(reward_micro, included, excluded_count, gate_failed_count)``; an excluded
    candidate's reward is ``None`` (a missing value is never a number)."""
    if len(statuses) != len(scalars):
        raise RlLossRefused(f"reward_status/scalar_micro length {len(statuses)} != {len(scalars)}")
    rewards: list[int | None] = []
    included: list[bool] = []
    excluded = gate_failed = 0
    for i, (status, scalar) in enumerate(zip(statuses, scalars, strict=True)):
        if status == STATUS_SCORED:
            if not _is_int(scalar):
                raise RlLossRefused(f"candidate {i}: scored needs an int scalar, got {scalar!r}")
            rewards.append(scalar)
            included.append(True)
        elif isinstance(status, str) and status.startswith(STATUS_GATE_PREFIX):
            if scalar is not None:
                raise RlLossRefused(f"candidate {i}: {status!r} must carry scalar None")
            rewards.append(policy.gate_failed_reward_micro)
            included.append(True)
            gate_failed += 1
        elif status == STATUS_INSUFFICIENT:
            if scalar is not None:
                raise RlLossRefused(f"candidate {i}: {status!r} must carry scalar None")
            rewards.append(None)  # never read: `included` is False
            included.append(False)
            excluded += 1
        else:
            raise RlLossRefused(f"candidate {i}: unknown reward status {status!r}")
    return rewards, included, excluded, gate_failed


def _group_advantages(
    rewards: list[int | None], included: list[bool], group_ids: Tensor
) -> tuple[Tensor, int, int]:
    """Per-group normalised advantage (float64, zeros for excluded), plus
    ``(zero_advantage_groups, groups)``."""
    gids = group_ids.tolist()
    adv = torch.zeros(len(gids), dtype=torch.float64)
    zero = 0
    groups = sorted(set(gids))
    for g in groups:
        idx = [i for i, gid in enumerate(gids) if gid == g and included[i]]
        vals = [v for i in idx if (v := rewards[i]) is not None]
        if len(vals) < 2 or max(vals) == min(vals):
            zero += 1
            continue
        r = torch.tensor(vals, dtype=torch.int64).to(torch.float64)
        centred = r - r.sum() / len(vals)  # int64 sum is exact before the divide
        adv[idx] = centred / centred.pow(2).mean().sqrt()
    return adv, zero, len(groups)


def _aggregate(per_token: Tensor, mask: Tensor, how: str) -> Tensor:
    """Per-candidate aggregate over valid tokens -> ``(N,)``."""
    total = (per_token * mask).sum(dim=1)
    if how == "token_mean":
        return total / mask.sum(dim=1).clamp(min=1)
    return total


def _check_inputs(
    new: Tensor, old: Tensor, ref: Tensor, pad: Tensor, legal: Tensor, gids: Tensor, source: str
) -> None:
    if source != OLD_LOGPROBS_SOURCE:
        raise RlLossRefused(
            f"old_logprobs_source must be {OLD_LOGPROBS_SOURCE!r} (teacher-forced recompute on "
            f"the update path), got {source!r}"
        )
    for name, t in (("old_logprobs", old), ("ref_logprobs", ref)):
        if t.requires_grad:
            raise RlLossRefused(f"{name} must be detached (computed under no_grad)")
    if new.dim() != 2:
        raise RlLossRefused(f"new_logprobs must be (candidates, tokens), got {tuple(new.shape)}")
    for name, t in (
        ("old_logprobs", old),
        ("ref_logprobs", ref),
        ("padding_mask", pad),
        ("legal_mask", legal),
    ):
        if t.shape != new.shape:
            raise RlLossRefused(f"{name} shape {tuple(t.shape)} != new_logprobs {tuple(new.shape)}")
    if pad.dtype != torch.bool or legal.dtype != torch.bool:
        raise RlLossRefused("padding_mask and legal_mask must be bool")
    if gids.shape != (new.shape[0],):
        raise RlLossRefused(f"group_ids shape {tuple(gids.shape)} != ({new.shape[0]},)")


def l_rl(
    *,
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    ref_logprobs: Tensor,
    padding_mask: Tensor,
    legal_mask: Tensor,
    group_ids: Tensor,
    reward_status: Any,
    scalar_micro: Any,
    policy: RlLossPolicy,
    provenance: RlLossProvenance,
    old_logprobs_source: Literal["update_path_recomputed"],
) -> RlLossResult:
    """``L_rl = L_pg + (kl_coefficient / 1e6) * L_kl`` over ``(N candidates, T tokens)``.

    ``new_logprobs`` carries the gradient; ``old_logprobs`` and ``ref_logprobs``
    are detached tensors of the sampled tokens' log-probabilities under the
    update-path snapshot and the frozen reference. ``reward_status`` /
    ``scalar_micro`` are the per-candidate ``status`` / ``scalar_micro`` of the
    sealed ``rl_candidate_evaluation`` documents (T4).
    """
    _check_inputs(
        new_logprobs,
        old_logprobs,
        ref_logprobs,
        padding_mask,
        legal_mask,
        group_ids,
        old_logprobs_source,
    )
    rewards, included, excluded, gate_failed = _candidate_rewards(
        reward_status, scalar_micro, policy
    )
    if len(rewards) != new_logprobs.shape[0]:
        raise RlLossRefused(f"{len(rewards)} reward entries for {new_logprobs.shape[0]} candidates")
    inc = torch.tensor(included, dtype=torch.bool, device=new_logprobs.device)
    valid = padding_mask & legal_mask & inc[:, None]
    seen = valid & torch.isfinite(new_logprobs) & torch.isfinite(old_logprobs)
    seen = seen & torch.isfinite(ref_logprobs)
    if bool((valid & ~seen).any()):
        raise RlLossRefused("non-finite log-probability at a valid (unmasked) position")

    adv64, zero_groups, n_groups = _group_advantages(rewards, included, group_ids)
    adv = adv64.to(device=new_logprobs.device, dtype=new_logprobs.dtype)
    mask = valid.to(new_logprobs.dtype)

    zero = torch.zeros((), dtype=new_logprobs.dtype, device=new_logprobs.device)
    log_ratio = torch.where(valid, new_logprobs - old_logprobs, zero)
    ratio = torch.exp(log_ratio)
    eps = policy.clip_epsilon_micro / MICRO
    a = adv[:, None]
    surrogate = torch.minimum(ratio * a, torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * a)
    n_inc = max(int(inc.sum()), 1)
    policy_loss = -_aggregate(surrogate, mask, policy.length_normalisation).sum() / n_inc

    delta = torch.where(valid, ref_logprobs - new_logprobs, zero)
    k3 = torch.exp(delta) - delta - 1.0
    kl_loss = _aggregate(k3, mask, policy.length_normalisation).sum() / n_inc
    n_tok = max(int(valid.sum()), 1)
    kl_mean = float((k3.detach() * mask).sum() / n_tok)

    clip_active = valid & (((a > 0) & (ratio > 1.0 + eps)) | ((a < 0) & (ratio < 1.0 - eps)))
    total = policy_loss + (policy.kl_coefficient_micro / MICRO) * kl_loss
    return RlLossResult(
        loss=total,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        kl_mean=kl_mean,
        zero_advantage_rate=zero_groups / n_groups if n_groups else 0.0,
        zero_advantage_rate_micro=zero_groups * MICRO // n_groups if n_groups else None,
        groups_total=n_groups,
        groups_zero_advantage=zero_groups,
        excluded_insufficient_evidence=excluded,
        gate_failed_count=gate_failed,
        clipped_fraction=float(clip_active.sum()) / n_tok if valid.any() else 0.0,
        advantages=adv.detach(),
        provenance=provenance,
    )


def typed_categorical_logprobs(
    logits: Tensor, sampled_index: Tensor, *, kind: str, temperature_micro: int
) -> Tensor:
    """Log-prob of the sampled class for one categorical typed head, ``(...,)``.

    ``kind`` is one of ``policies.SAMPLEABLE_HEAD_KINDS``. ``boolean`` takes a
    single logit ``l`` and is the two-class ``[0, l]`` (as in
    ``rollout.select_head_value``). Integer kinds are argmax in rollouts and
    are not part of the policy gradient: refused. Temperature matches the
    rollout's, so old and new sit under the same tempered distribution.
    """
    if kind in INTEGER_HEAD_KINDS:
        raise RlLossRefused(f"integer head kind {kind!r} is argmax, not in the policy gradient")
    if kind not in SAMPLEABLE_HEAD_KINDS:
        raise RlLossRefused(
            f"unknown typed head kind {kind!r}; sampleable: {SAMPLEABLE_HEAD_KINDS}"
        )
    if not _is_int(temperature_micro) or temperature_micro <= 0:
        raise RlLossRefused(f"temperature_micro must be a positive int, got {temperature_micro!r}")
    if kind == "boolean":
        if logits.shape[-1] != 1:
            raise RlLossRefused(f"boolean head takes one logit, got last dim {logits.shape[-1]}")
        logits = torch.cat([torch.zeros_like(logits), logits], dim=-1)
    if sampled_index.shape != logits.shape[:-1]:
        raise RlLossRefused(
            f"sampled_index shape {tuple(sampled_index.shape)} != "
            f"logits batch {tuple(logits.shape[:-1])}"
        )
    if bool(((sampled_index < 0) | (sampled_index >= logits.shape[-1])).any()):
        raise RlLossRefused(f"sampled index outside [0, {logits.shape[-1]})")
    scaled = logits * (MICRO / temperature_micro)
    return torch.log_softmax(scaled, dim=-1).gather(-1, sampled_index[..., None])[..., 0]


def combine_rl_and_supervised(
    rl: RlLossResult, supervised_total: Tensor, *, supervised_weight_micro: int
) -> Tensor:
    """``L_rl + w * L_supervised`` for the non-policy heads (aligner, continuous
    heads). ``supervised_total`` is ``s4_typed_loss.s4_total_loss``'s scalar,
    unchanged; the two families are never merged into one LossUnit."""
    if not _is_int(supervised_weight_micro) or supervised_weight_micro < 0:
        raise RlLossRefused(
            f"supervised_weight_micro must be a non-negative int, got {supervised_weight_micro!r}"
        )
    return rl.loss + (supervised_weight_micro / MICRO) * supervised_total
