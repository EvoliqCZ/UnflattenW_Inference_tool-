"""Update-path log-probabilities of sampled rollouts (S4.5 design D2; dormant).

The rollout sampler draws from ``softmax(logits[legal] / T)`` (top-p aside). The
update path must score the very same distribution, so ``legal_logprobs``:

* **replays the grammar controller** over the trajectory's token ids and re-derives
  every step's legal id set -- the set is never read from the trajectory, only its
  digest is compared (``rollout._legal_digest``); any mismatch fails closed;
* takes the **temperature-scaled log-softmax over that legal set only** and gathers
  the sampled token (``temperature_micro`` from the rollout policy, float64 like
  the sampler);
* **refuses** a policy with ``top_p_micro != 1_000_000``: nucleus membership is not
  part of the log-prob, so a top-p sampler and this scorer would be two
  distributions (design D2, rejected: nucleus-renormalised log-prob);
* returns ``legal_mask`` -- ``False`` at ``substituted_positions`` (steps a loop
  guard replaced; excluded from PG and KL, kept as conditioning).

Rewinds are already truncated out of a trajectory's tokens/digests, and the caller
must pass the SAME controller the rollout used (``ForceContentController``-wrapped
when the guard arm forces content). The model forward is not here except for the
small ``teacher_forced_step_logits`` helper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from unflatten.model import events_vocab
from unflatten.model.candidate_model import TeacherForcedBatch
from unflatten.model.composite_vocab import CompositeVocabulary, QwenRef, StructuralRef
from unflatten.model.free_running_decode import _legal_packed_ids
from unflatten.model.rl.policies import verify_rollout_policy
from unflatten.model.rl.rollout import _legal_digest
from unflatten.model.rl.scorers import MICRO

REQUIRED_TOP_P_MICRO = MICRO


class LogprobsRefused(ValueError):
    """The update-path log-prob request violates the D2 contract."""


class LegalSetMismatch(LogprobsRefused):
    """The replayed legal set's digest differs from the one the rollout recorded."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_policy(policy: Mapping[str, Any]) -> int:
    verify_rollout_policy(policy)
    if policy["top_p_micro"] != REQUIRED_TOP_P_MICRO:
        raise LogprobsRefused(
            f"top_p_micro must be {REQUIRED_TOP_P_MICRO} on the update path "
            f"(top-p is not part of the log-prob), got {policy['top_p_micro']!r}"
        )
    temperature = policy["temperature_micro"]
    if not _is_int(temperature) or temperature <= 0:
        raise LogprobsRefused(f"temperature_micro must be a positive int, got {temperature!r}")
    return temperature


def _substituted_set(positions: Sequence[int], steps: int) -> frozenset[int]:
    for p in positions:
        if not _is_int(p) or not 0 <= p < steps:
            raise LogprobsRefused(
                f"substituted_positions entry {p!r} is not an int in [0, {steps})"
            )
    return frozenset(positions)


def legal_logprobs(
    *,
    logits: torch.Tensor,
    token_ids: Sequence[int],
    legal_set_digests: Sequence[str],
    controller: Any,
    vocabulary: CompositeVocabulary,
    capability_profile_id: str,
    policy: Mapping[str, Any],
    substituted_positions: Sequence[int] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(logprobs, legal_mask)``, both ``(steps,)``, for one trajectory.

    ``logits`` is ``(steps, vocab)``: row ``i`` is the composite-head output the
    sampler saw when it drew ``token_ids[i]`` (teacher-forced here). The result
    carries the gradient of ``logits``; ``legal_mask`` is bool, ``False`` at
    ``substituted_positions``."""
    temperature = _check_policy(policy)
    steps = len(token_ids)
    if logits.dim() != 2 or logits.shape[0] != steps:
        raise LogprobsRefused(f"logits {tuple(logits.shape)} must be ({steps}, vocab)")
    if len(legal_set_digests) != steps:
        raise LogprobsRefused(f"{len(legal_set_digests)} digests for {steps} tokens")
    substituted = _substituted_set(substituted_positions, steps)

    state = controller.fresh(capability_profile_id)
    picked: list[torch.Tensor] = []
    scale = MICRO / temperature
    for i, token in enumerate(token_ids):
        if state.mode == "terminated":
            raise LogprobsRefused(f"position {i}: grammar already terminated before this token")
        legal = sorted(_legal_packed_ids(controller, state, vocabulary))
        if _legal_digest(legal) != legal_set_digests[i]:
            raise LegalSetMismatch(
                f"position {i}: replayed legal set ({len(legal)} ids) does not match the "
                "recorded digest"
            )
        if token not in legal:
            raise LogprobsRefused(f"position {i}: token {token} is outside the legal set")
        index = torch.tensor(legal, dtype=torch.long, device=logits.device)
        scaled = logits[i].index_select(0, index).to(torch.float64) * scale
        picked.append(torch.log_softmax(scaled, dim=0)[legal.index(token)])
        state = controller.step(state, vocabulary.unpack(token))
    if not picked:
        empty = torch.zeros(0, dtype=torch.float64, device=logits.device)
        return empty, torch.zeros(0, dtype=torch.bool, device=logits.device)
    mask = torch.tensor(
        [i not in substituted for i in range(steps)], dtype=torch.bool, device=logits.device
    )
    return torch.stack(picked), mask


def teacher_forced_step_logits(
    *,
    model: Any,
    visual_features: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    vocabulary: CompositeVocabulary,
    token_ids: Sequence[int],
) -> torch.Tensor:
    """``(steps, vocab)`` composite-head logits that predict ``token_ids`` under teacher
    forcing, on the rollout's prefix (``image placeholders + BOS``): row ``i`` is the
    head output after ``BOS, token_ids[:i]``. Full-recompute ``encode_hidden`` path;
    the caller owns ``no_grad``/``train`` mode. Batch 1."""
    if visual_features.dim() != 3 or visual_features.shape[0] != 1:
        raise LogprobsRefused(
            f"teacher-forced scoring is single-page/batch=1, got {tuple(visual_features.shape)}"
        )
    if not token_ids:
        raise LogprobsRefused("token_ids is empty")
    n_visual = visual_features.shape[1]
    bos = vocabulary.pack(StructuralRef(local_id=events_vocab.TOKEN_IDS[events_vocab.END_DOCUMENT]))
    block = [vocabulary.pack(QwenRef(image_token_id))] * n_visual + [bos] + list(token_ids[:-1])
    device = visual_features.device
    batch = TeacherForcedBatch(
        visual_features=visual_features,
        composite_input_ids=torch.tensor([block], dtype=torch.long, device=device),
        composite_attention_mask=torch.ones(1, len(block), dtype=torch.long, device=device),
        image_grid_thw=image_grid_thw,
    )
    hidden, prefix_len = model.encode_hidden(batch)
    return model.composite_head(hidden[:, prefix_len + n_visual :, :])[0]


@dataclass(frozen=True)
class LogprobDrift:
    """``|old - rollout|`` over the unmasked positions (diagnostic ledger only)."""

    max_abs: float
    mean_abs: float
    positions: int


def logprob_drift(
    old: torch.Tensor, rollout: torch.Tensor, legal_mask: torch.Tensor
) -> LogprobDrift | None:
    """Pure diagnostic ``|old - rollout|``; ``None`` (never 0) with no unmasked
    position. Never fed to the loss."""
    if old.shape != rollout.shape or old.shape != legal_mask.shape or old.dim() != 1:
        raise LogprobsRefused(
            f"shape mismatch: old {tuple(old.shape)}, rollout {tuple(rollout.shape)}, "
            f"mask {tuple(legal_mask.shape)}"
        )
    diff = (old.detach() - rollout.detach()).abs()[legal_mask]
    if diff.numel() == 0:
        return None
    return LogprobDrift(
        max_abs=float(diff.max()), mean_abs=float(diff.mean()), positions=int(diff.numel())
    )
