"""Supervised auxiliary term of the S4.5 update (RFC005-ERRATA-010 E3.1/E4.3; design D7; dormant).

``L = L_rl + w * L_aux`` where ``L_aux`` covers ONLY the parts the policy
gradient does not train:

* ``L_alignment`` (the node-visual aligner), and
* the typed-property heads whose ``head_id`` is not in ``selected_typed_heads``
  (integer / continuous heads and every head outside the selection).

It is built from the S4 component functions (``typed_losses``,
``alignment_losses``), NEVER from ``s4_typed_loss.s4_total_loss``: that scalar is
``L_structure + L_text + L_typed + L_alignment`` and would put decoder token
logits and the sampled heads under both the PG family and the CE family
(E4.3: no logit enters two families). ``L_structure`` / ``L_text`` have no
slot here; a non-zero lambda for them is refused, not honoured.

The auxiliary forward is over the GOLD token stream (an SFT batch), never the
closed rollout batch. The decoder hidden state is computed without a graph, so
the term trains the aligner and the non-policy heads only -- nothing flows into
``composite_head`` or the decoder. ``supervised_weight_micro == 0`` returns
``None`` before any model call.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import torch
from torch import Tensor

from unflatten.model import s1_train
from unflatten.model import s4_typed_loss as stl
from unflatten.model.candidate_model import TeacherForcedBatch
from unflatten.model.rl.collate import RolloutBatch, assert_batch_closed
from unflatten.model.rl.loss import RlLossResult, combine_rl_and_supervised

#: lambdas (milli) that may be non-zero; structure/text are zero by construction.
ALLOWED_LAMBDAS: Final = ("L_typed", "L_alignment")
ZERO_ONLY_LAMBDAS: Final = ("L_structure", "L_text")


class AuxiliaryRefused(ValueError):
    """The auxiliary term's inputs break the D7 contract (fail closed)."""


@dataclass(frozen=True)
class SftAuxBatch:
    """An S4 batch over the GOLD token stream (``s1_train.collate_training_batch``
    + ``stl.collate_typed`` + ``stl.collate_regions``), separate from the rollout
    batch."""

    model_batch: TeacherForcedBatch
    visual_steps: int
    typed_batch: stl.TypedBatch
    region_batch: stl.RegionBatch

    def tensors(self) -> Iterator[Tensor]:
        mb = self.model_batch
        for t in (
            mb.visual_features,
            mb.composite_input_ids,
            mb.composite_attention_mask,
            mb.image_grid_thw,
            mb.visual_valid_lengths,
            *(mb.deepstack_visual_embeds or ()),
            self.region_batch.row_index,
            self.region_batch.anchor_step,
            self.region_batch.status_target,
            self.region_batch.bbox_target,
            self.region_batch.supervise_status,
            self.region_batch.supervise_box,
        ):
            if isinstance(t, Tensor):
                yield t
        for key_batch in self.typed_batch.keys.values():
            yield from (
                key_batch.row_index,
                key_batch.anchor_step,
                key_batch.target,
                key_batch.supervise,
            )


@dataclass(frozen=True)
class AuxiliaryResult:
    """``total`` is ``None`` when nothing was supervised in the batch (never 0)."""

    total: Tensor | None
    report: dict[str, Any] = field(default_factory=dict)


def _storage_ptr(t: Tensor) -> int | None:
    return t.untyped_storage().data_ptr() if t.numel() else None


def assert_disjoint(sft: SftAuxBatch, rollout: RolloutBatch | Mapping[str, Tensor]) -> None:
    """Refuse a rollout batch that is not closed (gold/reward fields) or that
    shares any tensor -- by identity or by storage -- with the SFT batch."""
    mapping = rollout.as_mapping() if isinstance(rollout, RolloutBatch) else dict(rollout)
    assert_batch_closed(mapping)
    rollout_ids = {id(t) for t in mapping.values()}
    rollout_ptrs = {p for t in mapping.values() if (p := _storage_ptr(t)) is not None}
    for t in sft.tensors():
        if id(t) in rollout_ids or _storage_ptr(t) in rollout_ptrs:
            raise AuxiliaryRefused(
                "SFT batch shares a tensor with the rollout batch (identity or storage): "
                "gold may enter tensors only in an SFT batch that shares none (E3.1); clone it"
            )


def _lambdas(lambdas_milli: Mapping[str, int] | None) -> dict[str, int]:
    given = dict(lambdas_milli or {})
    for name in ZERO_ONLY_LAMBDAS:
        if given.pop(name, 0) != 0:
            raise AuxiliaryRefused(
                f"{name} is zero by construction in the auxiliary term (E4.3: no logit in two "
                "loss families); it cannot be turned on"
            )
    unknown = sorted(set(given) - set(ALLOWED_LAMBDAS))
    if unknown:
        raise AuxiliaryRefused(f"unknown lambda keys {unknown}; allowed {list(ALLOWED_LAMBDAS)}")
    out = dict.fromkeys(ALLOWED_LAMBDAS, 1000)
    for name, value in given.items():
        if type(value) is not int or value < 0:
            raise AuxiliaryRefused(f"lambda {name} must be a non-negative int milli, got {value!r}")
        out[name] = value
    return out


def _non_policy_keys(
    typed_batch: stl.TypedBatch, selected: Collection[str]
) -> tuple[stl.TypedBatch, list[tuple[str, str]]]:
    chosen = frozenset(selected)
    kept = {k: v for k, v in typed_batch.keys.items() if k[0] not in chosen}
    masked = sorted(k for k in typed_batch.keys if k[0] in chosen)
    return stl.TypedBatch(keys=kept, excluded=dict(typed_batch.excluded)), masked


def auxiliary_loss(
    model: Any,
    sft_batch: SftAuxBatch | None,
    *,
    selected_typed_heads: Collection[str],
    supervised_weight_micro: int,
    lambdas_milli: Mapping[str, int] | None = None,
    rollout_batch: RolloutBatch | Mapping[str, Tensor] | None = None,
) -> AuxiliaryResult | None:
    """The supervised auxiliary scalar for the non-policy parts, or ``None``.

    ``supervised_weight_micro == 0`` returns ``None`` before the model is
    touched. Otherwise ``sft_batch`` is required; with ``rollout_batch`` given
    the two are checked for gold fields and shared tensors. ``total`` is
    ``lambda_typed * macro_mean(non-policy keys) + lambda_alignment * L_alignment``
    over the terms that scored anything; a family with no supervised example
    contributes nothing (``mean: None``) rather than a zero."""
    if type(supervised_weight_micro) is not int or supervised_weight_micro < 0:
        raise AuxiliaryRefused(
            f"supervised_weight_micro must be a non-negative int, got {supervised_weight_micro!r}"
        )
    if supervised_weight_micro == 0:
        return None
    lam = _lambdas(lambdas_milli)
    if sft_batch is None:
        raise AuxiliaryRefused("supervised_weight_micro > 0 needs an SFT batch")
    if rollout_batch is not None:
        assert_disjoint(sft_batch, rollout_batch)
    typed_batch, masked = _non_policy_keys(sft_batch.typed_batch, selected_typed_heads)

    with torch.no_grad():
        hidden, prefix_len = model.encode_hidden(sft_batch.model_batch)
    geom = {"prefix_len": prefix_len, "visual_steps": sft_batch.visual_steps}
    per_key = stl.typed_losses(model, hidden, typed_batch=typed_batch, **geom)
    contributing = {k: v for k, v in per_key.items() if v is not None}
    alignment_mean, alignment_count = stl.alignment_losses(
        model, hidden, sft_batch.region_batch, sft_batch.model_batch.image_grid_thw, **geom
    )

    terms: list[Tensor] = []
    report: dict[str, Any] = {
        "masked_policy_keys": [list(k) for k in masked],
        "typed_key_count": len(contributing),
        "typed_supervised_count": sum(c for _, c in contributing.values()),
        "typed_mean": None,
        "alignment_mean": None,
        "alignment_supervised_count": alignment_count,
    }
    if contributing:
        typed_mean = torch.stack([loss for loss, _ in contributing.values()]).mean()
        terms.append((lam["L_typed"] / 1000) * typed_mean)
        report["typed_mean"] = float(typed_mean.detach())
    if alignment_mean is not None:
        terms.append((lam["L_alignment"] / 1000) * alignment_mean)
        report["alignment_mean"] = float(alignment_mean.detach())
    if not terms:
        return AuxiliaryResult(total=None, report=report)
    total = torch.stack(terms).sum()
    if not bool(torch.isfinite(total)):
        raise s1_train.NonFiniteStepError(f"auxiliary loss is {float(total.detach())}")
    return AuxiliaryResult(total=total, report=report)


def combine_with_auxiliary(
    rl: RlLossResult, aux: AuxiliaryResult | None, *, supervised_weight_micro: int
) -> Tensor:
    """``rl.loss`` unchanged when there is no auxiliary scalar (weight 0 or an
    unsupervised batch), else ``loss.combine_rl_and_supervised``."""
    if aux is None or aux.total is None:
        return rl.loss
    return combine_rl_and_supervised(rl, aux.total, supervised_weight_micro=supervised_weight_micro)


__all__ = [
    "AuxiliaryRefused",
    "AuxiliaryResult",
    "SftAuxBatch",
    "assert_disjoint",
    "auxiliary_loss",
    "combine_with_auxiliary",
]
