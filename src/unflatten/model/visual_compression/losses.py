"""RFC005 §27.11 `L_visual_compressor` -- standalone, NOT a `LossUnit` family.

"Trénuje iba `visual_compressor` proti frozen primary teacherovi.
Feature/attention/task-logit distillation používa `sealed_derived` branch s
teacher checkpoint/Component Lock hashom, exact visual attachment points,
implementation hashom a output projection hashom." (§27.11)

This is deliberately a plain function over plain tensors, not wired into
:mod:`unflatten.model.training_materializer`'s `LOSS_FAMILY_IDS` /
`SUPPORTED_LOSS_FAMILIES` -- §27.11 itself says the compressor loss "nesmie
... nahradiť architecture-contract a free-running non-inferiority gates",
i.e. it is a distillation signal for the compressor's own opt-in S7 child
run, never a token-stream-packed family the M1 LossUnit machinery emits
(`test_losses.py::test_l_visual_compressor_not_in_loss_family_ids` pins
this). No gradient may reach `visual_compressor`'s host module until that
run is ratified (DEC-047 condition 5); this module only computes the
number.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SealedDerivedProvenance:
    """§27.11's required provenance for the teacher-derived branch.

    `task_preservation_family` names the *separate* `rfc003_supervised`
    family (`L_structure`/`L_text`/`L_typed`) task preservation reads
    ground truth from when it is present -- §27.11: "nesmie sa zamiešať do
    jedného teacher-derived LossUnit". `None` when no task-preservation term
    is computed alongside this one.
    """

    teacher_checkpoint_hash: str
    component_lock_hash: str
    attachment_points: tuple[str, ...]
    implementation_hash: str
    output_projection_hash: str
    task_preservation_family: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("teacher_checkpoint_hash", self.teacher_checkpoint_hash),
            ("component_lock_hash", self.component_lock_hash),
            ("implementation_hash", self.implementation_hash),
            ("output_projection_hash", self.output_projection_hash),
        ):
            if not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not self.attachment_points:
            raise ValueError("attachment_points must name at least one attachment point")
        if self.task_preservation_family not in (None, "L_structure", "L_text", "L_typed"):
            raise ValueError(
                "task_preservation_family must be one of None, 'L_structure', "
                f"'L_text', 'L_typed', got {self.task_preservation_family!r}"
            )


def l_visual_compressor(
    student_tokens: Tensor,
    teacher_tokens: Tensor,
    *,
    provenance: SealedDerivedProvenance,
) -> Tensor:
    """Feature-distillation term of §27.11: masked MSE between the
    compressor's student-side tokens (re-expanded to the teacher's original
    token count, e.g. via `expand_alignment`) and the frozen teacher's own
    tokens at the same positions, over the same `attachment_points`.

    `teacher_tokens` must already be detached by the caller (this run's
    frozen primary produced them, §27.11: "proti frozen primary
    teacherovi") -- this function does not detach on the caller's behalf,
    so a caller that forgets to freeze the teacher gets gradient flowing
    into it, which is exactly the mistake `provenance` exists to make
    auditable, not to silently correct.
    """
    if student_tokens.shape != teacher_tokens.shape:
        raise ValueError(
            f"student_tokens shape {tuple(student_tokens.shape)} must match "
            f"teacher_tokens shape {tuple(teacher_tokens.shape)} at the "
            f"attachment points {provenance.attachment_points}"
        )
    return F.mse_loss(student_tokens, teacher_tokens.detach())


def l_pre_encoder_compressor_signal_a(
    predicted_logits: Tensor, target_risk_class: Tensor, mask: Tensor
) -> Tensor:
    """§27.15 `L_importance_target` term only (signal a, RFC004-ERRATA-005
    E4 variant a): masked cross-entropy between the scout's per-cell risk
    logits and the `RegionImportanceTarget`-derived class index.

    `predicted_logits`: `(..., 5)`. `target_risk_class`, `mask`: `(...)`,
    matching `predicted_logits`'s leading dims. `mask` is availability
    (§29.1/§29.6: a missing target is never numeric zero), not confidence.
    """
    if predicted_logits.shape[:-1] != target_risk_class.shape:
        raise ValueError(
            f"predicted_logits leading shape {tuple(predicted_logits.shape[:-1])} "
            f"must match target_risk_class shape {tuple(target_risk_class.shape)}"
        )
    if target_risk_class.shape != mask.shape:
        raise ValueError(
            f"target_risk_class shape {tuple(target_risk_class.shape)} must match "
            f"mask shape {tuple(mask.shape)}"
        )
    per_cell = F.cross_entropy(
        predicted_logits.reshape(-1, predicted_logits.shape[-1]),
        target_risk_class.reshape(-1),
        reduction="none",
    )
    mask_flat = mask.reshape(-1).to(per_cell.dtype)
    denom = mask_flat.sum().clamp(min=1.0)
    return (per_cell * mask_flat).sum() / denom


class PreEncoderSignalBUnavailable(NotImplementedError):
    """§27.15 signal b (`L_task_preservation`, RFC004-ERRATA-005 E4 variant
    b) needs an online forward through the frozen Qwen3-VL vision encoder
    and backprop through it (RFC005-ERRATA-007 E5 rule 3) -- there is no
    backbone here to call. Refuses rather than returning a fabricated
    zero, which §21.2/§29.1's masking discipline would otherwise mistake
    for "no loss due here", not "not computed"."""


def l_pre_encoder_compressor_signal_b(*_args: object, **_kwargs: object) -> Tensor:
    raise PreEncoderSignalBUnavailable(
        "L_task_preservation (signal b, RFC005-ERRATA-007 §27.15) requires an "
        "online forward through the frozen Qwen3-VL vision encoder; this "
        "module has no backbone to call and refuses instead of faking the term"
    )
