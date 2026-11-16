"""Build the frozen S4.5 KL reference as a second model over the SAME candidate
(RFC005-ERRATA-010 E4.1; design D5; dormant).

The Qwen backbone is a closure plus two non-persistent buffers, not an
``nn.Module`` tree, so "share the backbone" means: both models are built from one
``QwenCandidate`` and hold the very same ``_decode`` / ``_inject_visual`` objects
and the very same embedding / lm-head tensors. Only the trainable document
branch is a second copy (loaded from the checkpoint state, then frozen).

Construction order is a requirement, not a convenience: ``Module.to(device)``
re-creates buffers, so the candidate is moved to the target device FIRST
(``place_candidate``), and both models are built afterwards from those tensor
objects. Building the reference from a candidate on another device, or
``.to``-ing a model afterwards, would hand the reference its own buffer copy.
``assert_reference_shares_frozen_backbone`` is the check; this module calls it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from unflatten.model.candidate_loader import QwenCandidate
from unflatten.model.candidate_model import QwenDenseCandidateModel, TeacherForcedBatch
from unflatten.model.candidate_runtime import build_candidate_model
from unflatten.model.level_router.policy import LevelRouterPolicy
from unflatten.model.rl.reference import (
    ReferenceIncompatible,
    ReferencePolicyIdentity,
    assert_reference_shares_frozen_backbone,
    check_reference_compatible,
    freeze_reference_parameters,
)


class ReferenceVariantMismatch(ReferenceIncompatible):
    """The reference's architecture variant / DeepStack flag differs from the policy's."""

    def __init__(self, message: str) -> None:
        super().__init__("architecture_variant", message)


def place_candidate(candidate: QwenCandidate, device: torch.device | str) -> None:
    """Move every candidate parameter to ``device`` in place (same ``Parameter``
    objects, new ``.data``), so models built afterwards share tensors on that
    device instead of each ``.to(device)`` creating its own buffer."""
    target = torch.device(device)
    for param in candidate.named_parameters.values():
        if param.device != target:
            param.data = param.data.to(target)


def _model_state(state: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    model_state = state.get("model_state_dict")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("checkpoint state has no non-empty 'model_state_dict'")
    return model_state


def _build_loaded(
    candidate: QwenCandidate,
    state: Mapping[str, Any],
    device: torch.device | str,
    *,
    level_router_policy: LevelRouterPolicy | None,
    deepstack: bool,
) -> QwenDenseCandidateModel:
    model = build_candidate_model(
        candidate,
        torch.device(device),
        level_router_policy=level_router_policy,
        deepstack=deepstack,
    )
    model.load_state_dict(_model_state(state), strict=True)
    return model


def build_policy_model(
    candidate: QwenCandidate,
    state: Mapping[str, Any],
    device: torch.device | str,
    *,
    level_router_policy: LevelRouterPolicy | None = None,
    deepstack: bool = False,
) -> QwenDenseCandidateModel:
    """The policy, initialised from the SAME checkpoint state as its reference
    (so KL == 0 at step 0). Trainable groups are left as the model builds them."""
    place_candidate(candidate, device)
    return _build_loaded(
        candidate, state, device, level_router_policy=level_router_policy, deepstack=deepstack
    )


def build_reference_model(
    candidate: QwenCandidate,
    state: Mapping[str, Any],
    device: torch.device | str,
    *,
    policy: QwenDenseCandidateModel,
    reference_identity: ReferencePolicyIdentity,
    policy_parent_identity: ReferencePolicyIdentity,
    run_class: str,
    level_router_policy: LevelRouterPolicy | None = None,
    deepstack: bool = False,
) -> QwenDenseCandidateModel:
    """Frozen, eval-mode second model from ``candidate`` + ``state["model_state_dict"]``.

    Order: identity refusal (``check_reference_compatible``, no model touched) ->
    place candidate on ``device`` -> build -> strict load -> freeze -> variant
    equality with ``policy`` -> shared-backbone assertion. ``policy`` must have
    been built from the same ``candidate`` (see module docstring).
    """
    check_reference_compatible(reference_identity, policy_parent_identity, run_class=run_class)
    place_candidate(candidate, device)
    reference = _build_loaded(
        candidate, state, device, level_router_policy=level_router_policy, deepstack=deepstack
    )
    freeze_reference_parameters(reference)
    if reference.deepstack != policy.deepstack:
        raise ReferenceVariantMismatch(
            f"reference deepstack={reference.deepstack} != policy deepstack={policy.deepstack}"
        )
    if reference.architecture_variant != policy.architecture_variant:
        raise ReferenceVariantMismatch(
            "reference architecture variant (level router / expert adapter / ...) "
            "differs from the policy's"
        )
    assert_reference_shares_frozen_backbone(
        policy, reference, candidate_parameters=candidate.named_parameters
    )
    return reference


def reference_forward_logits(reference: nn.Module, batch: TeacherForcedBatch) -> torch.Tensor:
    """``(batch, steps, vocab)`` composite logits of the frozen reference under
    ``torch.no_grad()``. No legal-set logic: ``legal_logprobs`` (B1) consumes these."""
    with torch.no_grad():
        return reference(batch)
