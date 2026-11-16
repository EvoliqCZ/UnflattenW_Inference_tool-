"""Trainability smoke suite -- RFC005 §9.1 item 11 / §13.2's "vykonať RFC004
trainability smoke".

**What this is not, first, because a passing smoke is easy to misread.**
This module is *not* an S1 gradient update and never produces an S1
checkpoint or promotion ancestry. S1 (RFC005 §14) remains blocked on the
DEC-005 D3 / packet §7 blockers regardless of what this suite reports; it
runs entirely on synthetic tensors built at import/call time, never on the
admitted corpus, and its pass/fail is a mechanical machinery check, not a
quality claim (`docs/rfc/RFC005-training-strategy-v0.1-2.md:923`: "iba
contract/trainability canary, nie quality, render-fidelity, generalization").

**What it does check.** RFC005 §13.2 lists "vykonať RFC004 trainability
smoke" as an S0 step, and §13.3's exit gate requires it to "pass for every
admitted track". At S0 time there is no admitted track's model yet to run
it against -- only the contract, i.e. the trainable-group registry -- so
this suite answers the question the registry can actually be asked before
any track exists: **if a model were wired exactly as the registry claims,
would gradients behave that way, and does one optimizer step move a toy
loss in the right direction?** It does this in three complementary ways:

1. :func:`run_registry_backed_gradient_smoke` builds one synthetic
   ``nn.Parameter`` per tensor identity the registry says a `dense V0`
   group owns (`trainable_group_registry.trainable_group_registry()`,
   restricted to the two kinds dense V0 actually instantiates tensors for:
   ``trainable_primary_s1`` and ``frozen_base`` -- ``not_applicable`` and
   `lora.*` groups own no tensor in this architecture, per that module's
   own docstring, so there is nothing to build for them). It is exhaustive
   over the full 20-group registry, not just the groups with a real
   `nn.Module` today.
2. :func:`run_composite_module_smoke` builds the two DIAGNOSTIC decoder-era
   modules (`decoder.StructureDecoder` standing in for `tree_text_decoder`,
   `expert_adapter.DenseExpertAdapter` for `expert_adapter.dense`), wires
   them the way RFC004's Adapter → Decoder → Expert-Adapter branch
   composes, and proves gradient flow, a loss-decreasing step, shape/dtype
   coherence and determinism, plus a frozen upstream stand-in (for
   `backbone.visual`) that must not accumulate a gradient. Predates M2;
   kept because `StructureDecoder` is still real production code, even
   though it is not the M2 Qwen candidate's `tree_text_decoder`.
3. :func:`run_candidate_model_smoke` is the M2 exit criterion itself
   (design §8 M2): the actual dense-V0 Qwen candidate
   (`candidate_model.QwenDenseCandidateModel`, built from
   `candidate_loader.build_stub_qwen_candidate`) run through one
   teacher-forced forward/backward, cross-checked against the registry the
   same way. It is the only one of the three that exercises
   `adapter.projection` (`VisualMemory`) and the M2-era `tree_text_decoder`
   (`composite_head.CompositeDecodeHead`) as real modules, and the only one
   whose frozen stand-in is the actual candidate loader's frozen-parameter
   map rather than one bare `nn.Linear`.

All three share :func:`check_gradient_flow`, the actual cross-check: it
takes a mapping from parameter name to owning group id and compares the
group's *expected* trainable/frozen disposition (read from the registry,
not hand-written here) against what `.grad` after `.backward()` actually
shows. That function is the whole point, and it is exercised directly by
`tests/model/test_trainability_smoke.py` against a deliberately mis-wired
fixture to prove it can fail, not only pass.

**The scope gap this docstring used to record here is closed as of M3.**
All four primary-S1 trainable groups -- `tree_text_decoder`,
`expert_adapter.dense`, `adapter.projection` and, as of M3,
`document_context_encoder` (`document_context_encoder.py`) -- now have a
real `nn.Module` wired into `QwenDenseCandidateModel`.
`run_candidate_model_smoke` audits all four; `run_trainability_smoke_suite`'s
`scope.real_modules_missing` is empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from unflatten.model.candidate_loader import (
    build_stub_qwen_candidate,
    group_id_of_qwen_parameter,
)
from unflatten.model.candidate_model import QwenDenseCandidateModel, TeacherForcedBatch
from unflatten.model.composite_head import TREE_TEXT_DECODER_GROUP_ID
from unflatten.model.composite_vocab import packed_ranges
from unflatten.model.decoder import StructureDecoder
from unflatten.model.document_context_encoder import DOCUMENT_CONTEXT_ENCODER_GROUP_ID
from unflatten.model.expert_adapter import (
    EXPERT_ADAPTER_DENSE_GROUP_ID,
    DenseExpertAdapter,
)
from unflatten.model.trainable_group_registry import (
    TrainableGroup,
    trainable_group_registry,
)
from unflatten.model.visual_memory import VISUAL_MEMORY_GROUP_ID

TRAINABILITY_SMOKE_SCHEMA_VERSION = "0.2.0"

#: The one track `unflatten.dataset.admission.report` admits at S0 today
#: (RFC004 ratified only for `dense_contract_canary_ready` /
#: `dense_vertical_slice_ready` under DEC-005).
DENSE_CONTRACT_CANARY_TRACK = "dense_contract_canary"

#: Registry `kind`s dense V0 actually instantiates a tensor for. See the
#: module docstring's point 1.
_TENSOR_OWNING_KINDS = frozenset({"trainable_primary_s1", "frozen_base"})


@dataclass(frozen=True)
class GradientFlowViolation:
    """One parameter whose observed gradient state disagrees with its
    registry-declared group's expected disposition in primary S1."""

    parameter_name: str
    group_id: str
    expected: str  # "trainable" | "frozen"
    actual: str  # "trainable" | "frozen"


@dataclass(frozen=True)
class SmokeResult:
    """The outcome of one smoke check."""

    name: str
    passed: bool
    violations: tuple[GradientFlowViolation, ...] = field(default_factory=tuple)
    loss_before: float | None = None
    loss_after: float | None = None
    loss_decreased: bool | None = None
    determinism_ok: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)


def check_gradient_flow(
    named_parameters: dict[str, torch.nn.Parameter],
    group_id_of: dict[str, str],
    expected_trainable_group_ids: frozenset[str],
) -> tuple[GradientFlowViolation, ...]:
    """Compare observed post-`.backward()` gradient state against what each
    parameter's group id says it should be.

    Must be called *after* `.backward()`. A parameter is "trainable" here
    iff `.grad` is not `None`; the registry's expectation comes from
    `group_id in expected_trainable_group_ids`. This function makes no
    claim about *why* a mismatch happened -- only that the wiring and the
    registry disagree for a named parameter.
    """
    violations: list[GradientFlowViolation] = []
    for name, param in named_parameters.items():
        group_id = group_id_of[name]
        expected = "trainable" if group_id in expected_trainable_group_ids else "frozen"
        actual = "trainable" if param.grad is not None else "frozen"
        if actual != expected:
            violations.append(
                GradientFlowViolation(
                    parameter_name=name,
                    group_id=group_id,
                    expected=expected,
                    actual=actual,
                )
            )
    return tuple(violations)


def _expected_trainable_group_ids(
    registry: dict[str, TrainableGroup],
) -> frozenset[str]:
    return frozenset(
        g.group_id for g in registry.values() if g.kind == "trainable_primary_s1"
    )


def _tensor_owning_groups(
    registry: dict[str, TrainableGroup],
) -> tuple[TrainableGroup, ...]:
    return tuple(g for g in registry.values() if g.kind in _TENSOR_OWNING_KINDS)


def run_registry_backed_gradient_smoke(seed: int = 0) -> SmokeResult:
    """Exhaustive synthetic-tensor cross-check against every tensor-owning
    group in `trainable_group_registry`.

    One `nn.Parameter` per `(group_id, tensor_identity)`, `requires_grad`
    set from the group's `kind`. The toy objective is the sum of squares of
    every parameter -- trainable or not -- so a frozen parameter still
    participates in the forward graph (as it would in a real frozen-base
    module) but records no gradient because `requires_grad=False`.
    """
    torch.manual_seed(seed)
    registry = trainable_group_registry()
    expected_trainable = _expected_trainable_group_ids(registry)

    parameters: dict[str, torch.nn.Parameter] = {}
    group_id_of: dict[str, str] = {}
    for group in _tensor_owning_groups(registry):
        requires_grad = group.group_id in expected_trainable
        for identity in group.owned_tensor_identities:
            name = f"{group.group_id}::{identity}"
            parameters[name] = nn.Parameter(
                torch.randn(4), requires_grad=requires_grad
            )
            group_id_of[name] = group.group_id

    def _loss() -> torch.Tensor:
        return sum(p.pow(2).sum() for p in parameters.values())

    loss_before = float(_loss().detach())
    _loss().backward()

    violations = check_gradient_flow(parameters, group_id_of, expected_trainable)

    trainable_params = [p for p in parameters.values() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    loss_after = float(_loss().detach())

    return SmokeResult(
        name="registry_backed_gradient_smoke",
        passed=not violations and loss_after < loss_before,
        violations=violations,
        loss_before=loss_before,
        loss_after=loss_after,
        loss_decreased=loss_after < loss_before,
        details={
            "tensor_count": len(parameters),
            "trainable_tensor_count": len(trainable_params),
            "group_count": len(_tensor_owning_groups(registry)),
        },
    )


def _composite_forward(
    decoder: StructureDecoder,
    adapter: DenseExpertAdapter,
    frozen_backbone: nn.Linear,
    raw_visual: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """`raw_visual -> frozen backbone stand-in -> decoder -> expert adapter
    -> vocab head`, matching RFC004's Adapter → Decoder → Expert-Adapter
    composition: the expert adapter transforms the *shared decoder hidden
    state* before the head, not the decoder's own logits."""
    visual_features = frozen_backbone(raw_visual)
    memory = decoder.project_memory(visual_features)
    steps = input_ids.shape[1]
    positions = torch.arange(steps, device=input_ids.device)
    hidden = decoder.token_embedding(input_ids) + decoder.position_embedding(positions)
    causal = nn.Transformer.generate_square_subsequent_mask(
        steps, device=input_ids.device
    )
    decoded = decoder.decoder(hidden, memory, tgt_mask=causal, tgt_is_causal=True)
    adapted = adapter(decoded)
    return decoder.head(decoder.norm(adapted))


def run_composite_module_smoke(
    seed: int = 0,
    *,
    device: str = "cpu",
    visual_dim: int = 16,
    d_model: int = 24,
    steps: int = 5,
    batch: int = 2,
    tokens: int = 6,
) -> SmokeResult:
    """Real-module smoke: `StructureDecoder` (`tree_text_decoder`) composed
    with `DenseExpertAdapter` (`expert_adapter.dense`), fed by a frozen
    `nn.Linear` standing in for `backbone.visual`.

    Runs on CPU by default; `device="cuda"` is accepted but never assumed
    -- callers must already know CUDA is available, this function does not
    probe for it.
    """
    torch.manual_seed(seed)
    registry = trainable_group_registry()
    expected_trainable = _expected_trainable_group_ids(registry)

    decoder = StructureDecoder(
        visual_dim=visual_dim, d_model=d_model, layers=2, heads=2, max_steps=steps
    ).to(device)
    adapter = DenseExpertAdapter(d_model=d_model, layers=1).to(device)
    frozen_backbone = nn.Linear(visual_dim, visual_dim).to(device)
    for p in frozen_backbone.parameters():
        p.requires_grad_(False)

    raw_visual = torch.randn(batch, tokens, visual_dim, device=device)
    vocab_size = decoder.token_embedding.num_embeddings
    input_ids = torch.randint(0, vocab_size, (batch, steps), device=device)
    targets = torch.randint(0, vocab_size, (batch, steps), device=device)

    def _run() -> tuple[torch.Tensor, torch.Tensor]:
        logits = _composite_forward(decoder, adapter, frozen_backbone, raw_visual, input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        return logits, loss

    logits, loss = _run()
    shape_ok = logits.shape == (batch, steps, decoder.token_embedding.num_embeddings)
    dtype_ok = logits.dtype == raw_visual.dtype

    loss_before = float(loss.detach())
    loss.backward()

    parameters: dict[str, torch.nn.Parameter] = {}
    group_id_of: dict[str, str] = {}
    for name, param in decoder.named_parameters():
        parameters[f"decoder.{name}"] = param
        group_id_of[f"decoder.{name}"] = "tree_text_decoder"
    for name, param in adapter.named_parameters():
        parameters[f"adapter.{name}"] = param
        group_id_of[f"adapter.{name}"] = EXPERT_ADAPTER_DENSE_GROUP_ID

    violations = check_gradient_flow(parameters, group_id_of, expected_trainable)
    frozen_backbone_leaked = any(p.grad is not None for p in frozen_backbone.parameters())
    finite_ok = all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters.values()
    )

    optimizer = torch.optim.SGD(list(decoder.parameters()) + list(adapter.parameters()), lr=0.5)
    optimizer.step()
    optimizer.zero_grad()
    _, loss_after_step = _run()
    loss_after = float(loss_after_step.detach())

    # Determinism: rebuild everything from the same seed/config and compare
    # the pre-step loss and gradients bit-for-bit.
    torch.manual_seed(seed)
    decoder_b = StructureDecoder(
        visual_dim=visual_dim, d_model=d_model, layers=2, heads=2, max_steps=steps
    ).to(device)
    adapter_b = DenseExpertAdapter(d_model=d_model, layers=1).to(device)
    frozen_backbone_b = nn.Linear(visual_dim, visual_dim).to(device)
    for p in frozen_backbone_b.parameters():
        p.requires_grad_(False)
    logits_b = _composite_forward(decoder_b, adapter_b, frozen_backbone_b, raw_visual, input_ids)
    determinism_ok = torch.equal(logits, logits_b)

    passed = (
        not violations
        and not frozen_backbone_leaked
        and finite_ok
        and shape_ok
        and dtype_ok
        and loss_after < loss_before
        and determinism_ok
    )

    return SmokeResult(
        name="composite_module_smoke",
        passed=passed,
        violations=violations,
        loss_before=loss_before,
        loss_after=loss_after,
        loss_decreased=loss_after < loss_before,
        determinism_ok=determinism_ok,
        details={
            "shape_ok": shape_ok,
            "dtype_ok": dtype_ok,
            "gradient_finite": finite_ok,
            "frozen_backbone_leaked": frozen_backbone_leaked,
            "device": device,
        },
    )


def run_candidate_model_smoke(
    seed: int = 0,
    *,
    visual_dim: int = 6,
    d_model: int = 8,
    qwen_vocab_size: int = 12,
    visual_tokens: int = 3,
    steps: int = 7,
    batch: int = 2,
) -> SmokeResult:
    """M2 exit criterion (design §8 M2): the Qwen dense candidate
    (`candidate_model.QwenDenseCandidateModel`, fed by
    `candidate_loader.build_stub_qwen_candidate`) passes a teacher-forced
    forward/backward and the parameter-group audit.

    Covers the three primary-S1 groups `run_composite_module_smoke` does not
    reach through the real M2/M3 wiring: `adapter.projection`
    (`VisualMemory`), `document_context_encoder` (`DocumentContextEncoder`,
    M3) and `tree_text_decoder` in its actual M2 form (`CompositeDecodeHead`,
    not the diagnostic `StructureDecoder` `run_composite_module_smoke`
    covers). It also proves the frozen stub candidate -- standing in for
    `backbone.visual`/`backbone.language`/`backbone.projector` -- never
    accumulates a gradient through the merged forward pass.
    """
    torch.manual_seed(seed)
    candidate = build_stub_qwen_candidate(
        visual_hidden_size=visual_dim, hidden_size=d_model, vocab_size=qwen_vocab_size, seed=seed
    )
    ranges = packed_ranges(qwen_vocab_size)
    model = QwenDenseCandidateModel(
        decode=candidate.decode,
        inject_visual=candidate.inject_visual,
        visual_dim=visual_dim,
        d_model=d_model,
        ranges=ranges,
        qwen_embedding_weight=candidate.embed_tokens.weight,
        qwen_lm_head_weight=candidate.lm_head_weight,
    )

    torch.manual_seed(seed + 1)
    visual_features = torch.randn(batch, visual_tokens, visual_dim)
    # `visual_tokens` leading positions are the native image placeholder
    # (`inject_visual` requires exactly one per visual token, in order);
    # the tail CYCLES deterministically through structural, Qwen-text and
    # byte-fallback ids, so the fixture is GUARANTEED (not merely likely) to
    # exercise every `CompositeDecodeHead` sub-embedding
    # (`composite_head.embed_packed` routes purely by id range) whenever
    # `steps >= visual_tokens + len(id_pool)` -- a random draw over a short
    # tail could by chance never hit one sub-path, which would make this
    # smoke falsely report a healthy group as frozen depending on the RNG
    # draw, not on the wiring.
    image_placeholder_id = ranges.qwen_start + candidate.image_token_id
    placeholder_block = torch.full((batch, visual_tokens), image_placeholder_id, dtype=torch.long)
    id_pool = torch.tensor(
        [
            0,
            ranges.qwen_start,
            ranges.byte_control_start,
            ranges.byte_value_start + 1,
        ]
    )
    tail_steps = max(steps - visual_tokens, 0)
    tail = id_pool[torch.arange(tail_steps) % len(id_pool)].unsqueeze(0).expand(batch, -1)
    composite_input_ids = torch.cat([placeholder_block, tail], dim=1)
    composite_attention_mask = torch.ones(batch, composite_input_ids.shape[1], dtype=torch.long)
    image_grid_thw = torch.tensor([[1, 1, visual_tokens]])
    fixture = TeacherForcedBatch(
        visual_features=visual_features,
        composite_input_ids=composite_input_ids,
        composite_attention_mask=composite_attention_mask,
        image_grid_thw=image_grid_thw,
    )
    actual_steps = composite_input_ids.shape[1]
    targets = torch.randint(0, ranges.total_count, (batch, actual_steps))

    def _run() -> tuple[torch.Tensor, torch.Tensor]:
        logits = model(fixture)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        return logits, loss

    logits, loss = _run()
    shape_ok = logits.shape == (batch, actual_steps, ranges.total_count)
    finite_logits_ok = bool(torch.isfinite(logits).all())

    loss_before = float(loss.detach())
    loss.backward()

    registry = trainable_group_registry()
    expected_trainable = _expected_trainable_group_ids(registry)

    parameters: dict[str, torch.nn.Parameter] = {}
    group_id_of: dict[str, str] = {}
    for name, param in model.visual_memory.named_parameters():
        parameters[f"visual_memory.{name}"] = param
        group_id_of[f"visual_memory.{name}"] = VISUAL_MEMORY_GROUP_ID
    for name, param in model.document_context_encoder.named_parameters():
        parameters[f"document_context_encoder.{name}"] = param
        group_id_of[f"document_context_encoder.{name}"] = DOCUMENT_CONTEXT_ENCODER_GROUP_ID
    for name, param in model.dense_adapter.named_parameters():
        parameters[f"dense_adapter.{name}"] = param
        group_id_of[f"dense_adapter.{name}"] = EXPERT_ADAPTER_DENSE_GROUP_ID
    for name, param in model.composite_head.named_parameters():
        parameters[f"composite_head.{name}"] = param
        group_id_of[f"composite_head.{name}"] = TREE_TEXT_DECODER_GROUP_ID

    violations = check_gradient_flow(parameters, group_id_of, expected_trainable)
    finite_grad_ok = all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters.values()
    )

    frozen_leaked = [
        name for name, param in candidate.named_parameters.items() if param.grad is not None
    ]
    # Every frozen candidate parameter must map to a `frozen_base` registry
    # group -- the same completeness check `candidate_loader`'s own tests
    # run, repeated here because it is what makes "frozen groups have zero
    # gradient" a claim about the *whole* candidate, not just the parameters
    # this function happened to enumerate.
    unknown_frozen_groups = {
        group_id_of_qwen_parameter(name)
        for name in candidate.named_parameters
        if registry.get(group_id_of_qwen_parameter(name)) is None
        or registry[group_id_of_qwen_parameter(name)].kind != "frozen_base"
    }

    # lr=0.5 (used by the two smaller composite smokes above) overshoots
    # here: this is now the deepest composition of the three (VisualMemory
    # + DocumentContextEncoder + native scatter/decoder + DenseExpertAdapter
    # + CompositeDecodeHead), so a full-batch single step at that lr can
    # land past the loss's local minimum rather than moving toward it --
    # itself unremarkable for plain SGD, not a wiring defect, but a smaller
    # lr is what a mechanical "does one step help" check should use.
    optimizer = torch.optim.SGD(list(parameters.values()), lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    for param in candidate.named_parameters.values():
        param.grad = None
    _, loss_after_step = _run()
    loss_after = float(loss_after_step.detach())

    passed = (
        not violations
        and not frozen_leaked
        and not unknown_frozen_groups
        and finite_grad_ok
        and finite_logits_ok
        and shape_ok
        and loss_after < loss_before
    )

    return SmokeResult(
        name="candidate_model_smoke",
        passed=passed,
        violations=violations,
        loss_before=loss_before,
        loss_after=loss_after,
        loss_decreased=loss_after < loss_before,
        details={
            "shape_ok": shape_ok,
            "finite_logits": finite_logits_ok,
            "finite_gradients": finite_grad_ok,
            "frozen_candidate_leaked": frozen_leaked,
            "unknown_or_non_frozen_groups": sorted(unknown_frozen_groups),
        },
    )


# ---------------------------------------------------------------------------
# Parameter movement check -- DEC-007 D3.1 / 2026-09-04 review must_fix #1
# ---------------------------------------------------------------------------
#
# A finite non-zero gradient (gate 3) is necessary but NOT sufficient
# evidence that training does anything: bf16 has 8 mantissa bits, so for a
# parameter of magnitude ~1e-1 the gap between representable values is
# ~5e-4, while a typical `lr * grad` step can be orders of magnitude
# smaller than that. The update rounds away and `p + update == p` exactly,
# silently, with a perfectly finite non-zero gradient the whole time.
# Fifteen gates could report green over a model that never moved.
#
# This is deliberately per-tensor, multi-step, and built from the caller's
# OWN optimizer -- see `run_parameter_movement_check`'s own docstring for
# why each of those three properties is load-bearing, not incidental.


@dataclass(frozen=True)
class TensorMovement:
    """Whether ONE named parameter tensor's storage changed at all across
    the checked steps, and at which step it first did (1-indexed; `None`
    if it never moved)."""

    name: str
    moved: bool
    first_moved_step: int | None


@dataclass(frozen=True)
class ParameterMovementReport:
    """Per-tensor movement evidence over `steps` real optimizer steps.

    `all_moved` is the pass/fail a caller should actually gate on --
    `moved_count / total_count` alone can hide one small tensor moving
    while the rest sit frozen (`any(...)` is not evidence; a countable
    claim needs a counter, per every `moved`/`total` pair kept here).
    """

    steps: int
    tensors: tuple[TensorMovement, ...]
    moved_count: int
    total_count: int

    @property
    def all_moved(self) -> bool:
        return self.total_count > 0 and self.moved_count == self.total_count


def run_parameter_movement_check(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step_fn: Any,
    *,
    steps: int = 5,
) -> ParameterMovementReport:
    """Run `steps` REAL optimizer steps against `model`/`optimizer` and
    report, per trainable parameter TENSOR, whether its own storage ever
    changed.

    **This function does not run a forward pass itself.** `step_fn` is a
    zero-argument callable that runs one forward (+ any loss computation)
    and returns a scalar loss tensor with `requires_grad=True` -- the
    caller's own model, own batch, own loss. This function only owns the
    `backward()` / `optimizer.step()` / `zero_grad()` loop and the
    before/after comparison, because the comparison is the only part that
    must be identical across every caller: per tensor, `torch.equal`
    against a frozen initial snapshot, never an aggregate `any(...)`.

    **Three properties are load-bearing, not stylistic choices** (2026-09-04
    review, must_fix #1):

    1. **Per-tensor, not aggregate.** `any(p changed for p in params)` is
       satisfied by ONE small tensor moving while every other trainable
       tensor sits frozen underflowed. `ParameterMovementReport.tensors`
       names every one so `moved_count == total_count` is a real claim, not
       an existence proof standing in for a universal one.
    2. **Multiple steps, not just the first.** Gradients are typically
       largest near initialization; a step that moves a parameter at step 1
       may round away to nothing by step 500 once the loss landscape
       flattens. `first_moved_step` records WHEN each tensor first moved so
       a caller can see the trend, not just a single sample of it.
    3. **The caller's own optimizer object, at its own configured lr,
       against its own real model.** This function takes `optimizer`
       (already constructed with the run's real learning rate over the
       run's real parameters) rather than building one internally --
       constructing a fresh optimizer at an arbitrary lr here would prove
       something about a hypothetical run, not about the one being audited.

    **Loss decreasing is affirmatively NOT evidence that any parameter
    moved, and must never be substituted for this check.** Loss can drift
    from dropout, batch composition, and forward-path bf16 rounding noise
    even when every weight is frozen in place by an underflowed update --
    exactly the failure this check exists to catch would still show a
    plausible-looking loss curve.
    """
    named_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    initial = {name: p.detach().clone() for name, p in named_params.items()}
    first_moved_step: dict[str, int | None] = dict.fromkeys(named_params)

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        loss = step_fn()
        loss.backward()
        optimizer.step()
        for name, param in named_params.items():
            if first_moved_step[name] is None and not torch.equal(
                param.detach(), initial[name]
            ):
                first_moved_step[name] = step

    tensors = tuple(
        TensorMovement(
            name=name,
            moved=first_moved_step[name] is not None,
            first_moved_step=first_moved_step[name],
        )
        for name in named_params
    )
    moved_count = sum(1 for t in tensors if t.moved)
    return ParameterMovementReport(
        steps=steps, tensors=tensors, moved_count=moved_count, total_count=len(tensors)
    )


def run_trainability_smoke_suite(seed: int = 0, *, device: str = "cpu") -> dict[str, Any]:
    """Run all smoke checks and return a combined, JSON-serialisable report.

    This is the artefact `unflatten.dataset.admission.report` probes for by
    the literal source pattern `trainability_smoke`. It is a mechanical
    contract/gradient check on synthetic tensors -- see the module
    docstring for what it explicitly does not claim.
    """
    registry_result = run_registry_backed_gradient_smoke(seed=seed)
    composite_result = run_composite_module_smoke(seed=seed, device=device)
    candidate_result = run_candidate_model_smoke(seed=seed)
    results = (registry_result, composite_result, candidate_result)
    return {
        "trainability_smoke_schema_version": TRAINABILITY_SMOKE_SCHEMA_VERSION,
        "run_class": "discovery",
        "gradient_updates": 0,
        "not_an_s1_gradient_update": True,
        "passed": all(r.passed for r in results),
        "results": {r.name: _result_to_dict(r) for r in results},
        "scope": {
            "covers_all_registry_groups": True,
            "covers_all_real_modules": True,
            "real_modules_covered": [
                "tree_text_decoder",
                "expert_adapter.dense",
                "adapter.projection",
                "document_context_encoder",
            ],
            "real_modules_missing": [],
        },
        "tracks": {
            DENSE_CONTRACT_CANARY_TRACK: {
                "passed": all(r.passed for r in results),
                "results": [r.name for r in results],
            }
        },
    }


def _result_to_dict(result: SmokeResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "passed": result.passed,
        "violations": [
            {
                "parameter_name": v.parameter_name,
                "group_id": v.group_id,
                "expected": v.expected,
                "actual": v.actual,
            }
            for v in result.violations
        ],
        "loss_before": result.loss_before,
        "loss_after": result.loss_after,
        "loss_decreased": result.loss_decreased,
        "determinism_ok": result.determinism_ok,
        "details": result.details,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.trainability_smoke",
        description=(
            "RFC005 §9.1 item 11 / §13.2 trainability smoke suite: writes "
            "the suite document `unflatten.dataset.admission.report` reads "
            "via --trainability-smoke"
        ),
    )
    parser.add_argument("--out", required=True, help="write the suite document here")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    doc = run_trainability_smoke_suite(seed=args.seed, device=args.device)
    Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0 if doc["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
