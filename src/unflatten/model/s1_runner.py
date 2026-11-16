"""S1 contract canary runner and its fifteen hard gates.

RFC005 §14.4 (ratified DEC-005/R5-1..R5-6, gates pinned verbatim by DEC-007
D8), `docs/plans/dense-v0-m2-m3-plan.md` task 3. **This module does not run
the canary** -- `evaluate_all_gates` is the entry point a caller invokes
with a built context (model, batch, vocabulary, controller, manifests,
lineage), and it returns a :class:`StageGateReport`. Whether that context
is wired to synthetic fixtures (the default, fast, offline path every gate
below is tested against) or to a real release/checkpoint is the CALLER's
decision -- this module never reaches into `release-v10` or loads a
checkpoint itself.

**D8's rule, mechanically enforced, not just quoted.** "Chýbajúci alebo
nedostatočný dôkaz znamená `blocked`, nie pass" (DEC-007 D8; RFC005 §14.4's
closing line): every gate function below returns `"blocked"` -- never
silently `"pass"` -- when the piece of context it needs is `None` or
otherwise absent. There is no gate here that defaults an absent input to a
passing state.

**Gate numbering matches RFC005 §14.4 exactly** (1-indexed, the order the
RFC lists them in); do not renumber even if a future gate is added, because
`docs/decisions/DEC-007-dense-v0-ratification.md` and this file's own
tests cite these numbers as identifiers, not as a sequence to keep tidy.

**What each gate reuses versus builds new**, because most of the mechanism
already exists elsewhere in this package and this module's job is to wire
it into one evidenced report, not to reimplement it:

- Gates 1-3 (no NaN/Inf, frozen groups zero grad, trainable groups finite
  non-zero grad): `trainability_smoke.run_candidate_model_smoke`.
- Gates 4-5 (masks exactly zero loss contribution, missing target never a
  default): `targets.IGNORE_INDEX` semantics, proven here directly against
  `torch.nn.functional.cross_entropy`.
- Gate 6 (checkpoint round-trip): reruns
  `QwenDenseCandidateModel.state_dict()` round-trip on the caller's model.
- Gate 7 (grammar cannot emit an invalid tree): `composite_grammar.replay`.
- Gate 8 (forbidden metadata leak): a real mutation -- swaps in a
  metadata-derived embedding and proves the unmutated model does NOT do
  this, then (in the test suite) proves the check catches it when it does.
- Gate 9 (loader lineage): `lineage_disjoint_from_eval_test`, a pure
  function over release split manifests -- no model, no pixels.
- Gate 10 (tokenizer round-trip): `exact_text.encode_text`/`decode_tokens`.
- Gate 11 (namespace collision): `composite_vocab.packed_ranges` +
  exhaustive `pack`/`unpack` bijectivity over the declared ranges.
- Gate 12 (batch padding equivalence): new, `candidate_model.encode_hidden`
  run twice at two padding lengths for one logical sample.
- Gate 13 (free-running canary never truncates silently):
  `free_running_decode.decode_free_running`'s own typed-failure contract.
- Gate 14 (Component Lock): `manifests.validate_exact_text_s1_component_lock`
  + `run_binding.RunManifestBindingVerifier`.
- Gate 15 (optimizer membership complete/disjoint):
  `trainable_group_registry` cross-checked against
  `QwenDenseCandidateModel.named_parameters()`.

This module authorizes no gradient update by itself: gates that need one
(1-3, 6, 12, 15) take an already-built model/optimizer from the caller and
never construct a training loop of their own.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from unflatten.model import (
    boundary_projector,
    candidate_model,
    manifests,
    s1_cohort,
    training_materializer,
)
from unflatten.model import composite_grammar as cg
from unflatten.model import composite_vocab as cv
from unflatten.model import exact_text as et
from unflatten.model import exact_text_cache as etc
from unflatten.model import qwen_text_allowlist as qtal
from unflatten.model import qwen_tokenizer_artifacts as qta
from unflatten.model import run_binding as rb
from unflatten.model import s1_text_corpus as stc
from unflatten.model.candidate_model import QwenDenseCandidateModel, TeacherForcedBatch
from unflatten.model.capability_profile import (
    PRIMARY_S1_CAPABILITY_PROFILE,
    CapabilityProfile,
    CapabilityProfileHashUnresolved,
    resolve_capability_profile_by_hash,
)
from unflatten.model.capability_profile import (
    # `as`-aliased: no longer used inside this module (every internal
    # caller now reads `.profile_id` off a `CapabilityProfile` object,
    # 2026-09-10 review item 2), but re-exported explicitly because
    # several tests import it as `s1_runner.PRIMARY_S1_PROFILE_ID`.
    PRIMARY_S1_PROFILE_ID as PRIMARY_S1_PROFILE_ID,
)
from unflatten.model.composite_head import TREE_TEXT_DECODER_GROUP_ID
from unflatten.model.document_context_encoder import DOCUMENT_CONTEXT_ENCODER_GROUP_ID
from unflatten.model.encode_fragment import encode_fragment_with_origins
from unflatten.model.exact_text_cache import ExactTextCacheIncompatible
from unflatten.model.expert_adapter import EXPERT_ADAPTER_DENSE_GROUP_ID
from unflatten.model.free_running_decode import (
    DecodeConfig,
    DecodeMaxStepsExceeded,
    DecodeVocabularyExhausted,
    decode_free_running,
)
from unflatten.model.targets import (
    IGNORE_INDEX,
    supervised_event_indices,
    unsupervised_target_steps,
)
from unflatten.model.trainability_smoke import run_parameter_movement_check
from unflatten.model.trainable_group_registry import trainable_group_registry
from unflatten.model.visual_memory import VISUAL_MEMORY_GROUP_ID

S1_RUNNER_SCHEMA_VERSION = "0.1.0"

GateStatus = Literal["pass", "fail", "blocked"]

#: `QwenDenseCandidateModel` child-module attribute name -> the
#: `trainable_group_registry` group id it owns. One place, reused by gate
#: 15's completeness check and by any caller building its own audit.
CANDIDATE_MODULE_GROUP_ID: dict[str, str] = {
    "visual_memory": VISUAL_MEMORY_GROUP_ID,
    "document_context_encoder": DOCUMENT_CONTEXT_ENCODER_GROUP_ID,
    "dense_adapter": EXPERT_ADAPTER_DENSE_GROUP_ID,
    "composite_head": TREE_TEXT_DECODER_GROUP_ID,
}


@dataclass(frozen=True)
class GateResult:
    """One gate's outcome. `status="blocked"` means insufficient evidence,
    never a silent pass (DEC-007 D8)."""

    number: int
    name: str
    status: GateStatus
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageGateReport:
    """RFC005 §14.4's "uložený Stage Gate Report". `promoted` is `True`
    iff all fifteen gates report `"pass"` -- any `"fail"` or `"blocked"`
    gate means `promoted=False`, per D8's closing rule."""

    schema_version: str
    run_class: str  # "discovery" | "promotion"
    gates: tuple[GateResult, ...]
    notes: tuple[str, ...] = ()

    @property
    def promoted(self) -> bool:
        # `len(...) == 15` alone would still be satisfied by e.g. two
        # "gate 3"s and a missing gate 7 -- require the FIFTEEN DISTINCT
        # numbers 1..15, not just a count (2026-09-05 review should_fix).
        numbers = {g.number for g in self.gates}
        return numbers == set(range(1, 16)) and all(g.status == "pass" for g in self.gates)

    def blocked_gates(self) -> tuple[GateResult, ...]:
        return tuple(g for g in self.gates if g.status == "blocked")

    def failed_gates(self) -> tuple[GateResult, ...]:
        return tuple(g for g in self.gates if g.status == "fail")


def _blocked(number: int, name: str, reason: str) -> GateResult:
    return GateResult(number=number, name=name, status="blocked", detail=reason)


# ---------------------------------------------------------------------------
# Gates 1-3: forward/backward health and gradient disposition, audited
# against the CALLER'S OWN model and batch -- not a freshly-built internal
# stub. (2026-09-05 review: `_smoke` used to discard `model`/`batch` and
# call `trainability_smoke.run_candidate_model_smoke`, which builds its
# OWN 12-token fp32 stub candidate every time. NaN/Inf under the real
# bf16 candidate, a real frozen tensor leaking, or a real group that
# never trains were none of them reachable through that path -- this
# module's own report described results as "evidenced from real page
# context" while never touching the real model at all.)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelGradientProbe:
    """N real forward/backward/optimizer-step passes against the CALLER's
    actual `model`, `batch` and `optimizer` -- the shared evidence gates
    1, 2 and 3 all read from, computed exactly ONCE per
    `evaluate_all_gates` call (a real backward pass through a real
    candidate is not something to pay for three times over).

    `moved_tensor_counts`/`total_trainable_tensors` are the reviewer's
    per-tensor, per-step design (2026-09-05), replacing an earlier
    `any(parameter moved)` boolean that four independent falsely-passing
    shapes could satisfy: an aggregate `any()` (one tensor moving hides 33
    frozen ones), a single step (gradients are largest at init), the wrong
    lr/model (a disposable stub optimizer proves nothing about the run's
    own), and loss-as-proxy (drifts from dropout/batch/bf16 forward noise
    with weights untouched -- **loss decreasing is affirmatively NOT
    evidence of movement, and nothing in this module treats it as such**).
    """

    finite_logits: bool
    finite_gradients: bool
    frozen_violations: tuple[str, ...]  # frozen-group params that got a NONZERO gradient
    trainable_violations: tuple[str, ...]  # trainable-group params with no finite non-zero grad
    frozen_candidate_leaked: tuple[str, ...]  # candidate (backbone) params that got a gradient
    #: Per-step count of trainable parameter TENSORS whose value changed
    #: after that step's `optimizer.step()` -- the trend across `steps`
    #: real steps at the run's own lr, not one boolean from one step.
    moved_tensor_counts: tuple[int, ...]
    #: Every trainable tensor NAME that never moved across ALL `steps`
    #: steps -- if this is non-empty, "K of N tensor identities moved" is
    #: less than the full trainable set, which is itself the finding
    #: (reviewer's own words), not a detail to average away.
    never_moved: tuple[str, ...]
    total_trainable_tensors: int
    steps: int
    details: dict[str, Any] = field(default_factory=dict)


def run_model_gradient_probe(
    model: QwenDenseCandidateModel | None,
    batch: TeacherForcedBatch | None,
    candidate_named_parameters: dict[str, torch.nn.Parameter] | None,
    optimizer: torch.optim.Optimizer | None,
    *,
    steps: int = 5,
    seed: int = 0,
) -> ModelGradientProbe | None:
    """Real forward -> loss -> backward -> `optimizer.step()`, repeated
    `steps` times, against `model`/`batch`/`optimizer` -- auditing
    gradient/parameter-movement behaviour the same way
    `trainability_smoke.check_gradient_flow` does, but against the
    model/optimizer the caller actually built (real dtype, real weights,
    real lr, real cached page), not a synthetic stand-in.
    `candidate_named_parameters` is the FROZEN backbone's own parameters
    (`QwenCandidate.named_parameters`) -- `model` itself never holds them
    as child-module parameters (design: the frozen candidate is a plain
    attribute, not a submodule), so this is the only way to prove they
    leak no gradient (gate 2). `optimizer` MUST be the run's own optimizer
    object (real lr) -- a disposable one built here at an arbitrary lr
    would not be evidence about the run at all (reviewer's item 3).

    Random targets (matching `trainability_smoke.run_candidate_model_smoke`'s
    own convention): what is under test is whether gradients/parameter
    updates FLOW correctly through the real wiring, not whether the
    predicted distribution is meaningful -- that is a quality question,
    explicitly out of S1 scope (RFC005 §14.1).
    """
    if (
        model is None
        or batch is None
        or candidate_named_parameters is None
        or optimizer is None
    ):
        return None
    torch.manual_seed(seed)
    model.eval()  # dropout off; does not block backward

    registry = trainable_group_registry()
    expected_trainable = frozenset(
        g.group_id for g in registry.values() if g.kind == "trainable_primary_s1"
    )

    named_parameters: dict[str, torch.nn.Parameter] = {}
    group_id_of: dict[str, str] = {}
    for attr_name, group_id in CANDIDATE_MODULE_GROUP_ID.items():
        for pname, p in getattr(model, attr_name).named_parameters():
            key = f"{attr_name}.{pname}"
            named_parameters[key] = p
            group_id_of[key] = group_id
    # One representative forward/backward for the gradient-FLOW evidence
    # gates 1/2 need (finite/frozen/leaked) -- separate from the N-step
    # MOVEMENT evidence gate 3 needs, which is delegated below to
    # `trainability_smoke.run_parameter_movement_check` (2026-09-05 review:
    # built by `m2-qwen-candidate` to the reviewer's exact spec -- per-
    # tensor, multi-step, the caller's own optimizer -- wiring to it here
    # rather than forking a second implementation).
    def _forward_loss() -> tuple[torch.Tensor, torch.Tensor]:
        for p in model.parameters():
            p.grad = None
        for p in candidate_named_parameters.values():
            p.grad = None
        logits = model(batch)
        targets = torch.randint(0, logits.shape[-1], logits.shape[:-1], device=logits.device)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        return logits, loss

    logits, loss = _forward_loss()
    # KNOWN, not fixed tonight (2026-09-04): `torch.isfinite(logits).all()`
    # materializes a full bool tensor over `logits` (~531 MB for a real
    # page); chunking this check would give a bit-identical answer with
    # O(chunk) memory, same class as the LM-head fix -- but it only buys
    # back ~510 MB against a forward pass already at ~10.26 of 11.59 GiB
    # with backward not yet started, which needs far more than that. Fixing
    # it would move the failure one line further without changing the
    # answer, so it stays a real, owned inefficiency rather than a step
    # toward reaching a number.
    finite_logits = bool(torch.isfinite(logits).all())
    loss.backward()

    finite_gradients = True
    frozen_violations: set[str] = set()
    trainable_violations: set[str] = set()
    frozen_candidate_leaked: set[str] = set()
    for name, p in named_parameters.items():
        is_trainable_group = group_id_of[name] in expected_trainable
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            finite_gradients = False
        # "trainable" here means a FINITE NON-ZERO gradient -- `.grad is
        # not None` alone (the pre-review check) passes an all-zero
        # gradient, which DEC-007 D6's own argument for
        # `document_context_encoder` existing explicitly depends on this
        # gate catching (2026-09-05 review item 4).
        has_finite_nonzero_grad = (
            p.grad is not None
            and bool(torch.isfinite(p.grad).all())
            and bool((p.grad != 0).any())
        )
        if is_trainable_group and not has_finite_nonzero_grad:
            trainable_violations.add(name)
        if not is_trainable_group and p.grad is not None and bool((p.grad != 0).any()):
            frozen_violations.add(name)
    for name, p in candidate_named_parameters.items():
        if p.grad is None:
            continue
        if not bool(torch.isfinite(p.grad).all()):
            finite_gradients = False
        if bool((p.grad != 0).any()):
            frozen_candidate_leaked.add(name)

    last_loss = float(loss.detach())
    optimizer.zero_grad()  # discard this diagnostic pass before the real N-step run

    # Item 1 (team lead), rebuilt to the reviewer's per-tensor/multi-step/
    # real-optimizer design and now delegated to
    # `trainability_smoke.run_parameter_movement_check` rather than a
    # second implementation of the same check.
    def _movement_step_fn() -> torch.Tensor:
        _logits, movement_loss = _forward_loss()
        return movement_loss

    movement = run_parameter_movement_check(model, optimizer, _movement_step_fn, steps=steps)
    never_moved = tuple(sorted(t.name for t in movement.tensors if not t.moved))
    moved_tensor_counts = tuple(
        sum(
            1
            for t in movement.tensors
            if t.first_moved_step is not None and t.first_moved_step <= step
        )
        for step in range(1, steps + 1)
    )
    total_trainable_tensors = movement.total_count

    passed = (
        finite_logits
        and finite_gradients
        and not frozen_violations
        and not trainable_violations
        and not frozen_candidate_leaked
        and movement.all_moved
    )
    return ModelGradientProbe(
        finite_logits=finite_logits,
        finite_gradients=finite_gradients,
        frozen_violations=tuple(sorted(frozen_violations)),
        trainable_violations=tuple(sorted(trainable_violations)),
        frozen_candidate_leaked=tuple(sorted(frozen_candidate_leaked)),
        moved_tensor_counts=moved_tensor_counts,
        never_moved=never_moved,
        total_trainable_tensors=total_trainable_tensors,
        steps=steps,
        details={
            "passed": passed,
            "dtype": str(next(iter(named_parameters.values())).dtype),
            "last_loss": last_loss,
            "loss_is_not_movement_evidence": True,
        },
    )


def gate_1_forward_backward_finite(probe: ModelGradientProbe | None) -> GateResult:
    if probe is None:
        return _blocked(
            1, "forward_backward_finite", "no model/batch/candidate_named_parameters supplied"
        )
    ok = probe.finite_logits and probe.finite_gradients
    return GateResult(
        number=1,
        name="forward_backward_finite",
        status="pass" if ok else "fail",
        detail="forward/backward produced only finite values"
        if ok
        else "NaN/Inf found in logits or gradients",
        evidence={"finite_logits": probe.finite_logits, "finite_gradients": probe.finite_gradients},
    )


def gate_2_frozen_groups_zero_grad(probe: ModelGradientProbe | None) -> GateResult:
    if probe is None:
        return _blocked(
            2, "frozen_groups_zero_grad", "no model/batch/candidate_named_parameters supplied"
        )
    ok = not probe.frozen_violations and not probe.frozen_candidate_leaked
    return GateResult(
        number=2,
        name="frozen_groups_zero_grad",
        status="pass" if ok else "fail",
        detail="every frozen group recorded zero gradient"
        if ok
        else f"{len(probe.frozen_violations)} frozen-module violation(s), "
        f"{len(probe.frozen_candidate_leaked)} frozen-candidate leak(s)",
        evidence={
            "frozen_violations": list(probe.frozen_violations),
            "frozen_candidate_leaked": list(probe.frozen_candidate_leaked),
        },
    )


def gate_3_trainable_groups_finite_nonzero_grad(probe: ModelGradientProbe | None) -> GateResult:
    """Per-tensor, per-step evidence (2026-09-05 review): `pass` requires
    EVERY trainable parameter TENSOR to have measurably moved on EVERY one
    of `probe.steps` real optimizer steps -- "K of N tensor identities
    moved" must report `K == N` at the final step, not "at least one
    tensor moved once". `probe.never_moved` names exactly which tensors
    (if any) never moved across all steps; that list, not a boolean, is
    the finding this gate reports when it fails.
    """
    if probe is None:
        return _blocked(
            3,
            "trainable_groups_finite_nonzero_grad",
            "no model/batch/candidate_named_parameters/optimizer supplied",
        )
    final_moved = probe.moved_tensor_counts[-1] if probe.moved_tensor_counts else 0
    ok = (
        not probe.trainable_violations
        and probe.finite_gradients
        and not probe.never_moved
        and probe.total_trainable_tensors > 0
    )
    return GateResult(
        number=3,
        name="trainable_groups_finite_nonzero_grad",
        status="pass" if ok else "fail",
        detail=f"{final_moved} of {probe.total_trainable_tensors} trainable tensor "
        f"identities moved at step {probe.steps} (finite, non-zero gradient "
        "every step)"
        if ok
        else f"{len(probe.trainable_violations)} trainable group violation(s); "
        f"never_moved={list(probe.never_moved)} "
        f"({final_moved}/{probe.total_trainable_tensors} moved at final step)",
        evidence={
            "trainable_violations": list(probe.trainable_violations),
            "moved_tensor_counts": list(probe.moved_tensor_counts),
            "never_moved": list(probe.never_moved),
            "total_trainable_tensors": probe.total_trainable_tensors,
            "steps": probe.steps,
        },
    )


# ---------------------------------------------------------------------------
# Gates 4-5: masking exactness and no-default-for-missing-target
# ---------------------------------------------------------------------------


def gate_4_masks_zero_loss_contribution(
    logits: torch.Tensor | None, target_ids: torch.Tensor | None
) -> GateResult:
    """`target_ids` carries `IGNORE_INDEX` at masked positions
    (`targets.py`'s convention). Proof: the loss computed WITH those
    positions equals the loss computed after DELETING them from the batch
    entirely -- if a masked position contributed anything, deleting it
    would change the sum."""
    if logits is None or target_ids is None:
        return _blocked(4, "masks_zero_loss_contribution", "no logits/target_ids supplied")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = target_ids.reshape(-1)
    masked = flat_targets == IGNORE_INDEX
    if not bool(masked.any()):
        return _blocked(
            4,
            "masks_zero_loss_contribution",
            "target_ids carries no IGNORE_INDEX position -- fixture proves nothing",
        )
    loss_with_masked = torch.nn.functional.cross_entropy(
        flat_logits, flat_targets, ignore_index=IGNORE_INDEX, reduction="sum"
    )
    kept = ~masked
    loss_without_masked_rows = torch.nn.functional.cross_entropy(
        flat_logits[kept], flat_targets[kept], reduction="sum"
    )
    ok = torch.allclose(loss_with_masked, loss_without_masked_rows, atol=1e-5)
    return GateResult(
        number=4,
        name="masks_zero_loss_contribution",
        status="pass" if ok else "fail",
        detail="masked positions contributed exactly zero to the summed loss"
        if ok
        else "masked positions changed the loss when removed -- mask is leaking",
        evidence={
            "loss_with_masked": float(loss_with_masked),
            "loss_without_masked_rows": float(loss_without_masked_rows),
            "masked_count": int(masked.sum()),
        },
    )


def expected_missing_mask_from_withdrawal(
    structural: dict[str, Any], fragment: dict[str, Any]
) -> torch.Tensor:
    """The ground-truth "this position is missing" mask gate 5 needs,
    built independently of any `target_ids` tensor -- exactly what makes
    gate 5 non-tautological (see this module's docstring / the mutation
    proof in `test_s1_runner.py`). Thin wrapper over
    `targets.unsupervised_target_steps`, which derives withdrawal from the
    fragment's own `page_evidence`/`supervision_mask` (RFC003 §29.3), never
    from a target tensor that could itself already be wrong.

    **Scope, corrected from an earlier version of this docstring.** This
    mask is aligned to `targets.token_sequence`'s STRUCTURAL-only token
    stream (`events_vocab.TOKEN_IDS`, 71 ids) -- the same stream
    `build_cache.py`'s `decoder_targets` currently materializes into
    `CacheEntry.target_ids`. It is NOT aligned to the composite
    (structural + Qwen text + byte-fallback) packed sequence
    `candidate_model.TeacherForcedBatch.composite_input_ids` uses -- for
    that, use :func:`composite_expected_missing_mask` below, which wraps
    `training_materializer.TrainingTargetMaterializer` (already ratified
    M1 machinery, not a new alignment rule invented here). Use THIS
    function only where the target really is the structural-only stream.
    """
    return unsupervised_target_steps(structural, fragment)


def _covering_grant(
    span: training_materializer.TextSpanAuthority, start: int, end: int
) -> training_materializer.TextRangeGrant | None:
    """Re-derives which grant owns `[start, end)`, straight from `span`'s
    OWN `grants` -- the same lookup `boundary_projector._covering_grant`
    performs, reused rather than duplicated (it is pure interval geometry,
    not a masking verdict)."""
    return boundary_projector._covering_grant(span.grants, start, end)


def composite_expected_missing_mask(
    materialized: training_materializer.MaterializedSequence,
    *,
    structural: dict[str, Any],
    fragment: dict[str, Any] | None,
    text_spans: Sequence[training_materializer.TextSpanAuthority],
) -> torch.Tensor:
    """The composite-token-space ground truth gate 5 needs, built by
    recomputing withdrawal from `structural`/`fragment`/`text_spans` --
    the SAME source authority `materializer.materialize` was called with --
    never by reading `effective_loss_mask` (or any of its already-conjoined
    `LossUnit.effective_mask` fields) back out of `materialized`.

    **Why an earlier version of this function was tautological, found by
    review (2026-09-05).** It returned `~materialized.effective_loss_mask`
    directly. Any caller who then built `target_ids` via
    `composite_target_ids_with_sentinel` (`packed_token_ids.masked_fill(
    ~effective_loss_mask, IGNORE_INDEX)`) was comparing `effective_loss_mask`
    against itself -- gate 5 reduced to `masked_fill(~M)[~M] == IGNORE_INDEX`,
    true by construction for every mask, every content, every bug. The
    general property this function restores: **`effective_loss_mask` must
    appear on exactly one side of gate 5's comparison.** It appears on the
    `target_ids` side (via `composite_target_ids_with_sentinel`, unchanged);
    this function is the OTHER side, and must never read it.

    **What "recompute" means, per position.** `materialized.loss_units` is
    used only for ALIGNMENT (`LossUnit.loss_family`, and for a text
    position, `LossUnit.locator`'s `text_span_id`/`scalar_start`/
    `scalar_end`) -- never for its `effective_mask` verdict fields:

    - **Structural** (`loss_family == "L_structure"`): mirrors
      `expected_missing_mask_from_withdrawal` exactly.
      `fragment is None` (document-level materialization) means no
      structural position is ever withdrawn (`training_materializer.
      _structural_atoms`' own rule: `effective = origin is None or
      supervised is None or origin in supervised`); with a real
      `fragment`, `encode_fragment_with_origins(fragment, structural)` is
      called HERE, independently, to recover each structural atom's
      origin event index, checked against a freshly-called
      `targets.supervised_event_indices(fragment)` -- the exact fragment
      argument, re-queried, not a value the materializer already computed.
    - **Text** (`loss_family == "L_text"`): the position's `TextRangeLocator`
      names which span/scalar-range it came from; the covering
      `TextRangeGrant` is looked up FRESH in `text_spans` (`_covering_grant`,
      reusing `boundary_projector`'s own pure interval lookup, not
      duplicating it) and `training_materializer.compute_effective` is
      called on that grant's raw `known_mask`/`observable_mask`/
      `supervision_mask`/`text_range_owner` fields plus a freshly-derived
      `stage_enabled` (`boundary_projector._covering_stage` over `text_spans`'
      OWN `stage_ranges`) -- the identical formula
      `ScalarBoundaryProjector._project_span` uses, called a second time on
      the same raw inputs, never reading `ScalarPolicyRange.effective` or
      `LossUnit.effective_mask.effective`.

      **This TEXT `effective=False` path is UNREACHABLE by construction
      of the envelope builder, on ANY cohort (T-13, the
      `spec-viacstranove-dokumenty` design RFC's §1 C1; corrected
      2026-09-09).** An earlier version of this docstring said the
      opposite of what follows and must not be trusted: it claimed the
      branch was unreachable specifically "on real single-page data" and
      that "multi-page fragments... are the first place this branch could
      become reachable by real data." That second claim does **not**
      survive a read of `compute_effective`'s five conjuncts
      (`boundary_projector.py:134-161`): wherever an `L_text` atom exists
      at all, `ownership_enabled = bool(grant.text_range_owner)` is
      `True` (an atom only exists where `char_ranges` was projected),
      `observable_mask`/`supervision_mask` are both `True` (any node with
      `char_ranges` has its own observation, so `_page_evidence` marks it
      `"observed"`), `known_mask` is hardcoded `True` for text
      (`text_authority.py:143-145`), and `stage_enabled` is `True`
      because no caller supplies `stage_ranges`
      (`boundary_projector.py:85-89`, `default_stage_enabled`). None of
      those four facts is a property of the single-page cohort, and
      multi-page changes none of them -- so the branch stays unreachable
      for every window policy, not just `single_page`. Measured directly
      on a real multi-page cohort (release-v11/m2a-v6, train,
      `page_count > 1`) by
      `tests/model/test_masking_pipeline_real_data.py`'s
      `test_l_text_effective_false_branch_incidence_on_real_multi_page_pages`:
      0 `effective=False` `L_text` positions over 21,443 checked across 27
      real multi-page documents -- a measured zero, reported with its
      cohort, not an "unreachable" assertion standing in for evidence.
      `test_no_real_page_naturally_withdraws_a_text_range_it_still_owns`
      (same file) checks the construction-level invariant this relies on
      -- no slice may have non-empty `char_ranges` and a non-observable
      `page_evidence` -- against the same real multi-page cohort. The
      litmus-test sibling still injects the withdrawal condition because
      that remains the only way to exercise this branch at all, on any
      cohort this codebase can produce.

    Padding: `loss_units` is `materialize`'s UNPADDED sequence length;
    `effective_loss_mask`/`packed_token_ids` may be longer
    (`materialize(..., pad_to=...)`). Padded positions never carried real
    content and are always "missing" under `composite_target_ids_with_sentinel`'s
    own convention (`effective_loss_mask` pads `False`), so this function
    pads its own output with `True` to match -- not read from
    `effective_loss_mask` either, just the same declared padding contract
    every `MaterializedSequence` caller already relies on.
    """
    supervised: set[int] | None = None
    if fragment is not None:
        supervised = supervised_event_indices(fragment)
        _tokens, structural_origins = encode_fragment_with_origins(fragment, structural)
    else:
        _tokens, structural_origins = training_materializer.document_level_tokens_with_origins(
            structural
        )
    unowned_text_origins = training_materializer.unowned_text_event_origins(
        structural, text_spans
    )
    span_by_id = {span.text_span_id: span for span in text_spans}

    missing: list[bool] = []
    origin_cursor = 0
    for unit in materialized.loss_units:
        if unit.loss_family == "L_structure":
            origin = structural_origins[origin_cursor]
            origin_cursor += 1
            # D1 mirror (2026-09-07): an unowned text event's wrapper pair
            # is withdrawn the same way `not_supervised` is -- both from
            # the SAME source authority `materialize` was called with,
            # never from `effective_loss_mask` (module docstring).
            not_supervised = (
                fragment is not None and origin is not None and origin not in supervised
            )
            unowned = origin is not None and origin in unowned_text_origins
            missing.append(not_supervised or unowned)
            continue
        loc = unit.locator
        assert isinstance(loc, training_materializer.TextRangeLocator)
        span = span_by_id.get(loc.text_span_id)
        grant = _covering_grant(span, loc.scalar_start, loc.scalar_end) if span else None
        if grant is None:
            missing.append(True)  # no authority at all here -> definitely missing
            continue
        stage_enabled = boundary_projector._covering_stage(
            span.stage_ranges, loc.scalar_start, loc.scalar_end, span.default_stage_enabled
        )
        effective, _reasons = training_materializer.compute_effective(
            known_mask=grant.known_mask,
            observable_mask=grant.observable_mask,
            supervision_mask=grant.supervision_mask,
            ownership_enabled=bool(grant.text_range_owner),
            stage_enabled=stage_enabled,
        )
        missing.append(not effective)

    pad_count = materialized.effective_loss_mask.shape[0] - len(missing)
    missing.extend([True] * pad_count)
    return torch.tensor(missing, dtype=torch.bool)


def composite_target_ids_with_sentinel(
    materialized: training_materializer.MaterializedSequence,
) -> torch.Tensor:
    """Reference composite-space `target_ids`: `packed_token_ids` with
    `IGNORE_INDEX` at every position `effective_loss_mask` says is not
    scored. `MaterializedSequence.packed_token_ids` carries a real packed
    id at EVERY position -- composite space has no structural convention
    for "no token here" -- so a caller building actual training targets
    must apply this masking step themselves (design/RFC005 do not do it
    inside the materializer, `test_training_materializer.py`'s own
    criterion-9 tests consume `effective_loss_mask` directly as a
    multiplicative mask rather than expect it pre-applied to the ids).
    This is that step, named so a caller who forgot it is exactly what
    gate 5's mutation proof (`test_s1_runner.py`) catches.
    """
    return materialized.packed_token_ids.masked_fill(
        ~materialized.effective_loss_mask, IGNORE_INDEX
    )


def gate_5_missing_target_no_default(
    target_ids: torch.Tensor | None, expected_missing_mask: torch.Tensor | None
) -> GateResult:
    """A missing target is `IGNORE_INDEX` (-100), never `0` or any other
    in-vocabulary id standing in for "no label" (RFC005: "missing target
    nevytvorí default"). `0` IS a valid packed id (`StructuralRef(0)`), so
    "is it `IGNORE_INDEX`" cannot be read off `target_ids` alone -- that
    would only ever confirm positions that already happen to carry the
    sentinel, never catch one that was silently defaulted to something
    else. `expected_missing_mask` is the INDEPENDENT ground truth of which
    positions are missing (e.g. `targets.unsupervised_target_steps`'s
    output, or any withdrawal-derived boolean mask computed from the
    fragment/label pipeline, never re-derived from `target_ids` itself) --
    this gate checks `target_ids[expected_missing_mask] == IGNORE_INDEX`
    for every one of those positions."""
    if target_ids is None or expected_missing_mask is None:
        return _blocked(
            5, "missing_target_no_default", "no target_ids/expected_missing_mask supplied"
        )
    flat_targets = target_ids.reshape(-1)
    flat_mask = expected_missing_mask.reshape(-1).bool()
    if not bool(flat_mask.any()):
        return _blocked(
            5, "missing_target_no_default", "expected_missing_mask names no position"
        )
    missing_values = flat_targets[flat_mask]
    ok = bool((missing_values == IGNORE_INDEX).all())
    return GateResult(
        number=5,
        name="missing_target_no_default",
        status="pass" if ok else "fail",
        detail="every independently-known-missing target is the IGNORE_INDEX sentinel"
        if ok
        else "a known-missing target was encoded as an in-vocabulary value instead "
        "of IGNORE_INDEX",
        evidence={"ignore_index": IGNORE_INDEX, "expected_missing_count": int(flat_mask.sum())},
    )


# ---------------------------------------------------------------------------
# Gate 6: checkpoint round-trip
# ---------------------------------------------------------------------------


def _fresh_model_like(model: QwenDenseCandidateModel, *, seed: int) -> QwenDenseCandidateModel:
    """A NEW `QwenDenseCandidateModel` instance sharing `model`'s frozen
    references (decode/inject_visual closures, the frozen Qwen embedding/
    LM-head buffers) but with freshly, differently initialized trainable
    submodules -- the "different random instance" gate 6's round-trip
    needs so an omitted submodule shows up as ITS OWN untouched random
    weights rather than the original model's, which a same-instance
    reload can never reveal (2026-09-05 review).

    **The dtype move is per-PARAMETER, never `Module.to(dtype=...)`
    (2026-09-04, first real-hardware S1 run).** This function used to do
    `fresh.to(device=..., dtype=reference.dtype)` with `reference =
    next(model.parameters())` -- an fp32 TRAINABLE parameter under DEC-007
    D3.1. But `Module.to(dtype=...)` casts non-persistent BUFFERS too, and
    the two frozen Qwen references (`composite_head.py`'s
    `qwen_embedding_weight`/`qwen_lm_head_weight`, `persistent=False`
    exactly because they are ~1.45 GB of frozen backbone weight that must
    stay out of a checkpoint) are buffers. So the fresh instance computed
    the embedding lookup and the LM-head readout in fp32 while the
    original computed them in bf16, and gate 6 reported `fail`
    (`max_abs_diff=1.566` on the real candidate) with the message "some
    trainable state was not saved/loaded" -- which was never true: the
    strict `load_state_dict` had already proven every key matched. A
    defect in the measuring apparatus, reported as a defect in the model.

    Device placement therefore moves the module (a no-op for the frozen
    buffers, which already live on the reference device) and the dtype
    cast is applied to `fresh.parameters()` one at a time, so the shared
    frozen buffers keep both their dtype AND their identity -- the same
    tensor objects the original holds, not fresh copies."""
    torch.manual_seed(seed)
    fresh = QwenDenseCandidateModel(
        decode=model._decode,
        inject_visual=model._inject_visual,
        visual_dim=model.visual_memory.visual_dim,
        d_model=model.d_model,
        ranges=model.ranges,
        qwen_embedding_weight=model.composite_head.qwen_embedding_weight,
        qwen_lm_head_weight=model.composite_head.qwen_lm_head_weight,
    )
    reference = next(model.parameters())
    fresh = fresh.to(device=reference.device)
    with torch.no_grad():
        for param in fresh.parameters():
            param.data = param.data.to(dtype=reference.dtype)
    _assert_model_on_device(fresh, reference.device, context="_fresh_model_like")
    return fresh


def gate_6_checkpoint_roundtrip(
    model: QwenDenseCandidateModel | None, batch: TeacherForcedBatch | None
) -> GateResult:
    """Real serialization (`torch.save`/`torch.load` through actual
    bytes, not a same-instance state_dict clone) into a FRESH model
    instance with different random trainable weights -- a submodule
    excluded from `state_dict()` would then keep ITS OWN random init
    after "loading", changing the output, which reloading into the same
    live instance (2026-09-05 review finding) can never expose: the
    original instance's weights are already correct regardless of what
    the reload did.
    """
    if model is None or batch is None:
        return _blocked(6, "checkpoint_roundtrip", "no model/batch supplied")
    model.eval()
    with torch.no_grad():
        before = model(batch)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    loaded_state = torch.load(buffer, map_location="cpu", weights_only=True)

    fresh = _fresh_model_like(model, seed=1)
    try:
        fresh.load_state_dict(loaded_state)
    except RuntimeError as exc:
        # A key mismatch (missing/unexpected) between what was saved and
        # what a fresh instance expects IS round-trip evidence, not a
        # crash to let propagate -- this is what makes the mutation below
        # (a checkpoint missing a real key) a `fail`, not an uncaught
        # exception the caller has to handle specially.
        return GateResult(
            number=6,
            name="checkpoint_roundtrip",
            status="fail",
            detail=f"state_dict failed to load into a fresh instance: {exc}",
        )
    fresh.eval()
    with torch.no_grad():
        after = fresh(batch)
    ok = torch.equal(before, after)
    return GateResult(
        number=6,
        name="checkpoint_roundtrip",
        status="pass" if ok else "fail",
        detail="a freshly-instantiated model, loaded from real serialized bytes, "
        "reproduced identical output"
        if ok
        else "a freshly-instantiated model's output diverged after loading real "
        "serialized state_dict bytes -- some trainable state was not saved/loaded",
        evidence={"max_abs_diff": float((before - after).abs().max())},
    )


# ---------------------------------------------------------------------------
# Gate 7: grammar-constrained decode cannot emit an invalid V0 tree
# ---------------------------------------------------------------------------


def gate_7_grammar_constrained_valid_tree(
    controller: cg.CompositeGrammarController | None,
    capability_profile_id: str | None,
    tokens: tuple[cv.TokenRef, ...] | None,
) -> GateResult:
    """Replays `tokens` (e.g. a `RawDocumentHypothesis.accepted_tokens`
    stream, unpacked) through a FRESH controller state. `replay` raises on
    any illegal step or on a non-terminated stream -- there is no partial-
    success return, matching gate 13's own "never truncate-and-accept"."""
    if controller is None or capability_profile_id is None or tokens is None:
        return _blocked(7, "grammar_constrained_valid_tree", "no controller/tokens supplied")
    try:
        state = cg.replay(controller, capability_profile_id, tokens)
    except (cg.CompositeGrammarViolation, cg.IncompleteCompositeStream) as exc:
        return GateResult(
            number=7,
            name="grammar_constrained_valid_tree",
            status="fail",
            detail=f"replay rejected the stream: {exc}",
        )
    ok = state.mode == "terminated"
    return GateResult(
        number=7,
        name="grammar_constrained_valid_tree",
        status="pass" if ok else "fail",
        detail="stream replayed to a valid terminated V0 tree"
        if ok
        else f"stream replayed but ended in mode {state.mode!r}",
    )


# ---------------------------------------------------------------------------
# Gate 8: changed forbidden metadata, identical pixels, must not change
# model tensors
# ---------------------------------------------------------------------------


def gate_8_forbidden_metadata_no_leak(
    model: QwenDenseCandidateModel | None,
    batch_a: TeacherForcedBatch | None,
    batch_b: TeacherForcedBatch | None,
) -> GateResult:
    """`batch_a`/`batch_b` must be built by the CALLER from identical
    pixels/grammar content but DIFFERENT forbidden metadata (e.g. two
    different `document_bundle_id`s, timestamps, or any field outside the
    RFC001 render surface) -- exactly the pair a data-loading layer would
    produce for the same page seen under two different provenance records.
    This is a real leakage test, not a tautology: if a caller's batch
    construction ever let such metadata influence a tensor (e.g. a
    metadata-derived embedding folded into `visual_features`), `batch_a`
    and `batch_b` would differ and this gate would catch it. Passing this
    gate on two batches whose only difference IS forbidden metadata is what
    proves it can fail; the test suite constructs exactly that deliberately
    leaky pair to prove it does.
    """
    if model is None or batch_a is None or batch_b is None:
        return _blocked(8, "forbidden_metadata_no_leak", "no model/batches supplied")
    model.eval()
    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)
    ok = torch.equal(out_a, out_b)
    return GateResult(
        number=8,
        name="forbidden_metadata_no_leak",
        status="pass" if ok else "fail",
        detail="model output is identical across two batches that differ only "
        "in forbidden metadata"
        if ok
        else "model output changed between batches that differ only in forbidden "
        "metadata -- a hidden channel exists",
        evidence={"max_abs_diff": float((out_a - out_b).abs().max())},
    )


# ---------------------------------------------------------------------------
# Gate 9: loader never touches eval/test lineage
# ---------------------------------------------------------------------------


def lineage_disjoint_from_eval_test(
    cohort_lineage_root_ids: frozenset[str],
    eval_test_lineage_root_ids: frozenset[str],
) -> frozenset[str]:
    """The set of cohort lineage roots that also appear in an eval/test
    split -- empty means disjoint. Pure set arithmetic, no I/O, so the
    caller decides how `eval_test_lineage_root_ids` was built (e.g. from
    `release-v10/splits/{val_iid,val_template_ood,test}.jsonl`'s
    `lineage_root_id` field)."""
    return cohort_lineage_root_ids & eval_test_lineage_root_ids


def gate_9_loader_lineage_disjoint(
    cohort_lineage_root_ids: frozenset[str] | None,
    eval_test_lineage_root_ids: frozenset[str] | None,
    *,
    requested_cohort_bundle_count: int | None = None,
    resolved_cohort_bundle_count: int | None = None,
) -> GateResult:
    """**Fails closed on missing eval/test evidence, not open.** An empty
    `eval_test_lineage_root_ids` is ambiguous on its own -- it could mean
    "checked real split manifests, they genuinely name zero eval/test
    lineage roots" (true on real data: `release-v10`'s own
    `val_template_ood` split has 0 documents) or "the release directory
    was misresolved/missing and nothing was ever checked". This function
    cannot tell those apart from the set alone, so it refuses to read an
    empty result as a clean bill of health: an empty
    `eval_test_lineage_root_ids` is `blocked`, the same as a missing one
    (DEC-007 D8, "chýbajúci alebo nedostatočný dôkaz znamená blocked").
    A caller with a genuinely empty eval/test split must still be able to
    name WHICH split manifests it read and found empty -- that positive
    confirmation belongs in how the caller builds this set
    (`s1_cohort.eval_test_lineage_root_ids` raises rather than silently
    returning empty when no split file exists at all), not in this gate
    accepting silence as evidence.

    **The same reasoning, applied to the COHORT side** (2026-09-04): the
    two counts are DOCUMENTS (`s1_cohort.CohortLineage`), not lineage
    roots, and any gap between them is cohort documents that resolved to
    no manifest row at all. The empty cohort was already refused above;
    the partial case was not, so a cohort where one of a hundred and fifty
    documents resolved produced a non-empty root set and a `pass` that
    every reader took as a statement about the cohort. Unresolved lineage
    cannot be proven disjoint from anything, so a non-zero gap is
    `blocked`, and counts that were never supplied are `blocked` too --
    a caller that cannot say how much of its cohort resolved has not shown
    that any of it did. Order matters: the overlap is computed FIRST, so a
    demonstrated leak still reports `fail`; missing evidence never
    launders a proven violation into "insufficient evidence".
    """
    if cohort_lineage_root_ids is None or eval_test_lineage_root_ids is None:
        return _blocked(9, "loader_lineage_disjoint", "no cohort/eval-test lineage supplied")
    if not cohort_lineage_root_ids:
        return _blocked(9, "loader_lineage_disjoint", "cohort is empty -- nothing to prove")
    if not eval_test_lineage_root_ids:
        return _blocked(
            9,
            "loader_lineage_disjoint",
            "eval_test_lineage_root_ids is empty -- cannot distinguish 'genuinely "
            "zero eval/test lineage roots' from 'release directory misresolved and "
            "nothing was checked'; an empty result is never read as a clean pass",
        )
    overlap = lineage_disjoint_from_eval_test(
        cohort_lineage_root_ids, eval_test_lineage_root_ids
    )
    # `cohort_lineage_root_count` replaces the old `cohort_count`, which
    # counted ROOTS while reading as a count of the cohort's documents.
    evidence: dict[str, Any] = {
        "overlap_count": len(overlap),
        "cohort_lineage_root_count": len(cohort_lineage_root_ids),
        "requested_document_count": requested_cohort_bundle_count,
        "resolved_document_count": resolved_cohort_bundle_count,
    }
    if overlap:
        return GateResult(
            number=9,
            name="loader_lineage_disjoint",
            status="fail",
            detail=f"{len(overlap)} cohort lineage root(s) also appear in eval/test",
            evidence=evidence
            | {
                "unresolved_document_count": (
                    None
                    if requested_cohort_bundle_count is None
                    or resolved_cohort_bundle_count is None
                    else requested_cohort_bundle_count - resolved_cohort_bundle_count
                )
            },
        )
    if requested_cohort_bundle_count is None or resolved_cohort_bundle_count is None:
        return _blocked(
            9,
            "loader_lineage_disjoint",
            "no cohort resolution evidence supplied (requested/resolved document "
            "counts) -- lineage roots alone cannot show whether every cohort "
            "document resolved to a manifest row, and a cohort that partly "
            "resolved proves disjointness only for the part that did",
        )
    unresolved = requested_cohort_bundle_count - resolved_cohort_bundle_count
    evidence["unresolved_document_count"] = unresolved
    if unresolved > 0:
        return GateResult(
            number=9,
            name="loader_lineage_disjoint",
            status="blocked",
            detail=f"{unresolved} of {requested_cohort_bundle_count} cohort document(s) "
            "matched no row in any split manifest -- unresolved lineage cannot be "
            "proven disjoint from eval/test, and the roots that did resolve speak "
            "only for themselves",
            evidence=evidence,
        )
    return GateResult(
        number=9,
        name="loader_lineage_disjoint",
        status="pass",
        detail="train cohort lineage is disjoint from every eval/test lineage root",
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Gate 10: text tokenizer + fallback exact NFC round-trip
# ---------------------------------------------------------------------------


def gate_10_text_roundtrip(
    corpus: tuple[str, ...] | None,
    *,
    tokenizer: et.QwenTokenizerLike | None,
    vocabulary: cv.CompositeVocabulary | None,
    budget_ops: int = 10_000,
    corpus_source: str | None = None,
) -> GateResult:
    """`corpus_source` is `corpus`' provenance when it is present, and the
    reason it is absent when it is not -- recorded either way, because
    "gate 10 blocked" and "gate 10 passed" are both unreadable without it.
    A reader cannot tell cohort text from a hand-written fixture from
    `corpus_size` alone, and cannot tell "the caller forgot" from "the
    corpus lives in an object store this payload does not carry"
    (`s1_text_corpus`'s module docstring) from a bare `blocked`.
    """
    if corpus is None or tokenizer is None or vocabulary is None:
        missing = [
            name
            for name, value in (
                ("corpus", corpus),
                ("tokenizer", tokenizer),
                ("vocabulary", vocabulary),
            )
            if value is None
        ]
        reason = f"no {'/'.join(missing)} supplied"
        return _blocked(
            10, "text_roundtrip", f"{reason}: {corpus_source}" if corpus_source else reason
        )
    if not corpus:
        # `()` is not `None`, so it used to reach the loop below, find no
        # failure and report `pass` -- "0 fixture(s) round-tripped
        # exactly". `s1_text_corpus.build_cohort_text_corpus` refuses to
        # produce an empty corpus; this is the same refusal at the gate,
        # so no other caller can hand one in either.
        return _blocked(
            10,
            "text_roundtrip",
            "the corpus is empty -- a round-trip over zero strings is a pass "
            f"over nothing, not evidence ({corpus_source or 'no source recorded'})",
        )
    failures: list[str] = []
    for text in corpus:
        try:
            nfc = et.require_nfc(text)
            tokens = et.encode_text(
                nfc, tokenizer=tokenizer, vocabulary=vocabulary, budget_ops=budget_ops
            )
            decoded = et.decode_tokens(tokens, tokenizer=tokenizer, vocabulary=vocabulary)
        except ValueError as exc:
            # `exact_text.py`'s five round-trip failure types (TextNotNfc,
            # TextInvalidUnicode, ByteFrameInvalid, TextTokenizerRoundtripFailed,
            # TextEncodingBudgetExceeded) are ALL `ValueError` subclasses and
            # ARE evidence about this fixture's content -- a corpus claim.
            # 2026-09-05 review should_fix: the previous bare `except Exception`
            # also swallowed a broken `tokenizer`/`vocabulary` (a TypeError,
            # AttributeError, or any other setup bug) as if it were the SAME
            # kind of evidence, reporting "this text failed to round-trip"
            # when the real story was "the harness itself is broken". Only
            # `ValueError` is caught here; anything else propagates and
            # crashes the caller, which is the honest outcome for a setup
            # error -- silently reclassifying it as a corpus failure would
            # be its own small lie.
            failures.append(f"{text!r}: {exc}")
            continue
        if decoded != nfc:
            failures.append(f"{text!r} != {decoded!r}")
    ok = not failures
    return GateResult(
        number=10,
        name="text_roundtrip",
        status="pass" if ok else "fail",
        detail=f"{len(corpus)} fixture(s) round-tripped exactly"
        if ok
        else f"{len(failures)}/{len(corpus)} fixture(s) failed round-trip",
        evidence={
            "failures": failures[:20],
            "corpus_size": len(corpus),
            "corpus_source": corpus_source,
        },
    )


# ---------------------------------------------------------------------------
# Gate 11: structural/text token namespace has no collision
# ---------------------------------------------------------------------------


def gate_11_namespace_no_collision(qwen_vocab_size: int | None) -> GateResult:
    """Exhaustive bijectivity check over the declared packed ranges: every
    packed id in `[0, total_count)` unpacks to exactly one `TokenRef`, and
    packing that ref back gives the same id. A collision (two refs sharing
    one packed id) would break bijectivity, which this enumerates directly
    rather than trusting the additive arithmetic that built the ranges."""
    if qwen_vocab_size is None:
        return _blocked(11, "namespace_no_collision", "no qwen_vocab_size supplied")
    ranges = cv.packed_ranges(qwen_vocab_size)
    seen: dict[int, cv.TokenRef] = {}
    collisions: list[str] = []
    for local_id in range(ranges.structural_count):
        ref = cv.StructuralRef(local_id)
        collisions.extend(_record(seen, cv.pack(ref, ranges), ref, ranges))
    for native_id in range(ranges.qwen_count):
        ref = cv.QwenRef(native_id)
        collisions.extend(_record(seen, cv.pack(ref, ranges), ref, ranges))
    for kind in ("byte_begin", "byte_end"):
        ref = cv.ByteControlRef(kind)
        collisions.extend(_record(seen, cv.pack(ref, ranges), ref, ranges))
    for value in range(ranges.byte_value_count):
        ref = cv.ByteValueRef(value)
        collisions.extend(_record(seen, cv.pack(ref, ranges), ref, ranges))
    ok = not collisions and len(seen) == ranges.total_count
    return GateResult(
        number=11,
        name="namespace_no_collision",
        status="pass" if ok else "fail",
        detail=f"all {ranges.total_count} packed ids are bijective with their TokenRef"
        if ok
        else f"{len(collisions)} collision(s) found",
        evidence={"collisions": collisions[:20], "total_count": ranges.total_count},
    )


def _record(
    seen: dict[int, cv.TokenRef], packed_id: int, ref: cv.TokenRef, ranges: cv.PackedRanges
) -> list[str]:
    problems: list[str] = []
    if packed_id in seen and seen[packed_id] != ref:
        problems.append(f"packed id {packed_id} shared by {seen[packed_id]!r} and {ref!r}")
    seen[packed_id] = ref
    if cv.unpack(packed_id, ranges) != ref:
        problems.append(f"unpack(pack({ref!r})) != {ref!r}")
    return problems


# ---------------------------------------------------------------------------
# Gate 12: same sample, different batch padding, equivalent output
# ---------------------------------------------------------------------------


def s1_runner_visual_prefix_padding_pair(
    *,
    ranges: cv.PackedRanges,
    image_token_id: int,
    real_visual_features: torch.Tensor,
    junk_visual_features: torch.Tensor,
    image_grid_thw_correct: torch.Tensor,
    image_grid_thw_leaked: torch.Tensor,
) -> tuple[TeacherForcedBatch, TeacherForcedBatch, int]:
    """The canonical gate-12 mutation pair: same underlying visual content
    (`real_visual_features`, `(1, real_tokens, visual_dim)`), with junk
    rows (`junk_visual_features`, `(1, junk_tokens, visual_dim)`) appended
    -- once CORRECTLY excluded via `visual_valid_lengths`, once WRONGLY
    declared valid. Mirrors `test_candidate_model.py`'s
    `test_gate_12_visual_prefix_leak_is_detected` exactly, as a reusable
    primitive rather than a second, drifting copy of the fixture shape.

    `image_grid_thw_correct`/`image_grid_thw_leaked` are the caller's own
    (patch-grid, pre-merge) `image_grid_thw` for each token count -- this
    function does not derive one, because a wrong derivation would corrupt
    the real candidate's native mRoPE positional index silently, which is
    exactly the kind of defect this gate exists to catch, not introduce.

    Returns `(batch_correct, batch_leaked, real_tokens)` -- `real_tokens`
    is gate 12's `unpadded_steps`: both batches' first `real_tokens`
    composite-range output positions are what must be compared, since the
    leaked batch's composite range is longer (one placeholder per visual
    row it wrongly declares valid).
    """
    device = real_visual_features.device
    real_tokens = real_visual_features.shape[1]
    junk_tokens = junk_visual_features.shape[1]
    junk_visual = torch.cat([real_visual_features, junk_visual_features], dim=1)
    placeholder_id = ranges.qwen_start + image_token_id

    correct_ids = torch.full((1, real_tokens), placeholder_id, dtype=torch.long, device=device)
    batch_correct = TeacherForcedBatch(
        visual_features=junk_visual,
        composite_input_ids=correct_ids,
        composite_attention_mask=torch.ones(1, real_tokens, dtype=torch.long, device=device),
        image_grid_thw=image_grid_thw_correct.to(device),
        # correctly excludes the junk
        visual_valid_lengths=torch.tensor([real_tokens], device=device),
    )

    total_tokens = real_tokens + junk_tokens
    leaked_ids = torch.full((1, total_tokens), placeholder_id, dtype=torch.long, device=device)
    batch_leaked = TeacherForcedBatch(
        visual_features=junk_visual,
        composite_input_ids=leaked_ids,
        composite_attention_mask=torch.ones(1, total_tokens, dtype=torch.long, device=device),
        image_grid_thw=image_grid_thw_leaked.to(device),
        visual_valid_lengths=torch.tensor([total_tokens], device=device),  # WRONG: junk marked real
    )
    return batch_correct, batch_leaked, real_tokens


#: The only `attn_implementation` gate 12 accepts as evidence. Not the
#: production pin (`candidate_loader`'s DEC-007 D3 default is `"sdpa"`) --
#: RFC005 §14.4's own "reference režim" clause, and the only backend
#: measured to be exact (0.0 diff) rather than merely close.
GATE_12_REQUIRED_ATTN_IMPLEMENTATION = "eager"

#: The magnitude of the defect gate 12 exists to screen for: M2's
#: unmasked-pad-row shift, MEASURED at 0.176 on real logits
#: (`candidate_loader.load_qwen_candidate`'s docstring, the same
#: measurement that pinned this gate to eager attention). Not a tolerance
#: and not tunable -- it is the size of a known real defect, and gate 12's
#: own resolution is checked against it below: an instrument whose noise
#: floor exceeds the defect it screens for is blind, and this project's
#: rule is to fix the instrument or report `blocked`, never to report a
#: `pass` the comparison could not have withheld.
GATE_12_SCREENED_DEFECT_MAGNITUDE = 0.176


def _representation_floor(reference: torch.Tensor, dtype: torch.dtype) -> float:
    """One ULP of `dtype` at `reference`'s LARGEST magnitude -- the
    smallest non-zero difference two otherwise-identical computations in
    `dtype` can report anywhere in this tensor.

    A comparison tolerance below this floor is not strict, it is
    unsatisfiable: no implementation, correct or not, can land inside it,
    because the numbers cannot express a difference that small.

    **Per-TENSOR, not per-element, and that is the arithmetic, not a
    convenience.** `reference` is the output of one bf16 matmul
    (`composite_head.forward`'s Qwen readout); a dot product's rounding
    error scales with the magnitude of its accumulated terms, not with
    the magnitude of the result -- a logit that lands near zero by
    cancellation is off by an ULP of the readout's scale, not by an ULP
    of its own tiny value. An elementwise floor would therefore report
    hundreds of "ULPs" for a difference that is one rounding step of the
    computation that produced it (measured on the real candidate: an
    elementwise floor read 694 ULPs for the same `max_abs_diff = 0.0625`
    that is 0.5 ULP at the tensor's own scale of 23.9).
    """
    magnitude = reference.detach().abs().max().to(dtype)
    upward = torch.tensor(float("inf"), dtype=dtype, device=magnitude.device)
    return float((torch.nextafter(magnitude, upward) - magnitude).to(torch.float32))


def gate_12_padding_equivalence(
    model: QwenDenseCandidateModel | None,
    batch_unpadded: TeacherForcedBatch | None,
    batch_padded: TeacherForcedBatch | None,
    *,
    unpadded_steps: int | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-5,
    attn_implementation: str | None = None,
    actual_attn_implementation: str | None = None,
) -> GateResult:
    """Compares `model(batch_unpadded)` against `model(batch_padded)` over
    the first `unpadded_steps` COMPOSITE-range output positions -- declared
    tolerance is `(atol, rtol)` (gate 12's own "v deklarovanej tolerancii"
    clause, `torch.allclose`'s own defaults), raised to the compared
    tensor's REPRESENTATION FLOOR where that is larger.

    **The floor, and why it is not a loosened tolerance (measured
    2026-09-04, first real-hardware S1 run).** That run reported `fail`
    with `max_abs_diff = 0.0625` at `atol=1e-4` -- and the negative
    control fired correctly, so the instrument was working. Measured on
    the real eager Qwen3-VL-4B candidate, this fixture, an A100-class
    GPU's local twin (RTX 4070 SUPER), gate 12's own synthetic pair:

    - `get_rope_index`'s position ids at the compared positions were
      BIT-IDENTICAL between the two members of the pair (all three mRoPE
      axes, all 12 positions) -- the suspected index shift from
      `_with_trailing_composite_padding` does not exist.
    - the dense-adapted hidden state (fp32) differed by `1.9e-6` at
      magnitude `80.5`, i.e. ~2.4e-8 relative: fp32 reduction-order noise
      from the two batches' different widths, nothing structural.
    - the logits differed at `0.64 %` of elements (peak magnitude `23.9`;
      the team lead's independent run on the same fixture measured
      `0.47 %` and peak `26.75` -- the fixture's visual rows are random,
      so the scale varies a little per draw), worst case `-8.0` vs
      `-8.0625` -- exactly one bf16 ULP at magnitude 8, and half a bf16
      ULP at the tensor's own peak.
    - the negative control (the deliberately leaked pair) differed by
      `8.28` on the same fixture -- 66 ULPs at that scale, not a fraction
      of one.

    So the compared logits are bf16-quantized: `composite_head.forward`'s
    size exception runs the Qwen readout matmul in the frozen weight's
    dtype and upcasts the result, so the smallest non-zero difference
    those values can EXPRESS is one ULP of that dtype. `atol=1e-4` sits
    three orders of magnitude below that floor -- a criterion no correct
    implementation could ever satisfy. The fix is therefore not a larger
    `atol` (this project's most expensive recorded mistake) but a floor
    that is COMPUTED, from `model.composite_head.qwen_lm_head_weight.
    dtype` and the compared tensor's own scale (`_representation_floor`,
    which documents why the scale is per-tensor) -- read off the tensor
    the readout actually matmuls
    in, read off the model rather than declared by the caller, so it
    cannot be widened except by making the model itself coarser. Load
    fp32 weights and the floor tightens to fp32's ULP automatically.

    It costs the gate nothing it had: 1 ULP is the tightest criterion that
    is satisfiable at all, the run above lands at 0.5 ULP while its own
    negative control stays 66 ULPs outside, and the M2 defect this gate
    screens for (`GATE_12_SCREENED_DEFECT_MAGNITUDE`, 0.176) is 1.4 ULPs
    at that scale -- still a `fail`.

    **AND THE FLOOR IS CHECKED AGAINST THAT DEFECT, because 1.4x is not
    much margin and it shrinks as training grows the logits.** One bf16
    ULP is 0.125 throughout [16, 32) but 0.25 throughout [32, 64), so at
    `|logit| >= 32` the floor EXCEEDS the 0.176 defect and this
    comparison stops being able to tell a leak from rounding. Two
    independent real-candidate runs measured the peak at 23.9 and 26.75
    (the fixture's visual rows are random, so the scale varies a little
    per draw) -- already ~84 % of the way to that boundary. This function
    therefore compares its own floor against
    `GATE_12_SCREENED_DEFECT_MAGNITUDE` and returns `blocked` when the
    floor is no longer smaller, rather than a `pass` it could not have
    withheld; `evidence` carries the floor, the defect magnitude, the
    margin and the run's own `max_abs_logit` on EVERY evaluated path, so a
    reader watches the margin shrink instead of discovering it when it
    flips. The fix for a blocked gate 12 is never a smaller floor (the
    floor is arithmetic) -- it is a comparison with more resolution than
    a bf16 readout has.

    **The two batches must differ in a way the property can actually be
    violated by -- this is the caller's responsibility, not a check this
    function can make for itself.** A first version of this gate compared
    trailing COMPOSITE-tail padding (`batch_padded` = `batch_unpadded` plus
    extra padded steps appended after it, `composite_attention_mask` zero
    past `unpadded_steps`). That comparison window sits causally AFTER the
    padding it varies, and a causal decoder excludes future keys from a
    query's attention unconditionally -- so it reads `0.0` under EVERY
    implementation, correct or leaking, and cannot catch the defect class
    gate 12 exists for (2026-09-05 finding, cross-referenced in
    `test_candidate_model.py`'s gate-12 section: it documents that specific
    mutation as an intentional expected-0.0 result, not a red flag, and
    supplies the mutation that DOES reach a real position).

    **The mutation that CAN be caught is `VisualMemory`'s own padding
    contract** (`TeacherForcedBatch.visual_valid_lengths`): an incorrectly
    declared valid-row count changes which visual rows
    `DocumentContextEncoder` pools into the document-context PREFIX, and
    that prefix sits causally BEFORE every composite position this gate
    compares -- so a leak there propagates to every real output position,
    unlike composite-tail padding. Callers should build `batch_unpadded`/
    `batch_padded` (or, more accurately for this contract, a "correctly
    declared" / "incorrectly declared" pair) around `visual_valid_lengths`,
    not around trailing composite padding alone;
    `s1_runner_visual_prefix_padding_pair` below builds exactly that pair.

    **`attn_implementation` is a required, verified declaration, not a
    caller convention -- belt and braces, both required.** Root-caused
    2026-09-04/05: a real bf16 candidate run under the production SDPA
    attention kernel reported spurious `fail` even at `atol=rtol=0.05` on a
    CORRECTLY masked pair -- not a leak, not plain bf16 rounding, but
    SDPA's bf16 kernel picking a different, non-bitwise-reproducible
    execution path depending on total sequence length, with a measured
    ~0.59 max-diff noise floor. A tolerance loose enough to absorb that
    noise (2026-09-05's own `atol=1.0` real-data run) is blind to the
    ORIGINAL measured defect this gate exists for -- M2's unmasked-pad-row
    shift was `0.176`, well inside that noise floor. So a tolerance loose
    enough for SDPA is a tolerance that cannot see the defect, and a
    tolerance tight enough to see the defect is not survivable under SDPA
    -- there is no `(atol, rtol)` that works under the production backend.
    `attn_implementation="eager"` (`candidate_loader.load_qwen_candidate`)
    is RFC005 §14.4's own named "reference režim": exact, no fused kernel,
    `0.0` diff at `torch.allclose`'s tight fp32 defaults for correct wiring.

    Two separate arguments, on purpose: `attn_implementation` is what the
    CALLER declares it meant to use (proves the caller knew what it was
    asking for); `actual_attn_implementation` is read from the loaded
    candidate's own `QwenCandidate.attn_implementation` field (proves what
    the model actually is). Either missing is `blocked`. A MISMATCH between
    them is `blocked` too, not `fail` -- a caller declaring `"eager"` while
    actually holding an SDPA (or stub) model is a WIRING error, and
    reporting that as `fail` would misrepresent it as evidence about
    padding when it is evidence about the caller's own setup. Only once
    both agree AND that agreed value is `"eager"` does this function run
    the real comparison; anything else -- `None`, `"sdpa"`, `"stub"`, a
    typo, or a declared/actual disagreement -- is `blocked`, never silently
    evaluated: DEC-007 D8's rule that insufficient evidence is `blocked`,
    not a pass, applies exactly as hard to "ran under the wrong backend and
    reported pass anyway".
    """
    if (
        model is None
        or batch_unpadded is None
        or batch_padded is None
        or unpadded_steps is None
        or attn_implementation is None
        or actual_attn_implementation is None
    ):
        return _blocked(
            12,
            "padding_equivalence",
            "no model/batches/unpadded_steps/attn_implementation/"
            "actual_attn_implementation supplied",
        )
    if attn_implementation != actual_attn_implementation:
        return _blocked(
            12,
            "padding_equivalence",
            f"declared attn_implementation={attn_implementation!r} does not match "
            f"the model's actual {actual_attn_implementation!r} -- a wiring error, "
            "not evidence about padding",
        )
    if attn_implementation != GATE_12_REQUIRED_ATTN_IMPLEMENTATION:
        return _blocked(
            12,
            "padding_equivalence",
            f"attn_implementation={attn_implementation!r} is not the required "
            f"{GATE_12_REQUIRED_ATTN_IMPLEMENTATION!r} reference mode -- refusing "
            "rather than reporting a tolerance-laundered pass/fail under a kernel "
            "whose noise floor (~0.59) exceeds the defect this gate screens for "
            "(0.176, measured)",
        )
    model.eval()
    with torch.no_grad():
        out_unpadded = model(batch_unpadded)
        out_padded = model(batch_padded)
    real_unpadded = out_unpadded[:, :unpadded_steps, :]
    real_padded = out_padded[:, :unpadded_steps, :]
    compute_dtype = model.composite_head.qwen_lm_head_weight.dtype
    floor = _representation_floor(real_unpadded, compute_dtype)
    floor_evidence = {
        "representation_floor_dtype": str(compute_dtype),
        "representation_floor": floor,
        "screened_defect_magnitude": GATE_12_SCREENED_DEFECT_MAGNITUDE,
        "floor_margin": GATE_12_SCREENED_DEFECT_MAGNITUDE / floor,
        "max_abs_logit": float(real_unpadded.abs().max()),
    }
    if floor >= GATE_12_SCREENED_DEFECT_MAGNITUDE:
        # The floor is arithmetic, not a knob: shrinking it here would be
        # the tolerance mistake in reverse. What changes with logit scale
        # is what this comparison can RESOLVE, and once one ULP is bigger
        # than the defect, a `pass` would only mean "a leak and rounding
        # look the same at this scale".
        return dataclasses.replace(
            _blocked(
                12,
                "padding_equivalence",
                f"one {compute_dtype} ULP at this run's logit scale "
                f"({floor_evidence['max_abs_logit']:.3g}) is {floor}, which is no "
                f"smaller than the {GATE_12_SCREENED_DEFECT_MAGNITUDE} defect this "
                "gate screens for -- the comparison can no longer tell that defect "
                "from rounding, so it reports no evidence rather than a pass it "
                "could not have withheld",
            ),
            evidence=floor_evidence,
        )
    diff = (real_unpadded - real_padded).abs()
    # `torch.allclose`'s own elementwise criterion, raised to the
    # representation floor where (and only where) it falls below it.
    allowed = torch.clamp(atol + rtol * real_padded.abs(), min=floor)
    ok = bool((diff <= allowed).all())
    max_ulps = float(diff.max()) / floor
    return GateResult(
        number=12,
        name="padding_equivalence",
        status="pass" if ok else "fail",
        detail=f"real-position output matched within atol={atol}, rtol={rtol} "
        f"or one {compute_dtype} ULP, whichever is larger"
        if ok
        else "padding changed the output at real (non-padded) positions "
        f"beyond atol={atol}, rtol={rtol} and beyond one {compute_dtype} ULP "
        f"({max_ulps:.3g} ULPs at the worst element)",
        evidence={
            "max_abs_diff": float(diff.max()),
            "atol": atol,
            "rtol": rtol,
            "max_diff_in_ulps": max_ulps,
        }
        | floor_evidence,
    )


# ---------------------------------------------------------------------------
# Gate 13: free-running canary ends validly or raises a typed failure
# ---------------------------------------------------------------------------


def gate_13_free_running_no_silent_truncation(
    decode_fn: Any,
    decode_kwargs: dict[str, Any] | None,
) -> GateResult:
    """`decode_fn` is `decode_free_running` (or a caller's wrapper with the
    same contract) -- passed in rather than imported and called directly
    here so the mutation proof (a caller that swallows
    `DecodeMaxStepsExceeded`/`DecodeVocabularyExhausted` and returns a
    partial hypothesis instead) has something to substitute.

    **Checks `hypothesis.terminated`, not `accepted_tokens` non-emptiness
    (fixed 2026-09-05, review item 3).** An earlier version of this
    function treated ANY non-empty `hypothesis.accepted_tokens` as proof
    the stream "reached a valid mode terminator" -- but nothing on
    `RawDocumentHypothesis` said so at the time; that claim only held
    because `decode_free_running`'s OWN control flow happened to guarantee
    it, a guarantee this function could not see or verify. A caller whose
    truncation path returned a non-empty but genuinely incomplete
    hypothesis (unlike this gate's own emptied-hypothesis mutation test)
    silently read as `pass`. `RawDocumentHypothesis.terminated` (defaults
    `False`, fail-closed; `decode_free_running` sets it `True` only on its
    real success path) closes that: this function now reads the field
    directly instead of inferring completion from list length.
    """
    if decode_fn is None or decode_kwargs is None:
        return _blocked(
            13, "free_running_no_silent_truncation", "no decode_fn/decode_kwargs supplied"
        )
    # Design §3.2: "gate 13 passed" is two different claims, and the
    # callee's identity is what rules out a wrapper that swallowed the
    # typed failure (this function reads `terminated` and trusts it, so it
    # cannot detect that itself -- only a reader can, from this field).
    wiring = {
        "decode_fn": f"{decode_fn.__module__}.{decode_fn.__qualname__}",
        "max_steps": getattr(decode_kwargs.get("config"), "max_steps", None),
    }
    try:
        hypothesis = decode_fn(**decode_kwargs)
    except (DecodeMaxStepsExceeded, DecodeVocabularyExhausted) as exc:
        return GateResult(
            number=13,
            name="free_running_no_silent_truncation",
            status="pass",
            detail=f"decode raised the typed failure {type(exc).__name__} "
            "instead of returning a truncated hypothesis",
            evidence={"typed_failure": type(exc).__name__, "branch": "typed_failure"} | wiring,
        )
    if hypothesis is None or not getattr(hypothesis, "terminated", False):
        return GateResult(
            number=13,
            name="free_running_no_silent_truncation",
            status="fail",
            detail="decode returned neither a terminated hypothesis nor a typed failure",
            evidence={"branch": "silent_truncation"} | wiring,
        )
    return GateResult(
        number=13,
        name="free_running_no_silent_truncation",
        status="pass",
        detail="decode reached a valid mode terminator",
        evidence={
            "accepted_token_count": len(hypothesis.accepted_tokens),
            "branch": "terminated",
        }
        | wiring,
    )


# ---------------------------------------------------------------------------
# Gate 14: Component Lock binds exact base weights/processor/tokenizer/
# vocabulary/grammar/decoder additions/decode config
# ---------------------------------------------------------------------------


def gate_14_component_lock(
    component_lock: dict[str, Any] | None,
    architecture_manifest: dict[str, Any] | None,
    run_manifest: dict[str, Any] | None,
    cache_identity: dict[str, Any] | None,
    allowlist: dict[str, Any] | None,
    expectation: etc.CacheExpectation | None = None,
) -> GateResult:
    """Gate 14 -- Component Lock binds exact base weights, processor,
    tokenizer, vocabulary, grammar, decoder additions and decode config
    (RFC005 §14.4).

    **History, because "UNVERIFIED" is not "probably fine".** This gate
    was reported UNVERIFIED by review on 2026-09-04 -- not sound, not
    broken, simply not looked at; the reviewer named the exact question
    ("can a component be added without changing component_lock_hash?")
    they had not answered. Audited the same day and found unsound in all
    three directions asked: a component's *content* could change (an
    `artifact_sha256` mutated) or a component could be *added* without
    `component_lock_hash` moving, because nothing hashed the actual
    `component_lock` document and compared it to the claim -- shape was
    validated, content never was. Separately, `base_model_weights` and
    `image_preprocessor` -- both named explicitly by §14.4 -- were absent
    from the required-kind set, so a lock could omit the model's own
    weights entirely and still validate as complete. A Stage Gate Report
    built on the unaudited gate would have carried a Component Lock that
    did not bind the model that ran, and every downstream identity
    (`model_release_id`, checkpoint identity, cross-run comparison)
    descends from this lock -- so the false claim would have propagated
    silently rather than failing loudly. All three are fixed below and
    mutation-proven (`tests/model/test_s1_runner.py::TestGate14ComponentLock`).
    """
    if (
        component_lock is None
        or architecture_manifest is None
        or run_manifest is None
        or cache_identity is None
    ):
        return _blocked(
            14, "component_lock",
            "no component_lock/architecture_manifest/run_manifest/cache_identity supplied",
        )
    lock_issues = manifests.validate_exact_text_s1_component_lock(component_lock)
    manifest_issues = manifests.validate_exact_text_s1_architecture_manifest(architecture_manifest)
    if lock_issues or manifest_issues:
        return GateResult(
            number=14,
            name="component_lock",
            status="fail",
            detail=f"{len(lock_issues)} lock issue(s), {len(manifest_issues)} manifest issue(s)",
            evidence={
                "lock_issues": [i.message for i in lock_issues],
                "manifest_issues": [i.message for i in manifest_issues],
            },
        )
    # **The check this gate was missing before 2026-09-05 (review).**
    # Everything above validates `component_lock`'s own SHAPE; nothing yet
    # ties its CONTENT to the `component_lock_hash` the run_manifest/
    # cache_identity pair claims. Without this, a component's pinned hash
    # -- or the whole component list -- could change in the real lock and
    # a stale claim elsewhere would never be caught (measured: mutating a
    # real component's `artifact_sha256`, or adding a brand-new component,
    # both still reported `pass` here before this check existed). Compare
    # against `cache_identity`, not `run_manifest`: `RunManifestBindingVerifier`
    # already requires the two to agree (`_DIRECT_BINDING_FIELDS`), so
    # checking one transitively checks both.
    real_lock_hash = manifests.component_lock_hash(component_lock)
    claimed_lock_hash = cache_identity.get("component_lock_hash")
    if real_lock_hash != claimed_lock_hash:
        return GateResult(
            number=14,
            name="component_lock",
            status="fail",
            detail="component_lock_hash does not match the actual component_lock "
            "content -- the lock changed (or was never hashed) without the run "
            "manifest/cache identity binding noticing",
            evidence={"real_lock_hash": real_lock_hash, "claimed_lock_hash": claimed_lock_hash},
        )
    # Resolved from the cache identity's own `capability_profile_hash`
    # (DEC-024 wiring), never an import-time global: this gate must check
    # the profile the cache actually names, not whichever profile happened
    # to be bound when this module was written.
    try:
        capability_profile = resolve_capability_profile_by_hash(
            cache_identity.get("capability_profile_hash")
        )
    except CapabilityProfileHashUnresolved as exc:
        return GateResult(
            number=14,
            name="component_lock",
            status="fail",
            detail=f"cache identity capability profile refused: {exc}",
        )
    try:
        rb.RunManifestBindingVerifier().verify(
            run_manifest,
            cache_identity,
            allowlist=allowlist,
            capability_profile=capability_profile,
            # DEC-MP-4: presented, exactly like `allowlist`. A
            # gradient-bearing manifest that pins no cohort -- or a run that
            # cannot present the pinned one -- fails this gate, which is the
            # fail-closed outcome, not a gap to route around.
            expectation=expectation,
        )
    except ExactTextCacheIncompatible as exc:
        return GateResult(
            number=14,
            name="component_lock",
            status="fail",
            detail=f"run manifest binding refused: {exc}",
        )
    return GateResult(
        number=14,
        name="component_lock",
        status="pass",
        detail="Component Lock, Architecture Manifest and run-manifest binding all verified",
    )


# ---------------------------------------------------------------------------
# Gate 15: optimizer parameter membership is complete, disjoint, and
# matches the run manifest
# ---------------------------------------------------------------------------


def gate_15_optimizer_membership_complete(
    model: QwenDenseCandidateModel | None,
    optimizer: torch.optim.Optimizer | None,
) -> GateResult:
    """Every `nn.Parameter` in `optimizer.param_groups` must (a) belong to
    exactly one of `model`'s four registered trainable child modules, (b)
    cover ALL of that module's parameters (no orphan left out), (c) map to
    a `trainable_group_registry` group whose `kind == "trainable_primary_s1"`,
    and (d) the four modules plus the frozen-in-S1 children -- the
    registered typed property head children
    (`candidate_model.typed_head_children()`, frozen until S3) and the node
    visual aligner child (`node_visual_aligner`, DEC-027, frozen until S2)
    -- must PARTITION `model.named_parameters()`: every parameter the model
    itself reports must belong to one of them. The typed head children and
    the aligner child must exist and be entirely `requires_grad=False`, and
    none of their parameters may sit in the optimizer: the typed heads are
    frozen until S3 (RFC004 §24/§26 wiring) and the aligner until S2, so a
    trainable or optimized one is an S1 violation, not a partition detail.

    (d) is the ground-truth check the earlier version of this gate lacked
    (2026-09-05 review): it iterated `CANDIDATE_MODULE_GROUP_ID`'s four
    known module names directly, so a fifth silent trainable module added
    to `QwenDenseCandidateModel` later -- present in `model.named_parameters()`
    but never in any of the four -- would be invisible to this gate even
    though it holds real trainable weights the optimizer might (or might
    not) cover. Starting from `model.named_parameters()` instead makes
    that a real, catchable violation rather than a case this gate never
    looked at. The frozen children (typed heads + aligner) are named
    explicitly here so they are recognised as part of the partition instead
    of being reported as an unclaimed module.

    (d) was not enough on its own: `missing`/`extra` stayed empty for any
    model under any defect while `build_context_from_loaded` built the
    optimizer with the same `CANDIDATE_MODULE_GROUP_ID` loop over the same
    objects this function compares it against (2026-09-04). The optimizer
    now comes from `build_s1_optimizer`, which enumerates
    `model.named_parameters()` and filters on `requires_grad` -- a
    different door, so the two sets can genuinely disagree. This function
    is unchanged; what changed is that its inputs are no longer generated
    by its own rule.
    """
    if model is None or optimizer is None:
        return _blocked(15, "optimizer_membership_complete", "no model/optimizer supplied")
    registry = trainable_group_registry()
    optimizer_params: set[int] = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer_params.add(id(p))

    # Ground truth: everything the model itself reports as a parameter.
    model_params: dict[int, str] = dict(
        (id(p), name) for name, p in model.named_parameters()
    )
    typed_children = candidate_model.typed_head_children()
    # The frozen-in-S1 partition children: the registered typed property
    # head children (frozen until S3) plus the node visual aligner (frozen
    # until S2, DEC-027). Together with the four trainable modules they
    # must partition model.named_parameters().
    frozen_children = tuple(sorted((*typed_children, "node_visual_aligner")))
    known_prefixes = tuple(f"{name}." for name in CANDIDATE_MODULE_GROUP_ID)
    frozen_prefixes = tuple(f"{name}." for name in frozen_children)
    unpartitioned = sorted(
        name
        for name in model_params.values()
        if not name.startswith(known_prefixes) and not name.startswith(frozen_prefixes)
    )

    expected_params: dict[int, str] = {}
    for module_name, group_id in CANDIDATE_MODULE_GROUP_ID.items():
        module = getattr(model, module_name)
        for p in module.parameters():
            expected_params[id(p)] = group_id

    # The frozen children (typed heads + aligner) are part of the partition
    # too, but frozen: each must exist, every parameter must be
    # requires_grad=False, and none may sit in the optimizer. The typed
    # heads train from S3 and the aligner from S2, not S1.
    typed_violations: list[str] = []
    aligner_violations: list[str] = []
    frozen_param_names: dict[int, str] = {}
    for child_name in frozen_children:
        child = getattr(model, child_name, None)
        target_violations = (
            aligner_violations
            if child_name == "node_visual_aligner"
            else typed_violations
        )
        if not isinstance(child, torch.nn.Module):
            target_violations.append(f"{child_name!r} is not an nn.Module child")
            continue
        for name, p in child.named_parameters():
            # Fully-qualified name: `child.named_parameters()` yields names
            # RELATIVE to the child, but the `typed_in_optimizer` filter below
            # classifies by the `node_visual_aligner.` prefix, which only
            # works on the model-qualified form.
            qualified = f"{child_name}.{name}"
            frozen_param_names[id(p)] = qualified
            if p.requires_grad:
                note = (
                    "the node visual aligner is frozen until S2"
                    if child_name == "node_visual_aligner"
                    else "typed heads are frozen until S3"
                )
                target_violations.append(f"{qualified} is trainable in S1 ({note})")
    frozen_violations = [*typed_violations, *aligner_violations]
    frozen_in_optimizer = sorted(
        frozen_param_names[pid] for pid in optimizer_params & set(frozen_param_names)
    )
    typed_in_optimizer = [
        name for name in frozen_in_optimizer if not name.startswith("node_visual_aligner.")
    ]

    missing = [gid for pid, gid in expected_params.items() if pid not in optimizer_params]
    # "extra" is checked against the MODEL's own full parameter set, not
    # just the four known modules' union -- an optimizer parameter that
    # belongs to the model but sits outside all four modules is caught by
    # `unpartitioned` instead; one that belongs to NEITHER the model NOR
    # any of the four modules (e.g. a stray frozen candidate tensor) is
    # caught here.
    extra = optimizer_params - set(model_params)
    unknown_groups = [
        gid
        for gid in set(CANDIDATE_MODULE_GROUP_ID.values())
        if registry.get(gid) is None or registry[gid].kind != "trainable_primary_s1"
    ]
    ok = (
        not missing
        and not extra
        and not unknown_groups
        and not unpartitioned
        and not frozen_violations
        and not frozen_in_optimizer
    )
    return GateResult(
        number=15,
        name="optimizer_membership_complete",
        status="pass" if ok else "fail",
        detail="optimizer membership is exactly the four trainable_primary_s1 groups, "
        "which partition the model's own parameters together with the frozen "
        "S3 typed-head children and the node visual aligner"
        if ok
        else f"missing={len(missing)} extra={len(extra)} unknown_groups={unknown_groups} "
        f"unpartitioned={unpartitioned} frozen_violations={frozen_violations} "
        f"frozen_in_optimizer={frozen_in_optimizer}",
        evidence={
            "expected_count": len(expected_params),
            "optimizer_count": len(optimizer_params),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "unpartitioned": unpartitioned,
            "typed_head_children": sorted(typed_children),
            "frozen_partition_children": list(frozen_children),
            "typed_violations": typed_violations,
            "aligner_violations": aligner_violations,
            "frozen_violations": frozen_violations,
            "typed_in_optimizer": typed_in_optimizer,
            "frozen_in_optimizer": frozen_in_optimizer,
        },
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class S1RunnerContext:
    """Everything a caller may supply to evaluate all fifteen gates in one
    pass. Every field defaults to `None`/empty -- an absent field routes
    its gate(s) to `"blocked"`, never a silent pass (DEC-007 D8). This
    dataclass carries no defaults that fabricate evidence."""

    run_class: str = "discovery"
    model: QwenDenseCandidateModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    smoke_seed: int = 0
    #: The batch gates 1-3's shared `ModelGradientProbe` runs `model`
    #: against -- ideally the same real batch `checkpoint_batch` uses, so
    #: gates 1-3 audit the actual candidate on actual page content, not a
    #: synthetic stand-in (2026-09-05 review).
    gate123_batch: TeacherForcedBatch | None = None
    #: The FROZEN backbone's own parameters (`QwenCandidate.named_parameters`),
    #: needed only by gates 1-3's probe to prove they leak no gradient --
    #: `model` itself never holds them as child-module parameters.
    candidate_named_parameters: dict[str, torch.nn.Parameter] | None = None
    #: Real optimizer steps the probe runs before requiring EVERY
    #: trainable tensor to have moved -- one step is not enough (gradients
    #: are largest at init; a tensor that moves once can stall later).
    gate123_steps: int = 5

    logits: torch.Tensor | None = None
    target_ids: torch.Tensor | None = None
    expected_missing_mask: torch.Tensor | None = None

    checkpoint_batch: TeacherForcedBatch | None = None

    grammar_controller: cg.CompositeGrammarController | None = None
    capability_profile_id: str | None = None
    accepted_tokens: tuple[cv.TokenRef, ...] | None = None

    metadata_leak_batch_a: TeacherForcedBatch | None = None
    metadata_leak_batch_b: TeacherForcedBatch | None = None

    cohort_lineage_root_ids: frozenset[str] | None = None
    #: DOCUMENT counts behind `cohort_lineage_root_ids` -- how many cache
    #: documents were asked for, and how many resolved to a split-manifest
    #: row. Gate 9 blocks on a gap, and on their absence: a lineage-root
    #: set alone cannot show whether the cohort resolved (2026-09-04).
    cohort_requested_bundle_count: int | None = None
    cohort_resolved_bundle_count: int | None = None
    eval_test_lineage_root_ids: frozenset[str] | None = None

    text_corpus: tuple[str, ...] | None = None
    #: Where `text_corpus` came from -- or, when it is `None`, why it is
    #: absent. Gate 10 records it either way; see its docstring.
    text_corpus_source: str | None = None
    tokenizer: et.QwenTokenizerLike | None = None
    vocabulary: cv.CompositeVocabulary | None = None

    qwen_vocab_size: int | None = None

    #: Gate 12's OWN model, deliberately separate from `model` above. Gate
    #: 12 requires an `attn_implementation="eager"` instance (see
    #: `gate_12_padding_equivalence`'s docstring); `model` may legitimately
    #: be the production SDPA candidate for every other real-model gate
    #: (1-3 use their own internal stub regardless; 6/8/15 compare
    #: identical-length runs, which SDPA's cross-length non-reproducibility
    #: does not affect). Reusing `model` here would either force every
    #: OTHER gate onto a second, unnecessary eager load, or silently hand
    #: gate 12 the SDPA one -- this field makes the caller supply the right
    #: one explicitly instead of the runner guessing.
    padding_model: QwenDenseCandidateModel | None = None
    padding_batch_unpadded: TeacherForcedBatch | None = None
    padding_batch_padded: TeacherForcedBatch | None = None
    padding_unpadded_steps: int | None = None
    #: Required, verified declaration -- see
    #: `gate_12_padding_equivalence`'s docstring. `None` (the default)
    #: blocks the gate; only the literal `"eager"` runs it.
    padding_attn_implementation: str | None = None
    #: The loaded candidate's OWN recorded `attn_implementation`
    #: (`candidate_loader.QwenCandidate.attn_implementation`) -- cross-
    #: checked against `padding_attn_implementation` above; a mismatch
    #: blocks with a message naming both rather than running under an
    #: unverified backend. `None` blocks, same as the declaration.
    padding_actual_attn_implementation: str | None = None
    #: Declared tolerance for gate 12 -- see `gate_12_padding_equivalence`'s
    #: docstring. Defaults match `torch.allclose`'s own (fp32-appropriate)
    #: defaults, which is what `attn_implementation="eager"` measures as
    #: exact for correct wiring -- unlike `atol`/`rtol`, there is no
    #: dtype-appropriate looser pair to reach for under SDPA (see the
    #: docstring: no tolerance both survives SDPA's noise and still sees
    #: the 0.176 defect), so these defaults should rarely need overriding.
    #: They are also not the whole criterion: the gate raises them
    #: elementwise to the compared value's representation floor (one ULP
    #: of the model's own compute dtype), because on a real bf16 candidate
    #: `atol=1e-4` is below what the logits can represent at all -- see
    #: `gate_12_padding_equivalence`'s docstring for the measurement.
    padding_atol: float = 1e-4
    padding_rtol: float = 1e-5

    decode_fn: Any = None
    decode_kwargs: dict[str, Any] | None = None

    component_lock: dict[str, Any] | None = None
    architecture_manifest: dict[str, Any] | None = None
    run_manifest: dict[str, Any] | None = None
    cache_identity: dict[str, Any] | None = None
    allowlist: dict[str, Any] | None = None
    #: The frozen cohort (DEC-MP-4) the run reads its exact-text cache
    #: under. Gate 14 presents it to `RunManifestBindingVerifier`.
    cache_expectation: etc.CacheExpectation | None = None


def evaluate_all_gates(ctx: S1RunnerContext) -> StageGateReport:
    """Evaluate all fifteen RFC005 §14.4 gates against `ctx` and return the
    Stage Gate Report. Never runs a training loop, never touches a
    release directory or a checkpoint file -- every tensor/artifact in
    `ctx` is the caller's.

    Gates 1-3 share ONE `ModelGradientProbe` (computed here, once): a real
    backward pass through the real candidate is not something to pay for
    three separate times. The probe uses `ctx.optimizer` -- the run's OWN
    optimizer object, at its real lr -- not a disposable one, per the
    reviewer's item 3 (a probe-internal optimizer would not be evidence
    about the run).

    **Gate 12 here has NO negative control.** `gate_12_with_negative_control`
    -- which reruns the comparison over a deliberately leaked pair and
    downgrades a `pass` to `blocked` when that control does not `fail` --
    lives in `main`, because the leaked batch is not on `S1RunnerContext`
    and putting it there would widen the interface for every caller. So a
    caller evaluating gate 12 through THIS function gets a `pass` from a
    comparison never shown to detect the defect on that model. That is a
    weaker claim than `main`'s, and a future runner path built on
    `evaluate_all_gates` would lose the control silently while still
    reporting gate 12 as passing."""
    probe = run_model_gradient_probe(
        ctx.model,
        ctx.gate123_batch,
        ctx.candidate_named_parameters,
        ctx.optimizer,
        steps=ctx.gate123_steps,
        seed=ctx.smoke_seed,
    )
    gates = (
        gate_1_forward_backward_finite(probe),
        gate_2_frozen_groups_zero_grad(probe),
        gate_3_trainable_groups_finite_nonzero_grad(probe),
        gate_4_masks_zero_loss_contribution(ctx.logits, ctx.target_ids),
        gate_5_missing_target_no_default(ctx.target_ids, ctx.expected_missing_mask),
        gate_6_checkpoint_roundtrip(ctx.model, ctx.checkpoint_batch),
        gate_7_grammar_constrained_valid_tree(
            ctx.grammar_controller, ctx.capability_profile_id, ctx.accepted_tokens
        ),
        gate_8_forbidden_metadata_no_leak(
            ctx.model, ctx.metadata_leak_batch_a, ctx.metadata_leak_batch_b
        ),
        gate_9_loader_lineage_disjoint(
            ctx.cohort_lineage_root_ids,
            ctx.eval_test_lineage_root_ids,
            requested_cohort_bundle_count=ctx.cohort_requested_bundle_count,
            resolved_cohort_bundle_count=ctx.cohort_resolved_bundle_count,
        ),
        gate_10_text_roundtrip(
            ctx.text_corpus,
            tokenizer=ctx.tokenizer,
            vocabulary=ctx.vocabulary,
            corpus_source=ctx.text_corpus_source,
        ),
        gate_11_namespace_no_collision(ctx.qwen_vocab_size),
        gate_12_padding_equivalence(
            ctx.padding_model,
            ctx.padding_batch_unpadded,
            ctx.padding_batch_padded,
            unpadded_steps=ctx.padding_unpadded_steps,
            atol=ctx.padding_atol,
            rtol=ctx.padding_rtol,
            attn_implementation=ctx.padding_attn_implementation,
            actual_attn_implementation=ctx.padding_actual_attn_implementation,
        ),
        gate_13_free_running_no_silent_truncation(ctx.decode_fn, ctx.decode_kwargs),
        gate_14_component_lock(
            ctx.component_lock,
            ctx.architecture_manifest,
            ctx.run_manifest,
            ctx.cache_identity,
            ctx.allowlist,
            ctx.cache_expectation,
        ),
        gate_15_optimizer_membership_complete(ctx.model, ctx.optimizer),
    )
    return StageGateReport(
        schema_version=S1_RUNNER_SCHEMA_VERSION,
        run_class=ctx.run_class,
        gates=gates,
    )


def stage_gate_report_payload(
    report: StageGateReport,
    *,
    evaluated_checkpoint: dict[str, Any] | None = None,
    gate_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON-ready projection of `report` -- what the CLI writes as the
    "uložený Stage Gate Report" RFC005 §14.4 requires `promoted` to have.
    `promoted` is included explicitly (it is a computed property, not a
    dataclass field, so a plain `dataclasses.asdict` would silently drop
    the one value the report exists to answer).

    `evaluated_checkpoint`/`gate_inputs` are ADDITIVE, keyword-only, and
    default to `None`: an existing caller passing only `report` gets
    exactly today's payload shape (`schema_version` unchanged at
    `"0.1.0"`). When given, they are recorded verbatim under top-level
    `"evaluated_checkpoint"`/`"gate_inputs"` keys -- `freeze_stage_gate_report`
    reads `evaluated_checkpoint` back out of a frozen runner report (plan
    `spec-runner-inputs-c5b`)."""
    payload: dict[str, Any] = {
        "schema_version": report.schema_version,
        "run_class": report.run_class,
        "promoted": report.promoted,
        "gates": [
            {
                "number": g.number,
                "name": g.name,
                "status": g.status,
                "detail": g.detail,
                "evidence": g.evidence,
            }
            for g in report.gates
        ],
        "blocked_gate_numbers": [g.number for g in report.blocked_gates()],
        "failed_gate_numbers": [g.number for g in report.failed_gates()],
        "notes": list(report.notes),
    }
    if evaluated_checkpoint is not None:
        payload["evaluated_checkpoint"] = evaluated_checkpoint
    if gate_inputs is not None:
        payload["gate_inputs"] = gate_inputs
    return payload


# ---------------------------------------------------------------------------
# CLI: `python -m unflatten.model.s1_runner <release_dir> <store_dir> <cache_dir>`
#
# Reads an ALREADY-BUILT candidate feature cache (`build_cache.py
# --single-page-only --exclude-capability-profile`) -- this command never
# builds one itself, and never extracts pixels through a vision tower: the
# cached `CacheEntry.features`/`grid_thw` ARE the model input, which is why
# no `candidate_backbone.CandidateVisionBackbone` is loaded here at all,
# only the two `QwenCandidate` language-side loads gates 1-3/6/8/9/11/12/15
# need. Cohort selection (single-page, capability-filtered -- `inline_image`
# stays excluded, DEC-007 D1.1, not this command's business to "fix") already
# happened when the cache was built; this command reads whatever is in it.
# ---------------------------------------------------------------------------


class _CandidateLike(Protocol):
    """Minimal duck-typed surface this CLI's context-building needs from a
    `candidate_loader.QwenCandidate` (real or `build_stub_qwen_candidate`),
    narrow enough that `build_context_from_loaded` is testable against the
    stub without importing `candidate_loader` at test time."""

    model_id: str
    vocab_size: int
    hidden_size: int
    visual_hidden_size: int
    image_token_id: int
    attn_implementation: str
    embed_tokens: Any
    lm_head_weight: torch.Tensor
    decode: Any
    inject_visual: Any
    #: Every named parameter this candidate holds -- gates 1-3's probe
    #: needs it to prove the FROZEN backbone leaks no gradient.
    named_parameters: dict[str, torch.nn.Parameter]


class _CacheEntryLike(Protocol):
    """Minimal duck-typed surface this CLI needs from one
    `feature_cache.CacheEntry` -- narrow so a synthetic entry (no real
    cache reader) can stand in for tests."""

    document_bundle_id: str
    features: torch.Tensor
    grid_thw: torch.Tensor


def _assert_model_on_device(model: torch.nn.Module, device: torch.device, *, context: str) -> None:
    """Runtime guard at THIS module's own composition boundary.

    `build_context_from_loaded`/`build_gate12_fixture`/`_fresh_model_like`
    assemble a `QwenDenseCandidateModel` out of parts that already live on
    `device` (the frozen candidate's own weights) plus freshly constructed
    trainable submodules, which default to CPU. `model.to(device)` alone
    was invisible on CPU -- every default coincides with every other
    default there -- and survived 89 fast tests plus a CPU smoke run
    before a real `--device cuda` run broke `main()`'s own path outright
    (2026-09-04: `build_context_from_loaded` built the model and its
    input tensors with no `device=` at all, and `QwenDenseCandidateModel`
    has no internal device handling of its own).

    Same class of bug as `candidate_loader.load_qwen_candidate`'s device
    guard, added that same morning, one layer down: a `.to()` call that
    silently only partly moves an assembly needs its own runtime check,
    not just the call, or the next place that composes a model from parts
    repeats it and finds out on a real GPU run instead of a test.
    """
    want = torch.device(device).type
    stale = [name for name, param in model.named_parameters() if param.device.type != want]
    if stale:
        raise RuntimeError(
            f"{context}: model.to({device!r}) did not move every parameter -- "
            f"{len(stale)} still elsewhere (first: {stale[0]!r}); this is the "
            "runtime check for the exact bug that once let a CUDA composition "
            "silently keep pieces on CPU instead of failing (2026-09-04)"
        )


#: Total merged sequence length below which SDPA's bf16 attention kernel
#: IS bitwise reproducible (`candidate_loader.load_qwen_candidate`: drift
#: "appearing only past ~8 total tokens, exactly 0.0 at 4->6"). A gate-12
#: fixture at or under this length would report the same answer under the
#: production SDPA pin as under the eager reference mode -- i.e. it would
#: not exercise the thing the eager pin was bought for.
GATE_12_SDPA_DRIFT_THRESHOLD_TOKENS = 8

#: Gate 12's synthetic fixture sizes. Chosen so BOTH batches decode past
#: `GATE_12_SDPA_DRIFT_THRESHOLD_TOKENS` and at DIFFERENT total lengths;
#: still tiny, because gate 12 tests wiring, not pixels, and memory is
#: scarce (DEC-007 D3's 4070 headroom).
GATE_12_DEFAULT_VISUAL_TOKENS = 12
GATE_12_DEFAULT_JUNK_TOKENS = 4
GATE_12_DEFAULT_COMPOSITE_PAD_STEPS = 4


@dataclass(frozen=True)
class Gate12Fixture:
    """Gate 12's eager model and the three batches it needs.

    `clean` is the unpadded sample; `padded` is the SAME sample under a
    collate that padded it in both directions (masked-out junk visual rows
    AND trailing masked composite steps, so it decodes at a genuinely
    longer total length); `leaked` is the mutation -- the junk visual rows
    wrongly declared valid. `clean`/`padded` are the positive control;
    `clean`/`leaked` is the negative control that proves the comparison
    can still see a leak on THIS model, not only in the stub suite.
    """

    model: QwenDenseCandidateModel
    clean: TeacherForcedBatch
    padded: TeacherForcedBatch
    leaked: TeacherForcedBatch
    unpadded_steps: int


def _with_trailing_composite_padding(
    batch: TeacherForcedBatch, *, pad_steps: int, pad_id: int
) -> TeacherForcedBatch:
    """Append `pad_steps` MASKED composite positions (attention 0). This
    is the batch-padding axis RFC005 §14.4 names ("rovnaký sample s iným
    batch paddingom"): what a collate function adds when a longer
    neighbour sample sets the batch's width. `pad_id` must not be the
    image placeholder -- `inject_visual` counts placeholders exactly.

    On its own this axis cannot FAIL for a causal decoder (the compared
    positions sit causally before the padding), which is why it is not the
    defect channel; `visual_valid_lengths` is. What it buys is the regime:
    the two batches now decode at different total lengths, which is where
    SDPA stops being reproducible and where the eager pin therefore does
    real work.
    """
    if pad_steps <= 0:
        return batch
    device = batch.composite_input_ids.device
    pad_ids = torch.full((1, pad_steps), pad_id, dtype=torch.long, device=device)
    pad_mask = torch.zeros(1, pad_steps, dtype=batch.composite_attention_mask.dtype, device=device)
    return TeacherForcedBatch(
        visual_features=batch.visual_features,
        composite_input_ids=torch.cat([batch.composite_input_ids, pad_ids], dim=1),
        composite_attention_mask=torch.cat([batch.composite_attention_mask, pad_mask], dim=1),
        image_grid_thw=batch.image_grid_thw,
        visual_valid_lengths=batch.visual_valid_lengths,
    )


def build_gate12_fixture(
    eager_candidate: _CandidateLike,
    ranges: cv.PackedRanges,
    *,
    gate12_visual_tokens: int = GATE_12_DEFAULT_VISUAL_TOKENS,
    gate12_junk_tokens: int = GATE_12_DEFAULT_JUNK_TOKENS,
    gate12_composite_pad_steps: int = GATE_12_DEFAULT_COMPOSITE_PAD_STEPS,
    visual_features_dtype: torch.dtype = torch.bfloat16,
) -> Gate12Fixture:
    """Gate 12's OWN eager model and its healthy-run (positive-control)
    batch pair, split out of `build_context_from_loaded` so `main` can
    build it in a SEPARATE phase from the SDPA-candidate gates -- loading
    both candidates at once needs ~16.6 GB (two resident bf16 8.3 GB
    checkpoints), more than the RTX 4070 SUPER's measured ~11.3 GB free
    (DEC-007 D3); sequencing "load SDPA, run its gates, free it, load
    eager, run gate 12" keeps one model resident at a time.

    Uses a SYNTHETIC visual-prefix leak pair (see
    `s1_runner_visual_prefix_padding_pair`'s and gate 12's own docstrings
    -- deriving a mutated `image_grid_thw` for a real cached page's
    arbitrary patch geometry would risk silently mis-deriving positions
    for the real candidate's native mRoPE; gate 12 tests the model's
    wiring, not any particular page's pixels). Returns the leaked pair
    as well, so the caller can run the negative control on the SAME model
    the positive control ran on -- a `pass` from a comparison that has
    never been shown to detect this leak on THIS model is not evidence
    (`gate_12_with_negative_control`).
    """
    device = eager_candidate.embed_tokens.weight.device
    gate12_model = QwenDenseCandidateModel(
        decode=eager_candidate.decode,
        inject_visual=eager_candidate.inject_visual,
        visual_dim=eager_candidate.visual_hidden_size,
        d_model=eager_candidate.hidden_size,
        ranges=ranges,
        qwen_embedding_weight=eager_candidate.embed_tokens.weight,
        qwen_lm_head_weight=eager_candidate.lm_head_weight,
    )
    gate12_model.to(device)
    _assert_model_on_device(gate12_model, device, context="build_gate12_fixture")
    gate12_model.eval()
    real_gate12_visual = torch.randn(
        1,
        gate12_visual_tokens,
        eager_candidate.visual_hidden_size,
        dtype=visual_features_dtype,
        device=device,
    )
    junk_visual = (
        torch.randn(
            1,
            gate12_junk_tokens,
            eager_candidate.visual_hidden_size,
            dtype=visual_features_dtype,
            device=device,
        )
        * 1000
    )
    gate12_correct, gate12_leaked, gate12_steps = s1_runner_visual_prefix_padding_pair(
        ranges=ranges,
        image_token_id=eager_candidate.image_token_id,
        real_visual_features=real_gate12_visual,
        junk_visual_features=junk_visual,
        image_grid_thw_correct=torch.tensor([[1, 4, gate12_visual_tokens]], device=device),
        image_grid_thw_leaked=torch.tensor(
            [[1, 4, gate12_visual_tokens + gate12_junk_tokens]], device=device
        ),
    )
    placeholder_id = ranges.qwen_start + eager_candidate.image_token_id
    gate12_clean = TeacherForcedBatch(
        visual_features=real_gate12_visual,
        composite_input_ids=torch.full(
            (1, gate12_steps), placeholder_id, dtype=torch.long, device=device
        ),
        composite_attention_mask=torch.ones(1, gate12_steps, dtype=torch.long, device=device),
        image_grid_thw=torch.tensor([[1, 4, gate12_visual_tokens]], device=device),
    )
    # The padded member of the pair carries BOTH padding axes a collate
    # function produces: masked visual rows (from the pair above) and
    # trailing masked composite steps (here). `structural_start` is a
    # legal packed id that is not the image placeholder; nothing on this
    # path reads it as content, since the position is masked out.
    gate12_padded = _with_trailing_composite_padding(
        gate12_correct, pad_steps=gate12_composite_pad_steps, pad_id=ranges.structural_start
    )
    return Gate12Fixture(
        model=gate12_model,
        clean=gate12_clean,
        padded=gate12_padded,
        leaked=gate12_leaked,
        unpadded_steps=gate12_steps,
    )


def gate_12_with_negative_control(
    fixture: Gate12Fixture,
    *,
    attn_implementation: str | None,
    actual_attn_implementation: str | None,
) -> tuple[GateResult, GateResult]:
    """Gate 12 over `fixture`'s positive control, plus the negative
    control on the same model, and returns `(result, control)`.

    A `pass` is downgraded to `blocked` when the control did NOT report
    `fail` -- i.e. when this comparison, on this model at these
    tolerances, could not see a deliberately leaked pair. That is DEC-007
    D8's own rule (insufficient evidence is `blocked`, never a pass)
    applied to the instrument rather than to the input; it can only ever
    make gate 12 stricter, and it never turns a `fail` into anything else.
    The gate's own pass criteria are untouched.

    The control's status is written into the returned result's
    `evidence["negative_control_status"]`, so it reaches the Stage Gate
    Report JSON rather than only `main`'s stdout: a reader holding the
    report is otherwise back to trusting a `pass` that they cannot see
    was ever capable of failing.
    """
    result = gate_12_padding_equivalence(
        fixture.model,
        fixture.clean,
        fixture.padded,
        unpadded_steps=fixture.unpadded_steps,
        attn_implementation=attn_implementation,
        actual_attn_implementation=actual_attn_implementation,
    )
    control = gate_12_padding_equivalence(
        fixture.model,
        fixture.clean,
        fixture.leaked,
        unpadded_steps=fixture.unpadded_steps,
        attn_implementation=attn_implementation,
        actual_attn_implementation=actual_attn_implementation,
    )
    if result.status == "pass" and control.status != "fail":
        result = _blocked(
            12,
            "padding_equivalence",
            "the correct pair agreed, but the negative control (the same "
            f"comparison over a deliberately leaked pair) reported {control.status!r} "
            "instead of 'fail' -- an instrument that cannot see the defect it "
            "screens for provides no evidence",
        )
    return (
        dataclasses.replace(
            result,
            evidence=result.evidence | {"negative_control_status": control.status},
        ),
        control,
    )


def build_s1_optimizer(
    model: QwenDenseCandidateModel,
    *,
    lr: float = 0.1,
    algorithm: str = "sgd",
) -> torch.optim.Optimizer:
    """The run's optimizer: every parameter `model` itself reports as
    trainable, in one param group.

    **Built through a different door than gate 15 audits, on purpose**
    (2026-09-04). The previous construction iterated
    `CANDIDATE_MODULE_GROUP_ID` and collected
    `getattr(model, name).parameters()` -- and
    `gate_15_optimizer_membership_complete` builds its EXPECTED set with
    the same constant, the same loop and the same objects, so `missing`
    and `extra` were empty for any model under any defect. The gate could
    not fail the property it is named for. Starting from
    `model.named_parameters()` and filtering on `requires_grad` is how a
    real training path builds an optimizer (the frozen candidate is not a
    submodule, so nothing frozen can leak in this way), and it makes a
    disagreement with the four modules' membership observable: a trainable
    group silently frozen upstream drops out here and gate 15 reports it
    `missing`, which the old construction reported as `pass`.

    **The trade this makes, stated because a green suite would otherwise
    imply both halves are still measured.** Deriving the optimizer from
    `model.parameters()` necessarily makes gate 15's `extra`
    (`optimizer_params - model_params`) empty by construction for THIS
    caller: an optimizer holding a tensor the model does not own -- a
    frozen candidate weight, say -- can no longer happen here, so that
    defect class is now structurally prevented rather than detected.
    `gate_15_optimizer_membership_complete` still checks it, and it is
    still real for any caller that builds its own optimizer. A defect that
    cannot occur beats one that is merely caught, but the swap is only a
    trade while it is written down.

    `lr` is the run's real learning rate: gate 3's movement evidence is
    only evidence about THIS run if it is measured with the optimizer the
    run uses (2026-09-05 review). There is one optimizer, built here.

    `algorithm` defaults to `sgd` so every existing caller -- the gates
    included -- keeps measuring exactly what it measured before. The
    alternative exists because plain SGD at a small `lr` moves a freshly
    initialised head so little that a diagnostic run comparing real
    features against noise compares two nearly untrained models and can
    only report that they are similar (measured 2026-09-05: 200 steps at
    lr 1e-4 left `L_structure` flat at ~17.6, and the noise arm *won* on
    `L_text`, which is what an unlearned input projection predicts). That
    is a property of the optimizer, not of the backbone, and reading it as
    the latter is the confusion this parameter exists to avoid.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(
            "model reports no trainable parameters -- an optimizer over an empty "
            "parameter set would make gates 3 and 15 vacuous"
        )
    if algorithm == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if algorithm == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    raise ValueError(
        f"unknown optimizer algorithm {algorithm!r} -- expected 'sgd' or 'adamw'"
    )


def _build_teacher_forced_batch(
    entry: _CacheEntryLike,
    *,
    ranges: cv.PackedRanges,
    image_token_id: int,
    device: torch.device,
    visual_features_dtype: torch.dtype,
    document_bundle_id: str,
) -> TeacherForcedBatch:
    """One teacher-forced batch for the model-forward gates: the page's
    visual placeholder run, then a short composite continuation.

    **The continuation is not decoration** (2026-09-04). `CompositeDecodeHead.
    embed_packed` routes each position into one of THREE embeddings by
    packed-id range, each behind an `.any()` guard, and only
    `structural.embedding` and `byte_fallback.embedding` are trainable
    tensors this run owns. A batch of visual placeholders alone is entirely
    inside the (frozen) Qwen range, so those two embeddings never affect the
    output -- which made gate 6 report `pass` for a checkpoint that omitted
    either of them, and made gate 3 report `fail` for two tensors the batch
    simply never touched. One position per sub-range is enough to make each
    tensor's identity observable in the output and to give it a gradient;
    more would only cost activation memory on a card already near its limit.
    Ids come from `cv.pack`, so an out-of-RANGE id raises here rather than
    landing silently in the wrong branch. That is all `pack` buys: nothing
    on this path consults the grammar -- `encode_hidden`/`forward` never
    touch `composite_grammar`, and `embed_packed` bounds-checks only -- so
    this sequence is NOT a grammar-legal stream and must not be described
    as one. Stream legality is gate 7's property, and gate 7 takes its own
    `accepted_tokens`. A synthetic id sequence is legitimate evidence for
    gates 1, 2, 3, 6 and 8 precisely because none of them compares an
    output against a label (gate 3's probe generates random targets on
    purpose).

    **What gate 6 does and does not prove with this batch, stated because
    an unstated bound becomes a false claim later.** It proves every
    trainable tensor is SAVED and RESTORED (a tensor missing from the
    checkpoint keeps the fresh instance's own random init and moves the
    output). It proves the restored CONTENTS are correct only for what
    the forward pass reads: everything that participates densely --
    `visual_memory`, `document_context_encoder`, `dense_adapter`, the
    bridge MLP, both sub-head `.output` Linears -- is fully covered, since
    perturbing any single element moves the compared logits, but the two
    input EMBEDDINGS are covered only at the rows this continuation
    indexes (`structural.embedding[0]`, `byte_fallback`'s `byte_begin` and
    byte value 0). A checkpoint that restores those tensors but diverges
    on rows the batch never indexes is outside the bound: a composite-
    vocabulary extension or reindexing that shifts structural/byte local
    ids while leaving row 0 in place (the likeliest, since that vocabulary
    is versioned), a sharded or partially-merged checkpoint writing a
    subset of rows, or per-row quantization on save. Gate 11 covers the
    range ARITHMETIC exhaustively and says nothing about row contents, so
    it does not close this.

    Do NOT close it by indexing more rows: every added position costs
    activation memory on a path that already OOMs an 11.59 GiB card, and
    row-level checkpoint corruption is not the defect class gate 6 exists
    for. The bound is the deliverable.

    Three things stay as the cached page has them, none of them incidental:
    `entry.features` (the REAL page's visual rows -- a synthetic tensor
    would turn the whole run into a fixture and would also move the
    forward pass off the real memory regime), `entry.grid_thw` (it
    describes the FEATURES, not the sequence; resizing it to the longer
    sequence would corrupt Qwen's `get_rope_index` with nothing raising),
    and the placeholder run's ORDER -- `inject_visual`'s scatter consumes
    valid visual rows row-major, so the continuation is appended after
    them, never interleaved. `composite_attention_mask` is 1 at every
    appended position: those positions are real, not padding.

    **`document_bundle_id` is an INPUT that must never reach a tensor.**
    Gate 8 builds two batches from identical page content under two
    different provenance values and requires bit-identical model output;
    that test only has a channel to measure because provenance is passed
    in here. Nothing below may read it other than to validate it -- and
    `TeacherForcedBatch` must never gain a metadata field.
    """
    if not isinstance(document_bundle_id, str) or not document_bundle_id:
        raise ValueError(
            "document_bundle_id must be a non-empty string (gate 8's provenance "
            f"input), got {document_bundle_id!r}"
        )
    visual_tokens = int(entry.features.shape[0])
    # Exactly one placeholder per visual row -- `inject_visual` raises on
    # any other count (candidate_loader.py's exact-count contract).
    placeholder_id = cv.pack(cv.QwenRef(native_id=image_token_id), ranges)
    continuation = [
        cv.pack(cv.StructuralRef(local_id=0), ranges),
        cv.pack(cv.ByteControlRef(kind="byte_begin"), ranges),
        cv.pack(cv.ByteValueRef(value=0), ranges),
    ]
    composite_input_ids = torch.cat(
        [
            torch.full((1, visual_tokens), placeholder_id, dtype=torch.long, device=device),
            torch.tensor([continuation], dtype=torch.long, device=device),
        ],
        dim=1,
    )
    steps = composite_input_ids.shape[1]
    return TeacherForcedBatch(
        visual_features=entry.features.to(visual_features_dtype).to(device).unsqueeze(0),
        composite_input_ids=composite_input_ids,
        composite_attention_mask=torch.ones(1, steps, dtype=torch.long, device=device),
        image_grid_thw=entry.grid_thw.to(device).unsqueeze(0),
    )


#: Gate 13's step budget (design §3.1). Deliberately small: every step
#: rebuilds the WHOLE prefix through the real 4B decoder with no KV cache
#: (`free_running_decode.py`), over a ~875-visual-token page whose single
#: forward already measures ~10.26 GiB. Running out of budget raises
#: `DecodeMaxStepsExceeded`, which IS the property gate 13 checks ("never
#: truncate-and-accept"), so a small budget still exercises the gate. It
#: is NOT a decode-quality knob -- raising it to "get a better decode"
#: buys nothing this gate measures and costs a forward pass per step.
GATE_13_DEFAULT_MAX_STEPS = 8


def resolve_evaluated_capability_profile(
    *,
    checkpoint_path: str | None,
    identity: etc.ExactTextCacheIdentity | None,
) -> CapabilityProfile:
    """The capability profile gate 13's decode and gate 7's replay must
    bind to for this run (2026-09-10 review item 2): the SAME profile
    gate 14 checks the run against, resolved from the checkpoint's own
    cache identity -- never `PRIMARY_S1_CAPABILITY_PROFILE` hardcoded
    independently of what the evaluated cache actually names.

    `checkpoint_path is None` (no `--checkpoint`) means there is no
    checkpoint-bound cache identity to read a `capability_profile_hash`
    from, so `PRIMARY_S1_CAPABILITY_PROFILE` is the correct behaviour
    here, not a fallback papering over a missing resolution. Otherwise
    `identity` must be the checkpoint's dataset identity (`main` reads it
    off `ExactTextTrainingDataset.identity`, built from
    `--exact-text-cache-dir`); raises `CapabilityProfileHashUnresolved`,
    fail-closed, if `identity.capability_profile_hash` is not one of
    `KNOWN_CAPABILITY_PROFILES` -- `main` turns that into a printed
    refusal and a nonzero exit, never a silent default.
    """
    if checkpoint_path is None:
        return PRIMARY_S1_CAPABILITY_PROFILE
    assert identity is not None  # `main`'s contract: identity accompanies a checkpoint path
    return resolve_capability_profile_by_hash(identity.capability_profile_hash)


def build_controller_for_profile(
    vocabulary: cv.CompositeVocabulary,
    capability_profile: CapabilityProfile,
) -> cg.CompositeGrammarController:
    """The grammar controller bound to `capability_profile`.

    Every field is read off the profile object, never restated here --
    DEC-007 D1.1 forbids editing a profile, and a controller built from
    hand-copied ids would decode under a different capability set than the
    one the run declares while still naming its `profile_id`.

    Generalises `build_primary_s1_controller` (kept as the S1-bound
    special case): decode and decode-hypothesis-replay must be able to
    bind to whichever profile the cache/checkpoint under evaluation
    actually names (DEC-024; 2026-09-10 review item 2), not always
    `PRIMARY_S1_CAPABILITY_PROFILE`.
    """
    return cg.CompositeGrammarController(
        vocabulary=vocabulary,
        capability_profile_id=capability_profile.profile_id,
        structural_profile=capability_profile.ast_profile,
        capability_excluded_local_ids=capability_profile.excluded_structural_local_ids,
    )


def build_primary_s1_controller(
    vocabulary: cv.CompositeVocabulary,
) -> cg.CompositeGrammarController:
    """The grammar controller bound to `PRIMARY_S1_CAPABILITY_PROFILE`.

    The S1-bound special case of `build_controller_for_profile`, kept as
    its own name because most callers (fixtures, S1-only tests) always
    mean primary S1 specifically and should not have to import the
    profile constant themselves.
    """
    return build_controller_for_profile(vocabulary, PRIMARY_S1_CAPABILITY_PROFILE)


def build_gate13_decode_kwargs(
    *,
    model: QwenDenseCandidateModel,
    vocabulary: cv.CompositeVocabulary,
    entry: _CacheEntryLike,
    image_token_id: int,
    device: torch.device,
    visual_features_dtype: torch.dtype = torch.bfloat16,
    max_steps: int = GATE_13_DEFAULT_MAX_STEPS,
    capability_profile: CapabilityProfile,
) -> dict[str, Any]:
    """`decode_free_running`'s own kwargs for gate 13's real decode over
    `entry` -- one cached page, the same features every other real-model
    gate reads (design §3.1).

    `capability_profile` is required keyword-only with no default. Every
    caller must pass the profile resolved from the cache/checkpoint identity
    (tests pass `PRIMARY_S1_CAPABILITY_PROFILE` explicitly). Gate 13 must
    decode under the SAME profile gate 14 checks the run against, never a
    profile hardcoded independently of what the cache under evaluation
    actually names (2026-09-10 review item 2)."""
    return {
        "model": model,
        "controller": build_controller_for_profile(vocabulary, capability_profile),
        "vocabulary": vocabulary,
        "visual_features": entry.features.to(visual_features_dtype).to(device).unsqueeze(0),
        "image_grid_thw": entry.grid_thw.to(device).unsqueeze(0),
        "image_token_id": image_token_id,
        "config": DecodeConfig(
            capability_profile_id=capability_profile.profile_id, max_steps=max_steps
        ),
    }


def _gate13_and_gate7(
    *,
    model: QwenDenseCandidateModel,
    vocabulary: cv.CompositeVocabulary,
    entry: _CacheEntryLike,
    image_token_id: int,
    device: torch.device,
    max_steps: int,
    skip_real_decode: bool,
    capability_profile: CapabilityProfile,
) -> tuple[GateResult, GateResult, str]:
    """Run gate 13 then replay its hypothesis in gate 7 under one profile.

    Gate 13's decode and gate 7's replay must bind to the SAME profile gate
    14 checks the run against (2026-09-10 review item 2): a checkpoint
    trained under a wider DEC-024 profile must not silently decode/replay
    under `PRIMARY_S1_CAPABILITY_PROFILE`'s narrower exclusion set just
    because a constructor's default is S1's.
    """
    from unflatten.model import s1_runner_inputs as sri

    if skip_real_decode:
        gate13_result = _blocked(
            13,
            "free_running_no_silent_truncation",
            "--skip-real-decode was passed: the real free-running decode did not "
            "run, so this gate has no evidence (skipping never produces a pass)",
        )
        hypothesis_capture = None
        gate7_source = "not_run: --skip-real-decode was passed"
    else:
        print(f"[s1_runner] phase 3: gate 13 free-running decode, max_steps={max_steps} ...")
        hypothesis_capture = sri.HypothesisCapture(decode_free_running)
        gate13_result = gate_13_free_running_no_silent_truncation(
            hypothesis_capture,
            build_gate13_decode_kwargs(
                model=model,
                vocabulary=vocabulary,
                entry=entry,
                image_token_id=image_token_id,
                device=device,
                max_steps=max_steps,
                capability_profile=capability_profile,
            ),
        )
        print(
            f"[s1_runner] gate 13: {gate13_result.status} "
            f"(branch: {gate13_result.evidence.get('branch')})"
        )
        gate7_source = f"{hypothesis_capture.__module__}.{hypothesis_capture.__qualname__}"

    print("[s1_runner] gate 7: replaying gate 13's own free-running hypothesis ...")
    gate7_result = sri.gate_7_from_hypothesis(
        hypothesis_capture.hypothesis if hypothesis_capture is not None else None,
        vocabulary=vocabulary,
        gate13_result=gate13_result,
        capability_profile=capability_profile,
    )
    print(f"[s1_runner] gate 7: {gate7_result.status}")
    return gate13_result, gate7_result, gate7_source


#: Gate 10's `blocked` reason when the run was not asked to read the store.
#: Named rather than generic: "no corpus supplied" reads as an oversight,
#: and the real situation is that the only copy of the cohort's text is a
#: ~62 GB object store which a Colab payload deliberately does not carry
#: (`s1_text_corpus`'s module docstring).
GATE_10_STORE_ABSENT_REASON = (
    "the cohort's text lives only in the object store, which this run was not "
    "asked to read (--text-corpus-from-store); it is deliberately not shipped "
    "as an artefact, so a payload without the store cannot evidence gate 10"
)


def resolve_gate10_corpus(
    *,
    release_dir: Path,
    store_dir: Path,
    bundle_ids: Iterable[str],
    split: str | None,
    task_kind: str | None,
    enabled: bool,
    max_strings: int = stc.DEFAULT_MAX_STRINGS,
) -> tuple[tuple[str, ...] | None, str]:
    """`(corpus, provenance)` for gate 10, or `(None, reason)`.

    `split`/`task_kind` come from the feature cache's OWN identity, so the
    corpus is drawn from the same enumeration the cohort was: choosing them
    here would be a second, independent selection that could silently
    disagree with the cache's.
    """
    if not enabled:
        return None, GATE_10_STORE_ABSENT_REASON
    if split is None or task_kind is None:
        return None, (
            f"the cache identity does not record split={split!r}/task_kind={task_kind!r}, "
            "so the cohort's own enumeration cannot be reproduced -- refusing to "
            "pick one and call the result cohort text"
        )
    try:
        corpus = stc.build_cohort_text_corpus(
            release_dir, store_dir, bundle_ids,
            split=split, task_kind=task_kind, max_strings=max_strings,
        )
    except stc.S1TextCorpusUnavailable as exc:
        return None, str(exc)
    return corpus.texts, corpus.source


def build_context_from_loaded(
    *,
    sdpa_candidate: _CandidateLike,
    eager_candidate: _CandidateLike | None,
    entry: _CacheEntryLike,
    cohort_bundle_ids: Iterable[str],
    release_dir: Path,
    run_class: str = "discovery",
    gate12_visual_tokens: int = GATE_12_DEFAULT_VISUAL_TOKENS,
    gate12_junk_tokens: int = GATE_12_DEFAULT_JUNK_TOKENS,
    visual_features_dtype: torch.dtype = torch.bfloat16,
    optimizer_lr: float = 0.1,
    allowlist: dict[str, Any] | None = None,
    tokenizer: et.QwenTokenizerLike | None = None,
    vocabulary: cv.CompositeVocabulary | None = None,
    text_corpus: tuple[str, ...] | None = None,
    text_corpus_source: str | None = None,
) -> S1RunnerContext:
    """Build a full `S1RunnerContext` from already-loaded candidates and
    one representative cached page. Separated from `main` so it is
    testable against `build_stub_qwen_candidate` and a hand-built
    `CacheEntry`-shaped object, with no checkpoint load, no cache reader,
    and no real release directory needed for the mechanics themselves
    (`release_dir` here only needs `splits/*.jsonl` to exist for the
    lineage helpers -- a synthetic one in a test's `tmp_path` works).

    `eager_candidate=None` skips gate 12's wiring entirely (it then
    reports `blocked`, correctly -- DEC-007 D8) -- `main` uses this to
    keep only ONE candidate resident at a time, evaluating gate 12 in a
    later, separate phase via `build_gate12_fixture` once the SDPA
    candidate has been freed.

    **Mixed precision (DEC-007 D3.1, `m2-qwen-candidate`'s fix):** `model`
    is built and left at its DEFAULT (fp32) dtype -- no `.to(bfloat16)`
    cast here anymore. The frozen candidate stays bf16 (however it was
    loaded); `decode`'s own boundary casts a trainable fp32
    `inputs_embeds` to the frozen decoder's compute dtype internally, so
    this function never needs to know or match it. `visual_features_dtype`
    only controls what dtype the (frozen, cached) visual features are read
    as -- an fp32 master copy of the trainable groups is what makes
    `run_parameter_movement_check`'s per-tensor evidence meaningful; casting
    the whole model to bf16 here would silently reintroduce the exact
    underflow this fix exists to prevent.

    Only `entry` (one page) drives the model-forward gates (1-3, 6, 8) --
    gate 9's lineage evidence comes from `cohort_bundle_ids`, the FULL set
    the caller read off the cache, not just this one entry.

    **What those batches are, and why it is load-bearing** (2026-09-04):
    every one of them is built by `_build_teacher_forced_batch`, whose
    composite ids deliberately span all three of `embed_packed`'s ranges
    (structural, Qwen, byte fallback) instead of being one uniform run of
    visual placeholders. A placeholder-only batch leaves both trainable
    embeddings out of the forward pass, which let gate 6 pass over a
    checkpoint missing one of them and made gate 3 fail for a harness
    reason. Gate 8's pair differs ONLY in the `document_bundle_id` handed
    to that builder, so the gate measures the builder, not a tensor
    against its own clone. Both properties have mutation proofs in
    `tests/model/test_s1_runner.py`; narrowing the batch back to
    placeholders alone silently un-evidences gates 3, 6 and 8.

    **Gate 12 built here has NO negative control**, the same gap
    `evaluate_all_gates` carries and for the same reason: the leaked batch
    stays out of `S1RunnerContext`, so a caller wiring gate 12 through
    this function gets a `pass` from a comparison never shown to detect
    the defect on that model. `main` runs `gate_12_with_negative_control`
    instead. A future runner path built on this function would lose the
    control silently while still reporting gate 12 as passing.
    """
    if allowlist is not None and allowlist["native_vocab_size"] != sdpa_candidate.vocab_size:
        # Design §4.6, acceptance A3. `composite_head` is built over
        # `packed_ranges(sdpa_candidate.vocab_size)` and `_select_greedy`
        # indexes `logits_row` by packed id, so an allowlist built at another
        # width puts byte-control and byte-value ids at offsets the head does
        # not share -- a silently wrong logit column, with nothing raising.
        # Refused HERE, before any decode.
        raise qtal.TextTokenAllowlistUnacceptable(
            f"A3: allowlist native_vocab_size {allowlist['native_vocab_size']} != "
            f"candidate vocab_size {sdpa_candidate.vocab_size} -- the composite "
            "vocabulary's packed ranges would disagree with the head's (§4.6)"
        )

    ranges = cv.packed_ranges(sdpa_candidate.vocab_size)
    device = sdpa_candidate.embed_tokens.weight.device

    model = QwenDenseCandidateModel(
        decode=sdpa_candidate.decode,
        inject_visual=sdpa_candidate.inject_visual,
        visual_dim=sdpa_candidate.visual_hidden_size,
        d_model=sdpa_candidate.hidden_size,
        ranges=ranges,
        qwen_embedding_weight=sdpa_candidate.embed_tokens.weight,
        qwen_lm_head_weight=sdpa_candidate.lm_head_weight,
    )
    model.to(device)
    _assert_model_on_device(model, device, context="build_context_from_loaded")
    model.eval()

    cohort_lineage = s1_cohort.resolve_cohort_lineage(release_dir, cohort_bundle_ids)

    build_kwargs: dict[str, Any] = {
        "ranges": ranges,
        "image_token_id": sdpa_candidate.image_token_id,
        "device": device,
        "visual_features_dtype": visual_features_dtype,
    }
    batch = _build_teacher_forced_batch(
        entry, document_bundle_id=entry.document_bundle_id, **build_kwargs
    )
    # Gate 8: the SAME builder run twice over the same page content under
    # two DIFFERENT provenance values -- what a loader would produce for
    # one page seen under two document_bundle_ids. `batch_b` is not a
    # clone of `batch_a` (which compared a tensor with itself and could
    # never fail); the builder is the subject under test, so a future
    # change that folds provenance into any tensor makes this pair differ
    # and gate 8 catches it.
    batch_b = _build_teacher_forced_batch(
        entry,
        document_bundle_id=f"{entry.document_bundle_id}__gate8_alternate_provenance",
        **build_kwargs,
    )

    optimizer = build_s1_optimizer(model, lr=optimizer_lr)

    # Gate 12: `eager_candidate=None` (main's sequential-loading phase 1)
    # leaves this unwired -- gate 12 correctly reports `blocked`, not a
    # silent pass, until phase 2 supplies it (2026-09-05 review item 3's
    # sequencing fix; see `build_gate12_fixture`).
    gate12_fixture = None
    if eager_candidate is not None:
        gate12_fixture = build_gate12_fixture(
            eager_candidate,
            ranges,
            gate12_visual_tokens=gate12_visual_tokens,
            gate12_junk_tokens=gate12_junk_tokens,
            visual_features_dtype=visual_features_dtype,
        )

    return S1RunnerContext(
        run_class=run_class,
        model=model,
        optimizer=optimizer,
        gate123_batch=batch,
        candidate_named_parameters=sdpa_candidate.named_parameters,
        checkpoint_batch=batch,
        metadata_leak_batch_a=batch,
        metadata_leak_batch_b=batch_b,
        cohort_lineage_root_ids=cohort_lineage.lineage_root_ids,
        cohort_requested_bundle_count=len(cohort_lineage.requested_bundle_ids),
        cohort_resolved_bundle_count=len(cohort_lineage.resolved_bundle_ids),
        eval_test_lineage_root_ids=s1_cohort.eval_test_lineage_root_ids(release_dir),
        qwen_vocab_size=sdpa_candidate.vocab_size,
        padding_model=gate12_fixture.model if gate12_fixture is not None else None,
        padding_batch_unpadded=gate12_fixture.clean if gate12_fixture is not None else None,
        padding_batch_padded=gate12_fixture.padded if gate12_fixture is not None else None,
        padding_unpadded_steps=(
            gate12_fixture.unpadded_steps if gate12_fixture is not None else None
        ),
        padding_attn_implementation="eager" if eager_candidate is not None else None,
        padding_actual_attn_implementation=(
            eager_candidate.attn_implementation if eager_candidate is not None else None
        ),
        allowlist=allowlist,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        text_corpus=text_corpus,
        text_corpus_source=text_corpus_source,
    )


_NOT_A_QUALITY_CLAIM = (
    "S1 is a contract/trainability canary only (RFC005 §14.1) -- it makes NO quality, "
    "render-fidelity, generalization, or product-backbone claim. Passing S1 authorizes "
    "nothing beyond itself; S2 requires a separate ratification with seven further "
    "conditions (§14.7)."
)


def _print_report_summary(report: StageGateReport) -> None:
    print()
    print("RFC005 S1 Stage Gate Report")
    print("=" * 60)
    for gate in report.gates:
        print(f"  gate {gate.number:>2}  {gate.name:<38} {gate.status.upper()}")
    print("=" * 60)
    print(f"promoted: {report.promoted}")
    if not report.promoted:
        blocked = [g.number for g in report.blocked_gates()]
        failed = [g.number for g in report.failed_gates()]
        if blocked:
            print(f"  blocked gates (insufficient evidence, never a pass): {blocked}")
        if failed:
            print(f"  failed gates: {failed}")
    print()
    print(_NOT_A_QUALITY_CLAIM)


def main(argv: list[str] | None = None) -> int:
    """Evaluate all fifteen RFC005 §14.4 gates against an already-built
    candidate feature cache and write the Stage Gate Report. Exit code is
    non-zero unless `promoted` is `True` -- a `blocked` or `fail` gate is
    never mistaken for success by whatever invokes this from a shell.
    """
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.s1_runner",
        description=(
            "Evaluate the fifteen RFC005 §14.4 S1 hard gates against an "
            "already-built candidate feature cache and write the Stage Gate "
            "Report. Reads release_dir for split-manifest lineage (gate 9) and "
            "store_dir only when --text-corpus-from-store asks for gate 10's "
            "corpus -- pixels/labels come entirely from cache_dir, "
            "which this command never builds or writes into. Not the canary "
            "itself: a contract/trainability check only (see the printed "
            "summary's closing note)."
        ),
    )
    parser.add_argument(
        "release_dir", type=Path, help="release-v10-shaped release directory (read-only)"
    )
    parser.add_argument(
        "store_dir",
        type=Path,
        help="the release's paired object store, e.g. m2a-v5 (read-only). Read "
        "ONLY when --text-corpus-from-store is passed, which is what makes gate "
        "10 evidenceable: the cohort's text lives nowhere else (the feature "
        "cache carries no document text, and release-v10 carries none either). "
        "Until 2026-09-05 this argument was accepted and never read at all, "
        "while its help text claimed it supplied gate 9's lineage -- gate 9 "
        "reads release_dir alone",
    )
    parser.add_argument(
        "cache_dir",
        type=Path,
        help="an ALREADY-BUILT candidate feature cache (build_cache.py "
        "--single-page-only --exclude-capability-profile); this command reads "
        "it and never builds or writes into it",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device for both candidate loads (default: cpu). This command loads "
        "TWO full Qwen3-VL-4B checkpoints (~8.3 GB bf16 each, DEC-007 D3) but "
        "SEQUENTIALLY -- the SDPA candidate is freed before the eager one loads, "
        "so only one is ever resident. cpu is the default specifically so this "
        "never contends for the GPU another agent may be holding; pass 'cuda' "
        "to run for real (coordinate the window first). KNOWN 2026-09-04 "
        "FINDING: after device placement, decode-activation checkpointing and "
        "the LM-head fp32-cast fix (three independently verified memory fixes), "
        "one real single page's teacher-forced FORWARD PASS ALONE already "
        "consumes ~10.26 of the RTX 4070's 11.59 GiB visible, before backward "
        "is even called -- 875 visual tokens (grid_thw=1x70x50, a full page at "
        "the configured resolution, not an outlier), on top of the 8.3 GB "
        "resident bf16 candidate. Gate 1-3's probe does not fit this card in "
        "the current architecture; confirmed reproducible across four runs, "
        "each failing measurably deeper (placement -> decode activations "
        "~10.5 GiB -> LM-head readout 9.27 GiB -> a 510 MiB auxiliary "
        "allocation after a completed forward) -- a budget walked down to, "
        "not a wall hit once. Needs the H200 or a DEC-level change (CPU "
        "offload, sequence chunking, lower max_pixels); not something "
        "--device alone can fix.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("s1_stage_gate_report.json"),
        help="where the Stage Gate Report JSON is written (default: "
        "./s1_stage_gate_report.json)",
    )
    parser.add_argument(
        "--run-class",
        default="discovery",
        choices=("discovery", "promotion"),
        help="RFC005 run_class recorded in the report (default: discovery -- the "
        "first S1 run is expected to be discovery even while gradient-bearing, "
        "design §8 rule 3a)",
    )
    parser.add_argument(
        "--gate12-visual-tokens",
        type=int,
        default=GATE_12_DEFAULT_VISUAL_TOKENS,
        help="synthetic visual token count for gate 12's leak fixture (default: "
        f"{GATE_12_DEFAULT_VISUAL_TOKENS} -- gate 12 tests model wiring, not this "
        "run's real pixels; the default keeps both batches past the ~8-token "
        "length where SDPA stops being reproducible, which is what the eager "
        "reference mode is for)",
    )
    parser.add_argument(
        "--gate12-junk-tokens",
        type=int,
        default=GATE_12_DEFAULT_JUNK_TOKENS,
        help="synthetic junk visual row count appended for gate 12's leak fixture "
        f"(default: {GATE_12_DEFAULT_JUNK_TOKENS})",
    )
    parser.add_argument(
        "--text-corpus-from-store",
        action="store_true",
        help="read gate 10's round-trip corpus, AND (when --checkpoint is also "
        "given) gate 5's independent expected_missing_mask recomputation, from "
        "the cohort's own text spans/target bundle in store_dir "
        "(dataset.items.build_item). WITHOUT this flag gate 10 reports "
        "`blocked`, naming the absent store, and gate 5 reports `blocked` "
        "naming GATE_5_STORE_ABSENT_REASON -- neither ever a pass. The corpus "
        "is deliberately NOT shipped as an artefact: that would put document "
        "text into a payload the privacy decision assumes carries none, so this "
        "flag is what separates a local run (store present, gates 5/10 "
        "evidenced) from a Colab one (store absent, both honestly blocked)",
    )
    parser.add_argument(
        "--text-corpus-max-strings",
        type=int,
        default=stc.DEFAULT_MAX_STRINGS,
        help="how many distinct cohort strings gate 10 round-trips (default: "
        f"{stc.DEFAULT_MAX_STRINGS})",
    )
    parser.add_argument(
        "--gate13-max-steps",
        type=int,
        default=GATE_13_DEFAULT_MAX_STEPS,
        help=f"gate 13's free-running decode budget (default: {GATE_13_DEFAULT_MAX_STEPS}; "
        "see GATE_13_DEFAULT_MAX_STEPS -- not a decode-quality knob)",
    )
    parser.add_argument(
        "--skip-real-decode",
        action="store_true",
        help="skip gate 13's real free-running decode (design §3.3). It then "
        "reports `blocked` with the skip recorded; there is no path where "
        "skipping produces a pass",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="a trained checkpoint to load onto the SDPA candidate before "
        "gates run (plan spec-runner-inputs-c5b), via "
        "s1_runner_inputs.load_trained_checkpoint. Requires --run-dir (for "
        "the run manifest and checkpoint ancestry base it is verified "
        "against) and --exact-text-cache-dir (for the dataset the checkpoint "
        "was trained against). Omit it -- the default -- to evaluate the "
        "untrained base candidate exactly as before",
    )
    parser.add_argument(
        "--checkpoint-optimizer",
        choices=("sgd", "adamw"),
        default=None,
        help="the algorithm --checkpoint was trained with; required when "
        "--checkpoint is given (no default -- a caller must declare it, "
        "never inherit gate12/gate123's own SGD default)",
    )
    parser.add_argument(
        "--exact-text-cache-dir",
        type=Path,
        default=None,
        help="an exact-text cache (s1_train.ExactTextTrainingDataset) used "
        "for --checkpoint's dataset, gate 4/5's real batch, and gate 14's "
        "cache identity. Omit it -- the default -- to leave gates 4, 5 and "
        "14 blocked exactly as before",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="a frozen run directory carrying s1_run_manifest.json and "
        "checkpoint_ancestry_base.json (freeze_s1_run_manifest / "
        "freeze_checkpoint_ancestry output), needed by --checkpoint and by "
        "gate 14. Omit it -- the default -- to leave gate 14 blocked",
    )
    parser.add_argument(
        "--component-lock",
        type=Path,
        default=None,
        help="the Component Lock document gate 14 verifies. Omit it -- the "
        "default -- to leave gate 14 blocked",
    )
    parser.add_argument(
        "--architecture-manifest",
        type=Path,
        default=None,
        help="the architecture manifest gate 14 verifies. Omit it -- the "
        "default -- to leave gate 14 blocked",
    )
    parser.add_argument(
        "--gate45-max-rows",
        type=int,
        default=4,
        help="how many of --checkpoint's dataset rows gate 4/5's input "
        "builder scans for a masked position before giving up on the last "
        "one scanned (default: 4)",
    )
    args = parser.parse_args(argv)

    if args.checkpoint is not None and args.checkpoint_optimizer is None:
        parser.error("--checkpoint-optimizer is required when --checkpoint is given")
    if args.checkpoint is not None and (args.run_dir is None or args.exact_text_cache_dir is None):
        parser.error("--checkpoint requires --run-dir and --exact-text-cache-dir")

    from unflatten.model import s1_runner_inputs as sri
    from unflatten.model import s1_train
    from unflatten.model.candidate_loader import load_qwen_candidate
    from unflatten.model.feature_cache import FeatureCacheReader
    from unflatten.model.freeze_checkpoint_ancestry import CHECKPOINT_ANCESTRY_BASE_FILENAME
    from unflatten.model.freeze_s1_run_manifest import S1_RUN_MANIFEST_FILENAME

    # The cohort pin is read (and hash-checked) BEFORE the 4B load and the
    # feature-cache scan: a run dir that cannot present it must refuse in
    # seconds, not after both (2026-09-14, the S1 canary's first gate run).
    checkpoint_expectation = None
    if args.checkpoint is not None:
        checkpoint_expectation = sri.load_cache_expectation(
            args.run_dir, json.loads((args.run_dir / S1_RUN_MANIFEST_FILENAME).read_text())
        )

    # Sequential loading (2026-09-05 review, team lead's follow-up): only
    # ONE 8.3 GB candidate is ever resident at a time -- load SDPA, run
    # every gate except 12, free it, load eager, run gate 12, merge. This
    # is what makes `--device cuda` safe against the 4070's ~11.3 GB free
    # (two resident candidates need ~16.6 GB) and turns an unmeasurable
    # CPU run (two real backward passes through 4B params, one at a time
    # would still be expensive but at least each phase is now separately
    # timeable) into a GPU-feasible one.
    print(f"[s1_runner] phase 1: loading SDPA candidate (main path) on {args.device} ...")
    sdpa_candidate = load_qwen_candidate(device=args.device, dtype=torch.bfloat16)

    print(f"[s1_runner] reading cache at {args.cache_dir} ...")
    reader = FeatureCacheReader(args.cache_dir, expect_backbone=sdpa_candidate.model_id)
    if len(reader) == 0:
        print(f"[s1_runner] {args.cache_dir}: cache is empty, nothing to evaluate", file=sys.stderr)
        return 1
    entry = reader[0]
    bundle_ids = {reader[i].document_bundle_id for i in range(len(reader))}
    print(f"[s1_runner] cache has {len(reader)} page(s), {len(bundle_ids)} distinct document(s)")

    # The DEC-007 D3-pinned tokenizer and everything that hangs off it.
    # Gates 10 and 13 both need the REAL Qwen text allowlist: without it
    # `exact_text.encode_text` rejects every Qwen candidate and falls back
    # to bytes on every scalar (gate 10 would then evidence only the
    # fallback path), and the grammar offers no text token at all (gate
    # 13's text mode would be unreachable). Built once, here.
    print("[s1_runner] loading the pinned tokenizer and building the text allowlist ...")
    tokenizer, snapshot_dir = qta.load_pinned_tokenizer()
    artifact_set_hash = qta.tokenizer_artifact_set_hash(
        snapshot_dir,
        model_id=qta.QWEN_CANDIDATE_MODEL_ID,
        revision=qta.QWEN_CANDIDATE_REVISION,
    )
    binding = qtal.build_qwen_text_binding(
        tokenizer=tokenizer,
        tokenizer_artifact_set_hash=artifact_set_hash,
        native_vocab_size=sdpa_candidate.vocab_size,
    )
    vocabulary = cv.make_composite_vocabulary(
        tokenizer_artifact_set_hash=artifact_set_hash,
        allowlist_payload=binding.allowlist,
    )
    print(
        f"[s1_runner] allowlist: {len(binding.allowlist['allowed_native_ids'])} allowed / "
        f"{len(binding.allowlist['denied'])} denied of {sdpa_candidate.vocab_size}"
    )

    text_corpus, text_corpus_source = resolve_gate10_corpus(
        release_dir=args.release_dir,
        store_dir=args.store_dir,
        bundle_ids=bundle_ids,
        # `FeatureCacheReader.identity` is the manifest's `asdict`ed
        # `CacheIdentity` -- a plain dict, not the dataclass.
        split=reader.identity.get("split"),
        task_kind=reader.identity.get("task_kind"),
        enabled=args.text_corpus_from_store,
        max_strings=args.text_corpus_max_strings,
    )
    print(f"[s1_runner] gate 10 corpus: {text_corpus_source}")

    phase1_ctx = build_context_from_loaded(
        sdpa_candidate=sdpa_candidate,
        eager_candidate=None,  # gate 12 unwired here on purpose -- phase 2 supplies it
        entry=entry,
        cohort_bundle_ids=bundle_ids,
        release_dir=args.release_dir,
        run_class=args.run_class,
        gate12_visual_tokens=args.gate12_visual_tokens,
        gate12_junk_tokens=args.gate12_junk_tokens,
        allowlist=binding.allowlist,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        text_corpus=text_corpus,
        text_corpus_source=text_corpus_source,
    )
    # `spec-runner-inputs-c5b`: optional checkpoint load, gate 4/5 and gate
    # 14 wiring. `ranges` moved up here (was computed later, right before
    # `del sdpa_candidate, phase1_ctx` below) because gate 4/5's input
    # builder needs it now. Every new flag defaults to `None`; when none of
    # them are given this block builds nothing and `phase1_ctx` is exactly
    # what it always was.
    ranges = cv.packed_ranges(sdpa_candidate.vocab_size)
    evaluated_checkpoint: sri.EvaluatedCheckpoint | None = None
    gate45_inputs: sri.Gate45Inputs | None = None
    if args.checkpoint is not None:
        run_manifest_for_checkpoint = json.loads(
            (args.run_dir / S1_RUN_MANIFEST_FILENAME).read_text()
        )
        ancestry_for_checkpoint = json.loads(
            (args.run_dir / CHECKPOINT_ANCESTRY_BASE_FILENAME).read_text()
        )
        print(
            f"[s1_runner] building the exact-text training dataset at "
            f"{args.exact_text_cache_dir} ..."
        )
        checkpoint_dataset = s1_train.ExactTextTrainingDataset(
            args.exact_text_cache_dir,
            gradient_bearing=run_manifest_for_checkpoint["gradient_bearing"],
            expectation=checkpoint_expectation,
        )
        # Gate 13's decode and gate 7's replay must bind to the SAME profile
        # gate 14 checks the run against (2026-09-10 review item 2): a
        # checkpoint trained under a wider DEC-024 profile must not silently
        # decode/replay under `PRIMARY_S1_CAPABILITY_PROFILE`'s narrower
        # exclusion set just because this constructor's default is S1's.
        # Resolve it before checkpoint loading or gate 4/5 work so an
        # unresolvable profile hash spends neither on a run that must refuse.
        try:
            evaluated_capability_profile = resolve_evaluated_capability_profile(
                checkpoint_path=args.checkpoint,
                identity=checkpoint_dataset.identity,
            )
        except CapabilityProfileHashUnresolved as exc:
            print(
                f"[s1_runner] checkpoint's cache identity capability profile "
                f"refused: {exc}",
                file=sys.stderr,
            )
            return 1
        # `build_context_from_loaded` always builds an SGD optimizer
        # (`build_s1_optimizer`'s own default) -- rebuilt here at the
        # DECLARED algorithm so a checkpoint's `optimizer_state_dict`
        # resumes into the real class it was written under, not merely a
        # name string `load_trained_checkpoint` checks after the fact.
        phase1_ctx = dataclasses.replace(
            phase1_ctx,
            optimizer=build_s1_optimizer(
                phase1_ctx.model,
                lr=phase1_ctx.optimizer.param_groups[0]["lr"],
                algorithm=args.checkpoint_optimizer,
            ),
        )
        # The REBUILT optimizer's own class name, not a string derived from
        # `args.checkpoint_optimizer` -- the comparison below is against
        # the object actually resumed, not against what the flag claims it
        # should be (2026-09-05 review suggestion).
        declared_optimizer_name = type(phase1_ctx.optimizer).__name__
        print(f"[s1_runner] loading trained checkpoint {args.checkpoint} ...")
        evaluated_checkpoint = sri.load_trained_checkpoint(
            model=phase1_ctx.model,
            optimizer=phase1_ctx.optimizer,
            path=args.checkpoint,
            run_manifest=run_manifest_for_checkpoint,
            ancestry=ancestry_for_checkpoint,
            cache_identity=dataclasses.asdict(checkpoint_dataset.identity),
            feature_source=checkpoint_dataset.feature_source,
            declared_optimizer=declared_optimizer_name,
        )
        print(
            f"[s1_runner] checkpoint loaded: step={evaluated_checkpoint.step} "
            f"id={evaluated_checkpoint.checkpoint_id}"
        )
        print(
            f"[s1_runner] building gate 4/5 inputs (scanning up to "
            f"{args.gate45_max_rows} row(s)) ..."
        )
        gate45_inputs = sri.build_gate_4_5_inputs(
            model=phase1_ctx.model,
            dataset=checkpoint_dataset,
            ranges=ranges,
            image_token_id=sdpa_candidate.image_token_id,
            device=sdpa_candidate.embed_tokens.weight.device,
            release_dir=args.release_dir,
            store_dir=args.store_dir if args.text_corpus_from_store else None,
            max_rows=args.gate45_max_rows,
        )
    else:
        evaluated_capability_profile = resolve_evaluated_capability_profile(
            checkpoint_path=None,
            identity=None,
        )

    gate14_inputs: sri.Gate14Inputs | None = None
    if (
        args.run_dir is not None
        and args.component_lock is not None
        and args.architecture_manifest is not None
        and args.exact_text_cache_dir is not None
    ):
        print("[s1_runner] building gate 14 inputs ...")
        gate14_inputs = sri.build_gate_14_inputs(
            exact_text_cache_dir=args.exact_text_cache_dir,
            run_dir=args.run_dir,
            component_lock_path=args.component_lock,
            architecture_manifest_path=args.architecture_manifest,
        )

    if gate14_inputs is not None or gate45_inputs is not None:
        phase1_ctx = sri.with_gate_inputs(phase1_ctx, gate14=gate14_inputs, gate45=gate45_inputs)

    print("[s1_runner] phase 1: evaluating gates 1-11, 14, 15 (12 and 13 blocked for now) ...")
    # Gates 1-3's probe runs real optimizer steps against `phase1_ctx.model`
    # as part of `evaluate_all_gates` -- a snapshot/restore pair around it
    # is the ONLY way gate 13's free-running decode (right below) sees the
    # checkpoint's own trained weights rather than "checkpoint plus a few
    # probe steps" (`snapshot_trainable_state`'s docstring). Taken and
    # restored unconditionally -- cheap even without `--checkpoint`, and it
    # means this property does not silently depend on the flag being set.
    weights_snapshot = sri.snapshot_trainable_state(phase1_ctx.model)
    phase1_report = evaluate_all_gates(phase1_ctx)
    sri.restore_trainable_state(phase1_ctx.model, weights_snapshot)

    # Phase 3 BEFORE phase 2 (design §3.3): gate 13's decode reuses the
    # already-resident SDPA candidate, so it must run before the `del`
    # below -- and AFTER `evaluate_all_gates` has returned, because gates
    # 1-3's `ModelGradientProbe` holds gradient and optimizer state for as
    # long as it is alive. A real ~10 GiB forward alongside that live probe
    # is the worst available ordering; this is the only ordering where
    # neither the probe nor the eager candidate is resident during it.
    # `HypothesisCapture` wraps `decode_free_running` so gate 7 replays the
    # SAME free-running hypothesis gate 13 just decoded, rather than a
    # second, separate decode (`s1_runner_inputs.gate_7_from_hypothesis`'s
    # docstring, design finding).
    # Profile-binding rationale: `_gate13_and_gate7`'s docstring.
    gate13_result, gate7_result, gate7_source = _gate13_and_gate7(
        model=phase1_ctx.model,
        vocabulary=vocabulary,
        entry=entry,
        image_token_id=sdpa_candidate.image_token_id,
        device=sdpa_candidate.embed_tokens.weight.device,
        max_steps=args.gate13_max_steps,
        skip_real_decode=args.skip_real_decode,
        capability_profile=evaluated_capability_profile,
    )

    del sdpa_candidate, phase1_ctx
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print("[s1_runner] phase 1 candidate freed")

    print(
        f"[s1_runner] phase 2: loading eager candidate (gate 12 reference mode) "
        f"on {args.device} ..."
    )
    eager_candidate = load_qwen_candidate(
        device=args.device, dtype=torch.bfloat16, attn_implementation="eager"
    )
    gate12_fixture = build_gate12_fixture(
        eager_candidate,
        ranges,
        gate12_visual_tokens=args.gate12_visual_tokens,
        gate12_junk_tokens=args.gate12_junk_tokens,
    )
    print("[s1_runner] phase 2: evaluating gate 12 (with its negative control) ...")
    gate12_result, gate12_control = gate_12_with_negative_control(
        gate12_fixture,
        attn_implementation="eager",
        actual_attn_implementation=eager_candidate.attn_implementation,
    )
    print(
        f"[s1_runner] gate 12 negative control (deliberately leaked pair): "
        f"{gate12_control.status} -- 'fail' is the healthy answer"
    )
    del eager_candidate, gate12_fixture
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print("[s1_runner] phase 2 candidate freed")

    merged = {7: gate7_result, 12: gate12_result, 13: gate13_result}
    gate5_result = sri.gate_5_from_inputs(gate45_inputs)
    if gate5_result is not None:
        merged[5] = gate5_result
    merged_gates = tuple(merged.get(g.number, g) for g in phase1_report.gates)
    report = StageGateReport(
        schema_version=phase1_report.schema_version,
        run_class=phase1_report.run_class,
        gates=merged_gates,
    )

    evaluated_checkpoint_payload = (
        sri.evaluated_checkpoint_payload(evaluated_checkpoint)
        if evaluated_checkpoint is not None
        else None
    )
    gate_inputs_payload = None
    if gate45_inputs is not None or gate14_inputs is not None:
        gate_inputs_payload = {
            "gate45_row_index": gate45_inputs.row_index if gate45_inputs is not None else None,
            "document_bundle_id": (
                gate45_inputs.document_bundle_id if gate45_inputs is not None else None
            ),
            "page_index": gate45_inputs.page_index if gate45_inputs is not None else None,
            "gate_5_source": (
                gate45_inputs.gate5_blocked_reason or "store_dir"
                if gate45_inputs is not None
                else None
            ),
            "gate_7_source": gate7_source,
            "gate_14_cache_identity_source": "exact_text" if gate14_inputs is not None else None,
            "weights_restored_before_gate_13": True,
        }

    payload = stage_gate_report_payload(
        report,
        evaluated_checkpoint=evaluated_checkpoint_payload,
        gate_inputs=gate_inputs_payload,
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[s1_runner] Stage Gate Report written to {args.report_out}")

    _print_report_summary(report)
    return 0 if report.promoted else 1


if __name__ == "__main__":
    sys.exit(main())
