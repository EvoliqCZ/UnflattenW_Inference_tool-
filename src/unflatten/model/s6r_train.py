"""`S6R_reasoning_specialization` dormant phase runner -- RFC005-ERRATA-007 E2.

RFC005-ERRATA-007 E2 names four fail-closed preconditions for S6R
("chýbajúci znamená `blocked`"): (1) G2 accepted sparse MoE
(RFC004-ERRATA-005 E1 rule 9 -- a dense G2 winner blocks S6R outright), (2)
the owner selected and confirmed a reasoning specialization (E1, over the
G2 parent), (3) a ratified RFC003 erratum with the new `task_kind`s (E3,
`RFC003-ERRATA-003`) has a dataset release actually built from it, and (4)
the parent is a *promotion* checkpoint of the G2 winner, never a discovery
checkpoint (RFC005 §12.1). This module checks all four before running
anything, and today all four fail: DEC-045's own decision record leaves
"otvorene ... prahy G3" and RFC005-ERRATA-007 §E2's precondition 3 only
half-satisfied (the erratum is ratified, but no release exists yet built
from it). Only `--preconditions-report` and `--dry-run` (over the
in-process stub, tests only) are wired to a CLI; every real phase call goes
through :func:`require_s6r_preconditions` and refuses.

**Sub-steps (RFC005-ERRATA-007 E2, návrh ADRF §9, R-A..R-E):**

- **R-A** -- the existing contract canary
  (`sparse_expert_adapter.reasoning_canary.run_r_a_contract_canary`) run
  once more here only to attach it to an S6R-identified manifest;
  `run_phase_r_a` never re-implements a single one of its checks.
- **R-B** -- warm-up: only `reasoning.gate`, `reasoning.specializations
  .<spec_id>` and, if the policy names one, exactly one
  `expert_adapter.specialists.<expert_id>` are trainable; every other
  parameter is frozen and verified unchanged every step (§24).
- **R-C** -- a deterministic hard-example + replay mix at a policy-declared
  ratio (the ratio itself is `[OPEN]`; the draft mirrors DEC-034's 80/20
  real-first split as a starting point, not a frozen number).
- **R-D** -- a low-LR joint unfreeze of exactly the groups a (still-draft)
  policy names -- a new promotion cycle per RFC005 §21.2, never assumed to
  reuse R-B's groups.
- **R-E** -- builds arm configs A (dense control), B (G2 winner, reasoning
  forced off, same exposure), C (G2 winner + reasoning) and optional D
  (capacity control, `[OPEN]`), all sharing one exposure plan verbatim so a
  `build_g3_report` caller (`g3_reasoning_selection_gate_report.py`) can
  compare them.

This module's own manifest identity (`S6R_RUN_MANIFEST_DOMAIN`,
`RUN_ID_PREFIX`, `S6R_ARM_DOMAIN`) is separate from S4's and S6's -- see
`s4_run_manifest.py` / `s6_paired_exposure.py` for the sibling identities
this one is never mixed with.

DEC-047 dormant-code boundary: this module runs no real gradient step
outside a CPU test's tiny stub, moves no hashed identity, and is not read
by `s4_train.py`, `s4_plan.py` or any promoted checkpoint path.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from unflatten.dataset.enums import TASK_KIND
from unflatten.model.hashing import domain_hash, make_id
from unflatten.model.reasoning import registry as reasoning_registry
from unflatten.model.sparse_expert_adapter.reasoning_canary import (
    GatePerturbation,
    run_r_a_contract_canary,
)
from unflatten.model.sparse_expert_adapter.reasoning_integration import (
    SparseExpertAdapterWithReasoning,
)

#: RFC005-ERRATA-007 E2's own manifest identity -- never S4's or S6's.
S6R_RUN_MANIFEST_DOMAIN = "rfc005.s6r-run-manifest/v1"
S6R_RUN_MANIFEST_SCHEMA_VERSION = "0.1.0"
RUN_ID_PREFIX = "s6rrun"
RUN_ID_DOMAIN = "rfc005.s6r-run-id/v1"

#: R-E arm envelope identity -- mirrors `s6_paired_exposure.py`'s arm
#: pattern (composition, its own domain, never that module's).
S6R_ARM_SCHEMA_VERSION = "0.1.0"
S6R_ARM_DOMAIN = "rfc005.s6r-arm/v1"

#: Draft stage-policy identity (R-B/R-C/R-D). Every field is a placeholder
#: (`is_placeholder: True`) until an owner freezes it before R-B's first
#: gradient (RFC005-ERRATA-007 E2/E4).
S6R_STAGE_POLICY_SCHEMA_VERSION = "0.1.0"

PHASES: tuple[str, ...] = ("R-A", "R-B", "R-C", "R-D")

#: RFC005-ERRATA-007 E3, dependent on the now-ratified `RFC003-ERRATA-003`.
REASONING_TASK_KINDS: frozenset[str] = frozenset(
    {"reasoning_constructive_v0", "reasoning_corrective_v0"}
) & TASK_KIND

#: `s6_paired_exposure.build_g2_gate_policy_draft`'s gate name/status pair --
#: read here, never recomputed: precondition 1 only checks the *report's*
#: own verdict.
G2_GATE_NAME = "G2_moe_acceptance"
G2_ACCEPTED_STATUS = "accepted"

_DEC_REFERENCE_PATTERN = re.compile(r"^DEC-\d+$")

#: R-B's default trainable groups before any specialist is named
#: (RFC005-ERRATA-007 E2 R-B): `reasoning.gate` plus the module's own
#: `reasoning.specializations.<spec_id>` -- the `<spec_id>` half is filled
#: in by :func:`r_b_group_ids` from the module under training, never
#: hardcoded, since no specialization is ratified yet.
ARM_A_DENSE_CONTROL = "A"
ARM_B_G2_WINNER_NO_REASONING = "B"
ARM_C_G2_WINNER_WITH_REASONING = "C"
ARM_D_CAPACITY_CONTROL = "D"


class S6RPreconditionRefused(RuntimeError):
    """One or more of the four RFC005-ERRATA-007 E2 preconditions is not
    satisfied -- a phase call refuses rather than running "as if" it were."""


class UnknownParameterGroupError(ValueError):
    """A `group_id` this module does not resolve, or resolves to a
    `spec_id`/`expert_id` the module under test does not actually carry."""


@dataclass(frozen=True)
class PreconditionCheckResult:
    name: str
    satisfied: bool
    reason: str


@dataclass(frozen=True)
class S6RPreconditionsReport:
    checks: tuple[PreconditionCheckResult, ...]

    @property
    def all_satisfied(self) -> bool:
        return all(c.satisfied for c in self.checks)

    def unmet(self) -> tuple[PreconditionCheckResult, ...]:
        return tuple(c for c in self.checks if not c.satisfied)

    def as_dict(self) -> dict[str, Any]:
        return {
            "all_satisfied": self.all_satisfied,
            "checks": [
                {"name": c.name, "satisfied": c.satisfied, "reason": c.reason}
                for c in self.checks
            ],
        }


# --------------------------------------------------------------------------
# Preconditions
# --------------------------------------------------------------------------


def check_g2_accepted_moe(g2_report: Mapping[str, Any] | None) -> PreconditionCheckResult:
    """Precondition 1: "G2 prijal sparse MoE. Ak G2 zostane pri dense, S6R
    je blocked" (RFC004-ERRATA-005 E1 rule 9)."""
    name = "g2_accepted_moe"
    if g2_report is None:
        return PreconditionCheckResult(name, False, "no G2 report supplied")
    if g2_report.get("gate") != G2_GATE_NAME:
        return PreconditionCheckResult(
            name, False, f"report.gate is not {G2_GATE_NAME!r}: {g2_report.get('gate')!r}"
        )
    status = g2_report.get("gate_status")
    if status != G2_ACCEPTED_STATUS:
        return PreconditionCheckResult(
            name,
            False,
            f"G2 gate_status is {status!r}, not {G2_ACCEPTED_STATUS!r} -- G2 did not "
            "accept sparse MoE (RFC004-ERRATA-005 E1 rule 9)",
        )
    return PreconditionCheckResult(name, True, "G2 report accepted sparse MoE")


def check_owner_specialization_selected(
    specialization_record: Mapping[str, Any] | None,
) -> PreconditionCheckResult:
    """Precondition 2: "Vlastník vybral špecializáciu ... a výber je
    potvrdený nad rodičom z G2" -- a named DEC decision plus a `spec_id`
    actually registered in `reasoning.registry` (never a default,
    `registry.py` rule 7)."""
    name = "owner_specialization_selected"
    if specialization_record is None:
        return PreconditionCheckResult(name, False, "no specialization selection record supplied")
    dec_reference = specialization_record.get("dec_reference")
    spec_id = specialization_record.get("spec_id")
    if not dec_reference or not _DEC_REFERENCE_PATTERN.match(dec_reference):
        return PreconditionCheckResult(
            name, False, f"dec_reference {dec_reference!r} is not a DEC-<n> reference"
        )
    if not spec_id:
        return PreconditionCheckResult(name, False, "specialization_record names no spec_id")
    if spec_id not in reasoning_registry.registered_spec_ids():
        return PreconditionCheckResult(
            name, False, f"spec_id {spec_id!r} is not registered in reasoning.registry"
        )
    return PreconditionCheckResult(
        name, True, f"{spec_id!r} selected under {dec_reference!r} and registered"
    )


def check_release_has_reasoning_task_views(
    release_manifest: Mapping[str, Any] | None,
) -> PreconditionCheckResult:
    """Precondition 3: "Existuje ratifikované RFC003 erratum s novými
    task_kind ... a z neho postavený dataset release." `RFC003-ERRATA-003`
    is ratified; this only checks that a release manifest was actually
    built carrying at least one of its two `task_kind`s -- the erratum
    being ratified is not itself evidence a release exists."""
    name = "release_has_reasoning_task_views"
    if release_manifest is None:
        return PreconditionCheckResult(name, False, "no release manifest supplied")
    task_kinds = set(release_manifest.get("task_kinds", ()))
    present = task_kinds & REASONING_TASK_KINDS
    if not present:
        return PreconditionCheckResult(
            name,
            False,
            "release manifest carries none of RFC003-ERRATA-003's reasoning task_kinds "
            f"{sorted(REASONING_TASK_KINDS)}",
        )
    return PreconditionCheckResult(name, True, f"release manifest carries {sorted(present)}")


def check_parent_is_promotion_checkpoint(
    parent_ancestry: Mapping[str, Any] | None,
) -> PreconditionCheckResult:
    """Precondition 4: "Rodič je promotion checkpoint víťaza G2. Discovery
    checkpoint rodičom byť nesmie." `parent_ancestry` carries the same
    `run_class` field `checkpoint_ancestry.py`'s `_parent_entry` already
    puts on a `parents[0]` entry."""
    name = "parent_is_promotion_checkpoint"
    if parent_ancestry is None:
        return PreconditionCheckResult(name, False, "no parent checkpoint ancestry supplied")
    run_class = parent_ancestry.get("run_class")
    if run_class != "promotion":
        return PreconditionCheckResult(
            name,
            False,
            f"parent run_class is {run_class!r}, not 'promotion' -- a discovery "
            "checkpoint may not parent S6R (RFC005 §12.1)",
        )
    return PreconditionCheckResult(name, True, "parent is a promotion checkpoint")


def evaluate_s6r_preconditions(
    *,
    g2_report: Mapping[str, Any] | None,
    specialization_record: Mapping[str, Any] | None,
    release_manifest: Mapping[str, Any] | None,
    parent_ancestry: Mapping[str, Any] | None,
) -> S6RPreconditionsReport:
    """All four RFC005-ERRATA-007 E2 preconditions, each evaluated
    independently -- never short-circuited, so a caller always sees every
    unmet reason at once."""
    return S6RPreconditionsReport(
        checks=(
            check_g2_accepted_moe(g2_report),
            check_owner_specialization_selected(specialization_record),
            check_release_has_reasoning_task_views(release_manifest),
            check_parent_is_promotion_checkpoint(parent_ancestry),
        )
    )


def require_s6r_preconditions(
    *,
    g2_report: Mapping[str, Any] | None,
    specialization_record: Mapping[str, Any] | None,
    release_manifest: Mapping[str, Any] | None,
    parent_ancestry: Mapping[str, Any] | None,
) -> S6RPreconditionsReport:
    """`evaluate_s6r_preconditions`, raising `S6RPreconditionRefused` unless
    every precondition is satisfied. Returns the (fully satisfied) report
    on success, for a caller to embed in its run manifest."""
    report = evaluate_s6r_preconditions(
        g2_report=g2_report,
        specialization_record=specialization_record,
        release_manifest=release_manifest,
        parent_ancestry=parent_ancestry,
    )
    _require_satisfied(report, phase=None)
    return report


def _require_satisfied(report: S6RPreconditionsReport, *, phase: str | None) -> None:
    if report.all_satisfied:
        return
    unmet = report.unmet()
    where = f"S6R phase {phase} refused" if phase else "S6R refused"
    raise S6RPreconditionRefused(
        f"{where}: preconditions not satisfied: {[c.name for c in unmet]} -- "
        f"{[c.reason for c in unmet]}"
    )


# --------------------------------------------------------------------------
# Manifest identity
# --------------------------------------------------------------------------


def build_s6r_run_manifest(
    *,
    phase: str,
    preconditions_report: S6RPreconditionsReport,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """One S6R run manifest for `phase`, hashed and id'd under this
    module's own domains -- never S4's or S6's (`s4_run_manifest.py`,
    `s6_paired_exposure.py`)."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    base: dict[str, Any] = {
        "s6r_run_manifest_schema_version": S6R_RUN_MANIFEST_SCHEMA_VERSION,
        "stage": "S6R_reasoning_specialization",
        "phase": phase,
        "preconditions": preconditions_report.as_dict(),
        "payload": dict(payload),
    }
    manifest_hash = domain_hash(S6R_RUN_MANIFEST_DOMAIN, base)
    run_id = make_id(RUN_ID_PREFIX, RUN_ID_DOMAIN, {"s6r_run_manifest_hash": manifest_hash})
    return {**base, "s6r_run_manifest_hash": manifest_hash, "run_id": run_id}


# --------------------------------------------------------------------------
# R-A -- contract canary
# --------------------------------------------------------------------------


def run_phase_r_a(
    *,
    preconditions: S6RPreconditionsReport,
    module: SparseExpertAdapterWithReasoning,
    hidden: Tensor,
    visual_state: Tensor,
    attention_mask: Tensor,
    eligibility_mask: Tensor,
    gate_perturbations: Sequence[GatePerturbation],
    component_lock: Mapping[str, Any],
    available_weight_hashes: Sequence[str] = (),
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the existing R-A contract canary and attach it to an
    S6R-identified manifest. Never re-implements a single one of the
    canary's own checks -- see `reasoning_canary.run_r_a_contract_canary`.

    `dry_run=True` skips the precondition refusal (tests over the stub
    only); any real invocation must go through satisfied preconditions.
    """
    if not dry_run:
        _require_satisfied(preconditions, phase="R-A")
    canary_report = run_r_a_contract_canary(
        module=module,
        hidden=hidden,
        visual_state=visual_state,
        attention_mask=attention_mask,
        eligibility_mask=eligibility_mask,
        gate_perturbations=gate_perturbations,
        component_lock=component_lock,
        available_weight_hashes=available_weight_hashes,
    )
    manifest = build_s6r_run_manifest(
        phase="R-A",
        preconditions_report=preconditions,
        payload={
            "child_run_kind": canary_report.child_run_kind,
            "passed": canary_report.passed,
            "checks": [r.name for r in canary_report.results],
        },
    )
    return {"canary_report": canary_report, "manifest": manifest}


# --------------------------------------------------------------------------
# Parameter groups (R-B / R-D)
# --------------------------------------------------------------------------


def resolve_parameter_group(module: SparseExpertAdapterWithReasoning, group_id: str) -> Any:
    """RFC004 §24 parameter-group vocabulary, restricted to the groups S6R
    names on a `SparseExpertAdapterWithReasoning`: `reasoning.gate`,
    `reasoning.specializations.<spec_id>`, `expert_adapter.router`,
    `expert_adapter.specialists.<expert_id>`. Refuses
    (`UnknownParameterGroupError`) any other id, and any id naming a
    `spec_id`/`expert_id` `module` does not actually carry -- never a
    silently empty group."""
    if group_id == "reasoning.gate":
        return module.gate
    if group_id.startswith("reasoning.specializations."):
        spec_id = group_id[len("reasoning.specializations.") :]
        if module.reasoning_module.spec_id != spec_id:
            raise UnknownParameterGroupError(
                f"{group_id!r} names spec_id {spec_id!r}; module carries "
                f"{module.reasoning_module.spec_id!r}"
            )
        return module.reasoning_module
    if group_id == "expert_adapter.router":
        return module.sparse_adapter.router
    if group_id.startswith("expert_adapter.specialists."):
        expert_id = group_id[len("expert_adapter.specialists.") :]
        try:
            index = module.sparse_adapter.expert_ids.index(expert_id)
        except ValueError:
            raise UnknownParameterGroupError(
                f"{group_id!r} names expert_id {expert_id!r}; module carries "
                f"{module.sparse_adapter.expert_ids!r}"
            ) from None
        return module.sparse_adapter.specialists[index]
    raise UnknownParameterGroupError(f"unknown S6R parameter group id {group_id!r}")


def freeze_to_groups(
    module: SparseExpertAdapterWithReasoning, *, group_ids: Sequence[str]
) -> frozenset[int]:
    """Freeze every parameter of `module` except those in `group_ids`
    (RFC005-ERRATA-007 E2 R-B/R-D: "všetko ostatné je zmrazené"). Returns
    the set of Python `id()`s left trainable, for a caller to verify
    against after a step."""
    if not group_ids:
        raise ValueError("group_ids must name at least one trainable group")
    trainable_ids: set[int] = set()
    for group_id in group_ids:
        trainable_ids.update(id(p) for p in resolve_parameter_group(module, group_id).parameters())
    for p in module.parameters():
        p.requires_grad_(id(p) in trainable_ids)
    return frozenset(trainable_ids)


def r_b_group_ids(
    module: SparseExpertAdapterWithReasoning, *, named_specialist_expert_id: str | None = None
) -> tuple[str, ...]:
    """R-B's trainable groups: `reasoning.gate` +
    `reasoning.specializations.<spec_id>` (from `module` itself, never
    hardcoded), plus one optional policy-named
    `expert_adapter.specialists.<expert_id>`."""
    group_ids = ["reasoning.gate", f"reasoning.specializations.{module.reasoning_module.spec_id}"]
    if named_specialist_expert_id is not None:
        group_ids.append(f"expert_adapter.specialists.{named_specialist_expert_id}")
    return tuple(group_ids)


def build_r_b_stage_policy_draft(
    *, named_specialist_expert_id: str | None = None, low_lr: float = 1e-5
) -> dict[str, Any]:
    """R-B draft policy -- `is_placeholder: True` until an owner freezes it
    before R-B's first gradient (RFC005-ERRATA-007 E4)."""
    return {
        "s6r_stage_policy_schema_version": S6R_STAGE_POLICY_SCHEMA_VERSION,
        "stage": "S6R_reasoning_specialization",
        "sub_step": "R-B",
        "named_specialist_expert_id": named_specialist_expert_id,
        "low_lr": low_lr,
        "is_placeholder": True,
    }


def run_frozen_group_dry_run_step(
    module: SparseExpertAdapterWithReasoning,
    *,
    hidden: Tensor,
    visual_state: Tensor,
    attention_mask: Tensor,
    eligibility_mask: Tensor,
    group_ids: Sequence[str],
    lr: float,
) -> dict[str, Any]:
    """One CPU-only optimizer step over exactly `group_ids`' parameters,
    proving every other parameter is bit-identical before and after
    (RFC005-ERRATA-007 E2: "nulová zmena zmrazených skupín sa overuje
    každým krokom", §24). A discovery-class CPU smoke step only -- never a
    claim about a real R-B/R-D training loop."""
    freeze_to_groups(module, group_ids=group_ids)
    frozen = [(name, p) for name, p in module.named_parameters() if not p.requires_grad]
    trainable = [p for p in module.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(f"group_ids {list(group_ids)!r} resolved to zero trainable parameters")
    before = [p.detach().clone() for _, p in frozen]

    optimizer = torch.optim.SGD(trainable, lr=lr)
    result = module(
        hidden,
        visual_state=visual_state,
        attention_mask=attention_mask,
        eligibility_mask=eligibility_mask,
    )
    loss = result.hidden.pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    frozen_params_unchanged = all(
        torch.equal(before_value, after_param.detach())
        for before_value, (_, after_param) in zip(before, frozen, strict=True)
    )
    return {"loss": float(loss.item()), "frozen_params_unchanged": frozen_params_unchanged}


# --------------------------------------------------------------------------
# R-C -- hard examples + replay
# --------------------------------------------------------------------------


def build_r_c_stage_policy_draft(*, replay_fraction: float = 0.2) -> dict[str, Any]:
    """RFC005-ERRATA-007 E2 R-C: replay ratio is `[OPEN]`; this draft
    mirrors DEC-034's 80% real / 20% synthetic split as a starting point,
    not a frozen number (`is_placeholder: True`)."""
    if not 0.0 < replay_fraction < 1.0:
        raise ValueError(f"replay_fraction must be in (0, 1), got {replay_fraction!r}")
    return {
        "s6r_stage_policy_schema_version": S6R_STAGE_POLICY_SCHEMA_VERSION,
        "stage": "S6R_reasoning_specialization",
        "sub_step": "R-C",
        "replay_fraction": replay_fraction,
        "is_placeholder": True,
    }


def build_r_c_mix(
    *,
    hard_examples: Sequence[Any],
    replay_examples: Sequence[Any],
    policy: Mapping[str, Any],
    seed: int,
) -> list[tuple[str, Any]]:
    """A deterministic `[("hard", ex), ("replay", ex), ...]` mix honouring
    `policy["replay_fraction"]`: every hard example is included once, and
    enough replay examples are drawn (in order, never resampled) to make
    up `replay_fraction` of the full mix, then the whole sequence is
    shuffled with a `random.Random(seed)` -- same `seed` and inputs always
    produce the same order (RFC005-ERRATA-007 E2 R-C: frozen hard-example
    artefact + replay of the ordinary mix, run together)."""
    if not hard_examples:
        raise ValueError("R-C requires at least one hard example (§30.5, frozen producer artefact)")
    if not replay_examples:
        raise ValueError("R-C requires at least one replay example")
    replay_fraction = policy["replay_fraction"]
    if not 0.0 < replay_fraction < 1.0:
        raise ValueError(f"policy replay_fraction must be in (0, 1), got {replay_fraction!r}")

    num_hard = len(hard_examples)
    num_replay = round(num_hard * replay_fraction / (1.0 - replay_fraction))
    num_replay = max(1, min(num_replay, len(replay_examples)))

    mix: list[tuple[str, Any]] = [("hard", ex) for ex in hard_examples]
    mix.extend(("replay", ex) for ex in replay_examples[:num_replay])
    random.Random(seed).shuffle(mix)
    return mix


# --------------------------------------------------------------------------
# R-D -- joint low-LR unfreeze
# --------------------------------------------------------------------------


def build_r_d_stage_policy_draft(
    *, unfrozen_group_ids: Sequence[str], low_lr: float = 1e-5
) -> dict[str, Any]:
    """R-D draft policy: the groups a promotion cycle unfreezes must be
    *named*, never inherited from R-B (RFC005 §21.2: a new promotion
    cycle). `is_placeholder: True` until an owner freezes it."""
    if not unfrozen_group_ids:
        raise ValueError("R-D policy must name at least one unfrozen group")
    return {
        "s6r_stage_policy_schema_version": S6R_STAGE_POLICY_SCHEMA_VERSION,
        "stage": "S6R_reasoning_specialization",
        "sub_step": "R-D",
        "unfrozen_group_ids": tuple(unfrozen_group_ids),
        "low_lr": low_lr,
        "is_placeholder": True,
    }


def run_r_d_dry_run_step(
    module: SparseExpertAdapterWithReasoning,
    *,
    hidden: Tensor,
    visual_state: Tensor,
    attention_mask: Tensor,
    eligibility_mask: Tensor,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """R-D's own dry-run step, over exactly `policy["unfrozen_group_ids"]`
    at `policy["low_lr"]` -- shares `run_frozen_group_dry_run_step` with
    R-B, but never shares R-B's group list."""
    return run_frozen_group_dry_run_step(
        module,
        hidden=hidden,
        visual_state=visual_state,
        attention_mask=attention_mask,
        eligibility_mask=eligibility_mask,
        group_ids=tuple(policy["unfrozen_group_ids"]),
        lr=policy["low_lr"],
    )


# --------------------------------------------------------------------------
# R-E -- comparison arms
# --------------------------------------------------------------------------


def build_r_e_arms(
    *,
    exposure_plan: Mapping[str, Any],
    dense_architecture_manifest_ref: str,
    g2_winner_architecture_manifest_ref: str,
    reasoning_spec_id: str,
    include_capacity_control: bool = False,
    capacity_control_architecture_manifest_ref: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """RFC005-ERRATA-007 E2 R-E arms A/B/C(+D), all sharing `exposure_plan`
    verbatim so they differ only in `arm_id`, `architecture_manifest_ref`,
    `reasoning_enabled` and `reasoning_spec_id`:

    - **A** (`dense_architecture_manifest_ref`, reasoning off) -- dense control.
    - **B** (`g2_winner_architecture_manifest_ref`, reasoning off) -- the G2
      winner retrained on the *same* exposure plan with the reasoning gate
      forced to `g ≡ 0` (RFC004-ERRATA-005 E1 rule 1's architectural
      bypass); controls for exposure, not just parameter count.
    - **C** (`g2_winner_architecture_manifest_ref`, reasoning on,
      `reasoning_spec_id`) -- the experiment arm.
    - **D** (optional, `[OPEN]`) -- a capacity control with the same added
      parameter count as C's named specialist but no gate.
    """
    if include_capacity_control and capacity_control_architecture_manifest_ref is None:
        raise ValueError(
            "include_capacity_control=True requires "
            "capacity_control_architecture_manifest_ref (arm D is [OPEN], RFC005-"
            "ERRATA-007 E2 R-E)"
        )
    if not include_capacity_control and capacity_control_architecture_manifest_ref is not None:
        raise ValueError(
            "capacity_control_architecture_manifest_ref was given but "
            "include_capacity_control is False"
        )

    def _arm(
        arm_id: str,
        architecture_manifest_ref: str,
        *,
        reasoning_enabled: bool,
        reasoning_spec_id_value: str | None,
    ) -> dict[str, Any]:
        arm: dict[str, Any] = {
            "s6r_arm_schema_version": S6R_ARM_SCHEMA_VERSION,
            "arm_id": arm_id,
            "architecture_manifest_ref": architecture_manifest_ref,
            "reasoning_enabled": reasoning_enabled,
            "reasoning_spec_id": reasoning_spec_id_value,
            "exposure_plan": dict(exposure_plan),
        }
        arm["s6r_arm_hash"] = domain_hash(S6R_ARM_DOMAIN, arm)
        return arm

    arms = [
        _arm(
            ARM_A_DENSE_CONTROL,
            dense_architecture_manifest_ref,
            reasoning_enabled=False,
            reasoning_spec_id_value=None,
        ),
        _arm(
            ARM_B_G2_WINNER_NO_REASONING,
            g2_winner_architecture_manifest_ref,
            reasoning_enabled=False,
            reasoning_spec_id_value=None,
        ),
        _arm(
            ARM_C_G2_WINNER_WITH_REASONING,
            g2_winner_architecture_manifest_ref,
            reasoning_enabled=True,
            reasoning_spec_id_value=reasoning_spec_id,
        ),
    ]
    if include_capacity_control:
        assert capacity_control_architecture_manifest_ref is not None
        arms.append(
            _arm(
                ARM_D_CAPACITY_CONTROL,
                capacity_control_architecture_manifest_ref,
                reasoning_enabled=False,
                reasoning_spec_id_value=None,
            )
        )
    return tuple(arms)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s6r_train",
        description=(
            "S6R_reasoning_specialization dormant phase runner (RFC005-ERRATA-007 "
            "E2). Every real invocation is refused until all four preconditions "
            "are satisfied -- see --preconditions-report."
        ),
    )
    parser.add_argument("--g2-report", type=Path, default=None)
    parser.add_argument("--specialization-record", type=Path, default=None)
    parser.add_argument("--release-manifest", type=Path, default=None)
    parser.add_argument("--parent-ancestry", type=Path, default=None)
    parser.add_argument(
        "--preconditions-report",
        action="store_true",
        help="print the four preconditions and exit without running any phase",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="acknowledge the in-process stub path only (tests) -- no real "
        "phase runner reads a dataset or checkpoint from this flag today",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report = evaluate_s6r_preconditions(
        g2_report=_load_json(args.g2_report),
        specialization_record=_load_json(args.specialization_record),
        release_manifest=_load_json(args.release_manifest),
        parent_ancestry=_load_json(args.parent_ancestry),
    )

    if args.preconditions_report:
        print(json.dumps(report.as_dict(), indent=1, sort_keys=True))
        return 0 if report.all_satisfied else 1

    if not report.all_satisfied:
        print(json.dumps(report.as_dict(), indent=1, sort_keys=True), file=sys.stderr)
        print(
            "S6R preconditions are not satisfied -- refusing to run any phase "
            "(RFC005-ERRATA-007 E2).",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print(
            "preconditions satisfied, but no real S6R phase runner is wired to a "
            "dataset or checkpoint yet -- this flag only acknowledges the stub path"
        )
        return 0

    print(
        "S6R has no real phase runner wired to data/checkpoints yet; only "
        "--preconditions-report and --dry-run are supported",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
