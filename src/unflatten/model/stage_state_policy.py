"""RFC005 stage-dependent trainable/frozen policy (S1 / S2 / S3).

The closed trainable-group registry
(:mod:`unflatten.model.trainable_group_registry`) pins S1 exactly. S2 and S3
change which groups may be trainable, but the RFC text does not amend the
registry (RFC005-ERRATA-002 states explicitly that no code, no
``TRAINABLE_GROUP_REGISTRY_HASH`` and no cache/manifest is touched by the
erratum). This module is the additive, stage-scoped layer on top of the
registry: it never rewrites S1 and it never touches
``TRAINABLE_GROUP_REGISTRY_HASH`` — it binds to it instead, so any registry
change invalidates this policy's hash by construction.

Normative sources per stage
----------------------------
- S1: ``trainable_group_registry.TrainableGroup.s1_allowed_states`` — the
  S1 column is *derived* from the registry, not restated, so S1 invariance
  holds by construction.
- S2: RFC005 §15.3 (trainable/frozen policy of the structure & exact-text
  bootstrap).
- S3: RFC005 §16.3 as replaced by RFC005-ERRATA-002 E3 **[DECIDED, DEC-021]**
  (the ratified S3 trainable list), plus the unchanged backbone sentence
  ("Backbone zostáva frozen v referenčnom passe…").
- S4: same trainable/frozen policy as S3 — a single-page scale-up run (RFC005
  §17 as rescoped by DEC-033/DEC-035) that continues training all S3-trainable
  groups on the v14 release with the 80/20 real/synthetic data mix — EXCEPT
  that S4 `promotion` alone additionally allows `resource_recovery` to stay
  `frozen` (RFC005-ERRATA-005 E2: it is deterministic, no trainable
  parameters, so the literal S3 promotion rule would refuse every model).

Documented inferences (resolved here, not guessed silently)
-----------------------------------------------------------
1. ``table_plan`` is named by neither the S2 trainable nor frozen list. It
   keeps its S1 allowed states (``frozen``, ``diagnostic_only``) — i.e. it is
   not trainable in S2. This matches §15.2's "table_v0 (structure/text
   projection only)" and the §15.1 stage goal; E3 makes it trainable in S3.
2. ``visual_compressor`` / ``speculators`` / ``confidence_heads`` appear in
   the §15.3 frozen list, but the already-ratified registry + capability
   profile resolve the dense track as *never instantiating* them
   (``kind = "not_applicable"``; the profile pins the literal
   ``"not_applicable"``). The registry's ambiguity resolution #1 governs:
   they stay ``not_applicable`` in S2 and S3, never ``frozen``.
3. ``backbone.projector`` is trainable in S2 *iff* the adapter profile
   declares it as a standalone compatible group (§15.3). This policy allows
   both states in S2; the profile gate — not this module — enforces the
   condition at run binding. In S3 the unchanged backbone sentence keeps it
   frozen (E3 does not list it).
4. LoRA delta groups stay ``not_applicable`` in every stage: the dense
   track runs before any LoRA decision, and no ratified text promotes them.
5. Learning-rate class: the policy declares ``standard`` / ``reduced`` per
   trainable group; the concrete rates are pinned by the future S2/S3 run
   manifests. S2: ``node_visual_aligner`` is the low-LR alignment bootstrap.
   S3: ``tree_text_decoder`` gets "nižší LR než nové hlavy" (E3);
   ``node_visual_aligner`` carries its S2 low-LR bootstrap forward (E3 is
   silent on it, so it does not jump to standard).
"""

from __future__ import annotations

from typing import Any

from unflatten.model.hashing import domain_hash
from unflatten.model.trainable_group_registry import (
    TRAINABLE_GROUP_REGISTRY_HASH,
    trainable_group_registry,
)

_POLICY_DOMAIN = "rfc005.stage-state-policy/v1"
SCHEMA_VERSION = "0.5.0"  # 0.5.0: S4 promotion resource_recovery frozen (RFC005-ERRATA-005 E2)

#: The closed stage set. No other stage may ever validate.
STAGES: tuple[str, ...] = ("s1", "s2", "s3", "s4")

STANDARD = "standard"
REDUCED = "reduced"

#: The closed learning-rate class set.
LR_CLASSES: tuple[str, ...] = (STANDARD, REDUCED)

_TRAINABLE = "trainable"

#: S2 explicit state overrides (RFC005 §15.3). Groups absent from this map
#: keep their S1 (registry) allowed states.
_S2_OVERRIDES: dict[str, frozenset[str]] = {
    # "backbone.projector, ak ho adapter profile deklaruje ako samostatný
    # compatible group" — both states allowed here; the bound profile
    # gate decides (see docstring inference 3).
    "backbone.projector": frozenset({"frozen", _TRAINABLE}),
    # "node_visual_aligner — low-LR alignment bootstrap". The aligner is now
    # a real, instantiated frozen_base child (DEC-027), so it rests `frozen`
    # until S2 trains it; `frozen` is therefore admissible here alongside
    # `trainable` (the intended S2 bootstrap) and `not_applicable`
    # (RFC005-ERRATA-003 E2, DEC-027 D4, while the dense track does not
    # instantiate it). S2 promotion admission requires `trainable` -- this
    # policy knows no run class, so that split lives in the admission.
    "node_visual_aligner": frozenset({_TRAINABLE, "frozen", "not_applicable"}),
    # "Frozen: all typed_heads.*"
    "typed_heads.inline_image": frozenset({"frozen"}),
    "typed_heads.anchor": frozenset({"frozen"}),
    "typed_heads.table": frozenset({"frozen"}),
    "typed_heads.row_cell": frozenset({"frozen"}),
    # "Frozen: resource_recovery"
    "resource_recovery": frozenset({"frozen"}),
}

#: E3's ratified S3 trainable list, exactly (RFC005-ERRATA-002 E3, DEC-021).
_S3_TRAINABLE: frozenset[str] = frozenset(
    {
        "node_visual_aligner",
        "typed_heads.anchor",
        "typed_heads.inline_image",
        "typed_heads.row_cell",
        "typed_heads.table",
        "table_plan",
        "resource_recovery",
        "adapter.projection",
        "document_context_encoder",
        "tree_text_decoder",
        "expert_adapter.dense",
    }
)

#: The unchanged backbone sentence keeps all three backbone slices frozen in
#: the S3 reference pass.
_S3_FROZEN: frozenset[str] = frozenset(
    {"backbone.visual", "backbone.language", "backbone.projector"}
)

#: RFC005-ERRATA-004 E2: relaxed S3 **discovery** states for groups the dense
#: track does not yet instantiate as trainable, plus typed-head groups the
#: pinned census shows below the supervision floor. Applied ONLY when the
#: caller identifies the run as S3 discovery (``run_class="discovery"``);
#: S3 **promotion** keeps the strict `_S3_TRAINABLE` requirement of
#: `trainable` for every one of these groups.
_S3_DISCOVERY_OVERRIDES: dict[str, frozenset[str]] = {
    # The aligner is an instantiated frozen_base child (DEC-027), so it may
    # rest `frozen` in S3 discovery too (trainable once S2 has bootstrapped
    # it); `not_applicable` is kept for the dense track that does not
    # instantiate it.
    "node_visual_aligner": frozenset({_TRAINABLE, "frozen", "not_applicable"}),
    "resource_recovery": frozenset({_TRAINABLE, "frozen"}),
    "typed_heads.anchor": frozenset({_TRAINABLE, "frozen"}),
    "typed_heads.inline_image": frozenset({_TRAINABLE, "frozen"}),
    "typed_heads.row_cell": frozenset({_TRAINABLE, "frozen"}),
    "typed_heads.table": frozenset({_TRAINABLE, "frozen"}),
}

#: The closed run-class set understood by the S3 discovery/promotion split.
#: ``None`` behaves like ``"promotion"`` (the strict, ERRATA-002 E3 states) —
#: the relaxed states are opt-in, never a silent default.
#: S4 reuses S3 discovery overrides for the same groups.
RUN_CLASSES: tuple[str, ...] = ("discovery", "promotion")

#: RFC005-ERRATA-005 E2: `resource_recovery` is a deterministic crop/verify
#: step with no trainable parameters (`trainable_group_registry`: kind
#: `frozen_base`, allowed states only `frozen`/`not_applicable`); the
#: literal ERRATA-002 E3 promotion rule ("trainable for every group in
#: `_S3_TRAINABLE`") would refuse EVERY S4 promotion model over an
#: unreachable state. S4 promotion alone may leave it `frozen`; every other
#: `_S3_TRAINABLE` group keeps the strict promotion requirement unchanged,
#: and S3 promotion is not touched by this override at all.
_S4_PROMOTION_OVERRIDES: dict[str, frozenset[str]] = {
    "resource_recovery": frozenset({_TRAINABLE, "frozen"}),
}

#: Trainable groups carrying the reduced LR class, per stage (docstring
#: inference 5).
_LR_REDUCED: dict[str, frozenset[str]] = {
    "s1": frozenset(),
    "s2": frozenset({"node_visual_aligner"}),
    "s3": frozenset({"node_visual_aligner", "tree_text_decoder"}),
    "s4": frozenset({"node_visual_aligner", "tree_text_decoder"}),
}


def _s1_states() -> dict[str, frozenset[str]]:
    return {
        group.group_id: group.s1_allowed_states
        for group in trainable_group_registry().values()
    }


def _s2_states() -> dict[str, frozenset[str]]:
    return {g: _S2_OVERRIDES.get(g, states) for g, states in _s1_states().items()}


def _s3_states() -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for group_id, s1_states in _s1_states().items():
        if group_id in _S3_TRAINABLE:
            out[group_id] = frozenset({_TRAINABLE})
        elif group_id in _S3_FROZEN:
            out[group_id] = frozenset({"frozen"})
        else:
            out[group_id] = s1_states
    return out


_STAGE_STATE_BUILDERS: dict[str, Any] = {
    "s1": _s1_states,
    "s2": _s2_states,
    "s3": _s3_states,
    "s4": _s3_states,  # S4 uses same trainable/frozen policy as S3
}

# Built once at import; frozenset values are immutable and shared.
_STAGE_STATES: dict[str, dict[str, frozenset[str]]] = {
    stage: builder() for stage, builder in _STAGE_STATE_BUILDERS.items()
}


def _require_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(
            f"stage {stage!r} is not one of the closed stage set {sorted(STAGES)}"
        )


def _require_run_class(run_class: str | None) -> None:
    if run_class is not None and run_class not in RUN_CLASSES:
        raise ValueError(
            f"run_class {run_class!r} is not one of {RUN_CLASSES} or None"
        )


def stage_allowed_states(
    stage: str, run_class: str | None = None
) -> dict[str, frozenset[str]]:
    """The closed allowed-state map for `stage`, keyed by exact group ID.

    `run_class` only affects ``stage="s3"`` or ``stage="s4"``:
    ``"discovery"`` applies the RFC005-ERRATA-004 E2 relaxed states
    (`_S3_DISCOVERY_OVERRIDES`); ``None`` (the default) and ``"promotion"``
    are identical — the strict ERRATA-002 E3 states, where every group in
    `_S3_TRAINABLE` allows only ``trainable`` — EXCEPT at ``stage="s4"``,
    where RFC005-ERRATA-005 E2 additionally allows `resource_recovery` to
    be ``frozen`` (`_S4_PROMOTION_OVERRIDES`); every other group keeps the
    strict requirement. S3 promotion is unchanged by E2.

    Raises ``ValueError`` for a stage outside the closed stage set or a
    `run_class` outside `RUN_CLASSES`.
    """
    _require_stage(stage)
    _require_run_class(run_class)
    base = _STAGE_STATES[stage]
    if stage in ("s3", "s4") and run_class == "discovery":
        return {
            group_id: _S3_DISCOVERY_OVERRIDES.get(group_id, states)
            for group_id, states in base.items()
        }
    if stage == "s4" and run_class != "discovery":
        return {
            group_id: _S4_PROMOTION_OVERRIDES.get(group_id, states)
            for group_id, states in base.items()
        }
    return base


def stage_trainable_groups(
    stage: str, run_class: str | None = None
) -> frozenset[str]:
    """Groups whose allowed states include ``trainable`` at `stage`.

    For S2 this includes ``backbone.projector``, whose trainability is
    additionally conditioned on the bound adapter profile (docstring
    inference 3) — the profile gate, not this set, makes the final call.
    """
    return frozenset(
        g
        for g, states in stage_allowed_states(stage, run_class).items()
        if _TRAINABLE in states
    )


def stage_lr_class(stage: str, group_id: str, run_class: str | None = None) -> str:
    """The declared learning-rate class for a trainable group at `stage`.

    Returns :data:`STANDARD` or :data:`REDUCED`. Raises ``ValueError`` if
    `group_id` is not trainable at `stage` (a frozen / not_applicable group
    has no learning-rate class) and ``KeyError`` for an unknown group.
    """
    _require_stage(stage)
    states = stage_allowed_states(stage, run_class)
    if group_id not in states:
        raise KeyError(group_id)
    if _TRAINABLE not in states[group_id]:
        raise ValueError(
            f"group {group_id!r} is not trainable at {stage!r} "
            f"(allowed states: {sorted(states[group_id])}); it has no lr class"
        )
    return REDUCED if group_id in _LR_REDUCED[stage] else STANDARD


def validate_stage_state(
    stage: str, group_id: str, state: str, run_class: str | None = None
) -> None:
    """Raise if `state` is not allowed for `group_id` at `stage`.

    ``ValueError`` for an unknown stage, an unknown `run_class`, or a
    disallowed state; ``KeyError`` for a group outside the closed registry.
    """
    _require_stage(stage)
    states = stage_allowed_states(stage, run_class)
    if group_id not in states:
        raise KeyError(group_id)
    if state not in states[group_id]:
        raise ValueError(
            f"group {group_id!r} may not be assigned {stage!r} state {state!r}; "
            f"allowed states are {sorted(states[group_id])}"
        )


def stage_state_policy_projection() -> dict[str, Any]:
    """Deterministic hash projection of the stage-state policy."""
    return {
        "stage_state_policy_schema_version": SCHEMA_VERSION,
        "derived_from_trainable_group_registry_hash": TRAINABLE_GROUP_REGISTRY_HASH,
        "stages": {
            stage: {
                "allowed_states": {
                    group_id: sorted(states)
                    for group_id, states in sorted(
                        _STAGE_STATES[stage].items()
                    )
                },
                "lr_class": {
                    group_id: stage_lr_class(stage, group_id)
                    for group_id in sorted(stage_trainable_groups(stage))
                },
            }
            for stage in sorted(STAGES)
        },
    }


STAGE_STATE_POLICY_HASH: str = domain_hash(
    _POLICY_DOMAIN, stage_state_policy_projection()
)
