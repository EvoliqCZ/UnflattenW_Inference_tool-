"""S4.5 allowed-state map (RFC005-ERRATA-010 E4.2; dormant).

Derived, never restated: the group universe, the ``not_applicable`` ("off")
groups and the frozen backbone slices are read from
``stage_state_policy.stage_allowed_states("s4", run_class)`` and each S4.5
state must be a state S4 itself allows for that group (or a refusal is
raised). Nothing here is registered in ``stage_state_policy`` (``STAGES`` and
``_STAGE_STATE_BUILDERS`` stay closed) and no group is added to the trainable
group registry; the sparse MoE router/shared/specialist groups are not in the
dense registry at all, so they cannot appear.

Narrowing rules (ERRATA-010 E4.2, one group = one state set):

* trainable: ``tree_text_decoder``, ``expert_adapter.dense`` and the explicitly
  selected ``typed_heads.*``;
* ``backbone.visual`` / ``backbone.language``: frozen, active in the forward;
* ``node_visual_aligner``: trainable only through a separate supervised
  auxiliary loss, never through ``L_rl`` (``SUPERVISED_AUXILIARY_ONLY``);
  continuous typed heads follow the same rule at loss level;
* every other S4-trainable group the row does not name (``adapter.projection``,
  ``document_context_encoder``, ``table_plan``, ``resource_recovery``,
  unselected ``typed_heads.*``) is frozen: this is a deliberate narrowing of S4.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from unflatten.model.hashing import domain_hash
from unflatten.model.rl.ids import STAGE_POLICY_DOMAIN
from unflatten.model.stage_state_policy import RUN_CLASSES, stage_allowed_states

PARENT_STAGE: Final = "s4"
RL_TRAINABLE_GROUPS: Final = ("tree_text_decoder", "expert_adapter.dense")
ACTIVE_FROZEN_GROUPS: Final = ("backbone.visual", "backbone.language")
SUPERVISED_AUXILIARY_ONLY: Final = ("node_visual_aligner",)
_TYPED_HEAD_PREFIX: Final = "typed_heads."
_TRAINABLE: Final = "trainable"
_FROZEN: Final = "frozen"
_OFF: Final = "not_applicable"


class StateMapInvalid(ValueError):
    """The S4.5 state map cannot be derived from S4 or a state is not allowed."""


def _require_s4_allows(group: str, s4_states: frozenset[str], state: str) -> None:
    if state not in s4_states:
        raise StateMapInvalid(
            f"{group}: S4.5 needs state {state!r} but S4 allows only {sorted(s4_states)}"
        )


def _check_selection(universe: Iterable[str], selected: Iterable[str]) -> frozenset[str]:
    known = set(universe)
    chosen = frozenset(selected)
    for group in sorted(chosen):
        if group not in known:
            raise StateMapInvalid(f"unknown group {group!r} selected as a typed head")
        if not group.startswith(_TYPED_HEAD_PREFIX):
            raise StateMapInvalid(f"{group!r} is not a typed_heads.* group")
    return chosen


def s45_allowed_states(
    run_class: str, *, selected_typed_heads: Iterable[str] = ()
) -> dict[str, frozenset[str]]:
    if run_class not in RUN_CLASSES:
        raise ValueError(f"run_class {run_class!r} not in {RUN_CLASSES}")
    s4 = stage_allowed_states(PARENT_STAGE, run_class)
    selected = _check_selection(s4, selected_typed_heads)
    trainable = set(RL_TRAINABLE_GROUPS) | selected
    out: dict[str, frozenset[str]] = {}
    for group, s4_states in s4.items():
        if s4_states == frozenset({_OFF}):
            out[group] = s4_states
        elif group in ACTIVE_FROZEN_GROUPS:
            _require_s4_allows(group, s4_states, _FROZEN)
            out[group] = frozenset({_FROZEN})
        elif group in trainable:
            _require_s4_allows(group, s4_states, _TRAINABLE)
            out[group] = frozenset({_TRAINABLE})
        elif group in SUPERVISED_AUXILIARY_ONLY:
            _require_s4_allows(group, s4_states, _TRAINABLE)
            out[group] = frozenset({_FROZEN, _TRAINABLE})
        else:
            out[group] = frozenset({_FROZEN})
    missing = [g for g in RL_TRAINABLE_GROUPS + ACTIVE_FROZEN_GROUPS if g not in out]
    if missing:
        raise StateMapInvalid(f"S4 does not name required group(s) {missing}")
    return out


def s45_trainable_groups(
    run_class: str, *, selected_typed_heads: Iterable[str] = ()
) -> frozenset[str]:
    """Groups an ``L_rl`` gradient may reach (auxiliary-only groups excluded)."""
    states = s45_allowed_states(run_class, selected_typed_heads=selected_typed_heads)
    return frozenset(
        g
        for g, s in states.items()
        if s == frozenset({_TRAINABLE}) and g not in SUPERVISED_AUXILIARY_ONLY
    )


def validate_s45_state(
    group: str,
    state: str,
    run_class: str = "promotion",
    *,
    selected_typed_heads: Iterable[str] = (),
) -> None:
    states = s45_allowed_states(run_class, selected_typed_heads=selected_typed_heads)
    if group not in states:
        raise StateMapInvalid(f"unknown group {group!r}")
    if state not in states[group]:
        raise StateMapInvalid(
            f"{group}: state {state!r} not allowed in S4.5, allowed {sorted(states[group])}"
        )


def s45_state_map_hash(run_class: str, selected_typed_heads: Iterable[str] = ()) -> str:
    states = s45_allowed_states(run_class, selected_typed_heads=selected_typed_heads)
    return domain_hash(
        STAGE_POLICY_DOMAIN,
        {
            "kind": "allowed_state_map",
            "parent_stage": PARENT_STAGE,
            "run_class": run_class,
            "allowed_states": {g: sorted(s) for g, s in sorted(states.items())},
        },
    )
