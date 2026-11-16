"""S4.5 RL training policy: the hashed home of every gradient-affecting number
(RFC005-ERRATA-010 E7.1/E7.4, DEC-047 addendum A v2; PROPOSED, dormant).

Without this artefact the optimizer, schedule, clipping and loss coefficients
live in no hash, so a ratification of the composite policy would not bind them.
The document is closed (``schemas/rl_training_policy.schema.json``), self-hashing
under ``TRAINING_POLICY_DOMAIN`` and enters ``composite_policy_hash`` as the
sixth key.

* Every number is a required keyword with no default. ``None`` means "not
  frozen yet": legal in a PROPOSED policy, refused in a RATIFIED one -- the same
  shape as the reward policy's null severe floors. A float is refused by name.
* ``lr_micro`` is per trainable group id (units of 1e-6, as in S1). The ids are
  checked by name against :func:`s45_allowed_states`, never against a pinned
  registry: every group an ``L_rl`` gradient may reach needs an LR (no silent
  default), and no frozen / off group may carry one.
* ``execution_profile`` must equal the rollout policy's: rollout and update are
  one distribution (D2/D9). :func:`check_execution_profile_matches` is the one
  comparison.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Final

from unflatten.model.rl.ids import TRAINING_POLICY_DOMAIN
from unflatten.model.rl.loss import RlLossPolicy
from unflatten.model.rl.policies import (
    RlPolicyInvalid,
    _reject_floats,
    _require_status,
    _seal,
    _verify,
    validate_execution_profile,
)
from unflatten.model.rl.state_map import s45_allowed_states, s45_trainable_groups

SCHEMA_VERSION: Final = "1.0.0"
TRAINING_HASH_FIELD: Final = "training_policy_hash"
_SCHEMA: Final = "rl_training_policy"
OPTIMIZER: Final = "adamw"
SCHEDULES: Final = ("constant_after_warmup", "linear_warmup_cosine")
_LOSS_KEYS: Final = (
    "clip_epsilon_micro",
    "kl_coefficient_micro",
    "length_normalisation",
    "gate_failed_reward_micro",
)


def _placeholders(doc: Mapping[str, Any]) -> list[str]:
    out = [k for k, v in doc.items() if v is None]
    out += [f"lr_micro.{g}" for g, v in doc["lr_micro"].items() if v is None]
    out += [f"loss_policy.{k}" for k, v in doc["loss_policy"].items() if v is None]
    return sorted(out)


def _check_semantics(doc: Mapping[str, Any]) -> None:
    nulls = _placeholders(doc)
    if doc["status"] == "RATIFIED" and nulls:
        raise RlPolicyInvalid(f"RATIFIED training policy carries unfrozen field(s) {nulls}")
    warmup, steps = doc["warmup_steps"], doc["max_update_steps"]
    if warmup is not None and steps is not None and warmup > steps:
        raise RlPolicyInvalid(f"warmup_steps {warmup} exceeds max_update_steps {steps}")


def build_rl_training_policy(
    *,
    weight_decay_micro: int | None,
    lr_micro: Mapping[str, int | None],
    warmup_steps: int | None,
    schedule: str | None,
    grad_clip_micro: int | None,
    accumulation_steps: int | None,
    max_update_steps: int | None,
    checkpoint_every: int | None,
    clip_epsilon_micro: int | None,
    kl_coefficient_micro: int | None,
    length_normalisation: str | None,
    gate_failed_reward_micro: int | None,
    supervised_weight_micro: int | None,
    execution_profile: Mapping[str, Any],
    run_class: str,
    selected_typed_heads: Iterable[str] = (),
    status: str = "PROPOSED",
) -> dict[str, Any]:
    """Seal a training policy. ``run_class`` / ``selected_typed_heads`` only
    select the S4.5 group universe the ``lr_micro`` keys are checked against."""
    _require_status(status)
    numbers = {
        "weight_decay_micro": weight_decay_micro,
        "warmup_steps": warmup_steps,
        "grad_clip_micro": grad_clip_micro,
        "accumulation_steps": accumulation_steps,
        "max_update_steps": max_update_steps,
        "checkpoint_every": checkpoint_every,
        "supervised_weight_micro": supervised_weight_micro,
        "lr_micro": dict(lr_micro),
    }
    _reject_floats(numbers, "training_policy")
    loss = {
        "clip_epsilon_micro": clip_epsilon_micro,
        "kl_coefficient_micro": kl_coefficient_micro,
        "length_normalisation": length_normalisation,
        "gate_failed_reward_micro": gate_failed_reward_micro,
    }
    _reject_floats(loss, "loss_policy")
    if all(v is not None for v in loss.values()):
        RlLossPolicy(**loss)  # type: ignore[arg-type]  # one source of truth for the ranges
    doc = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "optimizer": OPTIMIZER,
        **numbers,
        "schedule": schedule,
        "loss_policy": loss,
        "execution_profile": validate_execution_profile(execution_profile),
    }
    _check_semantics(doc)
    check_lr_groups(doc, run_class, selected_typed_heads)
    return _seal(_SCHEMA, TRAINING_POLICY_DOMAIN, TRAINING_HASH_FIELD, doc)


def verify_rl_training_policy(doc: Mapping[str, Any]) -> None:
    _verify(_SCHEMA, TRAINING_POLICY_DOMAIN, TRAINING_HASH_FIELD, doc)
    _check_semantics(doc)


def check_lr_groups(
    doc: Mapping[str, Any], run_class: str, selected_typed_heads: Iterable[str] = ()
) -> None:
    """``lr_micro`` keys == the S4.5 trainable groups, plus any auxiliary-only
    group that S4.5 allows to train; a frozen/off/unknown group is refused."""
    heads = tuple(selected_typed_heads)
    states = s45_allowed_states(run_class, selected_typed_heads=heads)
    allowed = {g for g, s in states.items() if "trainable" in s}
    given = set(doc["lr_micro"])
    unknown = sorted(given - allowed)
    if unknown:
        raise RlPolicyInvalid(
            f"lr_micro names group(s) {unknown} that are not trainable in S4.5 "
            f"(trainable: {sorted(allowed)})"
        )
    missing = sorted(s45_trainable_groups(run_class, selected_typed_heads=heads) - given)
    if missing:
        raise RlPolicyInvalid(f"lr_micro has no learning rate for trainable group(s) {missing}")


def check_execution_profile_matches(
    training_policy: Mapping[str, Any], rollout_policy: Mapping[str, Any]
) -> None:
    """Rollout and update must run the very same execution profile (D2/D9)."""
    train, roll = training_policy["execution_profile"], rollout_policy["execution_profile"]
    if dict(train) != dict(roll):
        diff = sorted(k for k in set(train) | set(roll) if train.get(k) != roll.get(k))
        raise RlPolicyInvalid(
            f"training_policy.execution_profile differs from rollout_policy's at {diff}: "
            "rollout and update would not be one distribution"
        )


def loss_policy_from(doc: Mapping[str, Any]) -> RlLossPolicy:
    """The ``RlLossPolicy`` the policy freezes; refuses an unfrozen (null) field."""
    loss = doc["loss_policy"]
    nulls = [k for k in _LOSS_KEYS if loss[k] is None]
    if nulls:
        raise RlPolicyInvalid(f"loss_policy field(s) {nulls} are not frozen")
    return RlLossPolicy(**{k: loss[k] for k in _LOSS_KEYS})
