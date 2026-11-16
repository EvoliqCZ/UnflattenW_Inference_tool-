"""``policy-numbers.json``: the one file the owner signs (DEC-081) and from which
the four S4.5 policies are built (design P3, DEC-082 A v4 point 9).

Closed key space: a foreign or missing key is refused by name, a float anywhere is
refused (integer micro units only). There is no number validation here beyond that;
the existing builders keep every range and ``RATIFIED``-with-``null`` guard. The
execution profile is not an input: the training policy takes the rollout policy's.

Torch-free at import; ``training_policy`` is imported lazily in :func:`build_policies`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from unflatten.model.rl.gate_policy import build_rl_stage_gate_policy
from unflatten.model.rl.policies import (
    RlPolicyInvalid,
    _reject_floats,
    build_execution_profile,
    build_reward_policy,
    build_rollout_policy,
)

SCHEMA: Final = "unflatten.rl.s4_5-policy-numbers/1"
_TOP: Final = (
    "schema", "decision_id", "run_class", "run_seed", "selected_typed_heads",
    "reward", "rollout", "training", "stage_gate",
)  # fmt: skip
_REWARD: Final = ("component_weights_micro", "chain_policy_flags", "severe_structural_error")
_REWARD_OPTIONAL: Final = ("text_gate",)
_ROLLOUT: Final = (
    "temperature_micro", "top_p_micro", "group_size", "rollout_budget", "max_steps",
    "loop_guard_arm", "typed_head_sampling", "render_probe", "loop_guards_in_policy",
    "prefill_chunk_tokens",
)  # fmt: skip
_TRAINING: Final = (
    "weight_decay_micro", "lr_micro", "warmup_steps", "schedule", "grad_clip_micro",
    "accumulation_steps", "max_update_steps", "checkpoint_every", "clip_epsilon_micro",
    "kl_coefficient_micro", "length_normalisation", "gate_failed_reward_micro",
    "supervised_weight_micro",
)  # fmt: skip
_STAGE_GATE: Final = (
    "critical_slice_ids", "thresholds", "min_correlation_micro", "min_correlation_lineages",
    "max_zero_advantage_rate_micro",
)  # fmt: skip


class PolicyNumbersInvalid(RlPolicyInvalid):
    """``policy-numbers.json`` has a foreign, missing or mistyped key."""


@dataclass(frozen=True)
class PolicyNumbers:
    decision_id: str
    run_class: str
    run_seed: int
    selected_typed_heads: tuple[str, ...]
    reward: Mapping[str, Any]
    rollout: Mapping[str, Any]
    training: Mapping[str, Any]
    stage_gate: Mapping[str, Any]
    sha256: str  # of the file bytes, "sha256:<hex>"


def _exact(
    doc: Any, keys: tuple[str, ...], where: str, optional: tuple[str, ...] = ()
) -> Mapping[str, Any]:
    if not isinstance(doc, Mapping):
        raise PolicyNumbersInvalid(f"{where} must be an object, got {type(doc).__name__}")
    foreign = sorted(set(doc) - set(keys) - set(optional))
    missing = sorted(set(keys) - set(doc))
    if foreign or missing:
        raise PolicyNumbersInvalid(f"{where}: foreign key(s) {foreign}, missing key(s) {missing}")
    return doc


def load_policy_numbers(path: Path | str) -> PolicyNumbers:
    raw = Path(path).read_bytes()
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise PolicyNumbersInvalid(f"{path} is not JSON ({exc})") from exc
    _reject_floats(doc, "policy-numbers")
    _exact(doc, _TOP, "policy-numbers")
    if doc["schema"] != SCHEMA:
        raise PolicyNumbersInvalid(f"schema must be {SCHEMA!r}, got {doc['schema']!r}")
    seed = doc["run_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PolicyNumbersInvalid(f"run_seed must be an integer, got {seed!r}")
    for key in ("decision_id", "run_class"):
        if not isinstance(doc[key], str) or not doc[key]:
            raise PolicyNumbersInvalid(f"{key} must be a non-empty string, got {doc[key]!r}")
    heads = doc["selected_typed_heads"]
    if not isinstance(heads, list) or not all(isinstance(h, str) for h in heads):
        raise PolicyNumbersInvalid("selected_typed_heads must be a list of strings")
    return PolicyNumbers(
        decision_id=doc["decision_id"],
        run_class=doc["run_class"],
        run_seed=seed,
        selected_typed_heads=tuple(heads),
        reward=_exact(doc["reward"], _REWARD, "reward", _REWARD_OPTIONAL),
        rollout=_exact(doc["rollout"], _ROLLOUT, "rollout"),
        training=_exact(doc["training"], _TRAINING, "training"),
        stage_gate=_exact(doc["stage_gate"], _STAGE_GATE, "stage_gate"),
        sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def build_policies(numbers: PolicyNumbers, *, status: str) -> dict[str, dict[str, Any]]:
    """The four sealed policies, keyed ``reward`` / ``rollout`` / ``training`` / ``stage_gate``."""
    from unflatten.model.rl.training_policy import build_rl_training_policy

    r, o, t, g = numbers.reward, numbers.rollout, numbers.training, numbers.stage_gate
    profile = build_execution_profile(prefill_chunk_tokens=o["prefill_chunk_tokens"])
    return {
        "reward": build_reward_policy(
            component_weights_micro=r["component_weights_micro"],
            chain_policy_flags=r["chain_policy_flags"],
            severe_structural_error=r["severe_structural_error"],
            status=status,
            text_gate=r.get("text_gate"),
        ),
        "rollout": build_rollout_policy(
            temperature_micro=o["temperature_micro"],
            top_p_micro=o["top_p_micro"],
            group_size=o["group_size"],
            rollout_budget=o["rollout_budget"],
            max_steps=o["max_steps"],
            loop_guard_arm=o["loop_guard_arm"],
            execution_profile=profile,
            typed_head_sampling=o["typed_head_sampling"],
            loop_guards_in_policy=o["loop_guards_in_policy"],
            render_probe=o["render_probe"],
            status=status,
        ),
        "training": build_rl_training_policy(
            **{k: t[k] for k in _TRAINING},
            execution_profile=profile,
            run_class=numbers.run_class,
            selected_typed_heads=numbers.selected_typed_heads,
            status=status,
        ),
        "stage_gate": build_rl_stage_gate_policy(
            critical_slice_ids=g["critical_slice_ids"],
            thresholds=g["thresholds"],
            min_correlation_micro=g["min_correlation_micro"],
            min_correlation_lineages=g["min_correlation_lineages"],
            max_zero_advantage_rate_micro=g["max_zero_advantage_rate_micro"],
            status=status,
        ),
    }
