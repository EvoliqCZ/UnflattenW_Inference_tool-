"""The `conditioning_policy` producer (RFC005 §26.5): the artifact that
declares, per attachment, how optimization is conditioned, and the
`conditioning_policy_hash` `ExactTextCacheIdentity` and `s1_run_manifest`
bind.

**This module builds and hashes a policy; it does not choose one.** §26.5
gives three closed modes -- `gold_prefix`, `predicted_prefix_stop_gradient`,
`versioned_mixed_policy` -- and requires each run to declare one per
attachment. `gold_prefix` is the obvious mode for a teacher-forced
exact-text canary, and obvious is not ratified: :func:`build_conditioning_policy`
therefore has **no default mode**, and a missing declaration is refused by
name rather than filled in with the likely answer.

The §26.5 prohibitions are carried *inside* the document, as required
const-false fields, so they land in the policy identity: gold crops,
regions, continuation edges, TablePlan and property values/applicability
are not production neural inputs, and `oracle_diagnostic` does not update
product-ancestor weights. They are arguments too, for the same reason the
mode is: a producer that wrote them itself would convert the run owner's
declaration into this file's opinion, and a hash over an opinion binds
nobody. The one thing §26.5 *permits* --
`gold_structural_prefix_teacher_forced_slots` -- is a free boolean, still
required, so it is declared either way.

Hashed under `rfc005.conditioning-policy/v1` (design §4.9), the same
`domain_hash` convention as every other identity in this package.

Ships no `nn.Module`, no optimizer, and authorizes no gradient update.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.model.hashing import domain_hash

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_SCHEMA_NAME = "conditioning_policy"

#: Design §4.9 registry row `rfc005.conditioning-policy/v1`. Owned here:
#: no other module computes a `conditioning_policy_hash`.
_CONDITIONING_POLICY_DOMAIN = "rfc005.conditioning-policy/v1"

CONDITIONING_POLICY_SCHEMA_VERSION = "1.0.0"

#: The three ratified §26.5 modes, read from the schema's own enum so the
#: two cannot drift.
CONDITIONING_MODES: tuple[str, ...]

#: The forbidden-gold-input fields §26.5 enumerates, read from the
#: schema's own `required` list for the same reason.
GOLD_INPUT_PROHIBITIONS: tuple[str, ...]


@cache
def _schema() -> dict[str, Any]:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        return json.load(f)


@cache
def _validator() -> jsonschema.protocols.Validator:
    schema = _schema()
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


CONDITIONING_MODES = tuple(
    _schema()["properties"]["attachments"]["items"]["properties"]["mode"]["enum"]
)
GOLD_INPUT_PROHIBITIONS = tuple(
    _schema()["properties"]["gold_input_prohibitions"]["required"]
)


class ConditioningPolicyInvalid(ValueError):
    """A conditioning policy is incomplete, declares a mode for no
    attachment or two for one, or fails its closed schema (RFC005 §26.5).

    M1-internal (design §9.3); not an RFC004 registry code.
    """


def validate_conditioning_policy(policy: Mapping[str, Any]) -> None:
    """Validate `policy` against `conditioning_policy.schema.json`, plus
    the one rule the schema cannot express: an attachment may be declared
    only once.

    `uniqueItems` on the array is not that rule -- two *different* rows
    for `decoder_step` (say `gold_prefix` and
    `predicted_prefix_stop_gradient`) are unique items and would leave the
    run's conditioning ambiguous while looking declared.
    """
    errors = list(_validator().iter_errors(policy))
    if errors:
        raise ConditioningPolicyInvalid(
            f"schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )
    seen: set[str] = set()
    for row in policy["attachments"]:
        attachment = row["attachment"]
        if attachment in seen:
            raise ConditioningPolicyInvalid(
                f"attachment {attachment!r} is declared twice -- RFC005 §26.5 "
                "requires exactly one mode per attachment"
            )
        seen.add(attachment)


def build_conditioning_policy(
    *,
    policy_version: str,
    attachments: Sequence[Mapping[str, Any]],
    gold_structural_prefix_teacher_forced_slots: bool,
    gold_input_prohibitions: Mapping[str, Any],
    oracle_diagnostic_updates_product_ancestor_weights: bool,
) -> dict[str, Any]:
    """Assemble one conditioning policy, validated on the way out.

    Every argument is keyword-only and required; there are no defaults and
    no mode is assumed (module docstring). A prohibition missing from
    `gold_input_prohibitions` is refused by name, as is one this policy
    does not know.
    """
    missing = sorted(set(GOLD_INPUT_PROHIBITIONS) - set(gold_input_prohibitions))
    if missing:
        raise ConditioningPolicyInvalid(
            f"missing gold-input prohibition(s): {', '.join(missing)} -- RFC005 "
            "§26.5 prohibitions are declared by the run owner, and this producer "
            "refuses to declare them on their behalf"
        )
    unknown = sorted(set(gold_input_prohibitions) - set(GOLD_INPUT_PROHIBITIONS))
    if unknown:
        raise ConditioningPolicyInvalid(
            f"unknown gold-input prohibition(s): {', '.join(unknown)} -- expected "
            f"exactly {', '.join(GOLD_INPUT_PROHIBITIONS)}"
        )
    policy = {
        "conditioning_policy_schema_version": CONDITIONING_POLICY_SCHEMA_VERSION,
        "policy_version": policy_version,
        "attachments": [dict(row) for row in attachments],
        "gold_structural_prefix_teacher_forced_slots": (
            gold_structural_prefix_teacher_forced_slots
        ),
        "gold_input_prohibitions": {
            name: gold_input_prohibitions[name] for name in GOLD_INPUT_PROHIBITIONS
        },
        "oracle_diagnostic_updates_product_ancestor_weights": (
            oracle_diagnostic_updates_product_ancestor_weights
        ),
    }
    validate_conditioning_policy(policy)
    return policy


def load_conditioning_policy(path: Path | str) -> dict[str, Any]:
    """Read a frozen conditioning policy from disk, validated on the way in."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise ConditioningPolicyInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConditioningPolicyInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_conditioning_policy(payload)
    return payload


def conditioning_policy_hash(policy: Mapping[str, Any]) -> str:
    """`conditioning_policy_hash` over the whole validated policy."""
    validate_conditioning_policy(policy)
    return domain_hash(_CONDITIONING_POLICY_DOMAIN, dict(policy))
