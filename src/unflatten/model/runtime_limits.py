"""The `runtime_limits_policy` producer (design §4.7): the one artifact
carrying every pinned M1 limit, its `runtime_limits_policy_hash`, and the
`budget_policy_hash` the composite grammar descriptor binds.

**This module builds and hashes a policy; it does not choose one.**
Design §4.7 and §18.1 item 3 are explicit that the concrete limit values
are frozen run/security policy set before use, "not quality thresholds
invented here". So `build_runtime_limits_policy` has **no defaults**: every
one of :data:`LIMIT_FIELDS` must be supplied, a missing one is refused
rather than filled in, and an unknown one is refused rather than ignored
(a caller who misspells a limit would otherwise get a silently defaulted
one under a name that looks bound). Until the owner freezes the values,
this producer can compute an identity for a policy but cannot supply the
policy -- that gap is real and is not closed by this module.

:data:`LIMIT_FIELDS` is read from the schema's own `required` list rather
than restated here: a limit added to `runtime_limits_policy.schema.json`
becomes required of every caller with nobody having to remember, which is
the same reasoning `grammar.controller_hash` gives for deriving its
payload from the modules at run time.

`budget_policy_hash` is the `text_encoding_budget_ops` *subset* hash of the
same policy (§4.7: "`composite_grammar.budget_policy_hash` covers **only**
the text-encoding budget and must equal the `text_encoding_budget_ops`
subset hash of this policy"). It is computed under the same
`rfc004.runtime-limits-policy/v1` domain over a one-key projection: the
projection and the full policy are maps of different size, which the
deterministic CBOR profile encodes differently, so the two can never
collide, and no second domain is minted for a subset of one artifact.
Moving an unrelated limit therefore does not move `budget_policy_hash` --
that is the "covers only" clause, and it is tested.

Ships no `nn.Module`, no optimizer, and authorizes no gradient update.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.model.hashing import domain_hash

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_SCHEMA_NAME = "runtime_limits_policy"

#: Design §4.9 registry row `rfc004.runtime-limits-policy/v1`. Owned here:
#: no other module computes a `runtime_limits_policy_hash`.
_RUNTIME_LIMITS_DOMAIN = "rfc004.runtime-limits-policy/v1"

RUNTIME_LIMITS_POLICY_SCHEMA_VERSION = "1.0.0"

#: The limit this policy's `budget_policy_hash` subset covers -- and only
#: this one (§4.7).
BUDGET_POLICY_FIELD = "text_encoding_budget_ops"


class RuntimeLimitsPolicyInvalid(ValueError):
    """A runtime limits policy is incomplete, carries an unknown limit, or
    fails its closed schema (design §4.7).

    M1-internal (§9.3); not an RFC004 registry code.
    """


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


@cache
def _limit_fields() -> tuple[str, ...]:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    return tuple(
        name
        for name in schema["required"]
        if name not in ("runtime_limits_policy_schema_version", "policy_version")
    )


#: Every limit a policy must pin, derived from the schema (module docstring).
LIMIT_FIELDS: tuple[str, ...] = _limit_fields()


def validate_runtime_limits_policy(policy: Mapping[str, Any]) -> None:
    """Validate `policy` against `runtime_limits_policy.schema.json`, plus
    the one rule the schema cannot express: every limit must be a *true*
    integer.

    JSON Schema Draft 2020-12 accepts `4096.0` where an integer is
    required (`manifests.validate_architecture_manifest` makes the same
    check for `d_model`), and Python's `True` is an `int`. A float or a
    boolean limit would be hashed into the policy identity as itself and
    then compared against an integer somewhere else.
    """
    errors = list(_validator().iter_errors(policy))
    if errors:
        raise RuntimeLimitsPolicyInvalid(
            f"schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )
    for name in LIMIT_FIELDS:
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeLimitsPolicyInvalid(
                f"{name} must be a true integer, got {value!r}"
            )


def build_runtime_limits_policy(
    *, policy_version: str, limits: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble one runtime limits policy from `limits`, validated on the
    way out.

    `limits` must carry exactly :data:`LIMIT_FIELDS`; there are no
    defaults (module docstring).
    """
    missing = sorted(set(LIMIT_FIELDS) - set(limits))
    if missing:
        raise RuntimeLimitsPolicyInvalid(
            f"missing required limit(s): {', '.join(missing)} -- concrete limits are "
            "owner-frozen run/security policy (design §4.7, §18.1 item 3) and this "
            "producer refuses to default them"
        )
    unknown = sorted(set(limits) - set(LIMIT_FIELDS))
    if unknown:
        raise RuntimeLimitsPolicyInvalid(
            f"unknown limit(s): {', '.join(unknown)} -- expected exactly "
            f"{', '.join(LIMIT_FIELDS)}"
        )
    policy = {
        "runtime_limits_policy_schema_version": RUNTIME_LIMITS_POLICY_SCHEMA_VERSION,
        "policy_version": policy_version,
        **{name: limits[name] for name in LIMIT_FIELDS},
    }
    validate_runtime_limits_policy(policy)
    return policy


def load_runtime_limits_policy(path: Path | str) -> dict[str, Any]:
    """Read a frozen policy document from disk, validated on the way in."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise RuntimeLimitsPolicyInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeLimitsPolicyInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_runtime_limits_policy(payload)
    return payload


def runtime_limits_policy_hash(policy: Mapping[str, Any]) -> str:
    """`runtime_limits_policy_hash` over the whole validated policy."""
    validate_runtime_limits_policy(policy)
    return domain_hash(_RUNTIME_LIMITS_DOMAIN, dict(policy))


def budget_policy_hash(policy: Mapping[str, Any]) -> str:
    """`composite_grammar.budget_policy_hash`: the text-encoding-budget
    subset hash of `policy`, and nothing else (module docstring)."""
    validate_runtime_limits_policy(policy)
    return domain_hash(
        _RUNTIME_LIMITS_DOMAIN, {BUDGET_POLICY_FIELD: policy[BUDGET_POLICY_FIELD]}
    )
