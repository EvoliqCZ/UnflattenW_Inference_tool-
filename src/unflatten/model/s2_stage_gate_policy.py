"""RFC005 §12.2 S2 Stage Gate Policy (DEC-012).

The first stage gate policy whose `quality` branch is `applies`. Like
`stage_gate_report.py`'s S1/S2 policy it carries **no embedded self-hash** --
its identity (`rfc005.s2-stage-gate-policy/v1`) is a `domain_hash` over the
whole validated document, computed here, never stored inside it.

S2 differs from S1 in one load-bearing way (DEC-012): S1's fifteen gates
were contract-only (owner R5 2026-09-07, D4), while S2 adds the exact-text
quality gate -- a mandatory `absolute_quality_floor_micro`, an optional
non-inferiority `relative_margin_micro` that may not compensate a failed
floor, the confidence-bound method and `alpha_micro`, the minimum
independent-lineage and minimum attempted denominators, and the critical
slices with their `minimum_fidelity_micro` floors. Every rate is carried in
micro-units (a rate r is r*1e6, an integer in [0, 1000000]) because the
deterministic CBOR profile these identities hash under
(`rfc8949-core-det-nfc-int-v0`) has no float in its data model: a float
floor could be validated but never hashed. The unit is in the name so the
scaling cannot be forgotten at a call site. The numbers are the owner's to
fill; this module freezes the shape.

`unflatten/eval/s2_gate/report.py`'s criterion 2 and criterion 4 read a
validated policy through this module's `s2_stage_gate_policy_hash` and
resolve their `policy_determination` from it; an invalid policy refuses
(`S2StageGatePolicyInvalid`), it never falls back to a default.

Torch-free by construction: nothing here imports torch.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.model.hashing import domain_hash

_SCHEMA_DIR = Path(__file__).parent / "schemas"

#: Design registry row `rfc005.s2-stage-gate-policy/v1`. Owned here: no other
#: module computes an S2 stage gate policy's identity hash.
_POLICY_DOMAIN = "rfc005.s2-stage-gate-policy/v1"

S2_STAGE_GATE_POLICY_SCHEMA_VERSION = "1.0.0"

#: Exactly fifteen distinct gate numbers, 1..15 (RFC005 §14.4) -- same
#: registry as `stage_gate_report._EXPECTED_GATE_NUMBERS`, re-declared here
#: so this module does not import the report builder's private state.
_EXPECTED_GATE_NUMBERS = frozenset(range(1, 16))


class S2StageGatePolicyInvalid(ValueError):
    """An S2 stage gate policy is malformed: the schema refuses it, or the
    semantic checks the schema cannot express (distinct gate numbers,
    distinct slice ids) do not hold.

    M1-internal; not an RFC004/RFC005 registry code.
    """


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / "s2_stage_gate_policy.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _validate_schema(doc: dict[str, Any]) -> None:
    errors = sorted(_validator().iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        pointer = "/" + "/".join(str(p) for p in first.absolute_path)
        raise S2StageGatePolicyInvalid(
            f"s2 stage gate policy invalid at {pointer or '/'}: {first.message}"
        )


def validate_s2_stage_gate_policy(doc: dict[str, Any]) -> None:
    _validate_schema(doc)
    gate_numbers = {g["gate"] for g in doc["gates"]}
    if gate_numbers != _EXPECTED_GATE_NUMBERS:
        raise S2StageGatePolicyInvalid(
            f"s2 stage gate policy gates are {sorted(gate_numbers)}, "
            "expected exactly 1..15 distinct"
        )
    slice_ids = [s["slice_id"] for s in doc["quality"]["critical_slices"]]
    if len(set(slice_ids)) != len(slice_ids):
        raise S2StageGatePolicyInvalid(
            f"s2 stage gate policy critical_slices repeat a slice_id: {slice_ids} -- "
            "one policy field per slice, never two competing floors"
        )


def s2_stage_gate_policy_hash(policy: dict[str, Any]) -> str:
    """`policy_hash` for `policy`: `domain_hash` over the whole validated
    document (module docstring) -- no embedded self-hash to project away."""
    validate_s2_stage_gate_policy(policy)
    return domain_hash(_POLICY_DOMAIN, policy)


def load_s2_stage_gate_policy(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise S2StageGatePolicyInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise S2StageGatePolicyInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_s2_stage_gate_policy(payload)
    return payload
