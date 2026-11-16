"""The DEC-040 augmentation policy: ordered op recipes per family, a
selection rule, and its own hash — frozen before use, versioned like
every other RFC003 policy (`split.allocator.SplitPolicy`,
`targets.masks` `FontPolicy`).

Loaded from `policies/augmentation_policy_v1.json` by default;
`load_policy` validates the shape so a malformed policy is refused at
load time, never discovered mid-run. `policy_hash` reuses
`identity.domain_hash` — no parallel identity scheme.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from unflatten.dataset.augment import ops_v2
from unflatten.dataset.augment.legibility import ENVELOPE_STATS
from unflatten.dataset.augment.ops import ALL_OPS
from unflatten.dataset.enums import VIEW_KIND
from unflatten.dataset.identity import domain_hash

#: `identity.domain_hash` domain for this policy's identity.
AUGMENTATION_POLICY_DOMAIN = "rfc003.augmentation-policy/v1"

#: Domain of the schema-2.0.0 identity (RFC003-ERRATA-004 E2).
AUGMENTATION_POLICY_DOMAIN_V2 = "rfc003.augmentation-policy/v2"

_DEFAULT_POLICY_PATH = Path(__file__).parent / "policies" / "augmentation_policy_v1.json"

#: The two DEC-040 families — must be exactly the `ViewKind` pair the
#: owner named, never a third.
FAMILY_NAMES: frozenset[str] = frozenset({"scan_simulation", "photo_simulation"})


class PolicyError(ValueError):
    """The policy JSON does not match the shape this module requires."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyError(message)


def _validate_param_range(op: str, name: str, value: Any) -> None:
    _require(
        isinstance(value, list) and len(value) == 2,
        f"family op {op!r} param {name!r} must be a [low, high] pair, got {value!r}",
    )
    low, high = value
    _require(
        isinstance(low, (int, float)) and isinstance(high, (int, float)) and low < high,
        f"family op {op!r} param {name!r} must have low < high, got {value!r}",
    )


def policy_schema_major(policy: Mapping[str, Any]) -> int:
    """`"1.0.0"` -> 1, `"2.0.0"` -> 2; anything else is a :class:`PolicyError`."""
    version = policy.get("policy_schema_version") if isinstance(policy, Mapping) else None
    majors = {"1.0.0": 1, "2.0.0": 2}
    _require(
        isinstance(version, str) and version in majors,
        f"policy_schema_version must be one of {sorted(majors)}, got {version!r}",
    )
    return majors[version]


def validate_policy(policy: dict[str, Any]) -> None:
    """Raise :class:`PolicyError` on the first shape violation (dispatch by major version)."""
    _require(isinstance(policy, dict), "policy must be a JSON object")
    if policy_schema_major(policy) == 2:
        validate_policy_v2(policy)
    else:
        _validate_policy_v1(policy)


def _validate_policy_v1(policy: dict[str, Any]) -> None:
    for key in (
        "policy_schema_version",
        "policy_id",
        "selection_numerator",
        "selection_denominator",
        "families",
    ):
        _require(key in policy, f"policy missing required key {key!r}")
    _require(
        isinstance(policy["selection_numerator"], int) and policy["selection_numerator"] > 0,
        "selection_numerator must be a positive int",
    )
    _require(
        isinstance(policy["selection_denominator"], int)
        and policy["selection_denominator"] >= policy["selection_numerator"],
        "selection_denominator must be an int >= selection_numerator",
    )
    families = policy["families"]
    _require(
        isinstance(families, dict) and set(families) == FAMILY_NAMES,
        f"families must be exactly {sorted(FAMILY_NAMES)}, got "
        f"{sorted(families) if isinstance(families, dict) else families!r}",
    )
    seen_ops: set[str] = set()
    for family_name, family in families.items():
        _require(
            isinstance(family, dict) and isinstance(family.get("ops"), list) and family["ops"],
            f"family {family_name!r} must carry a non-empty 'ops' list",
        )
        for entry in family["ops"]:
            _require(
                isinstance(entry, dict) and set(entry) == {"op", "params"},
                f"family {family_name!r} op entry must be {{'op', 'params'}}, got {entry!r}",
            )
            op_name = entry["op"]
            _require(
                op_name in ALL_OPS,
                f"family {family_name!r} names unknown op {op_name!r}, expected one of "
                f"{sorted(ALL_OPS)}",
            )
            seen_ops.add(op_name)
            params = entry["params"]
            _require(
                isinstance(params, dict) and params,
                f"op {op_name!r} in family {family_name!r} must carry non-empty params",
            )
            for param_name, param_range in params.items():
                _validate_param_range(op_name, param_name, param_range)
    _require(
        set(ALL_OPS) <= seen_ops,
        f"policy is missing op(s) {sorted(set(ALL_OPS) - seen_ops)} — all seven DEC-040 "
        "distortions must be covered by some family",
    )


# ------------------------------------------------------------ schema 2.0.0 --

_V2_REQUIRED_KEYS = (
    "policy_schema_version",
    "policy_id",
    "selection_numerator",
    "selection_denominator",
    "ops_per_view",
    "ink_free",
    "legibility",
    "families",
)
_OPS_PER_VIEW_MAX = 3


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_v2_param(op: str, name: str, value: Any) -> None:
    where = f"op {op!r} param {name!r}"
    _require(
        isinstance(value, list) and len(value) == 2 and all(_is_number(v) for v in value),
        f"{where} must be a numeric [low, high] pair, got {value!r}",
    )
    low, high = value
    _require(low < high, f"{where} must have low < high, got {value!r}")
    if ops_v2.is_int_param(op, name):
        _require(_is_int(low) and _is_int(high), f"{where} is an int param, got {value!r}")
    hard_low, hard_high = ops_v2.OP_PARAM_BOUNDS_V2[op][name]
    _require(
        hard_low <= low and high <= hard_high,
        f"{where} range {value!r} lies outside the hard bound [{hard_low}, {hard_high}]",
    )


def _validate_v2_entry(family_name: str, entry: Any) -> None:
    _require(
        isinstance(entry, dict) and set(entry) == {"op", "weight_milli", "params"},
        f"family {family_name!r} op entry must be {{'op', 'weight_milli', 'params'}}, "
        f"got {entry!r}",
    )
    op = entry["op"]
    _require(
        op in ops_v2.OPS_V2,
        f"family {family_name!r} names unknown op {op!r}, expected one of "
        f"{sorted(ops_v2.OPS_V2)}",
    )
    _require(
        ops_v2.GEOMETRY_CLASS[op] != "nonlinear",
        f"op {op!r} in family {family_name!r} is nonlinear — its geometry cannot be "
        "carried to the targets (RFC003-ERRATA-004 E4)",
    )
    _require(
        _is_int(entry["weight_milli"]) and entry["weight_milli"] >= 1,
        f"op {op!r} weight_milli must be an int >= 1, got {entry['weight_milli']!r}",
    )
    params = entry["params"]
    declared = set(ops_v2.OP_PARAMS_V2[op])
    _require(
        isinstance(params, dict) and set(params) == declared,
        f"op {op!r} params must be exactly {sorted(declared)}, got "
        f"{sorted(params) if isinstance(params, dict) else params!r}",
    )
    for name, value in params.items():
        _validate_v2_param(op, name, value)


def _validate_v2_sections(policy: dict[str, Any]) -> None:
    opv = policy["ops_per_view"]
    _require(
        isinstance(opv, list)
        and len(opv) == 2
        and all(_is_int(v) for v in opv)
        and 1 <= opv[0] <= opv[1] <= _OPS_PER_VIEW_MAX,
        f"ops_per_view must be [k_min, k_max] with 1 <= k_min <= k_max <= "
        f"{_OPS_PER_VIEW_MAX}, got {opv!r}",
    )
    ink_free = policy["ink_free"]
    _require(isinstance(ink_free, dict), "ink_free must be an object")
    for key in ("ink_threshold", "dilate_px", "max_overlap_ppm", "placement_tries"):
        _require(_is_int(ink_free.get(key)), f"ink_free.{key} must be an int")
    _require(
        0 <= ink_free["max_overlap_ppm"] <= _PPM,
        f"ink_free.max_overlap_ppm must be in [0, {_PPM}], got {ink_free['max_overlap_ppm']!r}",
    )
    _require(
        0 <= ink_free["dilate_px"] <= 32,
        f"ink_free.dilate_px must be in [0, 32], got {ink_free['dilate_px']!r}",
    )
    _require(
        0 <= ink_free["ink_threshold"] <= 255,
        f"ink_free.ink_threshold must be in [0, 255], got {ink_free['ink_threshold']!r}",
    )
    _require(
        ink_free["placement_tries"] >= 1,
        f"ink_free.placement_tries must be >= 1, got {ink_free['placement_tries']!r}",
    )
    legibility = policy["legibility"]
    _require(isinstance(legibility, dict), "legibility must be an object")
    for key in ("max_redraws", "min_retained_ink_milli", "max_rotation_deg"):
        _require(key in legibility, f"legibility missing key {key!r}")
    redraws = legibility["max_redraws"]
    _require(
        _is_int(redraws) and 0 <= redraws <= 10,
        f"legibility.max_redraws must be an int in [0, 10], got {redraws!r}",
    )
    retained = legibility["min_retained_ink_milli"]
    _require(
        _is_int(retained) and 0 <= retained <= 1000,
        f"legibility.min_retained_ink_milli must be an int in [0, 1000], got {retained!r}",
    )
    rotation = legibility["max_rotation_deg"]
    _require(
        _is_number(rotation) and rotation >= 0,
        f"legibility.max_rotation_deg must be a number >= 0, got {rotation!r}",
    )
    _require("envelope" in legibility, "legibility missing key 'envelope'")
    _validate_v2_envelope(legibility["envelope"])


def _validate_v2_envelope(envelope: Any) -> None:
    if envelope is None:
        return
    _require(
        isinstance(envelope, dict) and set(envelope) == set(ENVELOPE_STATS),
        f"legibility.envelope must be null or carry exactly {sorted(ENVELOPE_STATS)}, got "
        f"{sorted(envelope) if isinstance(envelope, dict) else envelope!r}",
    )
    for stat, band in envelope.items():
        _require(
            isinstance(band, dict)
            and set(band) == {"low", "high"}
            and all(_is_number(v) for v in band.values())
            and band["low"] < band["high"],
            f"legibility.envelope[{stat!r}] must be {{'low', 'high'}} with low < high, "
            f"got {band!r}",
        )


def validate_policy_v2(policy: dict[str, Any]) -> None:
    """Raise :class:`PolicyError` on the first schema-2.0.0 violation. Unlike
    v1 it does not require every op to be covered."""
    _require(isinstance(policy, dict), "policy must be a JSON object")
    for key in _V2_REQUIRED_KEYS:
        _require(key in policy, f"policy missing required key {key!r}")
    _require(policy["policy_schema_version"] == "2.0.0", "policy_schema_version must be 2.0.0")
    num, den = policy["selection_numerator"], policy["selection_denominator"]
    _require(_is_int(num) and num > 0, "selection_numerator must be a positive int")
    _require(_is_int(den) and den >= num, "selection_denominator must be an int >= numerator")
    _validate_v2_sections(policy)
    families = policy["families"]
    _require(
        isinstance(families, dict) and set(families) == FAMILY_NAMES,
        f"families must be exactly {sorted(FAMILY_NAMES)}, got "
        f"{sorted(families) if isinstance(families, dict) else families!r}",
    )
    for family_name, family in families.items():
        _require(
            isinstance(family, dict) and isinstance(family.get("ops"), list) and family["ops"],
            f"family {family_name!r} must carry a non-empty 'ops' list",
        )
        for entry in family["ops"]:
            _validate_v2_entry(family_name, entry)


def load_policy(path: Path | str | None = None) -> dict[str, Any]:
    """Load + validate the augmentation policy (default: the frozen
    `augmentation_policy_v1.json` shipped next to this module)."""
    resolved = Path(path) if path is not None else _DEFAULT_POLICY_PATH
    with open(resolved, encoding="utf-8") as f:
        policy = json.load(f)
    validate_policy(policy)
    return policy


@lru_cache(maxsize=8)
def _cached_default_policy() -> dict[str, Any]:
    return load_policy()


def default_policy() -> dict[str, Any]:
    """The frozen default policy, loaded and validated once per process."""
    return dict(_cached_default_policy())


#: Identity payloads are integer-only (RFC003 §12.1 / `canonical_json`
#: rejects floats), same reason `bbox_ppm` exists: every float param
#: range is projected to millionths before hashing.
_PPM = 1_000_000


def _policy_ppm_projection(policy: dict[str, Any]) -> dict[str, Any]:
    families = {
        family_name: {
            "ops": [
                {
                    "op": entry["op"],
                    "params": {
                        name: [round(low * _PPM), round(high * _PPM)]
                        for name, (low, high) in entry["params"].items()
                    },
                }
                for entry in family["ops"]
            ]
        }
        for family_name, family in policy["families"].items()
    }
    return {
        "policy_schema_version": policy["policy_schema_version"],
        "policy_id": policy["policy_id"],
        "selection_numerator": policy["selection_numerator"],
        "selection_denominator": policy["selection_denominator"],
        "families": families,
    }


def _policy_projection_v2(policy: Any) -> Any:
    """The whole body, recursively; floats become `round(x * 1e6)`, everything
    else is verbatim (identity payloads are integer-only)."""
    if isinstance(policy, dict):
        return {key: _policy_projection_v2(value) for key, value in policy.items()}
    if isinstance(policy, list):
        return [_policy_projection_v2(value) for value in policy]
    if isinstance(policy, float):
        return round(policy * _PPM)
    return policy


def policy_hash(policy: dict[str, Any]) -> str:
    """Major 1: `H("rfc003.augmentation-policy/v1", projection)` over the
    integer-ppm projection (see `_policy_ppm_projection`). Major 2:
    `H("rfc003.augmentation-policy/v2", _policy_projection_v2)`. The whole
    validated policy body is the payload, so any edit (a param range, an
    op order, the selection fraction) changes the hash."""
    if policy_schema_major(policy) == 2:
        return domain_hash(AUGMENTATION_POLICY_DOMAIN_V2, _policy_projection_v2(policy))
    return domain_hash(AUGMENTATION_POLICY_DOMAIN, _policy_ppm_projection(policy))


def family_view_kind(family: str) -> str:
    """`family` is already the exact `ViewKind` value — this only asserts
    that invariant, so a policy typo is caught before it reaches
    `identity.render_view_id`."""
    if family not in VIEW_KIND:
        raise PolicyError(f"family {family!r} is not a registered ViewKind")
    return family
