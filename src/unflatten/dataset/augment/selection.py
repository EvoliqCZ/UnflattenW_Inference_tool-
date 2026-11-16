"""Deterministic, population-stable page selection + recipe resolution.

Selection is a hash-threshold rule over `(document_bundle_id,
page_index)` alone: it is never re-derived from "the Nth page of a
sorted list" or any other construction that reads the whole candidate
population, so adding or removing unrelated pages never changes an
existing page's verdict (memory lesson "split deterministic, not
stable" — `unflatten.dataset.split.allocator`'s greedy fill is exactly
the shape this module avoids).

`resolve_recipe` turns a policy family's ordered `ops` list into
concrete parameter VALUES (still deterministic, hash-seeded per op) —
the same recipe, byte-identical, every time it is asked for the same
`(document_bundle_id, page_index, policy_hash)`.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from unflatten.dataset.augment import ops_v2
from unflatten.dataset.augment.policy import FAMILY_NAMES, policy_schema_major
from unflatten.dataset.identity import domain_hash, to_canonical_json

_SELECTION_DOMAIN = "rfc003.augmentation-selection/v1"
_FAMILY_DOMAIN = "rfc003.augmentation-family/v1"
_OP_SEED_DOMAIN = "rfc003.augmentation-op-seed/v1"
_RECIPE_V2_DOMAIN = "rfc003.augmentation-recipe-v2/v1"


def _digest_int(domain: str, payload: dict[str, Any]) -> int:
    """First 8 bytes of `H(domain, payload)`'s raw digest as an unsigned
    int — same construction as `identity.operation_seed`, but this
    module owns its own domains (an augmentation recipe is not a
    synthetic-generator operation)."""
    raw = hashlib.sha256(to_canonical_json([domain, payload])).digest()
    return int.from_bytes(raw[:8], "big")


def is_selected(
    document_bundle_id: str, page_index: int, policy_hash: str, policy: dict[str, Any]
) -> bool:
    """Whether this page is an augmentation candidate under `policy`.

    `selection_numerator / selection_denominator` (policy) of pages are
    selected, e.g. 1/5 == ~20%. Pure function of the page's own identity
    plus the policy hash — no RNG, no population scan.
    """
    value = _digest_int(
        _SELECTION_DOMAIN,
        {
            "document_bundle_id": document_bundle_id,
            "page_index": page_index,
            "policy_hash": policy_hash,
        },
    )
    denominator = policy["selection_denominator"]
    numerator = policy["selection_numerator"]
    return (value % denominator) < numerator


def select_family(document_bundle_id: str, page_index: int, policy_hash: str) -> str:
    """Deterministic family assignment (`scan_simulation` /
    `photo_simulation`), alternating by a hash bit rather than position
    in any list — stable under population changes for the same reason
    `is_selected` is."""
    value = _digest_int(
        _FAMILY_DOMAIN,
        {
            "document_bundle_id": document_bundle_id,
            "page_index": page_index,
            "policy_hash": policy_hash,
        },
    )
    ordered = sorted(FAMILY_NAMES)
    return ordered[value % len(ordered)]


def resolve_recipe(
    document_bundle_id: str,
    page_index: int,
    family: str,
    policy: dict[str, Any],
    policy_hash: str,
) -> tuple[list[dict[str, Any]], list[int]]:
    """The concrete, resolved `[{"op", "params"}, ...]` recipe for one
    page plus the per-op seeds used to resolve it — the seeds are what
    `ops.apply_recipe` needs to reproduce the exact same pixels, and
    what any noise-bearing op derives its own seed from again.

    Every param range in the policy is resolved to one value via
    `numpy.random.default_rng(op_seed).uniform(low, high)` — this makes
    the recipe fully determined by `(document_bundle_id, page_index,
    family, policy_hash)`, never by call order or wall-clock.

    Policy schema major 2 dispatches to `resolve_recipe_v2`; major 1 keeps
    the body below byte-for-byte.
    """
    if policy_schema_major(policy) == 2:
        return resolve_recipe_v2(document_bundle_id, page_index, family, policy, policy_hash)
    spec_ops = policy["families"][family]["ops"]
    resolved: list[dict[str, Any]] = []
    seeds: list[int] = []
    for index, entry in enumerate(spec_ops):
        op_seed = _digest_int(
            _OP_SEED_DOMAIN,
            {
                "document_bundle_id": document_bundle_id,
                "page_index": page_index,
                "family": family,
                "policy_hash": policy_hash,
                "op_index": index,
                "op": entry["op"],
            },
        )
        rng = np.random.default_rng(op_seed)
        params = {
            name: round(float(rng.uniform(low, high)), 6)
            for name, (low, high) in entry["params"].items()
        }
        resolved.append({"op": entry["op"], "params": params})
        seeds.append(op_seed)
    return resolved, seeds


def resolve_recipe_v2(
    document_bundle_id: str,
    page_index: int,
    family: str,
    policy: dict[str, Any],
    policy_hash: str,
) -> tuple[list[dict[str, Any]], list[int]]:
    """k-of-N recipe (RFC003-ERRATA-004 E3): draw `k` in `ops_per_view`,
    pick `k` ops without replacement weighted by `weight_milli`, then apply
    them in policy-list order. Each chosen op's parameters come from its own
    `op_seed` (`op_index` = index in the policy list); integer params
    (`ops_v2.is_int_param`) are drawn inclusive of `high`.
    """
    spec_ops = policy["families"][family]["ops"]
    k_min, k_max = policy["ops_per_view"]
    rng = np.random.default_rng(
        _digest_int(
            _RECIPE_V2_DOMAIN,
            {
                "document_bundle_id": document_bundle_id,
                "page_index": page_index,
                "family": family,
                "policy_hash": policy_hash,
            },
        )
    )
    k = min(int(rng.integers(k_min, k_max + 1)), len(spec_ops))
    weights = np.array([entry["weight_milli"] for entry in spec_ops], dtype=np.float64)
    chosen = sorted(
        int(i) for i in rng.choice(len(spec_ops), size=k, replace=False, p=weights / weights.sum())
    )
    resolved: list[dict[str, Any]] = []
    seeds: list[int] = []
    for index in chosen:
        entry = spec_ops[index]
        op_seed = _digest_int(
            _OP_SEED_DOMAIN,
            {
                "document_bundle_id": document_bundle_id,
                "page_index": page_index,
                "family": family,
                "policy_hash": policy_hash,
                "op_index": index,
                "op": entry["op"],
            },
        )
        op_rng = np.random.default_rng(op_seed)
        params: dict[str, Any] = {}
        for name, (low, high) in entry["params"].items():
            if ops_v2.is_int_param(entry["op"], name):
                params[name] = int(op_rng.integers(low, high + 1))
            else:
                params[name] = round(float(op_rng.uniform(low, high)), 6)
        resolved.append({"op": entry["op"], "params": params})
        seeds.append(op_seed)
    return resolved, seeds


#: Same integer-only identity discipline as `policy._policy_ppm_projection`
#: — a resolved recipe still carries float param VALUES (for `ops.py`),
#: so the hash payload projects them to millionths.
_PPM = 1_000_000


def _resolved_ops_ppm(resolved_ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "op": entry["op"],
            "params": {name: round(value * _PPM) for name, value in entry["params"].items()},
        }
        for entry in resolved_ops
    ]


def transform_chain_hash(
    document_bundle_id: str,
    page_index: int,
    family: str,
    resolved_ops: list[dict[str, Any]],
    policy_hash: str,
) -> str:
    """`H("rfc003.augmentation-transform-chain/v1", ...)` — the identity
    of the concrete, resolved recipe applied to this page, bound into
    `identity.render_view_id`'s `transform_chain_hash` field."""
    return domain_hash(
        "rfc003.augmentation-transform-chain/v1",
        {
            "document_bundle_id": document_bundle_id,
            "page_index": page_index,
            "family": family,
            "policy_hash": policy_hash,
            "ops": _resolved_ops_ppm(resolved_ops),
        },
    )
