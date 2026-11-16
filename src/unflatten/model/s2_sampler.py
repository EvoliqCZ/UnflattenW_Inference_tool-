"""S2 hierarchical sampler and the precomputed ordered exposure plan
(RFC005 §30.1–§30.7).

The sampler is the mechanism §30.2 makes normative: a closed joint
StageDataMatrix is drawn FIRST, and only then a lineage inside the chosen
cell, by the §30.3 bounded weight `w_lineage = bounded(w_feature_coverage
× w_hardness)` with the §30.4 capped inverse-sqrt rare-feature factor on
top. The result is an ORDERED EXPOSURE PLAN — the seed-deterministic
sequence of lineage-root draws for one virtual epoch — which §30.5 makes
mandatory and identical for Track A, the G1 outer comparison and G2: a
candidate must not be able to change its examples based on its own errors,
so the plan is built BEFORE the run and frozen by hash.

Design decisions this module makes explicit (the RFC leaves them open):

- all weights are integer per-mille: the deterministic CBOR profile these
  hashes are computed under (`rfc8949-core-det-nfc-int-v0`) has no float,
  so 1.5 is 1500, never 1.5;
- `w_rare` for a lineage is its RAREST carried feature's (minimum
  lineage-count) capped inverse-sqrt, in per-mille: a lineage that is the
  only carrier of a feature is the one a rare-feature batch must include;
- a lineage carrying no features gets the neutral multiplier 1000;
- a cell that carries frozen mass but whose eligible pool is empty — or
  whose eligible lineages all weight to zero — REFUSES the plan (§30.7: no
  implicit substitution, the dataloader never tops a gap up with another
  pyramid level);
- the §30.6 view cap is enforced at draw time: a lineage that reached its
  cap is excluded from further eligibility, and a pool the cap exhausts
  mid-epoch is a refusal, not a silent over-exposure.

This module is torch-free on purpose: plan construction is pure
deterministic bookkeeping and must stay importable and testable without a
model stack.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema

from unflatten.model.hashing import domain_hash, make_id
from unflatten.model.stage_data_matrix import stage_data_matrix_hash, validate_stage_data_matrix

_SCHEMAS_DIR = Path(__file__).parent / "schemas"

S2_SAMPLER_POLICY_SCHEMA_VERSION = "1.0.0"
S2_EXPOSURE_POLICY_SCHEMA_VERSION = "1.0.0"
ORDERED_EXPOSURE_PLAN_SCHEMA_VERSION = "1.0.0"

#: Domain constants — each artefact kind hashes under its own domain so a
#: valid-but-wrong artefact can never collide across kinds.
SAMPLER_POLICY_DOMAIN = "rfc005.s2-sampler-policy/v1"
EXPOSURE_POLICY_DOMAIN = "rfc005.s2-exposure-policy/v1"
ORDERED_EXPOSURE_PLAN_DOMAIN = "rfc005.ordered-exposure-plan/v1"

#: The §30.4 formula's neutral per-mille (a multiplier of 1.0).
_NEUTRAL_MILLI = 1000

_CELL_AXES = ("split", "pyramid_level", "task_kind", "cohort")


class S2SamplerPolicyInvalid(ValueError):
    """The document is not a well-formed S2 sampler policy: schema
    violation, out-of-range quantity, or bounds that cannot hold (min >
    max). A refusal, not a repair — the loader never normalises."""


class S2ExposurePolicyInvalid(ValueError):
    """The document is not a well-formed S2 exposure policy: schema
    violation or out-of-range quantity. A refusal, not a repair."""


class S2ExposurePlanRefused(ValueError):
    """The inputs cannot produce an ordered exposure plan without a rule
    being bent: a positive-mass cell with no eligible lineage (§30.7), an
    all-zero-weight cell, a cap-exhausted pool, or a malformed lineage
    pool. The plan is refused, never partially built."""


@cache
def _sampler_validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMAS_DIR / "s2_sampler_policy.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


@cache
def _exposure_validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMAS_DIR / "s2_exposure_policy.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


@cache
def _plan_validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMAS_DIR / "ordered_exposure_plan.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _refuse(message: str) -> S2ExposurePlanRefused:
    return S2ExposurePlanRefused(f"ordered exposure plan refused: {message}")


def validate_ordered_exposure_plan(payload: Mapping[str, Any]) -> None:
    """Validate `payload` against the closed ordered-exposure-plan schema
    -- exposed for callers that build a plan through a mechanism OTHER
    than :func:`build_ordered_exposure_plan` (e.g. `s4_plan`'s
    page-uniform builder, DEC-041) but must still conform to the same
    closed envelope every consumer (`s2_run_manifest.
    verify_ordered_exposure_plan_against`) validates against."""
    errors = list(_plan_validator().iter_errors(payload))
    if errors:
        raise _refuse(
            f"plan schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )


def _check_bounds(bounds: Mapping[str, Any], name: str) -> None:
    if bounds["min_milli"] > bounds["max_milli"]:
        raise S2SamplerPolicyInvalid(
            f"{name}: min_milli ({bounds['min_milli']}) exceeds max_milli "
            f"({bounds['max_milli']})"
        )


def validate_s2_sampler_policy(policy: Mapping[str, Any]) -> None:
    """Validate `policy` against the closed sampler-policy schema plus the
    §30.3 cross-field rule the schema cannot express (min ≤ max)."""
    errors = list(_sampler_validator().iter_errors(policy))
    if errors:
        raise S2SamplerPolicyInvalid(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )
    _check_bounds(policy["feature_coverage_bounds"], "feature_coverage_bounds")
    _check_bounds(policy["hardness_bounds"], "hardness_bounds")


def s2_sampler_policy_hash(policy: Mapping[str, Any]) -> str:
    validate_s2_sampler_policy(policy)
    return domain_hash(SAMPLER_POLICY_DOMAIN, dict(policy))


def load_s2_sampler_policy(path: Path | str) -> dict[str, Any]:
    """Read a frozen S2 sampler policy from disk, validated on the way in."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise S2SamplerPolicyInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise S2SamplerPolicyInvalid(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise S2SamplerPolicyInvalid(f"{path} must hold a JSON object")
    validate_s2_sampler_policy(payload)
    return payload


def validate_s2_exposure_policy(policy: Mapping[str, Any]) -> None:
    """Validate `policy` against the closed §30.6 exposure-policy schema."""
    errors = list(_exposure_validator().iter_errors(policy))
    if errors:
        raise S2ExposurePolicyInvalid(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )


def s2_exposure_policy_hash(policy: Mapping[str, Any]) -> str:
    validate_s2_exposure_policy(policy)
    return domain_hash(EXPOSURE_POLICY_DOMAIN, dict(policy))


def load_s2_exposure_policy(path: Path | str) -> dict[str, Any]:
    """Read a frozen S2 exposure policy from disk, validated on the way in."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise S2ExposurePolicyInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise S2ExposurePolicyInvalid(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise S2ExposurePolicyInvalid(f"{path} must hold a JSON object")
    validate_s2_exposure_policy(payload)
    return payload


def _require_int(value: object, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    # `type is int` on purpose: `bool` is an `int` subclass and a `true`
    # quantity would otherwise pass an `isinstance` check.
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise _refuse(f"{name} must be an integer in [{minimum}, {maximum}], got {value!r}")
    return value


def _cell_key(cell: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (cell["split"], cell["pyramid_level"], cell["task_kind"], cell["cohort"])


def _format_cell(cell: Mapping[str, Any]) -> str:
    return "(" + ", ".join(f"{axis}={cell[axis]}" for axis in _CELL_AXES) + ")"


def _matrix_domains(matrix: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    return {
        "split": frozenset(matrix["splits"]),
        "pyramid_level": frozenset(matrix["pyramid_levels"]),
        "task_kind": frozenset(matrix["task_kinds"]),
        "cohort": frozenset(matrix["cohorts"]),
    }


def _normalise_lineage(
    raw: Mapping[str, Any],
    index: int,
    domains: Mapping[str, frozenset[str]],
) -> dict[str, Any]:
    fields = frozenset(raw)
    expected = frozenset(
        {
            "lineage_root_id",
            "eligible_cells",
            "feature_coverage_milli",
            "hardness_score_milli",
            "features",
        }
    )
    unknown = fields - expected
    if unknown:
        raise _refuse(f"lineages[{index}] has unknown field(s): {sorted(unknown)}")
    missing = expected - fields
    if missing:
        raise _refuse(f"lineages[{index}] is missing field(s): {sorted(missing)}")

    lineage_root_id = raw["lineage_root_id"]
    if not isinstance(lineage_root_id, str) or not lineage_root_id:
        raise _refuse(f"lineages[{index}].lineage_root_id must be a non-empty string")

    eligible = raw["eligible_cells"]
    if not isinstance(eligible, Sequence) or isinstance(eligible, str) or not eligible:
        raise _refuse(f"lineages[{index}].eligible_cells must be a non-empty list of cells")
    cells: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for j, cell in enumerate(eligible):
        if not isinstance(cell, Mapping) or set(cell) != set(_CELL_AXES):
            raise _refuse(
                f"lineages[{index}].eligible_cells[{j}] must be a closed "
                "{split, pyramid_level, task_kind, cohort} object"
            )
        for axis in _CELL_AXES:
            if cell[axis] not in domains[axis]:
                raise _refuse(
                    f"lineages[{index}].eligible_cells[{j}] {axis}={cell[axis]!r} lies outside "
                    f"the matrix's declared domain"
                )
        key = (cell["split"], cell["pyramid_level"], cell["task_kind"], cell["cohort"])
        if key in cells:
            raise _refuse(f"lineages[{index}] lists cell {_format_cell(cell)} twice")
        cells[key] = {axis: cell[axis] for axis in _CELL_AXES}

    coverage = raw["feature_coverage_milli"]
    _require_int(coverage, f"lineages[{index}].feature_coverage_milli", maximum=1_000_000)
    hardness = raw["hardness_score_milli"]
    _require_int(hardness, f"lineages[{index}].hardness_score_milli")

    features = raw["features"]
    if (
        not isinstance(features, Sequence)
        or isinstance(features, str)
        or any(not isinstance(f, str) or not f for f in features)
        or len(set(features)) != len(features)
    ):
        raise _refuse(f"lineages[{index}].features must be a list of unique non-empty strings")

    return {
        "lineage_root_id": lineage_root_id,
        "cells": cells,
        "feature_coverage_milli": coverage,
        "hardness_score_milli": hardness,
        "features": list(features),
    }


def _normalise_pool(
    lineages: Sequence[Mapping[str, Any]], matrix: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not lineages:
        raise _refuse("the lineage pool is empty — a non-zero-mass matrix cannot be served")
    domains = _matrix_domains(matrix)
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(lineages):
        if not isinstance(raw, Mapping):
            raise _refuse(f"lineages[{i}] must be an object")
        record = _normalise_lineage(raw, i, domains)
        if record["lineage_root_id"] in seen:
            raise _refuse(f"duplicate lineage_root_id {record['lineage_root_id']!r} in the pool")
        seen.add(record["lineage_root_id"])
        pool.append(record)
    return pool


def compute_lineage_weight(
    *,
    feature_coverage_milli: int,
    hardness_score_milli: int,
    features: Sequence[str],
    feature_lineage_counts: Mapping[str, int],
    sampler_policy: Mapping[str, Any],
) -> dict[str, int]:
    """The §30.3/§30.4 weight of one lineage, in per-mille, under
    `sampler_policy`'s bounds.

    Pure and deterministic: clamping is the ONLY place the policy acts, so
    a weight that looks wrong is auditable from the recorded components
    alone. `feature_lineage_counts` is the §30.4 input over the pool; a
    feature absent from it is a caller error (the counts come from the
    same pool as the lineage).
    """
    fc = sampler_policy["feature_coverage_bounds"]
    hard = sampler_policy["hardness_bounds"]
    w_fc = min(max(feature_coverage_milli, fc["min_milli"]), fc["max_milli"])
    w_h = min(max(hardness_score_milli, hard["min_milli"]), hard["max_milli"])
    w_lineage = (w_fc * w_h) // 1000

    if features:
        rarest = min(feature_lineage_counts[f] for f in features)
        w_rare = min(sampler_policy["w_rare_cap_milli"], math.isqrt(1_000_000 // rarest))
    else:
        w_rare = _NEUTRAL_MILLI
    return {
        "feature_coverage_milli": w_fc,
        "hardness_milli": w_h,
        "w_lineage": w_lineage,
        "w_rare_milli": w_rare,
        "total_weight": (w_lineage * w_rare) // 1000,
    }


def build_ordered_exposure_plan(
    *,
    matrix: Mapping[str, Any],
    lineages: Sequence[Mapping[str, Any]],
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
    run_seed: int,
) -> dict[str, Any]:
    """Build the seed-deterministic ordered exposure plan for one virtual
    epoch.

    Every input is validated on the way in (matrix, both policies, the
    lineage pool, the seed); any violation REFUSES the plan. The draw
    order per §30.1: the frozen joint cell first, then a lineage inside
    the chosen cell, mass applied before any fragment/view/augmentation
    expansion. Determinism is part of the contract: the same inputs and
    seed always produce the identical plan, and the plan seals itself
    with `plan_id` and `ordered_exposure_plan_hash`.
    """
    validate_stage_data_matrix(matrix)
    validate_s2_sampler_policy(sampler_policy)
    validate_s2_exposure_policy(exposure_policy)
    _require_int(run_seed, "run_seed")

    pool = _normalise_pool(lineages, matrix)
    feature_lineage_counts: dict[str, int] = {}
    for record in pool:
        for f in record["features"]:
            feature_lineage_counts[f] = feature_lineage_counts.get(f, 0) + 1

    # Mass draw is over the sorted non-zero cells: the matrix's own order
    # is arbitrary and must not leak into the seed stream. Zero-mass cells
    # are explicitly infeasible (§30.2) and never enter the draw.
    mass_cells = sorted(
        (_cell_key(cell), cell["mass_micro"], cell) for cell in matrix["cells"]
    )
    mass_cells = [item for item in mass_cells if item[1] > 0]
    coords = [key for key, _, _ in mass_cells]
    masses = [mass for _, mass, _ in mass_cells]
    cells_by_key = {key: cell for key, _, cell in mass_cells}

    max_views = exposure_policy["max_views_per_lineage_per_virtual_epoch"]
    epoch_size = sampler_policy["virtual_epoch_size"]

    rng = random.Random(run_seed)
    views: dict[str, int] = {}
    draws: list[dict[str, Any]] = []
    for draw_index in range(epoch_size):
        key = rng.choices(coords, weights=masses, k=1)[0]
        cell = cells_by_key[key]
        eligible = [
            record
            for record in pool
            if key in record["cells"] and views.get(record["lineage_root_id"], 0) < max_views
        ]
        if not eligible:
            raise _refuse(
                f"draw {draw_index}: cell {_format_cell(cell)} carries frozen mass but no "
                "lineage in the pool can serve it (missing from the pool or cap-exhausted) — "
                "RFC005 §30.7: the gap is refused, never implicitly substituted"
            )
        weights = [
            compute_lineage_weight(
                feature_coverage_milli=record["feature_coverage_milli"],
                hardness_score_milli=record["hardness_score_milli"],
                features=record["features"],
                feature_lineage_counts=feature_lineage_counts,
                sampler_policy=sampler_policy,
            )
            for record in eligible
        ]
        totals = [w["total_weight"] for w in weights]
        if all(total == 0 for total in totals):
            raise _refuse(
                f"draw {draw_index}: every lineage eligible for cell {_format_cell(cell)} "
                "weights to zero under the policy bounds — refusing rather than drawing "
                "unweighted"
            )
        offset = rng.choices(range(len(eligible)), weights=totals, k=1)[0]
        record = eligible[offset]
        weight = weights[offset]
        lineage_id = record["lineage_root_id"]
        views[lineage_id] = views.get(lineage_id, 0) + 1
        draws.append(
            {
                "draw_index": draw_index,
                **{axis: cell[axis] for axis in _CELL_AXES},
                "lineage_root_id": lineage_id,
                "feature_coverage_milli": weight["feature_coverage_milli"],
                "hardness_milli": weight["hardness_milli"],
                "w_lineage": weight["w_lineage"],
                "w_rare_milli": weight["w_rare_milli"],
                "total_weight": weight["total_weight"],
                "lineage_views_in_epoch": views[lineage_id],
            }
        )

    payload: dict[str, Any] = {
        "ordered_exposure_plan_schema_version": ORDERED_EXPOSURE_PLAN_SCHEMA_VERSION,
        "run_seed": run_seed,
        "virtual_epoch_size": epoch_size,
        "matrix_hash": stage_data_matrix_hash(matrix),
        "sampler_policy_hash": s2_sampler_policy_hash(sampler_policy),
        "exposure_policy_hash": s2_exposure_policy_hash(exposure_policy),
        "feature_lineage_counts": {
            feature: feature_lineage_counts[feature] for feature in sorted(feature_lineage_counts)
        },
        "draws": draws,
    }
    payload["plan_id"] = make_id("plan", ORDERED_EXPOSURE_PLAN_DOMAIN, payload)
    payload["ordered_exposure_plan_hash"] = domain_hash(ORDERED_EXPOSURE_PLAN_DOMAIN, payload)
    _plan_validator().validate(payload)
    return payload
