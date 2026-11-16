"""Compute-matched SFT-continuation arm for S4.5 (RFC005-ERRATA-010 E4.1, E7.3c).

The RL arm and the SFT-continuation arm must see the same frozen
``StageDataMatrix``, the same ordered exposure plan and the same number of
optimizer steps. The plan is built once through
``s6_paired_exposure.build_paired_exposure_plans`` (itself a composition of
``s2_sampler.build_ordered_exposure_plan``) and wrapped, unchanged, into two
envelopes under this package's own domain; the S6 arm ids (dense/sparse) and
domain are not reused. :func:`verify_paired_arms` compares the two envelopes
field by field, so a re-sealed arm with one reordered draw is still refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from unflatten.model.hashing import domain_hash
from unflatten.model.rl.ids import PAIRED_ARM_DOMAIN
from unflatten.model.s6_paired_exposure import build_paired_exposure_plans

ARM_RL: Final = "rl_arm"
ARM_SFT: Final = "sft_continuation_arm"
SCHEMA_VERSION: Final = "1.0.0"
_KEYS: Final = frozenset(
    {
        "paired_arm_schema_version",
        "arm_id",
        "architecture_manifest_ref",
        "optimizer_steps",
        "exposure_plan",
        "paired_arm_hash",
    }
)
_SHARED: Final = ("paired_arm_schema_version", "optimizer_steps", "exposure_plan")


class PairedPlanRefused(ValueError):
    """The two arms are not a compute-matched pair."""


def _seal(arm_id: str, ref: str, steps: int, plan: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "paired_arm_schema_version": SCHEMA_VERSION,
        "arm_id": arm_id,
        "architecture_manifest_ref": ref,
        "optimizer_steps": steps,
        "exposure_plan": dict(plan),
    }
    return {**body, "paired_arm_hash": domain_hash(PAIRED_ARM_DOMAIN, body)}


def build_rl_sft_paired_plans(
    *,
    matrix: Mapping[str, Any],
    lineages: Sequence[Mapping[str, Any]],
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
    run_seed: int,
    optimizer_steps: int,
    rl_architecture_manifest_ref: str,
    sft_architecture_manifest_ref: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(optimizer_steps) is not int or optimizer_steps < 1:
        raise PairedPlanRefused(f"optimizer_steps must be an int >= 1, got {optimizer_steps!r}")
    if rl_architecture_manifest_ref == sft_architecture_manifest_ref:
        raise PairedPlanRefused(
            "RL and SFT architecture manifest refs are identical "
            f"({rl_architecture_manifest_ref!r}): two labels for one arm, not a pair"
        )
    shared, _ = build_paired_exposure_plans(
        matrix=matrix,
        lineages=lineages,
        sampler_policy=sampler_policy,
        exposure_policy=exposure_policy,
        run_seed=run_seed,
        dense_architecture_manifest_ref=rl_architecture_manifest_ref,
        sparse_architecture_manifest_ref=sft_architecture_manifest_ref,
    )
    plan = shared["exposure_plan"]
    return (
        _seal(ARM_RL, rl_architecture_manifest_ref, optimizer_steps, plan),
        _seal(ARM_SFT, sft_architecture_manifest_ref, optimizer_steps, plan),
    )


def _check_arm(arm: Mapping[str, Any], expected_id: str) -> None:
    if set(arm) != _KEYS:
        raise PairedPlanRefused(f"{expected_id}: arm keys {sorted(set(arm) ^ _KEYS)} differ")
    body = {k: v for k, v in arm.items() if k != "paired_arm_hash"}
    if domain_hash(PAIRED_ARM_DOMAIN, body) != arm["paired_arm_hash"]:
        raise PairedPlanRefused(f"{expected_id}: paired_arm_hash does not verify")
    if arm["arm_id"] != expected_id:
        raise PairedPlanRefused(f"arm_id {arm['arm_id']!r} where {expected_id!r} was expected")


def verify_paired_arms(rl_arm: Mapping[str, Any], sft_arm: Mapping[str, Any]) -> None:
    _check_arm(rl_arm, ARM_RL)
    _check_arm(sft_arm, ARM_SFT)
    if rl_arm["architecture_manifest_ref"] == sft_arm["architecture_manifest_ref"]:
        raise PairedPlanRefused("both arms carry the same architecture_manifest_ref")
    for key in _SHARED:
        if rl_arm[key] != sft_arm[key]:
            raise PairedPlanRefused(f"arms differ in {key}: the comparison is not compute-matched")
