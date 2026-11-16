"""RFC004 §37.2 / DEC-047: the S7 `dflash_speculator` stage policy draft.

Block size, step count, hidden-state taps and anchor policy are read from
``policies/s7_dflash_policy.json`` -- never a code literal (DEC-047
condition 4), same discipline as
:mod:`unflatten.model.visual_compression.s7_policy`. This is a dedicated
file rather than an addition to the VC program's shared
``s7_stage_policy.json`` so the two dormant programs never collide on the
same on-disk document while both are under active, concurrent development.
Its ``status`` is ``"PROPOSED"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final

from unflatten.model.hashing import domain_hash

if TYPE_CHECKING:
    from unflatten.model.speculative.drafter import VerifyCostModel

_POLICY_PATH: Final = Path(__file__).parent / "policies" / "s7_dflash_policy.json"
_POLICY_DOMAIN: Final = "unflatten.speculative.s7-dflash-policy/v1"

RATIFIED_STATUS = "RATIFIED"


class S7DflashPolicyError(ValueError):
    """The policy document does not match the shape this module reads."""


def load_s7_dflash_policy() -> dict[str, object]:
    """Load the S7 `dflash_speculator` policy draft verbatim, no defaults injected."""
    payload = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    for required in ("s7_dflash_policy_schema_version", "policy_version", "status", "dflash"):
        if required not in payload:
            raise S7DflashPolicyError(
                f"s7_dflash_policy.json is missing required field {required!r}"
            )
    return payload


def is_policy_ratified(policy: dict[str, object]) -> bool:
    """True iff an owner has flipped ``status`` to ``"RATIFIED"``."""
    return policy.get("status") == RATIFIED_STATUS


def s7_dflash_policy_hash(policy: dict[str, object]) -> str:
    """Domain-separated identity of a loaded policy payload, for run manifests."""
    return domain_hash(_POLICY_DOMAIN, policy)


def verify_cost_model_from_policy(
    policy: dict[str, object],
) -> tuple[VerifyCostModel, bool]:
    """`(VerifyCostModel, is_placeholder)` from the policy's
    `verify_cost_model` block (2026-09-28, confidence-scheduled
    verification). `is_placeholder` mirrors the repo's other stage-gate
    placeholders (`general_stage_gate_policy.py`'s `insufficient_evidence`
    discipline): any speedup CLAIM made while it is `True` must be reported
    as not yet backed by a frozen, owner-ratified measurement, even though
    the numbers themselves (today's best real measurement) are usable."""
    from unflatten.model.speculative.drafter import VerifyCostModel

    block = policy["verify_cost_model"]
    model = VerifyCostModel(
        plain_step_cost_ms=block["plain_step_cost_ms"],
        verify_round_cost_ms=block["verify_round_cost_ms"],
    )
    return model, bool(block.get("is_placeholder", True))
