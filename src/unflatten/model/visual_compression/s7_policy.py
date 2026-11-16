"""RFC005-ERRATA-007 E5 / DEC-045: the S7 stage policy draft.

Every threshold, ratio and lambda the S7 VC-1/VC-2 dormant trainers read is
a field of this policy document, never a code literal (DEC-047 condition
4). The policy is loaded from ``policies/s7_stage_policy.json`` and its
``status`` is ``"PROPOSED"``: no S7 gradient step may treat a placeholder
value as a frozen number (see :mod:`unflatten.model.visual_compression.
s7_ratification` for the separate gate that keeps a placeholder policy from
ever reaching a real gradient without an explicit ratification record).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from unflatten.model.hashing import domain_hash

_POLICY_PATH: Final = Path(__file__).parent / "policies" / "s7_stage_policy.json"
_POLICY_DOMAIN: Final = "unflatten.visual-compression.s7-stage-policy/v1"

#: Only status this module has ever seen ratified: none yet. Kept as a
#: named constant so a future ratification is a one-line diff, not a
#: silent behaviour change buried in a comparison.
RATIFIED_STATUS = "RATIFIED"


class S7StagePolicyError(ValueError):
    """The policy document does not match the shape this module reads."""


def load_s7_stage_policy() -> dict[str, object]:
    """Load the S7 stage policy draft verbatim (no defaults injected).

    Returns the raw JSON payload so callers can read whichever section
    (``vc1``, ``vc2``, ...) they need without this module having to grow a
    field for every future child.
    """
    payload = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    for required in ("s7_stage_policy_schema_version", "policy_version", "status"):
        if required not in payload:
            raise S7StagePolicyError(f"s7_stage_policy.json is missing required field {required!r}")
    return payload


def is_policy_ratified(policy: dict[str, object]) -> bool:
    """True iff an owner has flipped ``status`` to ``"RATIFIED"``.

    A draft policy (``"PROPOSED"``, the only value on disk today) is never
    treated as ratified -- this function has no other branch to fall back
    to a permissive default.
    """
    return policy.get("status") == RATIFIED_STATUS


def s7_stage_policy_hash(policy: dict[str, object]) -> str:
    """Domain-separated identity of a loaded policy payload, for run manifests."""
    return domain_hash(_POLICY_DOMAIN, policy)
