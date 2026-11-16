"""RFC005-ERRATA-006 B3 policy: how many tokens make a "gap" or
"duplicate", and whether B3 means zero pages or a rate floor.

E6's own text leaves both open ("PLACEHOLDER values that make the gate
`insufficient_evidence` until the owner freezes them" -- this task's own
instruction, since the errata does not fix the rule itself): `min_gap_
tokens`, `min_duplicate_tokens`, and `verdict_mode` ("zero_pages": B3
passes only when NO measured page has a gap or duplicate; "rate": B3
passes when the clean-page rate is `>= rate_floor_micro`) are all owner
decisions. `PLACEHOLDER_TEXT_CONTINUITY_POLICY` (also shipped as
`text_continuity_policy.placeholder.json` beside this module) carries
`is_placeholder=True`, which `text_continuity_report.build_text_
continuity_report` reads to force `verdict="insufficient_evidence"`
regardless of what the measurement finds, and which `unflatten.model.
s4_stage_gate_report.gate_b3_no_text_gaps_or_duplicates` reads the same
way -- so a placeholder can never accidentally promote S4 (RFC005 §12.2
fail-closed).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TEXT_CONTINUITY_POLICY_SCHEMA_VERSION = "1.0.0"

_VALID_MODES = frozenset({"zero_pages", "rate"})

_REQUIRED_KEYS = frozenset(
    {
        "text_continuity_policy_schema_version",
        "min_gap_tokens",
        "min_duplicate_tokens",
        "verdict_mode",
        "rate_floor_micro",
        "is_placeholder",
    }
)

#: PLACEHOLDER (owner has not frozen B3's thresholds, 2026-09-26). N=1 is
#: deliberately the smallest possible unit rather than a considered
#: threshold -- `is_placeholder=True` is what actually withholds the
#: gate, not this value.
PLACEHOLDER_TEXT_CONTINUITY_POLICY: dict[str, Any] = {
    "text_continuity_policy_schema_version": TEXT_CONTINUITY_POLICY_SCHEMA_VERSION,
    "min_gap_tokens": 1,
    "min_duplicate_tokens": 1,
    "verdict_mode": "zero_pages",
    "rate_floor_micro": None,
    "is_placeholder": True,
}

DEFAULT_POLICY_PATH = Path(__file__).parent / "text_continuity_policy.placeholder.json"


class TextContinuityPolicyInvalid(ValueError):
    """A text-continuity policy document is malformed."""


def validate_text_continuity_policy(doc: Mapping[str, Any]) -> None:
    missing = _REQUIRED_KEYS - set(doc)
    if missing:
        raise TextContinuityPolicyInvalid(
            f"text continuity policy missing key(s): {sorted(missing)}"
        )
    if doc["verdict_mode"] not in _VALID_MODES:
        raise TextContinuityPolicyInvalid(
            f"text continuity policy verdict_mode must be one of {sorted(_VALID_MODES)}, "
            f"got {doc['verdict_mode']!r}"
        )
    for key in ("min_gap_tokens", "min_duplicate_tokens"):
        value = doc[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TextContinuityPolicyInvalid(
                f"text continuity policy {key} must be an integer >= 1, got {value!r}"
            )
    if doc["verdict_mode"] == "rate":
        rate = doc.get("rate_floor_micro")
        if isinstance(rate, bool) or not isinstance(rate, int) or not 0 <= rate <= 1_000_000:
            raise TextContinuityPolicyInvalid(
                "text continuity policy verdict_mode='rate' requires an integer "
                "rate_floor_micro in 0..1000000"
            )
    if not isinstance(doc["is_placeholder"], bool):
        raise TextContinuityPolicyInvalid("text continuity policy is_placeholder must be a bool")


def load_text_continuity_policy(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TextContinuityPolicyInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TextContinuityPolicyInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_text_continuity_policy(payload)
    return payload
