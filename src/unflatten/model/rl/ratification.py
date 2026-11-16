"""S4.5 RL gradient ratification gate (RFC005-ERRATA-010, DEC-047 addendum A).

A real RL gradient step needs an on-disk, owner-signed record that names stage
``S4_5``, binds the exact composite ``policy_hash`` and (when known) the parent
checkpoint. S7 and level-router records are structurally refused. Anything
short of a well-formed record refuses: a missing or malformed record is exactly
as authorizing as none.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REQUIRED_STAGE: Final = "S4_5"
#: The two S4.5 arms a record may name (RL arm and its paired SFT continuation).
S4_5_CHILDREN: Final = ("rl", "sft_continuation")
_REQUIRED_FIELDS: Final = (
    "decision_id",
    "stage",
    "child",
    "policy_hash",
    "ratified",
    "ratified_by",
    "ratified_at",
)
_STRING_FIELDS: Final = tuple(f for f in _REQUIRED_FIELDS if f != "ratified")


class RlRatificationRequired(RuntimeError):
    """An S4.5 gradient step was requested without a valid ratification record."""


@dataclass(frozen=True, slots=True)
class RlRatificationRecord:
    decision_id: str
    stage: str
    child: str
    policy_hash: str
    ratified: bool
    ratified_by: str
    ratified_at: str
    parent_checkpoint_id: str | None = None


def load_ratification_record(path: Path) -> RlRatificationRecord:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RlRatificationRequired(
            f"ratification record {path} could not be read as JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RlRatificationRequired(f"ratification record {path} is not a JSON object")
    missing = [name for name in _REQUIRED_FIELDS if name not in payload]
    if missing:
        raise RlRatificationRequired(
            f"ratification record {path} is missing required field(s): {missing}"
        )
    for name in _STRING_FIELDS:
        value = payload[name]
        if not isinstance(value, str) or not value:
            raise RlRatificationRequired(
                f"ratification record {path} field {name!r} must be a non-empty string"
            )
    if payload["stage"] != REQUIRED_STAGE:
        raise RlRatificationRequired(
            f"ratification record {path} names stage {payload['stage']!r}, not {REQUIRED_STAGE!r}"
        )
    if payload["child"] not in S4_5_CHILDREN:
        raise RlRatificationRequired(
            f"ratification record {path} names child {payload['child']!r}, "
            f"not one of {S4_5_CHILDREN}"
        )
    if payload["ratified"] is not True:
        raise RlRatificationRequired(
            f"ratification record {path} has ratified={payload['ratified']!r}, not true"
        )
    parent = payload.get("parent_checkpoint_id")
    if parent is not None and not isinstance(parent, str):
        raise RlRatificationRequired(
            f"ratification record {path} field 'parent_checkpoint_id' must be a string"
        )
    return RlRatificationRecord(
        decision_id=payload["decision_id"],
        stage=payload["stage"],
        child=payload["child"],
        policy_hash=payload["policy_hash"],
        ratified=True,
        ratified_by=payload["ratified_by"],
        ratified_at=payload["ratified_at"],
        parent_checkpoint_id=parent,
    )


def require_ratification(
    record_path: Path | None,
    *,
    policy_hash: str,
    policy_status: str,
    parent_checkpoint_id: str | None = None,
) -> RlRatificationRecord:
    if record_path is None:
        raise RlRatificationRequired(
            "a real S4.5 gradient step requires an explicit ratification record; "
            "no ratification record was given"
        )
    record = load_ratification_record(record_path)
    if record.policy_hash != policy_hash:
        raise RlRatificationRequired(
            f"ratification record {record_path} binds policy_hash {record.policy_hash!r}, "
            f"not {policy_hash!r}"
        )
    if policy_status != "RATIFIED":
        raise RlRatificationRequired(
            f"policy status is {policy_status!r}; only a policy frozen as RATIFIED may train"
        )
    if parent_checkpoint_id is not None and record.parent_checkpoint_id != parent_checkpoint_id:
        raise RlRatificationRequired(
            f"ratification record {record_path} binds parent checkpoint "
            f"{record.parent_checkpoint_id!r}, not {parent_checkpoint_id!r}"
        )
    return record
