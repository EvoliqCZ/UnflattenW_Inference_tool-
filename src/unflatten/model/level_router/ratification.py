"""RFC007 gradient ratification gate (DEC-047 condition 5).

Same shape as `visual_compression.s7_ratification`: a real gradient step
through the level router needs an explicit, on-disk, owner-signed record.
Two additions, both because this module can run inside a dense v16 run and
not only inside a dormant stage: the record must bind the EXACT policy it
ratifies (`policy_hash`), so ratifying one topology never authorizes
another; and a policy still marked `status="PROPOSED"` is refused even
with a record. No such record exists in the repo today, so every trainer
path that reaches this gate refuses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from unflatten.model.level_router.policy import LevelRouterPolicy

LEVEL_ROUTER_MODULE_ID: Final = "RFC007-level-router"

#: S4: the v16 dense fine-tune from the v15b winner, run through `s4_train`
#: (DEC-053 point 1.6, dense fallback). S6: both arms of the paired
#: dense/MoE run, if the owner ratifies S6 (`s6_train`).
KNOWN_STAGES: Final[frozenset[str]] = frozenset({"S4", "S6"})

_REQUIRED_FIELDS = (
    "decision_id",
    "module",
    "stage",
    "policy_hash",
    "ratified",
    "ratified_by",
    "ratified_at",
)


class LevelRouterRatificationRequired(RuntimeError):
    """A level-router gradient step was requested without a valid record."""


@dataclass(frozen=True, slots=True)
class LevelRouterRatificationRecord:
    decision_id: str
    module: str
    stage: str
    policy_hash: str
    ratified_by: str
    ratified_at: str


def require_level_router_ratification(
    *, stage: str, policy: LevelRouterPolicy, ratification_record: Path | None
) -> LevelRouterRatificationRecord:
    if stage not in KNOWN_STAGES:
        raise ValueError(f"stage must be one of {sorted(KNOWN_STAGES)}, got {stage!r}")
    if ratification_record is None:
        raise LevelRouterRatificationRequired(
            f"a {stage} gradient step through the RFC007 level router requires an explicit "
            "ratification record (--level-router-ratification); none was given"
        )
    try:
        payload = json.loads(Path(ratification_record).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LevelRouterRatificationRequired(
            f"ratification record {ratification_record} could not be read as JSON: {exc}"
        ) from exc
    missing = [name for name in _REQUIRED_FIELDS if name not in payload]
    if missing:
        raise LevelRouterRatificationRequired(
            f"ratification record {ratification_record} is missing field(s): {missing}"
        )
    expected = {
        "module": LEVEL_ROUTER_MODULE_ID,
        "stage": stage,
        "policy_hash": policy.policy_hash(),
    }
    for name, value in expected.items():
        if payload[name] != value:
            raise LevelRouterRatificationRequired(
                f"ratification record {ratification_record} has {name}={payload[name]!r}, "
                f"not {value!r}"
            )
    if payload["ratified"] is not True:
        raise LevelRouterRatificationRequired(
            f"ratification record {ratification_record} has ratified={payload['ratified']!r}, "
            "not true"
        )
    if policy.status != "RATIFIED":
        raise LevelRouterRatificationRequired(
            f"level router policy status is {policy.status!r} (PROPOSED placeholder values); "
            "only a policy the owner froze as RATIFIED may train"
        )
    return LevelRouterRatificationRecord(
        decision_id=payload["decision_id"],
        module=payload["module"],
        stage=payload["stage"],
        policy_hash=payload["policy_hash"],
        ratified_by=payload["ratified_by"],
        ratified_at=payload["ratified_at"],
    )
