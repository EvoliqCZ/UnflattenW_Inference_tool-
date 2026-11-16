"""`LevelRouterPolicy` -- RFC007 §6 topology as data (DEC-047 condition 4).

Every choice the router makes about its own shape lives here as a field,
never as a module constant: which visual levels it reads, the memory width,
the number of read heads, which AST containers get their own grammar-state
bucket, the state-embedding width, the routing temperature and where the
output is added. `LEVEL_ROUTER_POLICY_V0_PROPOSED` is RFC007's starting
point and carries `status="PROPOSED"`; no trainer accepts it for a real
gradient step without a ratification record (`ratification.py`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from unflatten.model import events_vocab
from unflatten.model.hashing import domain_hash

LEVEL_ROUTER_POLICY_ID = "document-aware-level-router.v0"
_POLICY_DOMAIN = "rfc007.level-router-policy/v0"

#: `"final"` is `TeacherForcedBatch.visual_features` (the merged tokens the
#: decoder already receives); `"deepstack.<k>"` is
#: `TeacherForcedBatch.deepstack_visual_embeds[k]` (DEC-056, native ViT
#: layers 5/11/17 on the 4B pin, after `deepstack_merger_list`). The Node
#: Visual Aligner level is RFC007 §5.3 OPEN and refused here until ratified.
FINAL_LEVEL = "final"
_DEEPSTACK_LEVEL = re.compile(r"^deepstack\.(0|[1-9][0-9]*)$")

#: Where the reader's output is added. Only one attach point exists in v0;
#: it is still a field so a later policy can add the head-input variant
#: without a code constant changing meaning.
ATTACH_POINTS: tuple[str, ...] = ("post_expert_adapter",)
POLICY_STATUSES: tuple[str, ...] = ("PROPOSED", "RATIFIED")


class LevelRouterPolicyError(ValueError):
    """An invalid `LevelRouterPolicy` field or payload."""


@dataclass(frozen=True)
class LevelRouterPolicy:
    level_names: tuple[str, ...]
    memory_dim: int
    num_heads: int
    state_containers: tuple[str, ...]
    state_embedding_dim: int
    router_temperature_milli: int
    attach_point: str
    status: str
    policy_id: str = LEVEL_ROUTER_POLICY_ID

    def __post_init__(self) -> None:
        if self.policy_id != LEVEL_ROUTER_POLICY_ID:
            raise LevelRouterPolicyError(
                f"policy_id must be {LEVEL_ROUTER_POLICY_ID!r}, got {self.policy_id!r}"
            )
        if not self.level_names:
            raise LevelRouterPolicyError("level_names must name at least one level")
        if len(set(self.level_names)) != len(self.level_names):
            raise LevelRouterPolicyError(f"duplicate level in level_names {self.level_names!r}")
        for name in self.level_names:
            if name != FINAL_LEVEL and not _DEEPSTACK_LEVEL.match(name):
                raise LevelRouterPolicyError(
                    f"unknown level {name!r}: only 'final' and 'deepstack.<k>' exist in v0 "
                    "(the Node Visual Aligner level, e.g. 'aligner', is RFC007 §5.3 OPEN)"
                )
        _positive("memory_dim", self.memory_dim)
        _positive("num_heads", self.num_heads)
        if self.memory_dim % self.num_heads:
            raise LevelRouterPolicyError(
                f"memory_dim {self.memory_dim} must be divisible by num_heads {self.num_heads}"
            )
        if len(set(self.state_containers)) != len(self.state_containers):
            raise LevelRouterPolicyError(
                f"duplicate container in state_containers {self.state_containers!r}"
            )
        for container in self.state_containers:
            if container not in events_vocab.CONTAINER_TYPES:
                raise LevelRouterPolicyError(
                    f"state_containers entry {container!r} is not an RFC004 container type"
                )
        _positive("state_embedding_dim", self.state_embedding_dim)
        _positive("router_temperature_milli", self.router_temperature_milli)
        if self.attach_point not in ATTACH_POINTS:
            raise LevelRouterPolicyError(
                f"attach_point must be one of {ATTACH_POINTS}, got {self.attach_point!r}"
            )
        if self.status not in POLICY_STATUSES:
            raise LevelRouterPolicyError(
                f"status must be one of {POLICY_STATUSES}, got {self.status!r}"
            )

    def deepstack_level_indices(self) -> tuple[int, ...]:
        """The `deepstack_visual_embeds` indices this policy reads, in order."""
        return tuple(
            int(match.group(1))
            for name in self.level_names
            if (match := _DEEPSTACK_LEVEL.match(name))
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["level_names"] = list(self.level_names)
        payload["state_containers"] = list(self.state_containers)
        return payload

    def policy_hash(self) -> str:
        return domain_hash(_POLICY_DOMAIN, self.to_payload())


def _positive(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise LevelRouterPolicyError(f"{name} must be a positive integer, got {value!r}")


def level_router_policy_from_payload(payload: Mapping[str, Any]) -> LevelRouterPolicy:
    """Inverse of `LevelRouterPolicy.to_payload`; an unknown or missing key
    is refused, never ignored."""
    fields = set(LevelRouterPolicy.__dataclass_fields__)
    unknown = set(payload) - fields
    if unknown:
        raise LevelRouterPolicyError(f"unknown level-router policy key(s): {sorted(unknown)}")
    missing = fields - set(payload) - {"policy_id"}
    if missing:
        raise LevelRouterPolicyError(f"missing level-router policy key(s): {sorted(missing)}")
    values = dict(payload)
    values["level_names"] = tuple(values["level_names"])
    values["state_containers"] = tuple(values["state_containers"])
    return LevelRouterPolicy(**values)


#: RFC007 §6 starting point (PROPOSED). Four levels: final + the three
#: native DeepStack levels; 256-wide memory, 4 read heads; the containers
#: whose visual evidence differs most by RFC007 §4.2 (table layout, row,
#: cell geometry, paragraph/list-item text, story regions).
LEVEL_ROUTER_POLICY_V0_PROPOSED = LevelRouterPolicy(
    level_names=("final", "deepstack.0", "deepstack.1", "deepstack.2"),
    memory_dim=256,
    num_heads=4,
    state_containers=(
        "table",
        "row",
        "cell",
        "paragraph",
        "list_item",
        "header",
        "footer",
        "footnote",
        "endnote",
        "text_frame",
    ),
    state_embedding_dim=32,
    router_temperature_milli=1000,
    attach_point="post_expert_adapter",
    status="PROPOSED",
)


def load_level_router_policy(path: Path | str) -> LevelRouterPolicy:
    """Read a policy JSON file (`LevelRouterPolicy.to_payload` shape)."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LevelRouterPolicyError(
            f"level router policy {path} could not be read: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise LevelRouterPolicyError(f"level router policy {path} must hold a JSON object")
    return level_router_policy_from_payload(payload)
