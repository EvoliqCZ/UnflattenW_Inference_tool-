"""Closed collation of S4.5 rollouts for the update path (RFC005-ERRATA-010 E3.1/E7.2).

What reaches the teacher-forced forward is tokens, masks and group indices.
Reward, gold, advantage and any log-probability never enter a tensor of this
batch (E3.1: reward/gold are not model inputs; E4: old log-probs are
recomputed on the update path). The envelope is closed in three ways:

1. an explicit allow-list of keys, each with a required dtype class, so a float
   reward smuggled under an integer key is refused by dtype, not trusted;
2. a forbidden-name scan that runs FIRST, so ``reward``/``gold_ast`` is named
   as forbidden rather than reported as merely unknown;
3. computed keys cannot be overridden through ``extra_fields``.

Rewards travel to ``loss.l_rl`` through its own arguments, never through this
batch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import torch
from torch import Tensor

from unflatten.model.rl.rollout import (
    FORBIDDEN_FIELD_PARTS,
    STATUS_TERMINATED,
    RolloutTrajectory,
)

#: key -> required dtype class ("int" | "bool" | "float").
ALLOWED_KEYS: Final = {
    "token_ids": "int",
    "padding_mask": "bool",
    "legal_mask": "bool",
    "legal_set_sizes": "int",
    "group_ids": "int",
    "rollout_index": "int",
    "terminated": "bool",
    "visual_features": "float",
    "image_grid_thw": "int",
}
COMPUTED_KEYS: Final = (
    "token_ids",
    "padding_mask",
    "legal_mask",
    "legal_set_sizes",
    "group_ids",
    "rollout_index",
    "terminated",
)
#: Names no key may contain, on top of the rollout's own list.
FORBIDDEN_KEY_PARTS: Final = (*FORBIDDEN_FIELD_PARTS, "reference")
_INT_DTYPES: Final = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


class RolloutBatchRefused(ValueError):
    """A rollout batch that leaks, or is not inside the closed envelope."""


@dataclass(frozen=True)
class RolloutBatch:
    token_ids: Tensor
    padding_mask: Tensor
    legal_mask: Tensor
    legal_set_sizes: Tensor
    group_ids: Tensor
    rollout_index: Tensor
    terminated: Tensor
    extras: Mapping[str, Tensor]

    def as_mapping(self) -> dict[str, Tensor]:
        out = {k: getattr(self, k) for k in COMPUTED_KEYS}
        out.update(self.extras)
        return out


def _dtype_class(t: Tensor) -> str:
    if t.dtype == torch.bool:
        return "bool"
    if t.dtype in _INT_DTYPES:
        return "int"
    return "float" if t.is_floating_point() else "other"


def assert_batch_closed(batch: Mapping[str, Any]) -> None:
    """Refuse any key outside :data:`ALLOWED_KEYS`, any forbidden name, and any
    tensor whose dtype class does not match its key."""
    for key in batch:
        low = str(key).lower()
        hit = [part for part in FORBIDDEN_KEY_PARTS if part in low]
        if hit:
            raise RolloutBatchRefused(
                f"forbidden field {key!r} (matches {hit}): reward/gold/advantage/log-prob "
                "never enter the rollout batch"
            )
    for key, value in batch.items():
        if key not in ALLOWED_KEYS:
            raise RolloutBatchRefused(f"unknown field {key!r}; allowed: {sorted(ALLOWED_KEYS)}")
        if not isinstance(value, Tensor):
            raise RolloutBatchRefused(f"field {key!r} must be a tensor, got {type(value).__name__}")
        want, got = ALLOWED_KEYS[key], _dtype_class(value)
        if got != want:
            raise RolloutBatchRefused(f"field {key!r} dtype {value.dtype} is {got}, must be {want}")


def collate_rollout_batch(
    trajectories: Sequence[RolloutTrajectory],
    *,
    group_ids: Sequence[int],
    pad_id: int,
    extra_fields: Mapping[str, Tensor] | None = None,
) -> RolloutBatch:
    """Pad ``trajectories`` into one batch. ``group_ids[i]`` is the page/group of
    ``trajectories[i]``. ``extra_fields`` may add only allow-listed,
    non-computed keys (``visual_features``, ``image_grid_thw``)."""
    if not trajectories:
        raise RolloutBatchRefused("empty trajectory list")
    if len(group_ids) != len(trajectories):
        raise RolloutBatchRefused(
            f"group_ids length {len(group_ids)} != {len(trajectories)} trajectories"
        )
    extras = dict(extra_fields or {})
    clash = sorted(set(extras) & set(COMPUTED_KEYS))
    if clash:
        raise RolloutBatchRefused(f"extra_fields may not override computed keys: {clash}")
    width = max(1, max(len(t.token_ids) for t in trajectories))
    n = len(trajectories)
    ids = torch.full((n, width), pad_id, dtype=torch.long)
    sizes = torch.zeros((n, width), dtype=torch.long)
    pad = torch.zeros((n, width), dtype=torch.bool)
    for i, t in enumerate(trajectories):
        k = len(t.token_ids)
        if len(t.legal_set_sizes) != k:
            raise RolloutBatchRefused(f"trajectory {i}: legal_set_sizes length != token count")
        ids[i, :k] = torch.tensor(t.token_ids, dtype=torch.long)
        sizes[i, :k] = torch.tensor(t.legal_set_sizes, dtype=torch.long)
        pad[i, :k] = True
    batch = RolloutBatch(
        token_ids=ids,
        padding_mask=pad,
        legal_mask=pad & (sizes > 0),
        legal_set_sizes=sizes,
        group_ids=torch.tensor(list(group_ids), dtype=torch.long),
        rollout_index=torch.tensor([t.rollout_index for t in trajectories], dtype=torch.long),
        terminated=torch.tensor([t.status == STATUS_TERMINATED for t in trajectories]),
        extras=extras,
    )
    assert_batch_closed(batch.as_mapping())
    return batch
