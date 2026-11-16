"""Band moments of pooled ViT tokens: the single feature function of the page-start head.

Qwen merged tokens are row-major over the merged grid (H' = h // merge, W' = w // merge).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import torch


def page_start_moments(
    pooled: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    merge_size: int,
    band_top: float,
    band_bottom: float,
) -> torch.Tensor:
    """float32[6D]: mean+population std over the top band, bottom band and full page."""
    if pooled.ndim != 2:
        raise ValueError(f"page_start_moments: pooled must be [V, D], got {tuple(pooled.shape)}")
    if merge_size < 1:
        raise ValueError(f"page_start_moments: merge_size must be >= 1, got {merge_size}")
    for name, value in (("band_top", band_top), ("band_bottom", band_bottom)):
        if not 0 < value <= 1:
            raise ValueError(f"page_start_moments: {name} must be in (0, 1], got {value!r}")
    if tuple(grid_thw.shape) != (3,):
        raise ValueError(f"page_start_moments: grid_thw must be [3], got {tuple(grid_thw.shape)}")
    t, h, w = (int(v) for v in grid_thw.tolist())
    rows, cols = h // merge_size, w // merge_size
    v, d = pooled.shape
    if t != 1:
        raise ValueError(f"page_start_moments: single images only (t == 1), got t={t}")
    if v != rows * cols or rows < 1:
        raise ValueError(
            f"page_start_moments: {v} tokens do not match grid_thw {(t, h, w)} "
            f"with merge_size {merge_size} ({rows}*{cols} tokens)"
        )
    x = pooled.float().reshape(rows, cols, d)
    k_top = min(rows, math.ceil(band_top * rows))
    k_bot = min(rows, math.ceil(band_bottom * rows))
    parts = []
    for region in (x[:k_top], x[rows - k_bot : rows], x):
        flat = region.reshape(-1, d)
        parts += [flat.mean(0), flat.std(0, unbiased=False)]
    return torch.cat(parts)


@dataclass(frozen=True)
class MomentsContract:
    visual_input_policy_hash: str
    feature_tap: str
    backbone_artifact_hash: str
    merge_size: int
    band_fraction_top: float
    band_fraction_bottom: float
    feature_identity_sha256: str
    visual_dim: int

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> MomentsContract:
        names = {f.name for f in fields(cls)}
        if set(obj) != names:
            raise ValueError(
                f"moments contract: fields differ, missing {sorted(names - set(obj))}, "
                f"unexpected {sorted(set(obj) - names)}"
            )
        return cls(**obj)

    def refuse_mismatch(self, other: MomentsContract) -> None:
        bad = [f.name for f in fields(self) if getattr(self, f.name) != getattr(other, f.name)]
        if bad:
            raise ValueError(f"moments contract mismatch in {bad}")
