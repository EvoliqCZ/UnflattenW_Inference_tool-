"""RFC004-ERRATA-005 E4: the pre-encoder scout model.

A tiny, CPU-sized conv/MLP that predicts a per-cell risk-class map from a
low-resolution global page view, feeding
:func:`unflatten.model.visual_compression.region_plan.plan_region_resolution`
before the page ever reaches the real vision encoder (E4 §12.1's flow: page
-> [pre_encoder_compressor: importance/risk map -> deterministic resolution
plan] -> Qwen3-VL). It must be cheap enough to run on every page before the
real encoder does any work, so it stays a handful of conv layers, not a
second backbone.

Step-0 identity (E4 rule 2, DEC-047 condition 2): the classification head is
zero-initialized, so `argmax` picks index 0 ("protected") for every cell
deterministically -- every region resolves to full resolution, matching the
`bypass` pre-encoder profile exactly.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

from unflatten.model.visual_compression.adaptive_v0 import RISK_CLASSES
from unflatten.model.visual_compression.region_plan import RegionProposal, RiskClass

_N_RISK_CLASSES: Final = len(RISK_CLASSES)


class ScoutModel(nn.Module):
    """Predicts `(batch, grid, grid, 5)` risk-class logits from a
    `(batch, in_channels, H, W)` low-resolution page view.

    `AdaptiveAvgPool2d` fixes the output grid regardless of the input's
    `H, W`, so callers may hand it any downsampled page size.
    """

    def __init__(self, *, in_channels: int = 3, hidden: int = 16, grid: int = 8) -> None:
        super().__init__()
        if grid <= 0:
            raise ValueError(f"grid must be positive, got {grid!r}")
        self.grid = grid
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid, grid)),
        )
        self.head = nn.Linear(hidden, _N_RISK_CLASSES)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def forward(self, low_res_view: Tensor) -> Tensor:
        features = self.backbone(low_res_view)  # (batch, hidden, grid, grid)
        features = features.permute(0, 2, 3, 1)  # (batch, grid, grid, hidden)
        return self.head(features)  # (batch, grid, grid, 5)


def scout_logits_to_region_proposals(
    logits: Tensor, page_width_px: int, page_height_px: int
) -> tuple[RegionProposal, ...]:
    """Argmax `logits` (`(1, grid, grid, 5)`) into a `grid x grid` tiling of
    the page in original-page pixel coordinates, one `RegionProposal` per
    cell. Batch must be 1 (one page)."""
    if logits.shape[0] != 1:
        raise ValueError(
            f"scout_logits_to_region_proposals requires batch == 1, got {logits.shape[0]}"
        )
    grid = logits.shape[1]
    if logits.shape[2] != grid:
        raise ValueError(f"logits must be square over the grid dims, got {tuple(logits.shape)}")
    risk_idx = torch.argmax(logits[0], dim=-1)  # (grid, grid)
    proposals: list[RegionProposal] = []
    for row in range(grid):
        y0 = (row * page_height_px) // grid
        y1 = ((row + 1) * page_height_px) // grid
        for col in range(grid):
            if y1 <= y0:
                continue
            x0 = (col * page_width_px) // grid
            x1 = ((col + 1) * page_width_px) // grid
            if x1 <= x0:
                continue
            risk_class: RiskClass = RISK_CLASSES[int(risk_idx[row, col].item())]  # type: ignore[assignment]
            proposals.append(RegionProposal(x0=x0, y0=y0, x1=x1, y1=y1, risk_class=risk_class))
    return tuple(proposals)
