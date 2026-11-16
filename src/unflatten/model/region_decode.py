"""Tensor-to-contract decoding for NodeVisualAligner predictions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from unflatten.model.node_lifecycle import NodeState
from unflatten.model.node_visual_aligner import (
    STATUS_VOCABULARY,
    NodeVisualAlignerHead,
    decode_bbox,
)
from unflatten.model.region_codec import PPM_MAX, Region

AlignmentStatus = Literal[
    "region", "abstain_not_visible", "abstain_container", "abstain_degenerate"
]


@dataclass(frozen=True, slots=True)
class RegionHypothesis:
    """One closed, page-local region prediction for one decoded node anchor."""

    local_node_id: str
    node_type: str
    alignment_status: AlignmentStatus
    regions: tuple[Region, ...]
    confidence_ppm: int

    def __post_init__(self) -> None:
        if not self.local_node_id or not self.node_type:
            raise ValueError("RegionHypothesis requires non-empty node identity")
        if self.alignment_status not in (*STATUS_VOCABULARY, "abstain_degenerate"):
            raise ValueError(f"unknown alignment status {self.alignment_status!r}")
        if not 0 <= self.confidence_ppm <= PPM_MAX:
            raise ValueError("confidence_ppm must be within 0..1000000")
        if self.alignment_status == "region" and len(self.regions) != 1:
            raise ValueError("region status requires exactly one region")
        if self.alignment_status != "region" and self.regions:
            raise ValueError("abstention must not carry a region")


@torch.inference_mode()
def decode_regions(
    head: NodeVisualAlignerHead,
    node_states: Sequence[NodeState],
    *,
    visual_hidden: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> tuple[RegionHypothesis, ...]:
    """Decode exactly one hypothesis for every node with a decode anchor.

    Geometry is materialized only *after* the status wins.  Thus abstention
    cannot accidentally become a plausible zero-area bounding box.
    """
    _validate_retained_visual_hidden(visual_hidden, image_grid_thw)
    anchored = tuple(node for node in node_states if node.anchor is not None)
    if not anchored:
        return ()
    anchors = torch.stack([node.anchor for node in anchored])
    # One visual row shared by every anchor: an expanded copy per node made the
    # aligner project keys/values per node (OOM next to production, 25.9.).
    visual_row = torch.zeros(len(anchored), dtype=torch.long, device=anchors.device)
    output = head(anchors, visual_hidden.unsqueeze(0), image_grid_thw, visual_row=visual_row)
    probabilities = torch.softmax(output.status_logits.float(), dim=-1)
    status_indices = probabilities.argmax(dim=-1)
    boxes = decode_bbox(output.bbox_logits)
    predictions: list[RegionHypothesis] = []
    for node, index, box, probabilities_row in zip(
        anchored, status_indices.tolist(), boxes.tolist(), probabilities
    ):
        status = STATUS_VOCABULARY[index]
        confidence = int(round(float(probabilities_row[index].detach()) * PPM_MAX))
        if status != "region":
            predictions.append(
                RegionHypothesis(node.local_node_id, node.node_type, status, (), confidence)
            )
            continue
        try:
            region = Region(0, box, "complete", confidence)
        except ValueError:
            predictions.append(
                RegionHypothesis(
                    node.local_node_id, node.node_type, "abstain_degenerate", (), confidence
                )
            )
        else:
            predictions.append(
                RegionHypothesis(node.local_node_id, node.node_type, status, (region,), confidence)
            )
    return tuple(predictions)


def _validate_retained_visual_hidden(
    visual_hidden: torch.Tensor, image_grid_thw: torch.Tensor
) -> None:
    """Require the retained decode memory to be exactly this page's grid.

    Training intentionally gives the aligner right-padded rows; free-running
    decoding has one page and no padding contract, so either a truncated row
    or a leaked prefix/suffix is provenance drift rather than data to slice.
    """
    if visual_hidden.dim() != 2:
        raise ValueError(
            "retained visual_hidden must be (steps, hidden), "
            f"got shape {tuple(visual_hidden.shape)}"
        )
    if image_grid_thw.shape != (1, 3) or image_grid_thw.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64
    ):
        raise ValueError("decode image_grid_thw must be one integer [t, h, w] row")
    t, h, w = (int(value) for value in image_grid_thw[0].tolist())
    if t != 1 or h <= 0 or w <= 0 or h % 2 or w % 2:
        raise ValueError("decode image_grid_thw must be [1, positive-even-h, positive-even-w]")
    expected_steps = (h // 2) * (w // 2)
    if visual_hidden.shape[0] != expected_steps:
        raise ValueError(
            "retained visual_hidden length does not exactly match decoded grid: "
            f"expected {expected_steps}, got {visual_hidden.shape[0]}"
        )
