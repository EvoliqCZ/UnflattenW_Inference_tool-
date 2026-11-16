"""Opt-in chain policy `image-region-snap.v0` (RATIFIED 2026-10-16, DEC-077;
`docs/rfc/RFC004-ERRATA-011.md` E1).

The model predicts an image node's page region (`SealedRegionPrediction`,
ppm of the page) and `run._recover_resources` cuts exactly that rectangle
out of the 300 DPI raster (`resource_recovery.exact_crop`). Measured on the
S4 v16 selection cohort (40 pages, 25 gold images, checkpoints 170k-179k,
2026-10-16): the predicted boxes have the right SIZE (median model/gold
width 1.06, height 1.12) but their centre is shifted by ~0.3 of the image
width and ~0.2 of its height, so 15 of 25 recovered images were cropped
(ink recall < 0.9); only 3-4 of 25 reached IoU >= 0.6 with the gold box.
Logos are small, so a small shift cuts half the logo away.

This policy tightens each predicted image box onto the ink the raster
actually shows there, BEFORE recovery, and counts every outcome:

- scope: region predictions of `floating_image`/`inline_image` nodes with
  `alignment_status == "region"` and exactly one region; abstained or
  multi-region predictions are skipped and counted, other node types are
  never touched;
- ink: the raster is analysed at 1/`ANALYSIS_SCALE` resolution; a pixel is
  ink when its gray level is below `INK_GRAY_MAX` or its RGB spread exceeds
  `INK_SATURATION_MIN` (coloured logos on white paper);
- search window: the predicted box grown by `SEARCH_GROW` of its own size
  per side (at least `SEARCH_MIN_FRACTION` of the page); ink outside the
  window is never considered;
- snap: the box first shrinks to the ink inside it, then grows edge by edge
  while the next row/column (within the current span) carries ink, bridging
  at most `GAP_PX` blank rows/columns -- so a text block a clear gap away is
  not absorbed; the two steps repeat until stable;
- refusals, each counted and leaving the prediction as it was: no ink inside
  the predicted box (`no_ink`), a snapped box wider or taller than
  `GROWTH_CAP` x the prediction (`exceeds_cap`), a snapped box that reaches
  the search window's edge while the ink continues outside it
  (`ink_beyond_window`: a rule, border or large block runs through the box,
  not a self-contained image), no verified 300 DPI raster
  (`no_verified_raster`, the same DPI check `recovered_image_materialization`
  uses);
- a box whose every edge moves less than `TOLERANCE_PX` is `left_as_predicted`
  and the page object is returned unchanged.

What it does NOT do: it never creates, removes or retypes an image node, never
splits a box that covers two images, and never touches a floating image's
`/anchor/*` -- a snapped box changes only the recovered pixels (and, through
`recovered-image-materialization.v0`, the displayed size). Offline test on the
cohort (2026-10-16, pure measurement, no decode): cropped images 15 -> 4 of
25, IoU >= 0.6 from 3 to 9, one box of 25 made worse.
"""

from __future__ import annotations

import dataclasses
import io
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from unflatten.eval.builder_chain import recovered_image_materialization as rim
from unflatten.eval.s2_gate import sealed

POLICY_ID = "image-region-snap.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-011.md"
COUNT_PREFIX = "image_region_snap:"

_IMAGE_TYPES = ("floating_image", "inline_image")
PPM = 1_000_000
#: The raster is analysed at 1/4 of 300 DPI (75 DPI): a 2480 x 3508 page
#: becomes 620 x 877, which keeps a 10 pt logo at ~10 px.
ANALYSIS_SCALE = 4
INK_GRAY_MAX = 235
INK_SATURATION_MIN = 28
SEARCH_GROW = 0.35
SEARCH_MIN_FRACTION = 0.015
GAP_PX = 3
GROWTH_CAP = 1.8
TOLERANCE_PX = 2
RULES = (
    "ink_mask_gray_or_saturation.v0",
    "search_window_grown_prediction.v0",
    "shrink_to_ink_then_grow_across_gaps.v0",
    "growth_cap_refuses.v0",
    "ink_beyond_window_refuses.v0",
    "anchor_and_node_type_untouched.v0",
)


def policy_record() -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "raster_dpi": rim.RASTER_DPI,
        "dpi_tolerance": str(rim.DPI_TOLERANCE),
        "analysis_scale": ANALYSIS_SCALE,
        "ink_gray_max": INK_GRAY_MAX,
        "ink_saturation_min": INK_SATURATION_MIN,
        "search_grow_milli": round(SEARCH_GROW * 1000),
        "search_min_fraction_milli": round(SEARCH_MIN_FRACTION * 1000),
        "gap_px": GAP_PX,
        "growth_cap_milli": round(GROWTH_CAP * 1000),
        "tolerance_px": TOLERANCE_PX,
        "scope": {"node_types": list(_IMAGE_TYPES), "alignment_status": "region"},
        "rules": list(RULES),
    }


def _ink_mask(raster: bytes) -> np.ndarray | None:
    """Boolean ink mask at analysis scale, or None for unreadable bytes."""
    try:
        with Image.open(io.BytesIO(raster)) as opened:
            rgb = opened.convert("RGB")
            width, height = rgb.size
            small = rgb.resize(
                (max(1, width // ANALYSIS_SCALE), max(1, height // ANALYSIS_SCALE)),
                Image.Resampling.BOX,
            )
    except (OSError, UnidentifiedImageError):
        return None
    pixels = np.asarray(small, dtype=np.int16)
    gray = pixels.mean(axis=2)
    saturation = pixels.max(axis=2) - pixels.min(axis=2)
    return (gray < INK_GRAY_MAX) | (saturation > INK_SATURATION_MIN)


def _grow(start: int, step: int, lo: int, hi: int, has_ink: Callable[[int], bool]) -> int:
    """The farthest index from `start` in direction `step` that still carries
    ink, bridging at most `GAP_PX` blank indices on the way."""
    edge, blank, pos = start, 0, start + step
    while lo <= pos < hi:
        if has_ink(pos):
            edge, blank = pos, 0
        else:
            blank += 1
            if blank > GAP_PX:
                break
        pos += step
    return edge


def _snap(
    mask: np.ndarray, bbox_ppm: tuple[int, ...]
) -> tuple[str, tuple[int, int, int, int] | None]:
    """`(outcome, new_bbox_ppm | None)` for one predicted box."""
    height, width = mask.shape
    fx0, fy0 = bbox_ppm[0] / PPM * width, bbox_ppm[1] / PPM * height
    fx1, fy1 = bbox_ppm[2] / PPM * width, bbox_ppm[3] / PPM * height
    pred_w, pred_h = fx1 - fx0, fy1 - fy0
    if pred_w <= 0 or pred_h <= 0:
        return "skipped_degenerate", None
    margin_x = max(SEARCH_GROW * pred_w, SEARCH_MIN_FRACTION * width)
    margin_y = max(SEARCH_GROW * pred_h, SEARCH_MIN_FRACTION * height)
    wx0, wy0 = max(0, int(fx0 - margin_x)), max(0, int(fy0 - margin_y))
    wx1, wy1 = min(width, int(fx1 + margin_x) + 1), min(height, int(fy1 + margin_y) + 1)
    sub = mask[wy0:wy1, wx0:wx1]
    sub_h, sub_w = sub.shape
    x0, y0 = max(0, int(fx0) - wx0), max(0, int(fy0) - wy0)
    x1, y1 = min(sub_w, int(np.ceil(fx1)) - wx0), min(sub_h, int(np.ceil(fy1)) - wy0)
    if x1 <= x0 or y1 <= y0 or not sub[y0:y1, x0:x1].any():
        return "no_ink", None
    for _ in range(8):
        before = (x0, y0, x1, y1)
        rows = np.flatnonzero(sub[y0:y1, x0:x1].any(axis=1))
        cols = np.flatnonzero(sub[y0:y1, x0:x1].any(axis=0))
        y0, y1 = y0 + int(rows[0]), y0 + int(rows[-1]) + 1
        x0, x1 = x0 + int(cols[0]), x0 + int(cols[-1]) + 1
        y0 = _grow(y0, -1, 0, sub_h, lambda y: bool(sub[y, x0:x1].any()))
        y1 = _grow(y1 - 1, 1, 0, sub_h, lambda y: bool(sub[y, x0:x1].any())) + 1
        x0 = _grow(x0, -1, 0, sub_w, lambda x: bool(sub[y0:y1, x].any()))
        x1 = _grow(x1 - 1, 1, 0, sub_w, lambda x: bool(sub[y0:y1, x].any())) + 1
        if (x0, y0, x1, y1) == before:
            break
    if (x1 - x0) > GROWTH_CAP * pred_w or (y1 - y0) > GROWTH_CAP * pred_h:
        return "exceeds_cap", None
    if _ink_beyond_window(mask, (wx0, wy0, wx1, wy1), (x0 + wx0, y0 + wy0, x1 + wx0, y1 + wy0)):
        return "ink_beyond_window", None
    new_px = (x0 + wx0, y0 + wy0, x1 + wx0, y1 + wy0)
    if all(abs(n - f) <= TOLERANCE_PX for n, f in zip(new_px, (fx0, fy0, fx1, fy1), strict=True)):
        return "left_as_predicted", None
    return "snapped", (
        round(new_px[0] / width * PPM),
        round(new_px[1] / height * PPM),
        round(new_px[2] / width * PPM),
        round(new_px[3] / height * PPM),
    )


def _ink_beyond_window(
    mask: np.ndarray, window: tuple[int, int, int, int], box: tuple[int, int, int, int]
) -> bool:
    """True when the snapped box sits on the search window's edge and the ink
    continues just outside it: a rule, border or large block runs through the
    prediction, not a self-contained image."""
    height, width = mask.shape
    wx0, wy0, wx1, wy1 = window
    x0, y0, x1, y1 = box
    if x0 == wx0 and wx0 > 0 and mask[y0:y1, wx0 - 1].any():
        return True
    if x1 == wx1 and wx1 < width and mask[y0:y1, wx1].any():
        return True
    if y0 == wy0 and wy0 > 0 and mask[wy0 - 1, x0:x1].any():
        return True
    return bool(y1 == wy1 and wy1 < height and mask[wy1, x0:x1].any())


def apply(
    page: sealed.PageDecodeResult, *, raster: bytes | None
) -> tuple[sealed.PageDecodeResult, Counter[str]]:
    """`page` with every image region prediction snapped onto the raster's
    ink, and one count per prediction outcome (plus `no_verified_raster` once
    when the page has image predictions but no usable raster)."""
    counts: Counter[str] = Counter()
    if page.node_states is None:
        return page, counts
    if not any(p.node_type in _IMAGE_TYPES for p in page.region_predictions):
        return page, counts
    if raster is None or not rim.verify_raster_dpi(rim.read_raster_dpi(raster)):
        counts[COUNT_PREFIX + "no_verified_raster"] += 1
        return page, counts
    mask = _ink_mask(raster)
    if mask is None:
        counts[COUNT_PREFIX + "no_verified_raster"] += 1
        return page, counts
    predictions = list(page.region_predictions)
    edited = False
    for i, prediction in enumerate(predictions):
        if prediction.node_type not in _IMAGE_TYPES:
            continue
        if prediction.alignment_status != "region" or not prediction.regions:
            counts[COUNT_PREFIX + "skipped_abstain"] += 1
            continue
        if len(prediction.regions) != 1:
            counts[COUNT_PREFIX + "skipped_multi_region"] += 1
            continue
        outcome, bbox = _snap(mask, prediction.regions[0].bbox_ppm)
        counts[COUNT_PREFIX + outcome] += 1
        if bbox is None:
            continue
        region = dataclasses.replace(prediction.regions[0], bbox_ppm=bbox)
        predictions[i] = dataclasses.replace(prediction, regions=(region,))
        edited = True
    if not edited:
        return page, counts
    return dataclasses.replace(page, region_predictions=tuple(predictions)), counts
