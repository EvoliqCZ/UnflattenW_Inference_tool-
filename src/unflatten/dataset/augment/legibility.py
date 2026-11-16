"""Legibility guard of augmentation-v2 (design §1.6, RFC003-ERRATA-004).

`view_statistics` measures one augmented view; `within_envelope` applies the
fixed thresholds (retained ink, rotation) and, when the policy carries one,
the P2 envelope; `envelope_from_samples` builds that envelope from the
statistics of real captures. Pure numpy + PIL, deterministic, no model.

Envelope statistics run on a BOX x4 downsample (x2 for sharpness). Retention
uses full-resolution ink and a background estimated at x4 so thin strokes stay
in its denominator. "Ink" is a pixel that is both darker
than `ink_threshold` and clearly darker than its local background, so a dark
page border (page_offset on a dark desk) is not mistaken for ink.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from unflatten.dataset.augment.geometry import Matrix3x3
from unflatten.dataset.augment.ink_retention import InkReference, retained_ink_milli

#: The statistics a P2 envelope covers - those measurable on a real capture
#: without aligning it to a clean render (design §15 b. 11).
ENVELOPE_STATS: tuple[str, ...] = ("ink_contrast", "ink_sharpness", "background_brightness")

#: Calibration ledgers must never mix measurements from before/after the frame fix.
STATISTICS_VERSION = "legibility.frame-background.v2"

#: Retention additionally verifies unchanged flat ink against the clean pixels.
RETENTION_VERSION = "legibility.reference-flat-ink.v1"

#: Percentiles of the real-capture statistics that bound the envelope.
ENVELOPE_LOW_PCT = 5
ENVELOPE_HIGH_PCT = 95

_REDUCE = 4
_BG_FILTER = 15
#: Local background = 75th percentile of the window (the median is swallowed by dense text lines).
_BG_RANK = ImageFilter.RankFilter(_BG_FILTER, _BG_FILTER * _BG_FILTER * 3 // 4)
#: A pixel is ink only if it is at least this much darker than its local background.
_MIN_INK_DELTA = 30


@dataclass(frozen=True)
class ViewStatistics:
    """Legibility statistics of one view.

    `retained_ink_milli` is None when no clean ink was given (real captures);
    `rotation_deg` is 0.0 when no matrix was given.
    """

    ink_contrast: float
    ink_sharpness: float
    background_brightness: float
    retained_ink_milli: int | None
    rotation_deg: float


def _retention_ink(image: Image.Image, local_bg4: np.ndarray, threshold: int) -> np.ndarray:
    width, height = image.size
    local_bg = np.repeat(np.repeat(local_bg4, _REDUCE, axis=0), _REDUCE, axis=1)
    gray = np.asarray(image)
    return (gray < threshold) & (
        local_bg[:height, :width].astype(np.int16) - gray.astype(np.int16) >= _MIN_INK_DELTA
    )


def ink_reference(clean_png: bytes, ink_threshold: int) -> InkReference:
    """Preserve ALL threshold ink; also identify its measurable local structure.

    Flat filled interiors can be darker than the threshold without a local
    contrast of 30, even before augmentation. They remain in the denominator.
    """
    image = Image.open(io.BytesIO(clean_png)).convert("L")
    gray4 = np.asarray(image.reduce(_REDUCE))
    _, delta = _local_ink(gray4, ink_threshold)
    structure = _retention_ink(image, gray4.astype(np.int16) + delta, ink_threshold)
    return InkReference(np.asarray(image) < ink_threshold, structure, image)


def _local_ink(gray: np.ndarray, ink_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """``(ink mask, local background)`` of a gray array."""
    local_bg = np.asarray(Image.fromarray(gray).filter(_BG_RANK))
    delta = local_bg.astype(np.int16) - gray.astype(np.int16)
    return (gray < ink_threshold) & (delta >= _MIN_INK_DELTA), delta


def _border_background(gray: np.ndarray, ink_threshold: int) -> np.ndarray:
    """Pixel-only estimate of a dark external frame, including its BOX fringe.

    Flood only the colour band connected to the canvas perimeter. A difference
    of 30 already separates ink from background in this measurement; reuse that
    boundary, rather than fit a new cutoff to calibration results. Opening at
    the existing background-window width prevents a thin true text branch of
    the same colour from being removed along with a broad frame. A one-pixel
    outward fringe covers BOX mixing at the page edge, but never darker ink.
    This assumes a reasonably uniform external background, not arbitrary scene
    segmentation. White borders need no correction.
    """
    perimeter = np.concatenate((gray[0], gray[-1], gray[1:-1, 0], gray[1:-1, -1]))
    level = float(np.median(perimeter))
    if level >= ink_threshold:
        return np.zeros_like(gray, dtype=bool)
    # A dark sheet is not by itself evidence of an external frame. Require a
    # brighter region wide enough for the same local-background window.
    brighter = gray.astype(np.float32) >= level + _MIN_INK_DELTA
    if not np.asarray(Image.fromarray(brighter).filter(ImageFilter.MinFilter(_BG_FILTER))).any():
        return np.zeros_like(gray, dtype=bool)
    similar = np.abs(gray.astype(np.float32) - level) < _MIN_INK_DELTA
    padded = Image.fromarray(np.pad(similar.astype(np.uint8), 1, constant_values=1)).copy()
    ImageDraw.floodfill(padded, (0, 0), 255)
    connected = np.asarray(padded)[1:-1, 1:-1] == 255
    broad = np.asarray(
        Image.fromarray(connected)
        .filter(ImageFilter.MinFilter(_BG_FILTER))
        .filter(ImageFilter.MaxFilter(_BG_FILTER))
    )
    expanded = np.asarray(Image.fromarray(broad).filter(ImageFilter.MaxFilter(3)))
    return broad | (expanded & (gray >= level))


def _laplacian_variance_on_rows(
    gray2: np.ndarray, row_mask: np.ndarray, border2: np.ndarray, ink2: np.ndarray
) -> float:
    g = gray2.astype(np.float32)
    lap = -4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    rows = row_mask[1:-1]
    if not rows.any():
        return 0.0
    # Neutralise the inferred external page edge without changing the row's
    # denominator. A flat white frame would contribute the same zero values.
    touching_border = np.asarray(Image.fromarray(border2).filter(ImageFilter.MaxFilter(3)))
    touching_ink = np.asarray(Image.fromarray(ink2).filter(ImageFilter.MaxFilter(3)))
    lap[(touching_border & ~touching_ink)[1:-1, 1:-1]] = 0.0
    return float(lap[rows].var())


def view_statistics(
    view_png: bytes,
    *,
    clean_ink: np.ndarray | None,
    matrix: Matrix3x3 | None,
    ink_threshold: int,
    reference: InkReference | None = None,
) -> ViewStatistics:
    """Legibility statistics of one view.

    `clean_ink` is the boolean full-resolution ink mask of the CLEAN page and
    `matrix` the view's forward pixel map; both are needed for
    `retained_ink_milli`, which is otherwise None. An `ink_reference` allows
    exact pixel verification of flat interiors while independently protecting
    the original contrast structure. It never removes ink from the denominator.
    """
    if reference is not None and (
        clean_ink is None or not np.array_equal(clean_ink, reference.ink)
    ):
        raise ValueError("clean_ink must match the complete reference ink mask")
    image = Image.open(io.BytesIO(view_png)).convert("L")
    gray4 = np.asarray(image.reduce(_REDUCE))
    ink4, delta = _local_ink(gray4, ink_threshold)
    # Preserve the existing brightness and full-resolution retention definitions.
    background = float(np.median(gray4[~ink4])) if (~ink4).any() else 0.0
    border4 = _border_background(gray4, ink_threshold)
    ink4 = ink4 & ~border4
    if ink4.any():
        contrast = float(np.median(delta[ink4]))
        row_mask2 = np.repeat(ink4.any(axis=1), _REDUCE // 2)
        gray2 = np.asarray(image.reduce(2))
        border2 = np.repeat(np.repeat(border4, 2, axis=0), 2, axis=1)
        ink2 = np.repeat(np.repeat(ink4, 2, axis=0), 2, axis=1)
        sharpness = _laplacian_variance_on_rows(
            gray2,
            row_mask2[: gray2.shape[0]],
            border2[: gray2.shape[0], : gray2.shape[1]],
            ink2[: gray2.shape[0], : gray2.shape[1]],
        )
    else:
        contrast, sharpness = 0.0, 0.0
    retained = (
        None
        if clean_ink is None
        else retained_ink_milli(
            image,
            _retention_ink(image, gray4.astype(np.int16) + delta, ink_threshold),
            clean_ink,
            matrix,
            reference,
        )
    )
    rotation = 0.0 if matrix is None else math.degrees(math.atan2(matrix[1][0], matrix[0][0]))
    return ViewStatistics(contrast, sharpness, background, retained, rotation)


def within_envelope(
    stats: ViewStatistics, legibility: Mapping[str, Any]
) -> tuple[bool, str | None]:
    """``(ok, reason)``: fixed thresholds first, then the envelope when it is not null.

    A None `retained_ink_milli` (nothing to compare with) skips that threshold.
    """
    if (
        stats.retained_ink_milli is not None
        and stats.retained_ink_milli < legibility["min_retained_ink_milli"]
    ):
        return False, "retained_ink"
    if abs(stats.rotation_deg) > legibility["max_rotation_deg"]:
        return False, "rotation"
    envelope = legibility.get("envelope")
    if envelope is not None:
        for name in ENVELOPE_STATS:
            band = envelope[name]
            if not band["low"] <= getattr(stats, name) <= band["high"]:
                return False, f"envelope:{name}"
    return True, None


def envelope_from_samples(
    stats: Sequence[ViewStatistics],
    low_pct: float = ENVELOPE_LOW_PCT,
    high_pct: float = ENVELOPE_HIGH_PCT,
) -> dict[str, dict[str, float]]:
    """``{stat: {"low": .., "high": ..}}`` percentiles of `ENVELOPE_STATS` over `stats`."""
    if not stats:
        raise ValueError("envelope_from_samples: no samples")
    out: dict[str, dict[str, float]] = {}
    for name in ENVELOPE_STATS:
        values = np.array([getattr(s, name) for s in stats], dtype=np.float64)
        low, high = np.percentile(values, [low_pct, high_pct])
        out[name] = {"low": round(float(low), 6), "high": round(float(high), 6)}
    return out
