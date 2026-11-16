"""Ink masks for the ink-free occluder rule (design §1.4).

Occluders may only land where they cover (almost) no ink. The ink mask is
taken from the CURRENT image (after the earlier ops of a recipe). Thresholding
runs before conservative x2 pooling so thin, faint strokes cannot disappear.
Dilation runs at half resolution to keep the check cheap on a 300 DPI A4 page.
Pure numpy + PIL, no global state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

#: Box of the ink in full-resolution pixels, ``(x1, y1, x2, y2)``, upper bounds exclusive.
Box = tuple[int, int, int, int]

_SCALE = 2


def _gray(image_rgb: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(image_rgb).convert("L"))


def ink_mask(image_rgb: np.ndarray, threshold: int) -> np.ndarray:
    """Boolean ``[H, W]`` mask of pixels whose gray value is below `threshold`."""
    return _gray(image_rgb) < threshold


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Square dilation of a boolean mask by `px` pixels in every direction.

    Separable shift-and-or (equivalent to a ``(2*px+1)`` MaxFilter, but linear in
    `px` and without PIL's quadratic rank filter).
    """
    if px < 0:
        raise ValueError(f"dilate: px must be >= 0, got {px}")
    if px == 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, ((0, 0), (px, px)))
    rows = np.zeros_like(mask)
    for shift in range(2 * px + 1):
        rows |= padded[:, shift : shift + width]
    padded = np.pad(rows, ((px, px), (0, 0)))
    out = np.zeros_like(mask)
    for shift in range(2 * px + 1):
        out |= padded[shift : shift + height, :]
    return out


def ink_bbox(mask: np.ndarray) -> Box | None:
    """``(x1, y1, x2, y2)`` of the True pixels (upper bounds exclusive), or None if empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@dataclass(frozen=True)
class InkContext:
    """Ink of one image at half resolution.

    `mask` / `dilated` are in half-resolution coordinates (`scale` = 2);
    `bbox` is the ink box scaled back to full resolution; `ink_pixels` counts
    the original full-resolution ink pixels, not occupied half-resolution cells.
    """

    mask: np.ndarray
    dilated: np.ndarray
    bbox: Box | None
    ink_pixels: int
    scale: int


def ink_context(image_rgb: np.ndarray, *, ink_threshold: int, dilate_px: int) -> InkContext:
    """Conservatively pooled ink, outward-rounded dilation and full-resolution count."""
    full = ink_mask(image_rgb, ink_threshold)
    ink_pixels = int(full.sum())
    height, width = full.shape
    padded = np.pad(full, ((0, height % _SCALE), (0, width % _SCALE)))
    mask = padded.reshape(
        padded.shape[0] // _SCALE, _SCALE, padded.shape[1] // _SCALE, _SCALE
    ).any(axis=(1, 3))
    dilated = dilate(mask, -(-dilate_px // _SCALE))
    half_box = ink_bbox(mask)
    bbox = None if half_box is None else tuple(v * _SCALE for v in half_box)
    return InkContext(
        mask=mask,
        dilated=dilated,
        bbox=bbox,  # type: ignore[arg-type]
        ink_pixels=ink_pixels,
        scale=_SCALE,
    )
