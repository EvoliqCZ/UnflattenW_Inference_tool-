"""Reference-aware retained-ink coverage; envelope statistics live in legibility.

Every original threshold-ink pixel remains in the denominator. Flat interiors
can supply extra coverage only through exact clean-pixel matches, and only while
the original contrast structure independently survives the same retention rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from unflatten.dataset.augment.geometry import Matrix3x3, identity_matrix
from unflatten.dataset.augment.ops import _warp


@dataclass(frozen=True)
class InkReference:
    """Full original ink, its locally detectable structure and the clean pixels."""

    ink: np.ndarray
    structure: np.ndarray
    image: Image.Image


def _visible_ink_pixels(clean_ink: np.ndarray, matrix: Matrix3x3) -> int:
    """Count original pixel centres in frame, independently of geometric scale."""
    height, width = clean_ink.shape
    visible = 0
    for row in range(0, height, 128):
        ys, xs = np.nonzero(clean_ink[row : row + 128])
        x = xs.astype(np.float64) + 0.5
        y = ys.astype(np.float64) + row + 0.5
        denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
        with np.errstate(divide="ignore", invalid="ignore"):
            projected_x = (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator
            projected_y = (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator
        visible += int(
            (
                (projected_x >= 0)
                & (projected_x < width)
                & (projected_y >= 0)
                & (projected_y < height)
            ).sum()
        )
    return visible


def _coverage(mask: np.ndarray, matrix: Matrix3x3 | None) -> np.ndarray:
    coverage = mask.astype(np.uint8) * 255
    if matrix is not None and matrix != identity_matrix():
        coverage = 255 - np.asarray(
            _warp(Image.fromarray(255 - coverage).convert("RGB"), matrix)
        )[..., 0]
    return coverage


def _unchanged_flat_coverage(
    image: Image.Image,
    view_ink: np.ndarray,
    coverage: np.ndarray,
    matrix: Matrix3x3 | None,
    reference: InkReference,
) -> int:
    flat_coverage = _coverage(reference.ink & ~reference.structure, matrix)
    expected = reference.image
    if matrix is not None and matrix != identity_matrix():
        expected = _warp(expected.convert("RGB"), matrix).convert("L")
    unchanged = np.asarray(image) == np.asarray(expected)
    return int(np.minimum(flat_coverage, coverage)[~view_ink & unchanged].sum())


def retained_ink_milli(
    image: Image.Image,
    view_ink: np.ndarray,
    clean_ink: np.ndarray,
    matrix: Matrix3x3 | None,
    reference: InkReference | None = None,
) -> int:
    """Retained original coverage times the fraction still visible after clipping."""
    width, height = image.size
    if clean_ink.shape != (height, width):
        raise ValueError(
            f"clean_ink shape {clean_ink.shape} does not match the view {(height, width)}"
        )
    original = int(clean_ink.sum())
    if original == 0:
        return 1000
    visible = original
    if matrix is not None and matrix != identity_matrix():
        visible = _visible_ink_pixels(clean_ink, matrix)
        if visible == 0:
            return 0
    coverage = _coverage(clean_ink, matrix)
    total = int(coverage.sum())
    if total == 0:
        return 0
    surviving = int(coverage[view_ink].sum())
    structure_retained = 1000
    if reference is not None and reference.structure.any():
        if (clean_ink & ~reference.structure).any():
            surviving += _unchanged_flat_coverage(image, view_ink, coverage, matrix, reference)
            # Ordinary text retains the original contrast test independently:
            # a uniform dark replacement cannot impersonate a preserved fill.
            structure_retained = retained_ink_milli(image, view_ink, reference.structure, matrix)
    retained = 1000 * surviving * visible // (total * original)
    return min(retained, structure_retained)
