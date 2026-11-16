"""Rendered arm of the `after_mpt` confusion cost matrix: what changed.

Design `categorical-word-properties` D3 licenses exactly this: "render
with class `i`, render with class `j`, compare", offline, deterministic,
no model, no GPU. This module owns the *compare*. The renders themselves
come from `render_pipeline`.

What "compare" means here, stated plainly:

* `page_count_delta` — an exact integer, the only signed quantity in the
  comparison and the only one that survives reflow.
* `changed_pixel_fraction` — the fraction of raster pixels whose 8-bit
  grey value is not byte-identical. Exact inequality, no tolerance: a
  tolerance would be an invented threshold (D3, RFC006 OPEN-006-012).
* `mean_abs_diff` — mean absolute 8-bit grey difference over the page.
* `first_differing_row` — the topmost raster row that differs, i.e.
  where the divergence starts.

What it does NOT measure: whether a human sees the difference, how far
anything moved (a shifted line reads as "many changed pixels", not as a
displacement), or render fidelity in the RFC002 §88 / RFC006 §26 sense —
none of which is built in this repo. It licenses no threshold and no
loss weight; RFC005 §37.9 files such signals as secondary evidence.
"""

from __future__ import annotations

import copy
import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from unflatten.eval.vocabulary import POINTER, walk_nodes


class RasterShapeMismatchError(ValueError):
    """Two rasters of the same page differ in size.

    Substituting paragraph spacing cannot change the paper size, so this
    means the two renders are not comparable — a caller bug, not a cost.
    """


class RasterDecodeError(ValueError):
    """A page raster could not be decoded as an image."""


def substitute_spacing_after(ast: dict[str, Any], value_mpt: int) -> dict[str, Any]:
    """Return a deep copy of `ast` with every paragraph's `after_mpt` set.

    The substitution is uniform over the document on purpose: it isolates
    the property, so the difference between two renders is attributable
    to it alone. It therefore measures a *document-level* substitution,
    not the marginal cost of one mispredicted node.
    """
    if value_mpt < 0:
        raise ValueError(f"{POINTER} must be non-negative, got {value_mpt}")
    result = copy.deepcopy(ast)
    for node in walk_nodes(result):
        if node.get("type") != "paragraph":
            continue
        spacing = node.get("layout", {}).get("spacing")
        if isinstance(spacing, dict) and "after_mpt" in spacing:
            spacing["after_mpt"] = value_mpt
    return result


@dataclass(frozen=True)
class PageDiff:
    """Difference between the two renders of one page."""

    changed_pixel_fraction: float
    mean_abs_diff: float
    first_differing_row: int | None


@dataclass(frozen=True)
class ClassDiff:
    """Difference between two whole-document renders (one ordered pair)."""

    page_count_truth: int
    page_count_predicted: int
    compared_pages: int
    pages_differing: int
    changed_pixel_fraction_mean: float
    changed_pixel_fraction_max: float
    mean_abs_diff_mean: float
    mean_abs_diff_max: float
    #: `(page_index, row)` of the topmost difference on the earliest
    #: differing page, or `None` when the compared pages are identical.
    first_difference: tuple[int, int] | None

    @property
    def page_count_delta(self) -> int:
        return self.page_count_predicted - self.page_count_truth

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_count_truth": self.page_count_truth,
            "page_count_predicted": self.page_count_predicted,
            "page_count_delta": self.page_count_delta,
            "compared_pages": self.compared_pages,
            "pages_differing": self.pages_differing,
            "changed_pixel_fraction_mean": self.changed_pixel_fraction_mean,
            "changed_pixel_fraction_max": self.changed_pixel_fraction_max,
            "mean_abs_diff_mean": self.mean_abs_diff_mean,
            "mean_abs_diff_max": self.mean_abs_diff_max,
            "first_difference": list(self.first_difference) if self.first_difference else None,
        }


def _grey(png_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(png_bytes)) as image:
            return np.asarray(image.convert("L"), dtype=np.int16)
    except (UnidentifiedImageError, OSError) as exc:
        raise RasterDecodeError(f"page raster is not a decodable image: {exc}") from exc


def compare_page_rasters(truth_png: bytes, predicted_png: bytes) -> PageDiff:
    """Compare two rasters of the same page; exact inequality, no tolerance."""
    truth, predicted = _grey(truth_png), _grey(predicted_png)
    if truth.shape != predicted.shape:
        raise RasterShapeMismatchError(
            f"page rasters differ in shape: {truth.shape} vs {predicted.shape}"
        )
    delta = np.abs(truth - predicted)
    changed = delta != 0
    rows = np.flatnonzero(changed.any(axis=1))
    return PageDiff(
        changed_pixel_fraction=float(changed.mean()),
        mean_abs_diff=float(delta.mean()),
        first_differing_row=int(rows[0]) if rows.size else None,
    )


def compare_class_rasters(
    truth_pages: Sequence[bytes], predicted_pages: Sequence[bytes]
) -> ClassDiff:
    """Compare two whole-document renders page by page.

    Pages present in only one render are counted (`page_count_delta`) but
    not compared: there is nothing to compare them against, and inventing
    a cost for them would be inventing a constant.
    """
    if not truth_pages or not predicted_pages:
        raise ValueError("a render with no pages cannot be compared")
    common = min(len(truth_pages), len(predicted_pages))
    diffs = [
        compare_page_rasters(truth_pages[index], predicted_pages[index])
        for index in range(common)
    ]
    first: tuple[int, int] | None = None
    for index, diff in enumerate(diffs):
        if diff.first_differing_row is not None:
            first = (index, diff.first_differing_row)
            break
    fractions = [diff.changed_pixel_fraction for diff in diffs]
    means = [diff.mean_abs_diff for diff in diffs]
    return ClassDiff(
        page_count_truth=len(truth_pages),
        page_count_predicted=len(predicted_pages),
        compared_pages=common,
        pages_differing=sum(1 for diff in diffs if diff.changed_pixel_fraction > 0),
        changed_pixel_fraction_mean=sum(fractions) / common,
        changed_pixel_fraction_max=max(fractions),
        mean_abs_diff_mean=sum(means) / common,
        mean_abs_diff_max=max(means),
        first_difference=first,
    )
