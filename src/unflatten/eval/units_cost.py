"""Units-derived lower bound on the cost of confusing two spacing values.

Design `categorical-word-properties` D3: the visible consequence of
predicting `after_mpt = j` when the truth is `i` has a floor that is a
*deterministic function of `renderer/units.py` and a recorded
`render_dpi`* — not a chosen weight. The renderer writes paragraph
spacing in twips (1 twip = 50 mpt), so a difference the twip grid
cannot represent has a rendered cost of exactly zero, and every other
difference has a cost of `|delta_twip| * dpi / 1440` pixels.

What this measures: the vertical displacement, in device pixels at
`render_dpi`, that one substitution of the property forces on the
content below it, assuming no reflow.

What it does NOT measure: reflow, page breaks, spacing collapse in
table cells, or anything about whether a human notices. It is a lower
bound, not render fidelity, and it licenses no threshold (RFC005 §37.9
files such signals as secondary; RFC006 OPEN-006-012 requires
human-cohort evidence).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from unflatten.renderer.units import mpt_to_twip

#: The AST pointer this module measures. Named, not implied: the cost of
#: a confusion is a property of the pointer, not of the number.
POINTER = "/layout/spacing/after_mpt"

#: 1 twip = 1/1440 inch, so `dpi` pixels per inch gives `dpi/1440` px.
_TWIPS_PER_INCH = 1440


def twip_to_px(twip: int, dpi: int) -> float:
    """Convert twips to device pixels at `dpi` (exact, no rounding)."""
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    return twip * dpi / _TWIPS_PER_INCH


def mpt_to_px(mpt: int, dpi: int) -> float:
    """Convert milli-points to device pixels at `dpi`.

    Unquantized: the IDEAL length, which the renderer never writes. The
    matrix therefore does not use it — `units_cost_matrix` quantizes
    through `mpt_to_twip` and converts with `twip_to_px`. It is kept as
    the stated reference for the design's unquantized figure (D3,
    "1 bod = 4,167 px" at 300 dpi), so that the gap between ideal and
    renderable stays measurable rather than asserted.
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    return mpt * dpi / (72 * 1000)


@dataclass(frozen=True)
class UnitsCostCell:
    """One ordered `(truth, predicted)` cell of the units-derived matrix."""

    truth_mpt: int
    predicted_mpt: int
    delta_twip: int
    delta_px: float
    cost_px: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "truth_mpt": self.truth_mpt,
            "predicted_mpt": self.predicted_mpt,
            "delta_twip": self.delta_twip,
            "delta_px": self.delta_px,
            "cost_px": self.cost_px,
        }


@dataclass(frozen=True)
class UnitsCostMatrix:
    """Every ordered pair of a recorded vocabulary at a recorded DPI."""

    pointer: str
    render_dpi: int
    vocabulary_mpt: tuple[int, ...]
    cells: tuple[UnitsCostCell, ...]

    def cell(self, truth_mpt: int, predicted_mpt: int) -> UnitsCostCell:
        for cell in self.cells:
            if cell.truth_mpt == truth_mpt and cell.predicted_mpt == predicted_mpt:
                return cell
        raise KeyError(f"({truth_mpt}, {predicted_mpt}) is not in this matrix")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": "units",
            "pointer": self.pointer,
            "render_dpi": self.render_dpi,
            "vocabulary_mpt": list(self.vocabulary_mpt),
            "cells": [cell.to_dict() for cell in self.cells],
        }


def units_cost_matrix(values_mpt: Sequence[int], dpi: int) -> UnitsCostMatrix:
    """Build the units-derived cost matrix over `values_mpt` at `dpi`.

    `values_mpt` is the *measured* vocabulary of the pointer; this
    function never supplies one. Raises `ValueError` on an empty or
    non-unique vocabulary, or a non-positive `dpi`.
    """
    if not values_mpt:
        raise ValueError("vocabulary is empty: units_cost_matrix needs measured values")
    if len(set(values_mpt)) != len(values_mpt):
        raise ValueError(f"vocabulary has duplicate values: {list(values_mpt)}")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    ordered = tuple(sorted(values_mpt))
    cells = []
    for truth in ordered:
        for predicted in ordered:
            delta_twip = mpt_to_twip(predicted) - mpt_to_twip(truth)
            delta_px = twip_to_px(delta_twip, dpi)
            cells.append(
                UnitsCostCell(
                    truth_mpt=truth,
                    predicted_mpt=predicted,
                    delta_twip=delta_twip,
                    delta_px=delta_px,
                    cost_px=abs(delta_px),
                )
            )
    return UnitsCostMatrix(
        pointer=POINTER,
        render_dpi=dpi,
        vocabulary_mpt=ordered,
        cells=tuple(cells),
    )
