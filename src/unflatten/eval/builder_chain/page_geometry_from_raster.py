"""Opt-in chain policy `page-geometry-from-raster.v0` (PROPOSED, owner
2026-09-27; draft `docs/rfc/RFC004-ERRATA-008.md`).

The page raster is a 300 DPI PNG whose pixel size is exact (the same
raster `recovered_image_materialization` reads for resource recovery --
this policy reuses that one read, `run.process_model_page` never opens the
file twice). This policy derives the page's physical size from that pixel
size, snaps it to the nearest standard page size within a tolerance, and
writes `/page/width_mpt`, `/page/height_mpt`, `/page/orientation` on every
section of the page's typed predictions when the model's own prediction
disagrees in orientation, differs from the snapped size beyond the
tolerance, or is missing outright. Margins are never touched in v0 (out of
scope; a future version may reconcile them against the same raster once
`table-geometry.v0`'s `available` width has been re-measured against it).

Rules, per section on the page:

1. the raster's pixel size converts to mpt at the pinned, VERIFIED DPI
   (`recovered_image_materialization.RASTER_DPI`/`verify_raster_dpi`,
   reused rather than re-declared) -- `width_mpt = round(px * 72000 /
   RASTER_DPI)`, same formula as that policy's image sizing;
2. the derived size snaps to the nearest of `STANDARD_SIZES_MPT` (both
   portrait and its landscape swap) within `TOLERANCE_MPT` of BOTH
   dimensions; the closest match wins ties. Without a match the section's
   target is the raster's own exact size instead, and `unmatched_
   nonstandard` is counted once for that section -- this is orthogonal to
   the edit decision below, since a nonstandard page may still already
   match the model's own prediction;
3. the section is edited to the target size/orientation, and exactly one
   of the following is counted per section: `size_set_from_missing` (any
   of the three pointers absent), `orientation_flipped` (present but
   portrait/landscape disagrees with the target), `size_snapped` (present,
   orientation agrees, but width or height is outside the tolerance of the
   target), `left_as_predicted` (already within tolerance, no edit).

Without a verified raster DPI (missing raster, unreadable bytes, or a DPI
outside `recovered_image_materialization.DPI_TOLERANCE`) the page is
returned unedited with one `no_verified_raster` count.
"""

from __future__ import annotations

import dataclasses
import io
from collections import Counter
from typing import Any

from PIL import Image, UnidentifiedImageError

from unflatten.eval.builder_chain import recovered_image_materialization as rim
from unflatten.eval.s2_gate import sealed

POLICY_ID = "page-geometry-from-raster.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-008.md"
COUNT_PREFIX = "page_geometry:"

#: Portrait `(width_mpt, height_mpt)`, at 50 mpt/twip (the same convention
#: `table_geometry.OVERFLOW_TOLERANCE_MPT` derives from): A4/A3/A5 the
#: common OOXML twip pair, Letter/Legal the exact inch size.
STANDARD_SIZES_MPT: dict[str, tuple[int, int]] = {
    "A4": (595_300, 841_900),
    "A3": (841_900, 1_190_550),
    "A5": (419_550, 595_300),
    "Letter": (612_000, 792_000),
    "Legal": (612_000, 1_008_000),
}
#: Policy constant, not measured (owner tolerance, 2026-09-27): ~0.28 in,
#: wide enough for scan trimming/DPI rounding without conflating A4 and
#: Letter (which differ by ~17 000 mpt on the short edge).
TOLERANCE_MPT = 20_000
RULES = (
    "size_from_raster_pixels.v0",
    "snapped_to_nearest_standard.v0",
    "orientation_or_size_or_missing_written.v0",
)


def policy_record(*, tolerance_mpt: int = TOLERANCE_MPT) -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "raster_dpi": rim.RASTER_DPI,
        "dpi_tolerance": str(rim.DPI_TOLERANCE),
        "tolerance_mpt": tolerance_mpt,
        "standard_sizes_mpt": {name: list(size) for name, size in STANDARD_SIZES_MPT.items()},
        "rules": list(RULES),
    }


def _raster_pixel_size(raster: bytes) -> tuple[int, int] | None:
    """`(width_px, height_px)`, or `None` for unreadable bytes. Header only,
    same guarantee as `recovered_image_materialization.read_raster_dpi`."""
    try:
        with Image.open(io.BytesIO(raster)) as opened:
            return opened.size
    except (OSError, UnidentifiedImageError):
        return None


def _target_size(
    width_px: int, height_px: int, *, tolerance_mpt: int
) -> tuple[int, int, str, bool]:
    """`(width_mpt, height_mpt, orientation, matched_standard)` for the
    raster's own exact pixel size."""
    width_mpt = round(width_px * 72_000 / rim.RASTER_DPI)
    height_mpt = round(height_px * 72_000 / rim.RASTER_DPI)
    best: tuple[int, int, int, str] | None = None
    for portrait_w, portrait_h in STANDARD_SIZES_MPT.values():
        for w, h, orientation in ((portrait_w, portrait_h, "portrait"),
                                   (portrait_h, portrait_w, "landscape")):
            distance = max(abs(width_mpt - w), abs(height_mpt - h))
            if distance <= tolerance_mpt and (best is None or distance < best[0]):
                best = (distance, w, h, orientation)
    if best is not None:
        return best[1], best[2], best[3], True
    orientation = "portrait" if height_mpt >= width_mpt else "landscape"
    return width_mpt, height_mpt, orientation, False


def apply(
    page: sealed.PageDecodeResult, *, raster: bytes | None, tolerance_mpt: int = TOLERANCE_MPT
) -> tuple[sealed.PageDecodeResult, Counter[str]]:
    """`page` with every section's `/page/width_mpt`/`height_mpt`/
    `orientation` reconciled against the raster, and one count per section
    outcome (plus `no_verified_raster` when nothing could be derived at
    all)."""
    counts: Counter[str] = Counter()
    if page.node_states is None:
        return page, counts
    dpi = rim.read_raster_dpi(raster) if raster is not None else None
    pixels = _raster_pixel_size(raster) if raster is not None else None
    if not rim.verify_raster_dpi(dpi) or pixels is None:
        counts[COUNT_PREFIX + "no_verified_raster"] += 1
        return page, counts
    target_w, target_h, target_o, matched = _target_size(*pixels, tolerance_mpt=tolerance_mpt)
    if not matched:
        counts[COUNT_PREFIX + "unmatched_nonstandard"] += 1
    predictions = list(page.typed_predictions)
    index = {(p["local_node_id"], p["pointer"]): i for i, p in enumerate(predictions)}
    edited = False
    for section in (s.local_node_id for s in page.node_states if s.node_type == "section"):
        what, values = _section_edit(predictions, index, section, target_w, target_h, target_o,
                                      tolerance_mpt)
        counts[COUNT_PREFIX + what] += 1
        if what == "left_as_predicted":
            continue
        for pointer, value in values.items():
            i = index.get((section, pointer))
            if i is None:
                index[(section, pointer)] = len(predictions)
                predictions.append({"head_id": "policy." + POLICY_ID, "local_node_id": section,
                                     "node_type": "section", "pointer": pointer, "value": value})
            else:
                predictions[i] = {**predictions[i], "value": value}
            edited = True
    if not edited:
        return page, counts
    return dataclasses.replace(page, typed_predictions=tuple(predictions)), counts


def _section_edit(
    predictions: list[dict[str, Any]],
    index: dict[tuple[str, str], int],
    section: str,
    target_w: int,
    target_h: int,
    target_o: str,
    tolerance_mpt: int,
) -> tuple[str, dict[str, Any]]:
    values = {"/page/width_mpt": target_w, "/page/height_mpt": target_h,
              "/page/orientation": target_o}

    def get(pointer: str) -> Any:
        i = index.get((section, pointer))
        return predictions[i]["value"] if i is not None else None

    width, height, orientation = get("/page/width_mpt"), get("/page/height_mpt"), get(
        "/page/orientation"
    )
    if width is None or height is None or orientation is None:
        return "size_set_from_missing", values
    if orientation != target_o:
        return "orientation_flipped", values
    if not (isinstance(width, int) and isinstance(height, int)
            and abs(width - target_w) <= tolerance_mpt
            and abs(height - target_h) <= tolerance_mpt):
        return "size_snapped", values
    return "left_as_predicted", {}
