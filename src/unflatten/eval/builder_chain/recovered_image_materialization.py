"""Opt-in chain policy `recovered-image-materialization.v0` (RATIFIED 2026-09-26,
RFC004-ERRATA-004; handoff 2026-09-26 §6/§10 item 2).

`resource_recovery.exact_crop` cuts the visible page region out of the
raster and tags the bytes `fidelity="render_recovered"`; nothing downstream
reads that tag. The builder writes the image node's predicted `/crop/*`,
`/rotation_mdeg` and `/size/*` verbatim and the renderer applies them to
bytes that already ARE the visible rectangle: the recovered crop is cropped
a second time (94 % of a worksheet image's height lost) and stretched into a
predicted rectangle (median aspect error x1.55 over 44 cohort images).

This policy materializes the image transform from the resource's own
provenance, only when the caller asks for it, between `_recover_resources`
and `build_strict` in `run.process_model_page`, and counts every step:

- scope: image nodes (`inline_image`, `floating_image`) whose recovery is
  `resolved` with `fidelity == "render_recovered"`; every other resource
  (gold arm, `RecoveredResource.resolved(...)` from another producer,
  abstained or refused slots) is skipped and counted;
- values: crop 0/0/0/0, rotation 0, size = pixel dims x 72000 / RASTER_DPI
  mpt (px x 240 at 300 DPI, exact integers) -- the bitmap displayed at the
  physical scale it had on the page, independent of the page-size head;
- DPI: `RASTER_DPI` is pinned and VERIFIED against the raster's pHYs chunk
  (pdftoppm writes 11811 px/m = 299.9994 DPI); a page whose raster carries no
  pHYs or a different one gets no edit and one counted refusal;
- `/anchor/*` of a floating image is never touched (counted per edit).

`build_strict` and the closed derivation registry (DEC-037) are unchanged;
the Builder Report's `predicted_by_pointer` snapshots `assembled.typed`, so
these values count as "predicted" there exactly as `headless-defaults.v0`'s
defaults do -- the chain report's `recovered_image:*` counts are the truth.
The record carries `dpi_tolerance` as a decimal string because canonical
JSON (RFC001 §35.1) has no floats.
"""

from __future__ import annotations

import dataclasses
import io
from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from unflatten.assembly.assemble import AssembledDocumentHypothesis, AssembledNode
    from unflatten.model.resource_binder import RecoveredResource

POLICY_ID = "recovered-image-materialization.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-004.md"
RASTER_DPI = 300  # pinned; verified against the raster's pHYs
DPI_TOLERANCE = 0.01
COUNT_PREFIX = "recovered_image:"

RENDER_RECOVERED = "render_recovered"
_IMAGE_TYPES = frozenset({"inline_image", "floating_image"})
_CROP_POINTERS = ("/crop/left_ppm", "/crop/top_ppm", "/crop/right_ppm", "/crop/bottom_ppm")
_ROTATION_POINTER = "/rotation_mdeg"
_SIZE_POINTERS = ("/size/width_mpt", "/size/height_mpt")


def policy_record() -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "raster_dpi": RASTER_DPI,
        "dpi_tolerance": str(DPI_TOLERANCE),
        "rules": [
            "raster_dpi_verified_against_phys.v0",
            "render_recovered_crop_zero.v0",
            "render_recovered_rotation_zero.v0",
            "render_recovered_size_from_pixel_dims.v0",
            "floating_anchor_untouched.v0",
        ],
        "scope": {"fidelity": RENDER_RECOVERED},
        "values": {"crop": 0, "rotation_mdeg": 0, "size": "pixel_dims*72000/raster_dpi"},
    }


def read_raster_dpi(raster: bytes) -> float | None:
    """The raster's pHYs horizontal DPI, or None when it carries none (or the
    bytes are no image). `Image.open` parses the header only; no pixel is
    decoded."""
    try:
        with Image.open(io.BytesIO(raster)) as opened:
            dpi = opened.info.get("dpi")
    except (OSError, UnidentifiedImageError):
        return None
    return None if dpi is None else float(dpi[0])


def verify_raster_dpi(dpi: float | None) -> bool:
    return dpi is not None and abs(dpi - RASTER_DPI) <= DPI_TOLERANCE


def apply(
    document: AssembledDocumentHypothesis,
    recovered: Mapping[str, RecoveredResource],
    *,
    raster_dpi: float | None,
    raster_dpi_by_slot: Mapping[str, float | None] | None = None,
) -> tuple[AssembledDocumentHypothesis, Counter[str]]:
    """`document` with every render-recovered image materialized, and one
    count per node outcome. Without a verified raster DPI the document is
    returned as is with the refusal counted once for the page. A multi-page
    caller supplies `raster_dpi_by_slot` so each image uses its own source
    raster's verified DPI; an absent slot never inherits a neighbour's DPI."""
    counts: Counter[str] = Counter()
    if raster_dpi_by_slot is None and not verify_raster_dpi(raster_dpi):
        # Refused only where there was something to materialize: a page
        # without a render-recovered image has nothing to refuse.
        if any(r.status == "resolved" and r.fidelity == RENDER_RECOVERED
               for r in recovered.values()):
            counts[COUNT_PREFIX + "raster_dpi_unverified"] += 1
        return document, counts
    sections = tuple(
        _walk(node, recovered, counts, raster_dpi_by_slot) for node in document.sections
    )
    notes = tuple(_walk(node, recovered, counts, raster_dpi_by_slot) for node in document.notes)
    stories = tuple(
        dataclasses.replace(story, node=_walk(story.node, recovered, counts, raster_dpi_by_slot))
        for story in document.stories
    )
    story_nodes = {story.node.assembled_id: story.node for story in stories}
    story_sets = tuple(
        dataclasses.replace(story_set, stories=tuple(
            dataclasses.replace(story, nodes=tuple(
                story_nodes[node.assembled_id] for node in story.nodes
            )) for story in story_set.stories
        )) for story_set in document.story_sets
    )
    edited = dataclasses.replace(
        document, sections=sections, notes=notes, stories=stories, story_sets=story_sets,
    )
    return edited, counts


def _walk(
    node: AssembledNode, recovered: Mapping[str, RecoveredResource], counts: Counter[str],
    raster_dpi_by_slot: Mapping[str, float | None] | None = None,
) -> AssembledNode:
    children = tuple(_walk(child, recovered, counts, raster_dpi_by_slot) for child in node.children)
    edited = node
    if node.node_type in _IMAGE_TYPES:
        resource = recovered.get(node.assembled_id)
        if (raster_dpi_by_slot is not None and resource is not None
                and resource.status == "resolved" and resource.fidelity == RENDER_RECOVERED
                and not verify_raster_dpi(raster_dpi_by_slot.get(node.assembled_id))):
            counts[COUNT_PREFIX + "raster_dpi_unverified"] += 1
        else:
            edited = _materialize(node, recovered, counts)
    if edited is node and all(c is o for c, o in zip(children, node.children, strict=True)):
        return node
    return dataclasses.replace(edited, children=children)


def _materialize(
    node: AssembledNode, recovered: Mapping[str, RecoveredResource], counts: Counter[str]
) -> AssembledNode:
    resource = recovered.get(node.assembled_id)
    if resource is None or resource.status != "resolved":
        counts[COUNT_PREFIX + "skipped_unresolved"] += 1
        return node
    if resource.fidelity != RENDER_RECOVERED:
        counts[COUNT_PREFIX + "skipped_foreign_fidelity"] += 1
        return node
    faithful: dict[str, int] = dict.fromkeys(_CROP_POINTERS, 0)
    faithful[_ROTATION_POINTER] = 0
    for pointer, pixels in zip(
        _SIZE_POINTERS, (resource.pixel_width, resource.pixel_height), strict=True
    ):
        faithful[pointer] = round(pixels * 72000 / RASTER_DPI)
    changed = {p for p, v in faithful.items() if node.typed.get(p, _ABSENT) != v}
    if not changed:
        counts[COUNT_PREFIX + "already_faithful"] += 1
        return node
    if changed.intersection(_CROP_POINTERS):
        counts[COUNT_PREFIX + "crop_zeroed"] += 1
    if _ROTATION_POINTER in changed:
        counts[COUNT_PREFIX + "rotation_zeroed"] += 1
    if changed.intersection(_SIZE_POINTERS):
        counts[COUNT_PREFIX + "size_set"] += 1
    if node.node_type == "floating_image":
        counts[COUNT_PREFIX + "floating_anchor_untouched"] += 1
    return dataclasses.replace(node, typed={**node.typed, **faithful})


_ABSENT = object()
