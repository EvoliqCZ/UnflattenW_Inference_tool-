"""RFC004 §14 VisualMemory metadata sidecar — page/tile ordinal, region_ppm,
scale level, global token mask, `scale_unknown`.

An additive record built from what `unflatten.model.input_projector.project`
already knows (`ModelSafeInput`'s planes, transforms and declared scale) --
never from the tensors `visual_memory.VisualMemory` produces. It sits NEXT
TO `visual_memory.MemoryBatch`, not inside it: `VisualMemory.forward` takes
no metadata argument and its output tensors never depend on this module, so
changing what this module reports can never change a checkpoint's numerics,
the feature cache identity, or `VisualMemory`'s own contract.

RFC004 §14's reference schema is per-VISUAL-TOKEN
(`Tensor[batch, visual_tokens]`); this module is per-PAGE-PLANE, the finest
grain the input path actually carries before a backbone ever runs. §14.1's
tile/scale positional encoding still has per-token work to do downstream;
this sidecar is the page-level ground truth a future per-token expansion
must agree with, not a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from unflatten.model.input_contract import CoordinateTransform, ModelSafeInput

#: A1 carries no tiling yet (`input_projector.project` docstring: "A1 uses
#: only the identity transform (no tiling yet)"), so every plane is tile 0
#: and scale level 0 until a tiling/pyramid path exists.
_UNTILED_ORDINAL = 0
_BASE_SCALE_LEVEL = 0

#: `region_ppm` denominator: parts-per-million of the page's own extent,
#: resolution-independent regardless of the page's actual pixel dimensions.
PPM_DENOMINATOR = 1_000_000

#: The full-page footprint in ppm -- what every plane gets under today's
#: identity-only transform (RFC004 §14 `region_ppm` is `(x0, y0, x1, y1)`).
_FULL_PAGE_REGION_PPM: tuple[int, int, int, int] = (0, 0, PPM_DENOMINATOR, PPM_DENOMINATOR)


@dataclass(frozen=True, slots=True)
class PageVisualMemoryMetadata:
    """Metadata for one page plane of a `ModelSafeInput` (RFC004 §14)."""

    page_ordinal: int
    tile_ordinal: int
    region_ppm: tuple[int, int, int, int]
    scale_level: int
    is_global_token: bool


@dataclass(frozen=True, slots=True)
class VisualMemoryMetadata:
    """One `PageVisualMemoryMetadata` per plane, plus the input's declared
    scale state (§14.2's `scale_unknown`: "adapter must use an explicit
    scale_unknown state" when runtime does not know a trustworthy DPI)."""

    pages: tuple[PageVisualMemoryMetadata, ...]
    scale_unknown: bool


def _region_ppm(transform: CoordinateTransform) -> tuple[int, int, int, int]:
    """The tile's footprint on the page, in ppm of the page's own extent.

    Today's input path carries only the identity transform (no tiling), so
    this is always the full page; a future tiling path would derive a
    sub-region footprint from a non-identity `transform` here instead of
    refusing it."""
    if transform != CoordinateTransform.identity():
        raise ValueError(
            "region_ppm is only defined for the identity transform today "
            "(the input path carries no tiling yet); got a non-identity "
            f"CoordinateTransform {transform!r}"
        )
    return _FULL_PAGE_REGION_PPM


def visual_memory_metadata_from_model_safe_input(msi: ModelSafeInput) -> VisualMemoryMetadata:
    """Build the §14 metadata sidecar from a `ModelSafeInput`.

    Deterministic and pure: the same input always yields the same metadata,
    and nothing here reads or writes a tensor -- so it can be computed
    before, after, or never, without changing `VisualMemory`'s output."""
    pages = tuple(
        PageVisualMemoryMetadata(
            page_ordinal=plane.page_ordinal,
            tile_ordinal=_UNTILED_ORDINAL,
            region_ppm=_region_ppm(transform),
            scale_level=_BASE_SCALE_LEVEL,
            is_global_token=False,
        )
        for plane, transform in zip(msi.planes, msi.transforms, strict=True)
    )
    return VisualMemoryMetadata(pages=pages, scale_unknown=msi.scale.status == "unknown")
