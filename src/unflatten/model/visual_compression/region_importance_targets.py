"""RFC004-ERRATA-005 E4 variant a / §27.15 signal a: per-region importance
targets for the pre-encoder compressor, derived from the AST and the
aligner's own observed boxes -- never predicted, per E4 rule 8 ("pri
tréningu smie byť cieľom mapa odvodená z AST alebo aligneru, nikdy
vstupom").

Reuses :mod:`unflatten.model.region_targets`'s ``RegionTarget`` records
(the aligner's raw per-node boxes, RFC003 §24.1) instead of re-deriving
boxes from the sidecar a second time -- a second reader of the same
sidecar rows would be a second rule, free to drift from the first (the
precedent `region_targets.py` itself states for its own anchor-token
reuse of `typed_property_targets.py`).

**Scope note.** E4 rule 7 enumerates protected classes: small font, numbers/
dates/sums, diacritics/punctuation, tables (incl. borderless), super/
subscript, bullets/numbering, image borders. `RegionTarget` carries
`node_type` and observed boxes but not typography (font size, script
position) -- that lives in the typed-property channel
(`typed_property_targets.py`), not here. This builder therefore classifies
only the `node_type`-derived members of that list (tables, lists, images);
the typography-derived members are `[OPEN]` until a caller joins this with
typed-property targets. Treat "not protected here" as "unclassified", never
as "safe to compress".
"""

from __future__ import annotations

from dataclasses import dataclass

from unflatten.model.region_targets import RegionTarget
from unflatten.model.visual_compression.region_plan import RegionProposal, RiskClass

_PPM_DENOMINATOR = 1_000_000

#: E4 rule 7's node-type-derived protected classes: tables (incl. rows/cells,
#: covering borderless tables since a border style is a property, not a
#: node type), lists (bullets/numbering), and image boundaries.
PROTECTED_NODE_TYPES: frozenset[str] = frozenset(
    {"table", "row", "cell", "list", "list_item", "inline_image", "floating_image"}
)


@dataclass(frozen=True, slots=True)
class RegionImportanceTarget:
    """One node's deterministic protection class on one page (signal a)."""

    node_id: str
    node_type: str
    risk_class: RiskClass
    #: Straight from `RegionTarget.regions` -- no re-derivation, no union.
    regions: tuple[tuple[int, int, int, int], ...]


def build_region_importance_targets(
    region_targets: list[RegionTarget] | tuple[RegionTarget, ...],
) -> tuple[RegionImportanceTarget, ...]:
    """One `RegionImportanceTarget` per `RegionTarget` that has at least one
    observed box -- a node the aligner never placed on this page has
    nothing for a pre-encoder resolution plan to protect or not.

    Classification (E4 rule 6 "neistota znamená viac detailu" applied at
    the node-evidence level, not yet at pixels): `PROTECTED_NODE_TYPES`
    always resolve to `"protected"`; any node whose `page_evidence` is
    `"unresolved"` resolves to `"uncertain"` (full resolution, same rule);
    everything else defaults to `"medium"`, since this builder has no
    typography signal to distinguish `"high"`/`"low"` (scope note above).
    """
    targets: list[RegionImportanceTarget] = []
    for target in region_targets:
        if not target.regions:
            continue
        if target.node_type in PROTECTED_NODE_TYPES:
            risk_class: RiskClass = "protected"
        elif target.page_evidence == "unresolved":
            risk_class = "uncertain"
        else:
            risk_class = "medium"
        targets.append(
            RegionImportanceTarget(
                node_id=target.node_id,
                node_type=target.node_type,
                risk_class=risk_class,
                regions=target.regions,
            )
        )
    return tuple(targets)


def importance_targets_to_region_proposals(
    targets: list[RegionImportanceTarget] | tuple[RegionImportanceTarget, ...],
) -> tuple[RegionProposal, ...]:
    """Flatten to one `RegionProposal` per observed box (a node may have
    several across chained observations), ready for
    `region_plan.plan_region_resolution`."""
    proposals: list[RegionProposal] = []
    for target in targets:
        for x0, y0, x1, y1 in target.regions:
            proposals.append(
                RegionProposal(x0=x0, y0=y0, x1=x1, y1=y1, risk_class=target.risk_class)
            )
    return tuple(proposals)


def importance_targets_to_pixel_region_proposals(
    targets: list[RegionImportanceTarget] | tuple[RegionImportanceTarget, ...],
    *,
    page_width_px: int,
    page_height_px: int,
    protect_all_observed: bool = False,
) -> tuple[RegionProposal, ...]:
    """Convert aligner ``bbox_ppm`` regions to page-pixel proposals.

    :class:`RegionTarget` deliberately stores resolution-independent
    millionths-of-page coordinates, while ``region_plan`` operates in page
    pixels. Passing the former directly to the latter makes ordinary boxes
    exceed the page by two orders of magnitude. This explicit boundary uses
    floor for the top/left and ceil for the bottom/right, preserving every
    covered source pixel after conversion.

    ``protect_all_observed`` is the rehearsal's ideal-map policy: every box
    the gold aligner observed (including text) stays full resolution while
    the unobserved background uses the global low-res view. The normal
    learned policy keeps each target's classified risk instead.
    """
    if page_width_px <= 0 or page_height_px <= 0:
        raise ValueError(
            f"page dimensions must be > 0, got {page_width_px}x{page_height_px}"
        )

    proposals: list[RegionProposal] = []
    for target in targets:
        risk_class: RiskClass = "protected" if protect_all_observed else target.risk_class
        for x0_ppm, y0_ppm, x1_ppm, y1_ppm in target.regions:
            x0 = max(0, min(page_width_px - 1, x0_ppm * page_width_px // _PPM_DENOMINATOR))
            y0 = max(
                0, min(page_height_px - 1, y0_ppm * page_height_px // _PPM_DENOMINATOR)
            )
            x1 = max(
                x0 + 1,
                min(
                    page_width_px,
                    (x1_ppm * page_width_px + _PPM_DENOMINATOR - 1)
                    // _PPM_DENOMINATOR,
                ),
            )
            y1 = max(
                y0 + 1,
                min(
                    page_height_px,
                    (y1_ppm * page_height_px + _PPM_DENOMINATOR - 1)
                    // _PPM_DENOMINATOR,
                ),
            )
            proposals.append(
                RegionProposal(x0=x0, y0=y0, x1=x1, y1=y1, risk_class=risk_class)
            )
    return tuple(proposals)
