"""Image builder (RFC002 "Inline images"): `w:drawing`/`wp:inline` for
`inline_image` and `w:drawing`/`wp:anchor` for `word-visual-v1`'s
`floating_image`, plus the node collection and `ImageRef` allocation
that must happen *before* the parts are built (the `r:embed`
relationship id and the `wp:docPr` id are both assigned from a
pre-scan, not while walking the tree paragraph by paragraph).

`r:embed` is resolved against the relationship table of the part the
image node lives in, never a document-wide map: a logo in
`header1.xml` and the same logo in `document.xml` share one media part
but get two different relationship ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lxml import etree

from unflatten.ast.issues import Issue
from unflatten.renderer.units import mdeg_to_ooxml_angle, mpt_to_emu, ppm_to_srcrect
from unflatten.renderer.xml_writer import qn

# AST anchor enums -> the ECMA-376 tokens of ST_RelFromH / ST_RelFromV /
# ST_AlignH / ST_AlignV / ST_WrapText (dml-wordprocessingDrawing.xsd).
_RELATIVE_FROM_MAP = {
    "margin": "margin",
    "page": "page",
    "column": "column",
    "character": "character",
    "paragraph": "paragraph",
    "line": "line",
    "left_margin": "leftMargin",
    "right_margin": "rightMargin",
    "top_margin": "topMargin",
    "bottom_margin": "bottomMargin",
    "inside_margin": "insideMargin",
    "outside_margin": "outsideMargin",
}
_WRAP_SIDE_MAP = {
    "both_sides": "bothSides",
    "left": "left",
    "right": "right",
    "largest": "largest",
}
# `text_frame` has no fixed/pixel extent by contract. The positive seed
# gives consumers that do not implement `a:spAutoFit` a usable initial
# canvas while conforming consumers derive the final shape from content.
_TEXT_FRAME_SEED_CX_EMU = mpt_to_emu(216_000)  # 3in
_TEXT_FRAME_SEED_CY_EMU = mpt_to_emu(72_000)  # 1in


@dataclass(frozen=True)
class AnchorSpec:
    """`wp:anchor` geometry for one `floating_image`, in OOXML units."""

    h_relative_from: str
    h_align: str | None
    h_offset_emu: int | None
    v_relative_from: str
    v_align: str | None
    v_offset_emu: int | None
    wrap_mode: str  # "none" | "top_and_bottom" | "square"
    wrap_side: str | None
    behind_text: bool
    allow_overlap: bool
    z_order: int


@dataclass(frozen=True)
class ImageRef:
    """Everything `append_drawing` needs for one image node, resolved
    ahead of time by `build_image_refs`.

    `anchor is None` means `inline_image` (`wp:inline`); an
    `AnchorSpec` means `floating_image` (`wp:anchor`).
    """

    rel_id: str
    docpr_id: int
    cx_emu: int
    cy_emu: int
    alt_text: str | None
    src_rect: tuple[int, int, int, int] | None  # (l, t, r, b) in srcRect units, or None
    anchor: AnchorSpec | None = None
    #: `a:xfrm/@rot` in 1/60000 of a degree; 0 means the attribute is
    #: omitted entirely, keeping unrotated output byte-identical.
    rotation_ooxml: int = 0


def build_anchor_spec(anchor: dict[str, Any]) -> AnchorSpec:
    """Convert an AST `anchor` object into OOXML units/tokens."""
    horizontal, vertical, wrap = anchor["horizontal"], anchor["vertical"], anchor["wrap"]
    return AnchorSpec(
        h_relative_from=_RELATIVE_FROM_MAP[horizontal["relative_from"]],
        h_align=horizontal.get("align"),
        h_offset_emu=(
            mpt_to_emu(horizontal["offset_mpt"]) if horizontal["mode"] == "offset" else None
        ),
        v_relative_from=_RELATIVE_FROM_MAP[vertical["relative_from"]],
        v_align=vertical.get("align"),
        v_offset_emu=(
            mpt_to_emu(vertical["offset_mpt"]) if vertical["mode"] == "offset" else None
        ),
        wrap_mode=wrap["mode"],
        wrap_side=_WRAP_SIDE_MAP[wrap["side"]] if wrap["mode"] == "square" else None,
        behind_text=anchor["behind_text"],
        allow_overlap=anchor["allow_overlap"],
        z_order=anchor["z_order"],
    )


def collect_inline_image_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect every `inline_image`/`floating_image` node under `nodes`,
    in preorder.

    Walks every container an image can be reached through: paragraphs
    (direct/hyperlink inline children), list items, table cells, and
    nested lists/tables within those.
    """
    found: list[dict[str, Any]] = []
    for node in nodes:
        node_type = node.get("type")
        if node_type == "paragraph":
            for child in node.get("children", []):
                found.extend(_collect_inline(child))
        elif node_type == "list":
            for item in node.get("items", []):
                found.extend(collect_inline_image_nodes(item.get("children", [])))
        elif node_type == "table":
            for row in node.get("rows", []):
                for cell in row.get("cells", []):
                    found.extend(collect_inline_image_nodes(cell.get("children", [])))
        elif node_type == "text_frame":
            found.extend(collect_inline_image_nodes(node.get("children", [])))
    return found


def _collect_inline(node: dict[str, Any]) -> list[dict[str, Any]]:
    if node.get("type") in ("inline_image", "floating_image"):
        return [node]
    if node.get("type") == "hyperlink":
        found: list[dict[str, Any]] = []
        for child in node.get("children", []):
            found.extend(_collect_inline(child))
        return found
    return []


def build_image_refs(
    image_nodes: list[dict[str, Any]],
    rel_id_by_sha256: dict[str, str],
    sha256_by_resource_id: dict[str, str],
    docpr_start: int = 1,
) -> tuple[dict[str, ImageRef], list[Issue]]:
    """Build `{node_id: ImageRef}` for every node in `image_nodes` (preorder).

    `wp:docPr/@id` must be unique across the whole package, not just
    within one part, so `docpr_start` lets the caller continue a single
    counter across `document.xml`, the header/footer stories and the
    notes parts. `rotation_mdeg` maps to `a:xfrm/@rot` (DrawingML's
    1/60000-degree unit) for inline and anchored images alike: the
    schema permits it on both, so refusing it would fail a whole
    document on one rotated logo.
    """
    refs: dict[str, ImageRef] = {}
    issues: list[Issue] = []
    for index, node in enumerate(image_nodes, start=docpr_start):
        node_id = node["id"]
        resource_ref = node["resource_ref"]
        sha256 = sha256_by_resource_id.get(resource_ref)
        if sha256 is None:
            issues.append(
                Issue(
                    code="RESOURCE_REF_UNRESOLVED",
                    severity="error",
                    node_id=node_id,
                    path="/resource_ref",
                    message=f"inline_image references resource '{resource_ref}', which was not "
                    "resolved.",
                    details={"resource_ref": resource_ref},
                )
            )
            continue

        refs[node_id] = ImageRef(
            rel_id=rel_id_by_sha256[sha256],
            docpr_id=index,
            cx_emu=mpt_to_emu(node["size"]["width_mpt"]),
            cy_emu=mpt_to_emu(node["size"]["height_mpt"]),
            alt_text=node.get("alt_text"),
            src_rect=_src_rect(node["crop"]),
            anchor=(
                build_anchor_spec(node["anchor"]) if node["type"] == "floating_image" else None
            ),
            rotation_ooxml=mdeg_to_ooxml_angle(node["rotation_mdeg"]),
        )
    return refs, issues


def _src_rect(crop: dict[str, int]) -> tuple[int, int, int, int] | None:
    if not any(crop.values()):
        return None
    return (
        ppm_to_srcrect(crop["left_ppm"]),
        ppm_to_srcrect(crop["top_ppm"]),
        ppm_to_srcrect(crop["right_ppm"]),
        ppm_to_srcrect(crop["bottom_ppm"]),
    )


def append_drawing(run: etree._Element, image_ref: ImageRef) -> None:
    """Append `w:drawing` to `run`: `wp:inline` for an inline image,
    `wp:anchor` for a floating one.
    """
    drawing = etree.SubElement(run, qn("w", "drawing"))
    if image_ref.anchor is None:
        _append_inline_body(drawing, image_ref)
    else:
        _append_anchor_body(drawing, image_ref, image_ref.anchor)


def append_text_frame_drawing(
    run: etree._Element, node: dict[str, Any], docpr_id: int,
    content_width_emu: int | None = None,
) -> etree._Element:
    """Append an anchored DrawingML word-processing shape.

    Returns its ``w:txbxContent`` so ``document_builder`` can render the
    renderer-neutral block children through the ordinary block dispatcher.
    The outer anchor follows the vendored Transitional schema; the shape
    uses Office's word-processing-shape extension namespace.

    `content_width_emu` is the story/section's usable width (page minus
    margins), used as the seed `cx` ONLY when the anchor's horizontal
    position is alignment-based (`h_align` set, e.g. `align="right"`).
    That case is where the seed width is load-bearing: `a:spAutoFit`
    only ever autofits height, never width (verified against a real
    document, not assumed — see the `fix-import-fidelity-defects`
    render probe), so an alignment-relative box keeps whatever width it
    is seeded with for its entire life. A 3in guess made a page-wide
    disclaimer band collapse into a narrow strip pinned to one edge —
    "shifted to the right side" is exactly what an align="right", too
    -narrow box looks like. An offset-positioned box's corner is placed
    directly, so its seed width does not move it and keeps the old
    conservative default.
    """
    drawing = etree.SubElement(run, qn("w", "drawing"))
    anchor = build_anchor_spec(node["anchor"])
    cx_emu = (
        content_width_emu
        if anchor.h_align is not None and content_width_emu
        else _TEXT_FRAME_SEED_CX_EMU
    )
    anchor_el = _append_anchor_shell(drawing, anchor, docpr_id, f"Text Frame {docpr_id}", cx_emu)
    graphic = etree.SubElement(anchor_el, qn("a", "graphic"))
    graphic_data = etree.SubElement(graphic, qn("a", "graphicData"))
    graphic_data.set("uri", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
    shape = etree.SubElement(graphic_data, qn("wps", "wsp"))
    etree.SubElement(shape, qn("wps", "cNvSpPr")).set("txBox", "1")
    _append_text_frame_shape_properties(shape, cx_emu)
    txbx = etree.SubElement(shape, qn("wps", "txbx"))
    content = etree.SubElement(txbx, qn("w", "txbxContent"))
    body_pr = etree.SubElement(shape, qn("wps", "bodyPr"))
    etree.SubElement(body_pr, qn("a", "spAutoFit"))
    return content


def _append_anchor_shell(
    drawing: etree._Element, anchor: AnchorSpec, docpr_id: int, name: str, cx_emu: int,
) -> etree._Element:
    """CT_Anchor prefix shared by resource-free text-frame drawings."""
    anchor_el = etree.SubElement(drawing, qn("wp", "anchor"))
    _set_wrap_distances(anchor_el)
    for key, value in (
        ("simplePos", "0"),
        ("relativeHeight", str(anchor.z_order)),
        ("behindDoc", "1" if anchor.behind_text else "0"),
        ("locked", "0"),
        ("layoutInCell", "1"),
        ("allowOverlap", "1" if anchor.allow_overlap else "0"),
    ):
        anchor_el.set(key, value)
    simple_pos = etree.SubElement(anchor_el, qn("wp", "simplePos"))
    simple_pos.set("x", "0")
    simple_pos.set("y", "0")
    _append_position(
        anchor_el, "positionH", anchor.h_relative_from, anchor.h_align, anchor.h_offset_emu
    )
    _append_position(
        anchor_el, "positionV", anchor.v_relative_from, anchor.v_align, anchor.v_offset_emu
    )
    _append_text_frame_extent(anchor_el, cx_emu)
    _append_wrap(anchor_el, anchor)
    doc_pr = etree.SubElement(anchor_el, qn("wp", "docPr"))
    doc_pr.set("id", str(docpr_id))
    doc_pr.set("name", name)
    return anchor_el


def _append_text_frame_extent(parent: etree._Element, cx_emu: int) -> None:
    extent = etree.SubElement(parent, qn("wp", "extent"))
    extent.set("cx", str(cx_emu))
    extent.set("cy", str(_TEXT_FRAME_SEED_CY_EMU))


def _append_text_frame_shape_properties(shape: etree._Element, cx_emu: int) -> None:
    sp_pr = etree.SubElement(shape, qn("wps", "spPr"))
    xfrm = etree.SubElement(sp_pr, qn("a", "xfrm"))
    off = etree.SubElement(xfrm, qn("a", "off"))
    off.set("x", "0")
    off.set("y", "0")
    ext = etree.SubElement(xfrm, qn("a", "ext"))
    ext.set("cx", str(cx_emu))
    ext.set("cy", str(_TEXT_FRAME_SEED_CY_EMU))
    geom = etree.SubElement(sp_pr, qn("a", "prstGeom"))
    geom.set("prst", "rect")
    etree.SubElement(geom, qn("a", "avLst"))
    etree.SubElement(sp_pr, qn("a", "noFill"))


def _set_wrap_distances(element: etree._Element) -> None:
    for name in ("distT", "distB", "distL", "distR"):
        element.set(name, "0")


def _append_inline_body(drawing: etree._Element, image_ref: ImageRef) -> None:
    """CT_Inline element order (dml-wordprocessingDrawing.xsd
    xsd:sequence): extent, effectExtent?, docPr, cNvGraphicFramePr?,
    a:graphic.
    """
    inline = etree.SubElement(drawing, qn("wp", "inline"))
    _set_wrap_distances(inline)
    _append_extent(inline, image_ref)
    _append_doc_pr(inline, image_ref)
    _append_graphic(inline, image_ref)


def _append_anchor_body(
    drawing: etree._Element, image_ref: ImageRef, anchor: AnchorSpec
) -> None:
    """CT_Anchor element order (dml-wordprocessingDrawing.xsd
    xsd:sequence): simplePos, positionH, positionV, extent,
    effectExtent?, EG_WrapType, docPr, cNvGraphicFramePr?, a:graphic.
    `relativeHeight`, `behindDoc`, `locked`, `layoutInCell` and
    `allowOverlap` are all `use="required"`.
    """
    anchor_el = etree.SubElement(drawing, qn("wp", "anchor"))
    _set_wrap_distances(anchor_el)
    anchor_el.set("simplePos", "0")
    anchor_el.set("relativeHeight", str(anchor.z_order))
    anchor_el.set("behindDoc", "1" if anchor.behind_text else "0")
    anchor_el.set("locked", "0")
    anchor_el.set("layoutInCell", "1")
    anchor_el.set("allowOverlap", "1" if anchor.allow_overlap else "0")

    # `simplePos="0"` above tells the consumer to ignore these
    # coordinates, but CT_Anchor still declares the element mandatory.
    simple_pos = etree.SubElement(anchor_el, qn("wp", "simplePos"))
    simple_pos.set("x", "0")
    simple_pos.set("y", "0")
    _append_position(
        anchor_el, "positionH", anchor.h_relative_from, anchor.h_align, anchor.h_offset_emu
    )
    _append_position(
        anchor_el, "positionV", anchor.v_relative_from, anchor.v_align, anchor.v_offset_emu
    )
    _append_extent(anchor_el, image_ref)
    _append_wrap(anchor_el, anchor)
    _append_doc_pr(anchor_el, image_ref)
    _append_graphic(anchor_el, image_ref)


def _append_position(
    anchor_el: etree._Element,
    local: str,
    relative_from: str,
    align: str | None,
    offset_emu: int | None,
) -> None:
    position = etree.SubElement(anchor_el, qn("wp", local))
    position.set("relativeFrom", relative_from)
    if align is not None:
        etree.SubElement(position, qn("wp", "align")).text = align
    else:
        etree.SubElement(position, qn("wp", "posOffset")).text = str(offset_emu)


def _append_wrap(anchor_el: etree._Element, anchor: AnchorSpec) -> None:
    """EG_WrapType is a mandatory choice of exactly one element.

    `wrapTight`/`wrapThrough` are unreachable here: both require a
    `wp:wrapPolygon` (CT_WrapTight/CT_WrapThrough, `minOccurs="1"`),
    i.e. pixel geometry the AST deliberately does not carry, so the
    AST's `anchor_wrap` has no such variant to render.
    """
    if anchor.wrap_mode == "none":
        etree.SubElement(anchor_el, qn("wp", "wrapNone"))
    elif anchor.wrap_mode == "top_and_bottom":
        etree.SubElement(anchor_el, qn("wp", "wrapTopAndBottom"))
    else:
        wrap_square = etree.SubElement(anchor_el, qn("wp", "wrapSquare"))
        wrap_square.set("wrapText", anchor.wrap_side or "bothSides")


def _append_extent(parent: etree._Element, image_ref: ImageRef) -> None:
    extent = etree.SubElement(parent, qn("wp", "extent"))
    extent.set("cx", str(image_ref.cx_emu))
    extent.set("cy", str(image_ref.cy_emu))


def _append_doc_pr(parent: etree._Element, image_ref: ImageRef) -> None:
    doc_pr = etree.SubElement(parent, qn("wp", "docPr"))
    doc_pr.set("id", str(image_ref.docpr_id))
    doc_pr.set("name", f"Picture {image_ref.docpr_id}")
    if image_ref.alt_text:
        doc_pr.set("descr", image_ref.alt_text)


def _append_graphic(parent: etree._Element, image_ref: ImageRef) -> None:
    graphic = etree.SubElement(parent, qn("a", "graphic"))
    graphic_data = etree.SubElement(graphic, qn("a", "graphicData"))
    graphic_data.set(
        "uri", "http://schemas.openxmlformats.org/drawingml/2006/picture"
    )
    pic = etree.SubElement(graphic_data, qn("pic", "pic"))

    nv_pic_pr = etree.SubElement(pic, qn("pic", "nvPicPr"))
    c_nv_pr = etree.SubElement(nv_pic_pr, qn("pic", "cNvPr"))
    c_nv_pr.set("id", str(image_ref.docpr_id))
    c_nv_pr.set("name", f"Picture {image_ref.docpr_id}")
    etree.SubElement(nv_pic_pr, qn("pic", "cNvPicPr"))

    blip_fill = etree.SubElement(pic, qn("pic", "blipFill"))
    blip = etree.SubElement(blip_fill, qn("a", "blip"))
    blip.set(qn("r", "embed"), image_ref.rel_id)
    if image_ref.src_rect is not None:
        left, top, right, bottom = image_ref.src_rect
        src_rect = etree.SubElement(blip_fill, qn("a", "srcRect"))
        src_rect.set("l", str(left))
        src_rect.set("t", str(top))
        src_rect.set("r", str(right))
        src_rect.set("b", str(bottom))
    stretch = etree.SubElement(blip_fill, qn("a", "stretch"))
    etree.SubElement(stretch, qn("a", "fillRect"))

    sp_pr = etree.SubElement(pic, qn("pic", "spPr"))
    xfrm = etree.SubElement(sp_pr, qn("a", "xfrm"))
    if image_ref.rotation_ooxml:
        # CT_Transform2D/@rot rotates the picture about its own centre.
        # `wp:extent` and `a:ext` both stay the *unrotated* size the AST
        # carries: the AST has one `size`, so inventing a rotated
        # bounding box for `wp:extent` would be a second, unobserved
        # measurement (RFC002 "no unobserved source semantics").
        xfrm.set("rot", str(image_ref.rotation_ooxml))
    off = etree.SubElement(xfrm, qn("a", "off"))
    off.set("x", "0")
    off.set("y", "0")
    ext = etree.SubElement(xfrm, qn("a", "ext"))
    ext.set("cx", str(image_ref.cx_emu))
    ext.set("cy", str(image_ref.cy_emu))
    prst_geom = etree.SubElement(sp_pr, qn("a", "prstGeom"))
    prst_geom.set("prst", "rect")
    etree.SubElement(prst_geom, qn("a", "avLst"))
