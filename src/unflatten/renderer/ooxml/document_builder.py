"""`word/document.xml` builder — sections, paragraphs, runs, inline
controls, lists, tables, images, hyperlinks and note references
(RFC002 "OOXML mapping principles": Sections, Paragraphs, Runs, Inline
control nodes, Lists, Tables, Inline images).

Multi-section/multi-column documents are supported (RFC002
"Sections"). `word-visual-v1` (DEC-003) adds `w:hyperlink`,
`w:footnoteReference`/`w:endnoteReference` with an explicit custom
mark, and anchored (floating) images; the header/footer and notes
parts that reference this one are built by `story_builder`, which
reuses `append_block` from here so a paragraph in a header maps
exactly like a paragraph in the body.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from lxml import etree

from unflatten.ast.issues import Issue
from unflatten.ast.xml_text import find_illegal_xml_char
from unflatten.renderer.api import FontPolicy
from unflatten.renderer.fonts import resolve_font
from unflatten.renderer.ooxml.context import BuildContext
from unflatten.renderer.ooxml.drawing_builder import (
    append_drawing,
    append_text_frame_drawing,
    collect_inline_image_nodes,
)
from unflatten.renderer.ooxml.table_builder import append_table
from unflatten.renderer.units import (
    mpt_to_eighth_point,
    mpt_to_half_point,
    mpt_to_twip,
    round_half_away_from_zero,
)
from unflatten.renderer.xml_writer import make_root, qn

_ALIGNMENT_MAP = {
    "left": "left",
    "center": "center",
    "right": "right",
    "justify": "both",
    "distributed": "distribute",
}
_TAB_LEADER_MAP = {
    "dots": "dot",
    "hyphen": "hyphen",
    "underscore": "underscore",
    "middle_dot": "middleDot",
}
_UNDERLINE_MAP = {
    "none": "none",
    "single": "single",
    "double": "double",
    "dotted": "dotted",
    "dashed": "dash",
    "wave": "wave",
}
_BORDER_STYLE_MAP = {
    "none": "none",
    "single": "single",
    "double": "double",
    "dotted": "dotted",
    "dashed": "dashed",
}
_BREAK_TYPE_MAP = {
    "next_page": "nextPage",
    "continuous": "continuous",
    "even_page": "evenPage",
    "odd_page": "oddPage",
    "next_column": "nextColumn",
}
# Word's own default header/footer distance, used only when the section
# carries no `margins.header_mpt`/`footer_mpt` (a `word-core-v0`
# document, where the key does not exist at all).
_DEFAULT_HEADER_FOOTER_MARGIN_TWIP = 720  # 0.5in
_LINE_NUMBER_RESTART_MAP = {
    "continuous": "continuous",
    "new_page": "newPage",
    "new_section": "newSection",
}
_PAGE_NUMBER_FORMAT_MAP = {
    "decimal": "decimal",
    "lower_alpha": "lowerLetter",
    "upper_alpha": "upperLetter",
    "lower_roman": "lowerRoman",
    "upper_roman": "upperRoman",
}
# `field.field_type` -> the field instruction. The enum *is* the
# whitelist (RFC001 §30 forbids inventing a field code), so an
# unmapped value can only mean the schema grew and this map did not,
# which must fail loudly rather than emit a guessed instruction.
_FIELD_INSTRUCTION_MAP = {
    "page": " PAGE ",
    "num_pages": " NUMPAGES ",
}
_PAGE_BORDER_DISPLAY_MAP = {
    "all_pages": "allPages",
    "first_page": "firstPage",
    "not_first_page": "notFirstPage",
}
#: Namespace prefixes declared on a part root, in fixed order.
_IMAGE_PREFIXES = ["w", "wp", "a", "r", "pic"]


@dataclass
class DocumentBuildResult:
    root: etree._Element | None
    issues: list[Issue] = field(default_factory=list)
    used_fonts: set[str] = field(default_factory=set)
    fatal: bool = False


def build_document(
    ast: dict, font_policy: FontPolicy, ctx: BuildContext | None = None
) -> DocumentBuildResult:
    """Build the `w:document` root element for `ast` (RFC002 "Sections").

    `ctx` carries the numbering registry and resolved image references
    that `compiler.py` pre-computes over the whole document (RFC002
    stage S6); callers that never render lists/tables/images (existing
    R0/R1 tests) may omit it. A document with N sections emits N
    `w:sectPr`: the final section's as the last child of `w:body`, every
    other section's inside the `w:pPr` of that section's *own* last
    paragraph (RFC002 "Sections") — synthesizing a technical boundary
    paragraph when the section does not end with one.
    """
    if ctx is None:
        ctx = BuildContext(font_policy=font_policy)
    issues: list[Issue] = []
    used_fonts: set[str] = set()
    sections = ast.get("sections", [])

    top_nodes = [child for section in sections for child in section.get("children", [])]
    doc_root = make_root(
        "w",
        "document",
        used_prefixes=part_prefixes(
            top_nodes, needs_relationships=bool(ctx.story_refs_by_section)
        ),
    )
    body = etree.SubElement(doc_root, qn("w", "body"))
    fatal = False
    for s_index, section in enumerate(sections):
        section_path = f"/sections/{s_index}"
        if section.get("id") is not None:
            ctx.rendered_node_ids.add(section["id"])
        before_count = len(body)
        for i, child in enumerate(section.get("children", [])):
            block_path = f"{section_path}/children/{i}"
            block_issues, block_fatal = _append_block(
                body, child, block_path, font_policy, used_fonts, ctx
            )
            issues.extend(block_issues)
            fatal = fatal or block_fatal

        story_refs = ctx.story_refs_by_section.get(s_index, ())
        if s_index == len(sections) - 1:
            body.append(_build_sect_pr(section, story_refs))
        else:
            _append_non_final_sect_pr(body, section, before_count, ctx, story_refs)

    return DocumentBuildResult(root=doc_root, issues=issues, used_fonts=used_fonts, fatal=fatal)


def part_prefixes(blocks: list[dict], *, needs_relationships: bool = False) -> list[str]:
    """Namespace prefixes to declare on the root of a part containing
    `blocks`.

    Declared per part, not once for the package: a header that carries
    a logo needs `wp`/`a`/`r`/`pic`, a header that carries only text
    must not declare them, and keeping the rule local is what keeps a
    `word-core-v0` document byte-identical to the pre-DEC-003 renderer.

    `needs_relationships` covers `r:id` attributes that no block node
    implies — `w:headerReference`/`w:footerReference` live in `sectPr`,
    not in the block tree. Without it lxml invents an `ns1:` prefix per
    attribute, which is valid XML and unlike anything Word writes.
    """
    has_images = bool(collect_inline_image_nodes(blocks))
    has_text_frames = _contains_text_frame(blocks)
    if has_images or has_text_frames:
        prefixes = list(_IMAGE_PREFIXES if has_images else ["w", "wp", "a"])
        if "r" not in prefixes and (needs_relationships or _has_external_hyperlink(blocks)):
            prefixes.append("r")
        if has_text_frames:
            prefixes.extend(["mc", "wps"])
        return prefixes
    if needs_relationships or _has_external_hyperlink(blocks):
        return ["w", "r"]
    return ["w"]


def _contains_text_frame(nodes: list[dict]) -> bool:
    for node in nodes:
        if node.get("type") == "text_frame":
            return True
        for key in ("children", "items", "rows", "cells"):
            children = node.get(key)
            if isinstance(children, list) and _contains_text_frame(children):
                return True
    return False


def _has_external_hyperlink(nodes: list[dict]) -> bool:
    for node in nodes:
        node_type = node.get("type")
        if node_type == "paragraph":
            for child in node.get("children", []):
                target = child.get("target") or {}
                if child.get("type") == "hyperlink" and target.get("kind") != "internal_anchor":
                    return True
        elif node_type == "list":
            for item in node.get("items", []):
                if _has_external_hyperlink(item.get("children", [])):
                    return True
        elif node_type == "table":
            for row in node.get("rows", []):
                for cell in row.get("cells", []):
                    if _has_external_hyperlink(cell.get("children", [])):
                        return True
        elif node_type == "text_frame":
            if _has_external_hyperlink(node.get("children", [])):
                return True
    return False


def append_block(
    body: etree._Element,
    node: dict,
    path: str,
    font_policy: FontPolicy,
    used_fonts: set[str],
    ctx: BuildContext,
) -> tuple[list[Issue], bool]:
    """Render one block-level AST node into `body`.

    Public because `story_builder` renders header/footer/footnote
    bodies through exactly this mapping — a paragraph in a header is
    not a different kind of paragraph.
    """
    return _append_block(body, node, path, font_policy, used_fonts, ctx)


def _append_block(
    body: etree._Element,
    node: dict,
    path: str,
    font_policy: FontPolicy,
    used_fonts: set[str],
    ctx: BuildContext,
) -> tuple[list[Issue], bool]:
    node_type = node.get("type")
    if node.get("id") is not None:
        ctx.rendered_node_ids.add(node["id"])
    if node_type == "paragraph":
        issues = _append_paragraph(body, node, path, font_policy, used_fonts, ctx)
        return issues, False
    if node_type == "page_break":
        _append_technical_break(body, "page")
        return [], False
    if node_type == "column_break":
        _append_technical_break(body, "column")
        return [], False
    if node_type == "list":
        return _append_list(body, node, path, font_policy, used_fonts, ctx), False
    if node_type == "table":
        issues = append_table(body, node, path, font_policy, used_fonts, ctx, _append_block)
        return issues, False
    if node_type == "text_frame":
        return _append_text_frame(body, node, path, font_policy, used_fonts, ctx), False
    return (
        [
            Issue(
                code="UNSUPPORTED_NODE_R0",
                severity="error",
                node_id=node.get("id"),
                path=path,
                message=f"node type '{node_type}' is not supported by renderer chunk R0/R1.",
                details={"node_type": node_type},
            )
        ],
        True,
    )


def _append_text_frame(
    body: etree._Element,
    node: dict,
    path: str,
    font_policy: FontPolicy,
    used_fonts: set[str],
    ctx: BuildContext,
) -> list[Issue]:
    anchor_paragraph = etree.SubElement(body, qn("w", "p"))
    ctx.mark_technical_paragraph(anchor_paragraph)
    run = etree.SubElement(anchor_paragraph, qn("w", "r"))
    content = append_text_frame_drawing(
        run, node, ctx.text_frame_docpr_ids[node["id"]],
        ctx.text_frame_width_emu.get(node["id"]),
    )
    issues: list[Issue] = []
    for index, child in enumerate(node.get("children", [])):
        child_issues, _fatal = _append_block(
            content, child, f"{path}/children/{index}", font_policy, used_fonts, ctx
        )
        issues.extend(child_issues)
    if len(content) == 0:
        ctx.mark_technical_paragraph(etree.SubElement(content, qn("w", "p")))
    return issues


def _append_technical_break(body: etree._Element, break_type: str) -> None:
    """A block-level PageBreak/ColumnBreak AST node becomes its own
    technical paragraph containing a single run with `w:br` — Word
    requires breaks inside a run (RFC002 §41.3/§41.4).
    """
    p = etree.SubElement(body, qn("w", "p"))
    r = etree.SubElement(p, qn("w", "r"))
    br = etree.SubElement(r, qn("w", "br"))
    br.set(qn("w", "type"), break_type)


def _append_list(
    body: etree._Element,
    node: dict,
    path: str,
    font_policy: FontPolicy,
    used_fonts: set[str],
    ctx: BuildContext,
) -> list[Issue]:
    """Append every `list_item` of `list` AST node `node` (RFC002 "Lists").

    The first paragraph of an item carries `w:numPr` (the marker); any
    further paragraphs in the same item get the list's continuation
    indent and no marker. A `list_item` not starting with a paragraph
    is `unsupported` in `word-core-v0`.
    """
    issues: list[Issue] = []
    num_id = ctx.numbering.num_id_by_node_id.get(node["id"])
    for item_index, item in enumerate(node.get("items", [])):
        item_path = f"{path}/items/{item_index}"
        if item.get("id") is not None:
            ctx.rendered_node_ids.add(item["id"])
        children = item.get("children", [])
        starts_with_paragraph = bool(children) and children[0].get("type") == "paragraph"
        if not starts_with_paragraph:
            issues.append(
                Issue(
                    code="UNSUPPORTED_LIST_ITEM_NOT_PARAGRAPH",
                    severity="error",
                    node_id=item.get("id"),
                    path=item_path,
                    message="a list_item not starting with a paragraph is unsupported in "
                    "profile word-core-v0.",
                    details={},
                )
            )
        for child_index, child in enumerate(children):
            child_path = f"{item_path}/children/{child_index}"
            if child.get("type") != "paragraph":
                block_issues, _fatal = _append_block(
                    body, child, child_path, font_policy, used_fonts, ctx
                )
                issues.extend(block_issues)
                continue
            is_marker_paragraph = starts_with_paragraph and child_index == 0
            issues.extend(
                _append_paragraph(
                    body,
                    child,
                    child_path,
                    font_policy,
                    used_fonts,
                    ctx,
                    num_pr=(num_id, 0) if is_marker_paragraph else None,
                    marker_indent=(
                        (node["layout"]["left_indent_mpt"], node["layout"]["hanging_mpt"])
                        if is_marker_paragraph
                        else None
                    ),
                    continuation_indent_mpt=(
                        None if is_marker_paragraph else node["layout"]["left_indent_mpt"]
                    ),
                )
            )
    return issues


def _append_paragraph(
    body: etree._Element,
    node: dict,
    path: str,
    font_policy: FontPolicy,
    used_fonts: set[str],
    ctx: BuildContext,
    *,
    num_pr: tuple[int, int] | None = None,
    marker_indent: tuple[int, int] | None = None,
    continuation_indent_mpt: int | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    if node.get("id") is not None:
        ctx.rendered_node_ids.add(node["id"])
    p = etree.SubElement(body, qn("w", "p"))
    _append_paragraph_properties(
        p,
        node,
        num_pr=num_pr,
        marker_indent=marker_indent,
        continuation_indent_mpt=continuation_indent_mpt,
    )
    bookmark_id = ctx.bookmark_targets.get(node.get("id"))
    if bookmark_id is not None:
        start = etree.SubElement(p, qn("w", "bookmarkStart"))
        start.set(qn("w", "id"), str(bookmark_id))
        start.set(qn("w", "name"), node["id"])
    for i, child in enumerate(node.get("children", [])):
        issues.extend(
            _append_inline(p, child, f"{path}/children/{i}", font_policy, used_fonts, ctx)
        )
    if bookmark_id is not None:
        etree.SubElement(p, qn("w", "bookmarkEnd")).set(qn("w", "id"), str(bookmark_id))
    return issues


def _append_inline(
    parent: etree._Element,
    node: dict,
    path: str,
    font_policy: FontPolicy,
    used_fonts: set[str],
    ctx: BuildContext,
) -> list[Issue]:
    node_type = node.get("type")
    if node.get("id") is not None:
        ctx.rendered_node_ids.add(node["id"])
    if node_type == "run":
        return _append_run(parent, node, path, font_policy, used_fonts)
    if node_type == "tab":
        r = etree.SubElement(parent, qn("w", "r"))
        etree.SubElement(r, qn("w", "tab"))
        return []
    if node_type == "line_break":
        r = etree.SubElement(parent, qn("w", "r"))
        etree.SubElement(r, qn("w", "br"))
        return []
    if node_type in ("inline_image", "floating_image"):
        return _append_inline_image(parent, node, path, ctx)
    if node_type == "hyperlink":
        return _append_hyperlink(parent, node, path, font_policy, used_fonts, ctx)
    if node_type == "note_ref":
        return _append_note_ref(parent, node, path, ctx)
    if node_type == "field":
        return _append_field(parent, node, path)
    return [
        Issue(
            code="UNSUPPORTED_NODE_R0",
            severity="error",
            node_id=node.get("id"),
            path=path,
            message=f"inline node type '{node_type}' is not supported by renderer chunk R0/R1.",
            details={"node_type": node_type},
        )
    ]


def _append_hyperlink(
    parent: etree._Element,
    node: dict,
    path: str,
    font_policy: FontPolicy,
    used_fonts: set[str],
    ctx: BuildContext,
) -> list[Issue]:
    """`w:hyperlink` (CT_Hyperlink, wml.xsd): `@r:id` naming an
    `External` relationship of *this part*, or `@w:anchor` naming a
    bookmark for an `internal_anchor` (RFC001 ERRATA-001 E5's
    `#<node_id>` convention, so the anchor name is the target node id).
    """
    target = node["target"]
    hyperlink = etree.SubElement(parent, qn("w", "hyperlink"))
    if target["kind"] == "internal_anchor":
        hyperlink.set(qn("w", "anchor"), target["uri"].removeprefix("#"))
    else:
        rel_id = ctx.hyperlink_rel_ids.get(node["id"])
        if rel_id is None:
            return [
                Issue(
                    code="HYPERLINK_REL_UNRESOLVED",
                    severity="error",
                    node_id=node.get("id"),
                    path=path,
                    message="hyperlink has no relationship id; the compiler did not register "
                    "it against the part being built.",
                    details={"uri": target.get("uri")},
                )
            ]
        hyperlink.set(qn("r", "id"), rel_id)

    issues: list[Issue] = []
    for i, child in enumerate(node.get("children", [])):
        issues.extend(
            _append_inline(hyperlink, child, f"{path}/children/{i}", font_policy, used_fonts, ctx)
        )
    return issues


def _append_field(parent: etree._Element, node: dict, path: str) -> list[Issue]:
    """`w:fldSimple` carrying only the instruction (ECMA-376
    CT_SimpleField: `@w:instr` is required, the cached result is
    `minOccurs="0"`).

    No cached result is emitted, deliberately: a PAGE result differs on
    every page, so a stored one would be a second source of truth that
    is wrong on all pages but one — the very defect this node exists to
    remove. `settings.xml` therefore asks the consumer to update fields
    on open whenever the document contains one.
    """
    field_type = node["field_type"]
    instruction = _FIELD_INSTRUCTION_MAP.get(field_type)
    if instruction is None:
        return [
            Issue(
                code="UNSUPPORTED_FIELD_TYPE",
                severity="error",
                node_id=node.get("id"),
                path=path,
                message=f"field type '{field_type}' has no mapped Word field instruction.",
                details={"field_type": field_type},
            )
        ]
    etree.SubElement(parent, qn("w", "fldSimple")).set(qn("w", "instr"), instruction)
    return []


def _append_note_ref(
    parent: etree._Element, node: dict, path: str, ctx: BuildContext
) -> list[Issue]:
    """`w:footnoteReference`/`w:endnoteReference` with
    `@w:customMarkFollows="1"` and the visible mark as the run content
    that follows it in the same run (ECMA-376 CT_FtnEdnRef).

    The mark is always explicit: the AST carries the mark that was
    actually on the page, and Word's automatic numbering would silently
    renumber it.
    """
    note_id = node["note_id"]
    numeric_id = ctx.note_numeric_ids.get(note_id)
    kind = ctx.note_kinds.get(note_id)
    if numeric_id is None or kind is None:
        return [
            Issue(
                code="NOTE_REF_UNRESOLVED",
                severity="error",
                node_id=node.get("id"),
                path=path,
                message=f"note_ref targets '{note_id}', which the compiler did not register "
                "in footnotes.xml or endnotes.xml.",
                details={"note_id": note_id},
            )
        ]
    run = etree.SubElement(parent, qn("w", "r"))
    reference = etree.SubElement(run, qn("w", f"{kind}Reference"))
    reference.set(qn("w", "customMarkFollows"), "1")
    reference.set(qn("w", "id"), str(numeric_id))
    _append_text(run, node["marker_text"], node.get("id"), path)
    return []


def _append_inline_image(
    parent: etree._Element, node: dict, path: str, ctx: BuildContext
) -> list[Issue]:
    image_ref = ctx.images.get(node["id"])
    if image_ref is None:
        return [
            Issue(
                code="RESOURCE_REF_UNRESOLVED",
                severity="error",
                node_id=node.get("id"),
                path=path,
                message="inline_image has no resolved image reference (unresolved resource or "
                "resolution error).",
                details={"resource_ref": node.get("resource_ref")},
            )
        ]
    r = etree.SubElement(parent, qn("w", "r"))
    append_drawing(r, image_ref)
    return []


def _bool_val(value: bool) -> str:
    return "1" if value else "0"


def _append_bool(parent: etree._Element, tag: str, value: bool) -> None:
    el = etree.SubElement(parent, qn("w", tag))
    el.set(qn("w", "val"), _bool_val(value))


def _color_to_ooxml(value: str) -> str:
    if value == "auto":
        return "auto"
    return value.lstrip("#")


def _append_paragraph_properties(
    p: etree._Element,
    node: dict,
    *,
    num_pr: tuple[int, int] | None = None,
    marker_indent: tuple[int, int] | None = None,
    continuation_indent_mpt: int | None = None,
) -> None:
    semantic, layout, style = node["semantic"], node["layout"], node["style"]
    pPr = etree.SubElement(p, qn("w", "pPr"))
    # CT_PPrBase element order (wml.xsd xsd:sequence) — MUST be preserved
    # exactly, this is a strict schema, not a bag of attributes:
    # keepNext, keepLines, pageBreakBefore, widowControl, numPr, pBdr,
    # shd, tabs, spacing, ind, jc, outlineLvl, (sectPr appended by caller).
    pagination = layout["pagination"]
    _append_bool(pPr, "keepNext", pagination["keep_with_next"])
    _append_bool(pPr, "keepLines", pagination["keep_lines"])
    _append_bool(pPr, "pageBreakBefore", pagination["page_break_before"])
    _append_bool(pPr, "widowControl", pagination["widow_control"])
    if num_pr is not None:
        num_id, ilvl = num_pr
        num_pr_el = etree.SubElement(pPr, qn("w", "numPr"))
        ilvl_el = etree.SubElement(num_pr_el, qn("w", "ilvl"))
        ilvl_el.set(qn("w", "val"), str(ilvl))
        num_id_el = etree.SubElement(num_pr_el, qn("w", "numId"))
        num_id_el.set(qn("w", "val"), str(num_id))
    _append_borders(pPr, style.get("borders"))
    _append_shading(pPr, style.get("background"))
    _append_tabs(pPr, layout["tabs"])
    _append_spacing(pPr, layout["spacing"])
    _append_indent(
        pPr,
        layout["indent"],
        marker_indent=marker_indent,
        continuation_indent_mpt=continuation_indent_mpt,
    )
    jc = etree.SubElement(pPr, qn("w", "jc"))
    jc.set(qn("w", "val"), _ALIGNMENT_MAP[layout["alignment"]])
    if semantic["role"] == "heading" and semantic["heading_level"] is not None:
        outline = etree.SubElement(pPr, qn("w", "outlineLvl"))
        outline.set(qn("w", "val"), str(semantic["heading_level"] - 1))


def _append_indent(
    pPr: etree._Element,
    indent: dict,
    *,
    marker_indent: tuple[int, int] | None = None,
    continuation_indent_mpt: int | None = None,
) -> None:
    """Direct paragraph `w:ind` shadows the numbering level's `w:ind`
    (RFC002 "Lists"), so a list marker paragraph's own `left`/`hanging`
    must fold in the list's `left_indent_mpt`/`hanging_mpt`, not just
    the paragraph's authored indent, or the marker would collapse to
    the paragraph's own (usually zero) indent.
    """
    ind = etree.SubElement(pPr, qn("w", "ind"))
    hanging_mpt = indent["hanging_mpt"]
    if continuation_indent_mpt is not None:
        left_mpt = continuation_indent_mpt
    elif marker_indent is not None:
        list_left_mpt, list_hanging_mpt = marker_indent
        left_mpt = indent["left_mpt"] + list_left_mpt
        hanging_mpt = hanging_mpt or list_hanging_mpt
    else:
        left_mpt = indent["left_mpt"]
    ind.set(qn("w", "left"), str(mpt_to_twip(left_mpt)))
    ind.set(qn("w", "right"), str(mpt_to_twip(indent["right_mpt"])))
    if indent["first_line_mpt"]:
        ind.set(qn("w", "firstLine"), str(mpt_to_twip(indent["first_line_mpt"])))
    if hanging_mpt:
        ind.set(qn("w", "hanging"), str(mpt_to_twip(hanging_mpt)))


def _append_spacing(pPr: etree._Element, spacing: dict) -> None:
    spacing_el = etree.SubElement(pPr, qn("w", "spacing"))
    spacing_el.set(qn("w", "before"), str(mpt_to_twip(spacing["before_mpt"])))
    spacing_el.set(qn("w", "after"), str(mpt_to_twip(spacing["after_mpt"])))
    line = spacing["line"]
    mode = line["mode"]
    if mode == "single":
        spacing_el.set(qn("w", "line"), "240")
        spacing_el.set(qn("w", "lineRule"), "auto")
    elif mode == "multiple":
        value = round_half_away_from_zero(line["multiplier_ppm"] * 240, 1_000_000)
        spacing_el.set(qn("w", "line"), str(value))
        spacing_el.set(qn("w", "lineRule"), "auto")
    elif mode == "exact":
        spacing_el.set(qn("w", "line"), str(mpt_to_twip(line["value_mpt"])))
        spacing_el.set(qn("w", "lineRule"), "exact")
    elif mode == "at_least":
        spacing_el.set(qn("w", "line"), str(mpt_to_twip(line["value_mpt"])))
        spacing_el.set(qn("w", "lineRule"), "atLeast")


def _append_tabs(pPr: etree._Element, tabs: list[dict]) -> None:
    if not tabs:
        return
    tabs_el = etree.SubElement(pPr, qn("w", "tabs"))
    for tab_stop in tabs:
        tab_el = etree.SubElement(tabs_el, qn("w", "tab"))
        tab_el.set(qn("w", "val"), tab_stop["alignment"])
        tab_el.set(qn("w", "pos"), str(mpt_to_twip(tab_stop["position_mpt"])))
        if tab_stop["leader"] != "none":
            tab_el.set(qn("w", "leader"), _TAB_LEADER_MAP[tab_stop["leader"]])


def _append_shading(pPr: etree._Element, background: str | None) -> None:
    if background is None:
        return
    shd = etree.SubElement(pPr, qn("w", "shd"))
    shd.set(qn("w", "val"), "clear")
    shd.set(qn("w", "color"), "auto")
    shd.set(qn("w", "fill"), _color_to_ooxml(background))


def _append_borders(pPr: etree._Element, borders: dict | None) -> None:
    if borders is None:
        return
    pbdr = etree.SubElement(pPr, qn("w", "pBdr"))
    for side in ("top", "left", "bottom", "right"):
        border = borders.get(side)
        if not border:
            continue
        side_el = etree.SubElement(pbdr, qn("w", side))
        side_el.set(qn("w", "val"), _BORDER_STYLE_MAP[border["style"]])
        side_el.set(qn("w", "sz"), str(mpt_to_eighth_point(border["width_mpt"])))
        side_el.set(qn("w", "space"), "0")
        side_el.set(qn("w", "color"), _color_to_ooxml(border["color"]))


def _append_run(
    parent: etree._Element, node: dict, path: str, font_policy: FontPolicy, used_fonts: set[str]
) -> list[Issue]:
    style = node["style"]
    r = etree.SubElement(parent, qn("w", "r"))
    rPr = etree.SubElement(r, qn("w", "rPr"))

    used_font, font_issue = resolve_font(
        style["font_family"],
        node.get("id"),
        f"{path}/style/font_family",
        font_policy.mode,
        font_policy.fallback_map,
        font_policy.fallback_default,
    )
    issues = [font_issue] if font_issue is not None else []
    used_fonts.add(used_font)
    # CT_RPr element order (wml.xsd xsd:sequence) — MUST be preserved
    # exactly: rFonts, b, bCs, i, iCs, caps, smallCaps, strike, color,
    # spacing, sz, szCs, highlight (shd fallback), u, vertAlign, lang.
    rfonts = etree.SubElement(rPr, qn("w", "rFonts"))
    rfonts.set(qn("w", "ascii"), used_font)
    rfonts.set(qn("w", "hAnsi"), used_font)

    _append_bool(rPr, "b", style["bold"])
    _append_bool(rPr, "bCs", style["bold"])
    _append_bool(rPr, "i", style["italic"])
    _append_bool(rPr, "iCs", style["italic"])
    _append_bool(rPr, "caps", style["all_caps"])
    _append_bool(rPr, "smallCaps", style["small_caps"])
    _append_bool(rPr, "strike", style["strike"])
    etree.SubElement(rPr, qn("w", "color")).set(qn("w", "val"), _color_to_ooxml(style["color"]))
    etree.SubElement(rPr, qn("w", "spacing")).set(
        qn("w", "val"), str(mpt_to_twip(style["character_spacing_mpt"]))
    )
    size = str(mpt_to_half_point(style["font_size_mpt"]))
    etree.SubElement(rPr, qn("w", "sz")).set(qn("w", "val"), size)
    etree.SubElement(rPr, qn("w", "szCs")).set(qn("w", "val"), size)
    if style.get("highlight") is not None:
        _append_shading(rPr, style["highlight"])
    etree.SubElement(rPr, qn("w", "u")).set(qn("w", "val"), _UNDERLINE_MAP[style["underline"]])
    vert = etree.SubElement(rPr, qn("w", "vertAlign"))
    if style["superscript"]:
        vert_val = "superscript"
    elif style["subscript"]:
        vert_val = "subscript"
    else:
        vert_val = "baseline"
    vert.set(qn("w", "val"), vert_val)
    if node.get("language"):
        etree.SubElement(rPr, qn("w", "lang")).set(qn("w", "val"), node["language"])

    _append_text(r, node["text"], node.get("id"), path)
    return issues


class RendererContractError(ValueError):
    """The AST violated a renderer precondition (RFC001: the Canonical
    AST is defined as deterministically renderable — this is a contract
    breach, not a degradable formatting choice, so it fails loud here
    rather than surfacing 800 lines deep as an opaque lxml error)."""


def _append_text(
    run: etree._Element, text: str, node_id: str | None = None, path: str | None = None
) -> None:
    illegal = find_illegal_xml_char(text)
    if illegal is not None:
        raise RendererContractError(
            f"node {node_id!r} at {path!r} carries XML-1.0-illegal codepoint "
            f"U+{ord(illegal):04X} in its text — the importer must sanitize this "
            "before the AST reaches the renderer."
        )
    t = etree.SubElement(run, qn("w", "t"))
    if _needs_xml_space_preserve(text):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text


_INTERIOR_TAB_OR_NEWLINE = re.compile(r"[\t\n\r]")


def _needs_xml_space_preserve(text: str) -> bool:
    """`xml:space="preserve"` is required whenever XML whitespace
    normalization would alter `text`: empty, leading/trailing
    whitespace, a run of 2+ spaces anywhere, or any interior tab/
    newline/carriage-return — a single interior tab is otherwise
    invisible to the naive "leading/trailing/double-space" heuristic
    this replaces (RFC002 "Runs" — text preserved byte-semantically).
    """
    if text == "" or text != text.strip() or "  " in text:
        return True
    return bool(_INTERIOR_TAB_OR_NEWLINE.search(text))


def _build_sect_pr(
    section: dict, story_refs: Sequence[tuple[str, str, str]] = ()
) -> etree._Element:
    """Build a standalone `w:sectPr` for `section` (RFC002 "Sections").

    Element order is CT_SectPr (wml.xsd): `EG_HdrFtrReferences` first
    (maxOccurs 6), then EG_SectPrContents in its declared sequence —
    of which this renderer emits `type, pgSz, pgMar, pgBorders,
    lnNumType, cols, vAlign, titlePg`. Not yet attached to a parent —
    the caller decides whether it becomes the last child of `w:body`
    (final section) or a child of a paragraph's `w:pPr` (non-final).
    Both positions accept the references: `w:body/w:sectPr` and
    `w:pPr/w:sectPr` are both typed `CT_SectPr`, not `CT_SectPrBase`.

    `titlePg` is *derived* from the presence of a `first` story rather
    than stored as a boolean (DEC-003 §2.1), exactly as
    `evenAndOddHeaders` is derived in `settings.xml`.
    """
    sect_pr = etree.Element(qn("w", "sectPr"))
    for local_name, hdr_ftr_type, rel_id in story_refs:
        reference = etree.SubElement(sect_pr, qn("w", local_name))
        reference.set(qn("w", "type"), hdr_ftr_type)
        reference.set(qn("r", "id"), rel_id)
    etree.SubElement(sect_pr, qn("w", "type")).set(
        qn("w", "val"), _BREAK_TYPE_MAP[section["break_type"]]
    )
    page = section["page"]
    pg_sz = etree.SubElement(sect_pr, qn("w", "pgSz"))
    pg_sz.set(qn("w", "w"), str(mpt_to_twip(page["width_mpt"])))
    pg_sz.set(qn("w", "h"), str(mpt_to_twip(page["height_mpt"])))
    if page["orientation"] == "landscape":
        pg_sz.set(qn("w", "orient"), "landscape")
    margins = section["margins"]
    pg_mar = etree.SubElement(sect_pr, qn("w", "pgMar"))
    pg_mar.set(qn("w", "top"), str(mpt_to_twip(margins["top_mpt"])))
    pg_mar.set(qn("w", "right"), str(mpt_to_twip(margins["right_mpt"])))
    pg_mar.set(qn("w", "bottom"), str(mpt_to_twip(margins["bottom_mpt"])))
    pg_mar.set(qn("w", "left"), str(mpt_to_twip(margins["left_mpt"])))
    pg_mar.set(qn("w", "header"), str(_header_footer_twip(margins, "header_mpt")))
    pg_mar.set(qn("w", "footer"), str(_header_footer_twip(margins, "footer_mpt")))
    pg_mar.set(qn("w", "gutter"), str(mpt_to_twip(margins["gutter_mpt"])))
    _append_page_borders(sect_pr, section.get("page_borders"))
    _append_line_numbering(sect_pr, section.get("line_numbering"))
    _append_page_numbering(sect_pr, section.get("page_numbering"))
    _append_cols(sect_pr, section.get("columns", {}))
    vertical_alignment = section.get("vertical_alignment")
    if vertical_alignment is not None:
        etree.SubElement(sect_pr, qn("w", "vAlign")).set(qn("w", "val"), vertical_alignment)
    if any(hdr_ftr_type == "first" for _local, hdr_ftr_type, _rel in story_refs):
        etree.SubElement(sect_pr, qn("w", "titlePg")).set(qn("w", "val"), "true")
    return sect_pr


def _header_footer_twip(margins: dict, key: str) -> int:
    """`w:pgMar/@w:header`/`@w:footer` from real section geometry.

    `word-core-v0` has no header/footer stories, so it omits the key
    entirely and Word's 0.5in default is the only defensible value;
    `word-visual-v1` carries the measured distance and this must not
    stay hardcoded, or every header would render at the wrong height.
    """
    value_mpt = margins.get(key)
    if value_mpt is None:
        return _DEFAULT_HEADER_FOOTER_MARGIN_TWIP
    return mpt_to_twip(value_mpt)


def _append_line_numbering(sect_pr: etree._Element, line_numbering: dict | None) -> None:
    if line_numbering is None:
        return
    ln_num = etree.SubElement(sect_pr, qn("w", "lnNumType"))
    ln_num.set(qn("w", "countBy"), str(line_numbering["count_by"]))
    ln_num.set(qn("w", "start"), str(line_numbering["start"]))
    ln_num.set(qn("w", "distance"), str(mpt_to_twip(line_numbering["distance_mpt"])))
    ln_num.set(qn("w", "restart"), _LINE_NUMBER_RESTART_MAP[line_numbering["restart"]])


def _append_page_numbering(sect_pr: etree._Element, page_numbering: dict | None) -> None:
    """`w:pgNumType` (CT_PageNumber). `start: null` means "continue the
    previous section", which OOXML expresses by omitting `@w:start`.
    """
    if page_numbering is None:
        return
    pg_num = etree.SubElement(sect_pr, qn("w", "pgNumType"))
    pg_num.set(qn("w", "fmt"), _PAGE_NUMBER_FORMAT_MAP[page_numbering["format"]])
    if page_numbering["start"] is not None:
        pg_num.set(qn("w", "start"), str(page_numbering["start"]))


def _append_page_borders(sect_pr: etree._Element, page_borders: dict | None) -> None:
    """`w:pgBorders` (CT_PageBorders): `top, left, bottom, right` in
    that xsd:sequence order, `display`/`offsetFrom` as attributes.
    """
    if page_borders is None:
        return
    pg_borders = etree.SubElement(sect_pr, qn("w", "pgBorders"))
    pg_borders.set(qn("w", "display"), _PAGE_BORDER_DISPLAY_MAP[page_borders["display"]])
    pg_borders.set(qn("w", "offsetFrom"), page_borders["offset_from"])
    sides = page_borders["sides"]
    for side in ("top", "left", "bottom", "right"):
        border = sides.get(side)
        if not border:
            continue
        side_el = etree.SubElement(pg_borders, qn("w", side))
        side_el.set(qn("w", "val"), _BORDER_STYLE_MAP[border["style"]])
        side_el.set(qn("w", "sz"), str(mpt_to_eighth_point(border["width_mpt"])))
        side_el.set(qn("w", "space"), "0")
        side_el.set(qn("w", "color"), _color_to_ooxml(border["color"]))


def _append_cols(sect_pr: etree._Element, columns: dict) -> None:
    """Append `w:cols` (RFC002 "Sections" — Columns).

    `count == 1` omits `w:cols` entirely (Word's single-column default,
    and it keeps single-column output byte-identical to the pre-R5
    renderer). `equal_width True` uses the compact `@w:num`/`@w:space`/
    `@w:equalWidth` form; `False` emits explicit `w:col` children with
    `@w:w` from `widths_mpt` and the shared `@w:space` gap.
    """
    count = columns.get("count", 1)
    if count <= 1:
        return
    cols = etree.SubElement(sect_pr, qn("w", "cols"))
    cols.set(qn("w", "num"), str(count))
    gap_twip = str(mpt_to_twip(columns["gap_mpt"]))
    if columns["equal_width"]:
        cols.set(qn("w", "space"), gap_twip)
        cols.set(qn("w", "equalWidth"), "1")
    else:
        cols.set(qn("w", "equalWidth"), "0")
        for width_mpt in columns["widths_mpt"]:
            col = etree.SubElement(cols, qn("w", "col"))
            col.set(qn("w", "w"), str(mpt_to_twip(width_mpt)))
            col.set(qn("w", "space"), gap_twip)


def _append_non_final_sect_pr(
    body: etree._Element,
    section: dict,
    before_count: int,
    ctx: BuildContext,
    story_refs: Sequence[tuple[str, str, str]] = (),
) -> None:
    """Attach `section`'s `w:sectPr` inside the `w:pPr` of its own last
    paragraph (RFC002 "Sections"). If the section rendered no paragraph
    as its last block (empty section, or last block is a table/list-
    ending-in-non-paragraph — `word-core-v0` list items always start
    with a paragraph, so only an empty section or a trailing table can
    trigger this), a technical, non-visible boundary paragraph is
    synthesized to carry it (RFC002 "no AST node, no visible text,
    minimized spacing").
    """
    sect_pr = _build_sect_pr(section, story_refs)
    last = body[-1] if len(body) > before_count else None
    if last is not None and etree.QName(last).localname == "p":
        p_pr = last.find(qn("w", "pPr"))
        if p_pr is None:
            p_pr = etree.Element(qn("w", "pPr"))
            last.insert(0, p_pr)
        p_pr.append(sect_pr)
        return

    # Record this paragraph's body-child position *before* appending it
    # (`etree.SubElement` appends last, so `len(body)` here is exactly
    # its future index) — an explicit identity, not a shape inferred
    # later from "no runs + carries sectPr", which a genuinely empty
    # AST paragraph ending a section would also match (RFC002 review
    # finding: false TEXT_CONSERVATION_MISMATCH).
    position = len(body)
    p = etree.SubElement(body, qn("w", "p"))
    p_pr = etree.SubElement(p, qn("w", "pPr"))
    spacing = etree.SubElement(p_pr, qn("w", "spacing"))
    spacing.set(qn("w", "before"), "0")
    spacing.set(qn("w", "after"), "0")
    spacing.set(qn("w", "line"), "240")
    spacing.set(qn("w", "lineRule"), "auto")
    p_pr.append(sect_pr)
    ctx.technical_boundary_paragraph_positions.add(position)
    ctx.mark_technical_paragraph(p)
