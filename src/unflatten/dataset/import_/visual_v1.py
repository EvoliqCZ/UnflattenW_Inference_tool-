"""`word-visual-v1` import surfaces (DEC-003, RFC001 §42.2).

The constructs this module recovers were, under `word-core-v0`, either
dropped with a typed issue (header/footer stories, anchored drawings,
hyperlink targets) or rejected outright (nested tables). Everything here
runs only when `ImportOptions.profile == "word-visual-v1"`; under the
default profile the importer behaves exactly as it did before, so no
pinned `ast_hash` moves until a caller opts in.

Three rules earn their own note, because getting them backwards is
silent damage rather than a visible failure:

1. **Relationship ids are part-local.** Resolution goes through
   `ctx.rels`, which answers for the part being parsed — see
   `_Ctx.in_part`. Never merge a story's relationships into the body's.
2. **`titlePg` with no `first` reference means the first page has NO
   header**, which is imported as a `first` story with `children: []`.
   An absent `first` story means "inherit the default", which is the
   opposite claim.
3. **`wrapTight` / `wrapThrough` are mapped, not carried.** Both
   require a `wp:wrapPolygon` (CT_WrapTight/CT_WrapThrough,
   `wrapPolygon` minOccurs=1), i.e. pixel geometry that RFC001 §0.1
   keeps out of the AST. They become `square` with a declared
   `IMPORT_WRAP_APPROXIMATED` issue, following the
   `IMPORT_BORDER_STYLE_APPROXIMATED` precedent.

Imports from `.importer` are deferred into the call sites: the importer
owns the paragraph/table/run parsers this module reuses and imports this
module in turn, so a module-level import would be circular.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from lxml import etree

from unflatten.renderer.xml_writer import NSMAP

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .importer import _Ctx

_W = NSMAP["w"]
_WP_NS = NSMAP["wp"]
_R_NS = NSMAP["r"]
_VML_OFFICE_NS = "urn:schemas-microsoft-com:office:office"
_VML_WORD_NS = "urn:schemas-microsoft-com:office:word"

# ECMA-376 token -> AST enum member. Every map is the vendored XSD's
# enumeration in full (vendor/ecma376/transitional/*.xsd), so an
# unmapped value means a malformed source, not a gap.
_REL_FROM_H = {
    "margin": "margin", "page": "page", "column": "column",
    "character": "character", "leftMargin": "left_margin",
    "rightMargin": "right_margin", "insideMargin": "inside_margin",
    "outsideMargin": "outside_margin",
}
_REL_FROM_V = {
    "margin": "margin", "page": "page", "paragraph": "paragraph",
    "line": "line", "topMargin": "top_margin", "bottomMargin": "bottom_margin",
    "insideMargin": "inside_margin", "outsideMargin": "outside_margin",
}
_ALIGN_H = frozenset({"left", "right", "center", "inside", "outside"})
_ALIGN_V = frozenset({"top", "bottom", "center", "inside", "outside"})
_WRAP_TEXT = {
    "bothSides": "both_sides", "left": "left", "right": "right",
    "largest": "largest",
}
_LINE_NUMBER_RESTART = {
    "continuous": "continuous", "newPage": "new_page", "newSection": "new_section",
}
_PAGE_BORDER_DISPLAY = {
    "allPages": "all_pages", "firstPage": "first_page",
    "notFirstPage": "not_first_page",
}
_SECTION_VERTICAL_ALIGNMENT = frozenset({"top", "center", "both", "bottom"})
_PAGE_NUMBER_FORMAT = {
    "decimal": "decimal", "lowerLetter": "lower_alpha", "upperLetter": "upper_alpha",
    "lowerRoman": "lower_roman", "upperRoman": "upper_roman",
}
# THE ENUM IS THE WHITELIST (schema `field.field_type`): RFC001 §30
# forbids hallucinating a field code, so anything else must flatten
# rather than pass as a field. Measured over 1500 raw originals: PAGE in
# 38.0 % of documents, NUMPAGES in 9.5 %; the next candidate (TOC) is
# 4.0 % and stays [OPEN].
_FIELD_TYPES = {"PAGE": "page", "NUMPAGES": "num_pages"}
# `\* MERGEFORMAT` and friends preserve the manual formatting of the
# previous result — invisible in an AST that stores no cached result.
_COSMETIC_SWITCHES = frozenset({"MERGEFORMAT", "CHARFORMAT"})
_STORY_KINDS = ("default", "first", "even")


def _w(tag: str) -> str:
    return f"{{{_W}}}{tag}"


def _wp(tag: str) -> str:
    return f"{{{_WP_NS}}}{tag}"


# --- anchored drawings --------------------------------------------------


def parse_anchor(ctx: _Ctx, anchor: etree._Element, node_id: str) -> dict[str, Any]:
    """`wp:anchor` -> the AST `anchor` object (Word-native, no pixel box)."""
    from .importer import _int_attr

    return {
        "horizontal": _position(
            ctx, anchor.find(_wp("positionH")), node_id,
            _REL_FROM_H, _ALIGN_H, "margin", "left",
        ),
        "vertical": _position(
            ctx, anchor.find(_wp("positionV")), node_id,
            _REL_FROM_V, _ALIGN_V, "paragraph", "top",
        ),
        "wrap": _wrap(ctx, anchor, node_id),
        "behind_text": _parse_bool_attr(anchor, "behindDoc"),
        "allow_overlap": _parse_bool_attr(anchor, "allowOverlap", default=True),
        "z_order": max(
            0, _int_attr(ctx, anchor, "relativeHeight", default=0, node_id=node_id) or 0
        ),
    }


def parse_vml_anchor(ctx: _Ctx, shape: etree._Element, node_id: str) -> dict[str, Any]:
    """Normalize the supported VML positioning subset to the shared anchor.

    VML stores Word-native positioning in a CSS-like ``v:shape/@style``.
    Only those positioning declarations enter the AST; width/height and
    shape paint are deliberately outside the ``text_frame`` contract.
    """
    style = _vml_style(shape.get("style", ""))
    horizontal = _vml_position(
        ctx, shape, node_id, style, "horizontal",
        {"margin": "margin", "page": "page", "text": "column",
         "char": "character", "character": "character"},
        frozenset({"left", "right", "center", "inside", "outside"}),
        "margin", "left", "margin-left",
    )
    vertical = _vml_position(
        ctx, shape, node_id, style, "vertical",
        {"margin": "margin", "page": "page", "text": "paragraph",
         "line": "line"},
        frozenset({"top", "bottom", "center", "inside", "outside"}),
        "paragraph", "top", "margin-top",
    )
    z_index = _vml_integer(ctx, shape, node_id, style.get("z-index"), "z-index")
    overlap = shape.get(f"{{{_VML_OFFICE_NS}}}allowoverlap")
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "wrap": _vml_wrap(ctx, shape, node_id),
        "behind_text": z_index < 0,
        "allow_overlap": overlap not in ("0", "false", "off"),
        "z_order": max(0, z_index),
    }


def _vml_style(raw: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for declaration in raw.split(";"):
        key, separator, value = declaration.partition(":")
        if separator and key.strip():
            declarations[key.strip().lower()] = value.strip()
    return declarations


def _vml_position(
    ctx: _Ctx, shape: etree._Element, node_id: str, style: dict[str, str], axis: str,
    relatives: dict[str, str], aligns: frozenset[str], rel_default: str,
    align_default: str, offset_key: str,
) -> dict[str, Any]:
    raw_rel = style.get(f"mso-position-{axis}-relative")
    relative = relatives.get(raw_rel or "", rel_default)
    if raw_rel is not None and raw_rel not in relatives:
        _approximate(
            ctx, node_id, "IMPORT_ANCHOR_POSITION_DEFAULTED",
            f"VML {axis} relative value {raw_rel!r} is unsupported; "
            f"{rel_default!r} was used (approximate import).",
            el=shape, relative_from=raw_rel,
        )
    align = style.get(f"mso-position-{axis}")
    if align in aligns:
        return {"relative_from": relative, "mode": "align", "align": align}
    offset = style.get(offset_key)
    if offset is not None:
        return {
            "relative_from": relative,
            "mode": "offset",
            "offset_mpt": _vml_length_mpt(ctx, shape, node_id, offset, offset_key),
        }
    if align is not None:
        _approximate(
            ctx, node_id, "IMPORT_ANCHOR_POSITION_DEFAULTED",
            f"VML {axis} alignment {align!r} is unsupported; {align_default!r} "
            "was used (approximate import).",
            el=shape, align=align,
        )
    return {"relative_from": relative, "mode": "align", "align": align_default}


def _vml_length_mpt(
    ctx: _Ctx, shape: etree._Element, node_id: str, raw: str, property_name: str,
) -> int:
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(pt|in|cm|mm)", raw.strip())
    if match is None:
        _approximate(
            ctx, node_id, "IMPORT_ANCHOR_POSITION_DEFAULTED",
            f"VML {property_name} value {raw!r} is not a supported physical length; "
            "offset 0 was used (approximate import).",
            el=shape, value=raw,
        )
        return 0
    value, unit = float(match.group(1)), match.group(2)
    points = value * {"pt": 1.0, "in": 72.0, "cm": 72.0 / 2.54, "mm": 72.0 / 25.4}[unit]
    return round(points * 1000)


def _vml_integer(
    ctx: _Ctx, shape: etree._Element, node_id: str, raw: str | None, name: str,
) -> int:
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError:
        _approximate(
            ctx, node_id, "IMPORT_MALFORMED_VALUE",
            f"VML {name} value {raw!r} is not an integer; 0 was used "
            "(approximate import).",
            el=shape, value=raw,
        )
        return 0


def _vml_wrap(ctx: _Ctx, shape: etree._Element, node_id: str) -> dict[str, Any]:
    wrap = shape.find(f"{{{_VML_WORD_NS}}}wrap")
    if wrap is None:
        return {"mode": "none"}
    raw_type = wrap.get("type", "none")
    if raw_type == "none":
        return {"mode": "none"}
    if raw_type == "topAndBottom":
        return {"mode": "top_and_bottom"}
    if raw_type in ("square", "tight", "through"):
        if raw_type != "square":
            _approximate(
                ctx, node_id, "IMPORT_WRAP_APPROXIMATED",
                f"VML wrap type {raw_type!r} requires outline geometry the AST does "
                "not carry; imported as square (approximate import).",
                el=wrap, wrap=raw_type,
            )
        side = {"both": "both_sides", "left": "left", "right": "right",
                "largest": "largest"}.get(wrap.get("side", "both"), "both_sides")
        return {"mode": "square", "side": side}
    _approximate(
        ctx, node_id, "IMPORT_WRAP_APPROXIMATED",
        f"VML wrap type {raw_type!r} is unsupported; no wrap was used "
        "(approximate import).",
        el=wrap, wrap=raw_type,
    )
    return {"mode": "none"}


def _parse_bool_attr(el: etree._Element, attr: str, *, default: bool = False) -> bool:
    raw = el.get(attr)
    if raw is None:
        return default
    return raw not in ("0", "false", "off")


def _position(ctx: _Ctx, pos: etree._Element | None, node_id: str,
              rel_map: dict[str, str], aligns: frozenset[str],
              rel_default: str, align_default: str) -> dict[str, Any]:
    """CT_PosH / CT_PosV -> a relative_from plus an align or an offset.

    A missing or unmapped `relativeFrom`, and an unmapped `wp:align`
    token, fall back to the Word default with a typed issue: a malformed
    anchor must degrade the node, never fail the document (RFC003 §45).
    """
    from .importer import _emu_to_mpt

    if pos is None:
        _approximate(
            ctx, node_id, "IMPORT_ANCHOR_POSITION_DEFAULTED",
            "anchored drawing has no wp:positionH/wp:positionV; the Word default "
            f"alignment '{align_default}' relative to '{rel_default}' was used "
            "(approximate import).",
        )
        return {"relative_from": rel_default, "mode": "align", "align": align_default}

    raw_rel = pos.get("relativeFrom")
    relative_from = rel_map.get(raw_rel or "", rel_default)
    if raw_rel is not None and raw_rel not in rel_map:
        _approximate(
            ctx, node_id, "IMPORT_ANCHOR_POSITION_DEFAULTED",
            f"anchor relativeFrom '{raw_rel}' is not an ECMA-376 value; "
            f"'{rel_default}' was used (approximate import).",
            el=pos, relative_from=raw_rel,
        )

    offset = pos.find(_wp("posOffset"))
    if offset is not None:
        try:
            emu = int((offset.text or "0").strip())
        except ValueError:
            _approximate(
                ctx, node_id, "IMPORT_MALFORMED_VALUE",
                f"wp:posOffset value {offset.text!r} is not a valid integer; offset 0 "
                "was used (approximate import).",
                el=offset, value=offset.text,
            )
            emu = 0
        return {"relative_from": relative_from, "mode": "offset",
                "offset_mpt": _emu_to_mpt(emu)}

    align_el = pos.find(_wp("align"))
    align = (align_el.text or "").strip() if align_el is not None else ""
    if align not in aligns:
        if align_el is not None:
            _approximate(
                ctx, node_id, "IMPORT_ANCHOR_POSITION_DEFAULTED",
                f"anchor alignment {align!r} is not an ECMA-376 value; "
                f"'{align_default}' was used (approximate import).",
                el=align_el, align=align,
            )
        else:
            _approximate(
                ctx, node_id, "IMPORT_ANCHOR_POSITION_DEFAULTED",
                "anchor position carries neither wp:align nor wp:posOffset; "
                f"'{align_default}' was used (approximate import).",
                el=pos,
            )
        align = align_default
    return {"relative_from": relative_from, "mode": "align", "align": align}


def _wrap(ctx: _Ctx, anchor: etree._Element, node_id: str) -> dict[str, Any]:
    """EG_WrapType -> the AST wrap object.

    `wrapTight`/`wrapThrough` carry a `wp:wrapPolygon` the AST cannot
    hold, so they are DECLARED approximations to `square`, never a
    silent coercion.
    """
    if anchor.find(_wp("wrapNone")) is not None:
        return {"mode": "none"}
    if anchor.find(_wp("wrapTopAndBottom")) is not None:
        return {"mode": "top_and_bottom"}
    square = anchor.find(_wp("wrapSquare"))
    if square is not None:
        return {"mode": "square", "side": _wrap_side(ctx, square, node_id)}
    for tag in ("wrapTight", "wrapThrough"):
        element = anchor.find(_wp(tag))
        if element is None:
            continue
        _approximate(
            ctx, node_id, "IMPORT_WRAP_APPROXIMATED",
            f"wp:{tag} wraps text to a wp:wrapPolygon, which the AST deliberately "
            "does not carry (RFC001 §0.1 — no pixel geometry); imported as a square "
            "wrap with the same side (approximate import).",
            el=element, wrap=tag,
        )
        return {"mode": "square", "side": _wrap_side(ctx, element, node_id)}
    # EG_WrapType is a required group, so no branch means a malformed
    # anchor; Word renders such a drawing in front of the text.
    _approximate(
        ctx, node_id, "IMPORT_WRAP_APPROXIMATED",
        "anchored drawing carries no EG_WrapType child; imported as an unwrapped "
        "(floating) drawing (approximate import).",
        el=anchor,
    )
    return {"mode": "none"}


def _wrap_side(ctx: _Ctx, element: etree._Element, node_id: str) -> str:
    raw = element.get("wrapText")
    if raw is None:
        return "both_sides"
    side = _WRAP_TEXT.get(raw)
    if side is None:
        _approximate(
            ctx, node_id, "IMPORT_WRAP_APPROXIMATED",
            f"wrapText '{raw}' is not an ECMA-376 value; 'both_sides' was used "
            "(approximate import).",
            el=element, wrap_text=raw,
        )
        return "both_sides"
    return side


def _approximate(ctx: _Ctx, node_id: str, code: str, message: str,
                 el: etree._Element | None = None, **details: Any) -> None:
    ctx.issue(code, "warning", message, el=el, node_id=node_id, **details)
    ctx.degrade(node_id, "approximate")


# --- advanced section settings -----------------------------------------


def section_settings(ctx: _Ctx, sect_pr: etree._Element,
                     node_id: str) -> dict[str, Any]:
    """`w:vAlign`, `w:lnNumType` and `w:pgBorders` -> optional section keys.

    Each key is omitted when the source does not carry it, so a section
    without advanced settings canonicalizes exactly as it did under
    `word-core-v0`.
    """
    from .importer import _borders_from, _int_attr, _twip_to_mpt

    settings: dict[str, Any] = {}

    v_align = sect_pr.find(_w("vAlign"))
    if v_align is not None:
        value = v_align.get(_w("val"))
        if value in _SECTION_VERTICAL_ALIGNMENT:
            settings["vertical_alignment"] = value
        else:
            _approximate(
                ctx, node_id, "IMPORT_SECTION_FEATURE_APPROXIMATED",
                f"section vAlign '{value}' is not an ECMA-376 value; the section "
                "vertical alignment was not imported (approximate import).",
                el=v_align, element="vAlign", value=value,
            )

    pg_num = sect_pr.find(_w("pgNumType"))
    if pg_num is not None:
        settings["page_numbering"] = _page_numbering(ctx, pg_num, node_id)

    ln_num = sect_pr.find(_w("lnNumType"))
    if ln_num is not None:
        count_by = _int_attr(ctx, ln_num, _w("countBy"), default=1, node_id=node_id) or 1
        restart_raw = ln_num.get(_w("restart"), "newPage")
        restart = _LINE_NUMBER_RESTART.get(restart_raw)
        if restart is None:
            _approximate(
                ctx, node_id, "IMPORT_SECTION_FEATURE_APPROXIMATED",
                f"lnNumType restart '{restart_raw}' is not an ECMA-376 value; "
                "'new_page' (the Word default) was used (approximate import).",
                el=ln_num, element="lnNumType", restart=restart_raw,
            )
            restart = "new_page"
        settings["line_numbering"] = {
            "count_by": max(1, count_by),
            "start": max(0, _int_attr(
                ctx, ln_num, _w("start"), default=1, node_id=node_id) or 1),
            "restart": restart,
            "distance_mpt": max(0, _twip_to_mpt(_int_attr(
                ctx, ln_num, _w("distance"), default=0, node_id=node_id) or 0)),
        }

    pg_borders = sect_pr.find(_w("pgBorders"))
    if pg_borders is not None:
        sides = _borders_from(ctx, pg_borders, node_id)
        if sides is None:
            _approximate(
                ctx, node_id, "IMPORT_SECTION_FEATURE_APPROXIMATED",
                "w:pgBorders declares no importable side; the page border was not "
                "imported (approximate import).",
                el=pg_borders, element="pgBorders",
            )
        else:
            settings["page_borders"] = {
                "sides": sides,
                "display": _PAGE_BORDER_DISPLAY.get(
                    pg_borders.get(_w("display"), "allPages"), "all_pages"),
                "offset_from": (
                    "page" if pg_borders.get(_w("offsetFrom")) == "page" else "text"
                ),
            }
    return settings


def _page_numbering(ctx: _Ctx, pg_num: etree._Element,
                    node_id: str) -> dict[str, Any]:
    """`w:pgNumType` -> `section.page_numbering`.

    `start: null` means "continue the previous section", which is what
    an absent `w:start` states. Chapter numbering is not modelled.
    """
    from .importer import _int_attr

    raw = pg_num.get(_w("fmt"))
    fmt = _PAGE_NUMBER_FORMAT.get(raw or "decimal")
    if fmt is None:
        _approximate(
            ctx, node_id, "IMPORT_PAGE_NUMBER_FORMAT_APPROXIMATED",
            f"page number format '{raw}' has no word-visual-v1 mapping; decimal was "
            "used (approximate import).",
            el=pg_num, element="pgNumType", format=raw,
        )
        fmt = "decimal"
    start = (
        _int_attr(ctx, pg_num, _w("start"), default=None, node_id=node_id)
        if pg_num.get(_w("start")) is not None
        else None
    )
    return {"format": fmt, "start": max(0, start) if start is not None else None}


def parse_field_instruction(ctx: _Ctx, instruction: str | None,
                            el: etree._Element) -> str | None:
    """A field instruction -> a whitelisted `field_type`, or None.

    None means "not on the whitelist": the caller falls back to
    flattening the field to its visible result runs, which is what
    word-core-v0 always did and never loses ink.
    """
    if not instruction:
        return None
    tokens = instruction.split()
    if not tokens:
        return None
    field_type = _FIELD_TYPES.get(tokens[0].upper())
    if field_type is None:
        return None
    _report_switches(ctx, tokens[1:], el, field_type)
    return field_type


def _report_switches(ctx: _Ctx, arguments: list[str], el: etree._Element,
                     field_type: str) -> None:
    """Field switches are not modelled: the node carries the field's
    identity, and the numbering scheme comes from `section.page_numbering`.

    A cosmetic switch (`\\* MERGEFORMAT`, present in a fifth of the
    corpus) is invisible here, because this AST stores no cached result
    whose formatting could be preserved. A numeric-format switch
    (`\\* ROMAN`) is a DECLARED approximation: a per-page-correct number
    in the section's numerals beats the same frozen number on every page.
    """
    for argument in arguments:
        # The switch markers themselves (\*, \#, \@, \s) carry no
        # meaning without their argument, which is the next token.
        if argument.startswith("\\"):
            continue
        name = argument.upper()
        if name in _COSMETIC_SWITCHES:
            ctx.issue(
                "IMPORT_FIELD_SWITCH_DROPPED", "info",
                f"field switch argument '{name}' preserves the formatting of a "
                "cached result the AST does not store; dropped as invisible.",
                el=el, field_type=field_type, switch=name,
            )
        else:
            ctx.issue(
                "IMPORT_FIELD_SWITCH_APPROXIMATED", "warning",
                f"field switch argument '{name}' is not modelled; the field's "
                "numbering follows section.page_numbering instead (approximate "
                "import).",
                el=el, field_type=field_type, switch=name,
            )


def header_footer_margins(ctx: _Ctx, pg_mar: etree._Element | None,
                          node_id: str) -> dict[str, int]:
    """`w:pgMar/@w:header` and `@w:footer` -> the optional margin keys.

    Only emitted when the source states them: an absent key keeps the
    canonical JSON of a header-less document unchanged.
    """
    from .importer import _int_attr, _twip_to_mpt

    if pg_mar is None:
        return {}
    margins: dict[str, int] = {}
    for attr, key in (("header", "header_mpt"), ("footer", "footer_mpt")):
        if pg_mar.get(_w(attr)) is None:
            continue
        value = _twip_to_mpt(
            _int_attr(ctx, pg_mar, _w(attr), default=0, node_id=node_id) or 0
        )
        margins[key] = max(0, value)
    return margins


# --- header / footer stories -------------------------------------------


def import_stories(ctx: _Ctx, sect_pr: etree._Element,
                   node_id: str) -> dict[str, list[dict[str, Any]]]:
    """Import a section's header and footer stories.

    Word only RENDERS a `first` story when `w:titlePg` is set, and an
    `even` story when `w:evenAndOddHeaders` is set in settings.xml. A
    reference whose switch is off is ink that never appears, so it is
    dropped with a typed issue instead of being imported into an AST the
    renderer would then draw.

    The converse is the `titlePg` trap: `titlePg` set with NO `first`
    reference means the first page shows no header at all, which is an
    EMPTY `first` story — an absent story would mean "inherit the
    default" and put a header on a title page that must have none.
    """
    title_pg = sect_pr.find(_w("titlePg")) is not None and (
        sect_pr.find(_w("titlePg")).get(_w("val")) not in ("0", "false", "off")
    )
    stories: dict[str, list[dict[str, Any]]] = {}
    for tag, key, story_type in (
        ("headerReference", "headers", "header"),
        ("footerReference", "footers", "footer"),
    ):
        nodes = _stories_of_type(ctx, sect_pr, tag, story_type, node_id, title_pg)
        if nodes:
            stories[key] = nodes
    return stories


def _stories_of_type(ctx: _Ctx, sect_pr: etree._Element, tag: str, story_type: str,
                     node_id: str, title_pg: bool) -> list[dict[str, Any]]:
    references: dict[str, etree._Element] = {}
    for reference in sect_pr.findall(_w(tag)):
        kind = reference.get(_w("type"), "default")
        if kind not in _STORY_KINDS:
            _approximate(
                ctx, node_id, "IMPORT_STORY_REFERENCE_UNRESOLVED",
                f"w:{tag} type '{kind}' is not an ECMA-376 header/footer kind; the "
                "reference was not imported (approximate import).",
                el=reference, element=tag, kind=kind,
            )
            continue
        if kind in references:
            _approximate(
                ctx, node_id, "IMPORT_DUPLICATE_STORY_REFERENCE",
                f"section declares more than one '{kind}' w:{tag}; only the first is "
                "imported, since two stories of one kind make 'which one renders' "
                "unanswerable (approximate import).",
                el=reference, element=tag, kind=kind,
            )
            continue
        references[kind] = reference

    stories: list[dict[str, Any]] = []
    for kind in _STORY_KINDS:
        reference = references.get(kind)
        if kind == "first" and not title_pg:
            if reference is not None:
                _drop_unrendered(ctx, reference, node_id, tag, kind, "w:titlePg")
            continue
        if kind == "even" and not _even_and_odd_headers(ctx):
            if reference is not None:
                _drop_unrendered(
                    ctx, reference, node_id, tag, kind, "w:evenAndOddHeaders"
                )
            continue
        if reference is None:
            if kind != "first" or not stories:
                # An empty `first` story exists to BLOCK inheritance of
                # the section's default on page 1. With no other story
                # of this type there is nothing to inherit, so the
                # override would assert a difference that does not
                # exist — a section with no footers at all must not gain
                # an empty first footer.
                continue
            # titlePg is on and no first-page story exists: the first
            # page renders NO header/footer. That is an empty story, not
            # a missing one.
            story = _story_node(ctx, story_type, kind, [], sect_pr)
            ctx.issue(
                "IMPORT_TITLE_PAGE_STORY_EMPTY", "info",
                f"w:titlePg is set with no 'first' w:{tag}: the first page renders no "
                f"{story_type}, imported as an empty 'first' story (an absent story "
                "would mean the section's default is inherited).",
                el=sect_pr, node_id=story["id"],
            )
            stories.append(story)
            continue
        children = _story_children(ctx, reference, story_type, node_id)
        if children is None:
            continue
        stories.append(_story_node(ctx, story_type, kind, children, reference))
    return stories


def _drop_unrendered(ctx: _Ctx, reference: etree._Element, node_id: str, tag: str,
                     kind: str, switch: str) -> None:
    prov = ctx.prov(reference)
    ctx.source_map[f"dropped:{prov}"] = {
        "part": ctx.part,
        "provenance": prov,
        "import_status": "dropped",
    }
    _approximate(
        ctx, node_id, "IMPORT_STORY_REFERENCE_UNRENDERED",
        f"section declares a '{kind}' w:{tag} without {switch}, so Word never renders "
        "it; the story was not imported and the reference recorded in the source map "
        "(approximate import).",
        el=reference, element=tag, kind=kind,
    )


def _even_and_odd_headers(ctx: _Ctx) -> bool:
    """`w:evenAndOddHeaders` in settings.xml — the document-wide switch
    that decides whether `even` stories render at all."""
    if ctx.even_and_odd_headers is None:
        data = ctx.parts.get("word/settings.xml")
        enabled = False
        if data is not None:
            try:
                root = etree.fromstring(data)
            except etree.XMLSyntaxError as exc:
                ctx.issue(
                    "IMPORT_SETTINGS_MALFORMED", "warning",
                    "word/settings.xml is not well-formed XML; even-page headers are "
                    f"treated as disabled: {exc}",
                )
                root = None
            if root is not None:
                element = root.find(_w("evenAndOddHeaders"))
                enabled = element is not None and element.get(
                    _w("val")
                ) not in ("0", "false", "off")
        ctx.even_and_odd_headers = enabled
    return ctx.even_and_odd_headers


def _story_node(ctx: _Ctx, story_type: str, kind: str,
                children: list[dict[str, Any]], el: etree._Element) -> dict[str, Any]:
    node_id = ctx.new_id()
    ctx.map_node(node_id, el)
    return {"id": node_id, "type": story_type, "kind": kind, "children": children}


def _story_children(ctx: _Ctx, reference: etree._Element, story_type: str,
                    node_id: str) -> list[dict[str, Any]] | None:
    """Parse the story part a `w:headerReference`/`w:footerReference`
    points at, INSIDE that part's relationship scope.

    Returns None when the reference cannot be resolved — a missing part
    is a source defect, reported and skipped, never a crash.
    """
    from .importer import _resolve_target

    rel_id = reference.get(f"{{{_R_NS}}}id")
    target = ctx.rels.get(rel_id) if rel_id else None
    if target is None:
        _approximate(
            ctx, node_id, "IMPORT_STORY_REFERENCE_UNRESOLVED",
            f"{story_type} reference r:id={rel_id!r} resolves to no relationship in "
            f"'{ctx.part}'; the story was not imported (approximate import).",
            el=reference, rel_id=rel_id,
        )
        return None
    part = _resolve_target(ctx, target)
    data = ctx.parts.get(part)
    if data is None:
        _approximate(
            ctx, node_id, "IMPORT_STORY_PART_MISSING",
            f"{story_type} part '{part}' is missing from the package; the story was "
            "not imported (approximate import).",
            el=reference, story_part=part,
        )
        return None
    return parse_story_part(ctx, data, part, story_type, node_id)


def parse_story_part(ctx: _Ctx, data: bytes, part: str, story_type: str,
                     node_id: str) -> list[dict[str, Any]] | None:
    """Parse one story part's root children into story block nodes."""
    try:
        root = etree.fromstring(data)
    except etree.XMLSyntaxError as exc:
        _approximate(
            ctx, node_id, "IMPORT_STORY_PART_MALFORMED",
            f"{story_type} part '{part}' is not well-formed XML; the story was not "
            f"imported (approximate import): {exc}",
            story_part=part,
        )
        return None
    with ctx.in_part(part, etree.ElementTree(root)):
        return parse_story_blocks(ctx, root, story_type)


def parse_story_blocks(ctx: _Ctx, root: etree._Element,
                       story_type: str) -> list[dict[str, Any]]:
    """Block content of a story root, with page/column breaks removed.

    `story_block_node` is the body block set minus the two break types:
    a page break inside a header is syntactically expressible and
    semantically meaningless, so it is dropped with a typed issue rather
    than emitted into an AST the validator rejects.
    """
    from .importer import _assemble_blocks, _iter_content, _parse_p, _parse_table

    raw: list[dict[str, Any]] = []
    for child in _iter_content(ctx, root):
        tag = etree.QName(child).localname
        if tag == "p":
            raw.extend(_parse_p(ctx, child, in_cell=True))
        elif tag == "tbl":
            table = _parse_table(ctx, child)
            if table is not None:
                raw.append({"kind": "block", "node": table})
        elif tag in ("bookmarkStart", "bookmarkEnd", "proofErr",
                     "permStart", "permEnd"):
            ctx.issue(
                "IMPORT_IGNORED_ELEMENT", "info",
                f"non-content element 'w:{tag}' ignored.", el=child, element=tag,
            )
        else:
            ctx.issue(
                "IMPORT_UNSUPPORTED_ELEMENT", "error",
                f"{story_type} child element 'w:{tag}' is not supported by the "
                "word-visual-v1 importer.",
                el=child, element=tag,
            )
    blocks = _assemble_blocks(ctx, raw)
    kept: list[dict[str, Any]] = []
    for block in blocks:
        if block["type"] in ("page_break", "column_break"):
            ctx.issue(
                "IMPORT_STORY_BREAK_DROPPED", "warning",
                f"a {block['type']} inside a {story_type} has no rendered meaning and "
                "is outside the story content set; dropped (approximate import).",
                node_id=block["id"],
            )
            continue
        kept.append(block)
    return kept


# --- footnote / endnote stories ----------------------------------------

# w:footnote/@w:type: only "normal" (the default) is a real note body.
# separator / continuationSeparator / continuationNotice are the
# rendering furniture Word stores in the same part.
_NOTE_FURNITURE = frozenset({
    "separator", "continuationSeparator", "continuationNotice",
})


def load_note_bodies(ctx: _Ctx) -> None:
    """Pre-parse `word/footnotes.xml` and `word/endnotes.xml`.

    Bodies are parsed BEFORE the body text, so a `w:footnoteReference`
    can resolve to a real node id while the body is parsed linearly.
    Which bodies are actually EMITTED is decided afterwards by
    `referenced_notes`: an unreferenced body is content Word never
    renders, and `_check_note_refs` rejects it.
    """
    for kind, part in (("footnote", "word/footnotes.xml"),
                       ("endnote", "word/endnotes.xml")):
        data = ctx.parts.get(part)
        if data is None:
            continue
        try:
            root = etree.fromstring(data)
        except etree.XMLSyntaxError as exc:
            ctx.issue(
                "IMPORT_NOTE_PART_MALFORMED", "warning",
                f"'{part}' is not well-formed XML; its note bodies were not imported "
                f"(approximate import): {exc}",
            )
            continue
        with ctx.in_part(part, etree.ElementTree(root)):
            for note in root.findall(_w(kind)):
                note_id = note.get(_w("id"))
                if note_id is None or note.get(_w("type")) in _NOTE_FURNITURE:
                    continue
                _load_one_note(ctx, note, kind, note_id)


def _load_one_note(ctx: _Ctx, note: etree._Element, kind: str, note_id: str) -> None:
    node_id = ctx.new_id()
    ctx.map_node(node_id, note)
    saved = (ctx.in_note_body, ctx.note_mark_key)
    ctx.in_note_body = True
    ctx.note_mark_key = (kind, note_id)
    try:
        children = parse_story_blocks(ctx, note, kind)
    finally:
        ctx.in_note_body, ctx.note_mark_key = saved
    ctx.note_bodies[(kind, note_id)] = {
        "id": node_id, "type": kind, "children": children,
    }


def note_body_for(ctx: _Ctx, kind: str, note_id: str | None) -> dict[str, Any] | None:
    """The imported body a reference points at, marked as referenced."""
    if note_id is None:
        return None
    body = ctx.note_bodies.get((kind, note_id))
    if body is None:
        return None
    ctx.referenced_notes.add((kind, note_id))
    return body


def emitted_notes(ctx: _Ctx) -> dict[str, list[dict[str, Any]]]:
    """The note bodies some `note_ref` reached, keyed by AST array.

    An unreferenced body is reported and left out: `_check_note_refs`
    rejects an AST carrying one, and the content is not rendered by Word
    either.
    """
    arrays: dict[str, list[dict[str, Any]]] = {}
    for (kind, source_id), body in ctx.note_bodies.items():
        if (kind, source_id) not in ctx.referenced_notes:
            ctx.issue(
                "IMPORT_UNREFERENCED_NOTE_DROPPED", "warning",
                f"{kind} w:id={source_id} has no reference in the document body; Word "
                "renders no such note, so the body was not imported (approximate "
                "import).",
                node_id=body["id"],
            )
            continue
        arrays.setdefault(f"{kind}s", []).append(body)
    return arrays
