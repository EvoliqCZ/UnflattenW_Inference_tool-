"""Boilerplate OPC/OOXML parts: content types, relationships, styles,
settings, fontTable (RFC002 "Package structure", "Styles", "Fonts",
"settings.xml").

These parts do not depend on document content beyond the set of fonts
used and the relationship targets, so they are built independently of
the paragraph/run compiler.
"""

from __future__ import annotations

from lxml import etree

from unflatten.renderer.ooxml.relationships import (
    RELTYPE_CORE_PROPERTIES,
    RELTYPE_FONT_TABLE,
    RELTYPE_OFFICE_DOCUMENT,
    RELTYPE_SETTINGS,
    RELTYPE_STYLES,
    PartRelationships,
)
from unflatten.renderer.xml_writer import NSMAP, qn, serialize

_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

# Content type constants — public, consumed by compiler.py to build the
# [Content_Types].xml Overrides map.
CONTENT_TYPE_RELS = "application/vnd.openxmlformats-package.relationships+xml"
CONTENT_TYPE_XML = "application/xml"
CONTENT_TYPE_DOCUMENT = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
)
CONTENT_TYPE_STYLES = "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"
CONTENT_TYPE_SETTINGS = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"
)
CONTENT_TYPE_FONT_TABLE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml"
)
CONTENT_TYPE_NUMBERING = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"
)
CONTENT_TYPE_CORE_PROPERTIES = "application/vnd.openxmlformats-package.core-properties+xml"
CONTENT_TYPE_HEADER = "application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"
CONTENT_TYPE_FOOTER = "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"
CONTENT_TYPE_FOOTNOTES = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"
)
CONTENT_TYPE_ENDNOTES = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.endnotes+xml"
)

_CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
_DC_NS = "http://purl.org/dc/elements/1.1/"
_CORE_PROPERTIES_CREATOR = "unflatten"


def build_content_types(
    overrides: dict[str, str], extra_defaults: dict[str, str] | None = None
) -> bytes:
    """Build `[Content_Types].xml`.

    `overrides` maps part-name (e.g. "/word/document.xml") -> content
    type, sorted by PartName. `extra_defaults` maps a file extension
    (no dot, e.g. "png") -> content type, for parts identified by
    extension rather than by name (media parts); merged with the
    always-emitted `rels`/`xml` defaults, sorted by Extension.
    """
    root = etree.Element(f"{{{_CT_NS}}}Types", nsmap={None: _CT_NS})
    defaults = {"rels": CONTENT_TYPE_RELS, "xml": CONTENT_TYPE_XML, **(extra_defaults or {})}
    for extension, content_type in sorted(defaults.items()):
        default = etree.SubElement(root, f"{{{_CT_NS}}}Default")
        default.set("Extension", extension)
        default.set("ContentType", content_type)
    for part_name, content_type in sorted(overrides.items()):
        override = etree.SubElement(root, f"{{{_CT_NS}}}Override")
        override.set("PartName", part_name)
        override.set("ContentType", content_type)
    return serialize(root)


def build_root_rels(*, include_core_properties: bool = False) -> bytes:
    """Build `/_rels/.rels` referencing `/word/document.xml` and,
    optionally, `/docProps/core.xml` (RFC002 "Metadata").
    """
    rels = PartRelationships("/", seed=[(RELTYPE_OFFICE_DOCUMENT, "word/document.xml")])
    if include_core_properties:
        rels.add(RELTYPE_CORE_PROPERTIES, "docProps/core.xml")
    return rels.to_bytes()


def new_document_relationships() -> PartRelationships:
    """`word/_rels/document.xml.rels` seeded with the three mandatory
    R0/R1 parts, which always hold `rId1`-`rId3`. Everything else
    (numbering, media, header/footer/notes parts, hyperlinks) is added
    by the compiler in a fixed order.
    """
    return PartRelationships(
        "word/document.xml",
        seed=[
            (RELTYPE_STYLES, "styles.xml"),
            (RELTYPE_SETTINGS, "settings.xml"),
            (RELTYPE_FONT_TABLE, "fontTable.xml"),
        ],
    )


def build_styles() -> bytes:
    """Build minimal `/word/styles.xml`: Normal, Default Paragraph Font,
    TableNormal, NoList (RFC002 "Styles").
    """
    root = qn_root("styles")
    _append_style(root, style_id="Normal", style_type="paragraph", name="Normal", default=True)
    _append_style(
        root,
        style_id="DefaultParagraphFont",
        style_type="character",
        name="Default Paragraph Font",
        default=True,
    )
    _append_style(
        root, style_id="TableNormal", style_type="table", name="Normal Table", default=True
    )
    _append_style(root, style_id="NoList", style_type="numbering", name="No List", default=True)
    return serialize(root)


def qn_root(local: str) -> etree._Element:
    return etree.Element(qn("w", local), nsmap={"w": NSMAP["w"]})


def _append_style(
    root: etree._Element, *, style_id: str, style_type: str, name: str, default: bool
) -> None:
    style = etree.SubElement(root, qn("w", "style"))
    style.set(qn("w", "type"), style_type)
    style.set(qn("w", "styleId"), style_id)
    if default:
        style.set(qn("w", "default"), "1")
    name_el = etree.SubElement(style, qn("w", "name"))
    name_el.set(qn("w", "val"), name)


def build_settings(
    compatibility_mode: str,
    *,
    even_and_odd_headers: bool = False,
    update_fields: bool = False,
) -> bytes:
    """Build minimal `/word/settings.xml` (RFC002 §71).

    `compatibility_mode="office2013_plus"` maps to `w:compatSetting`
    value 15 (Word 2013+); `w:updateFields` is always explicit — false
    per the no-current-state determinism rule, true only when the
    document contains a `field` node, whose `w:fldSimple` deliberately
    carries no cached result and would otherwise render blank until the
    reader pressed F9.

    `even_and_odd_headers` is derived by the caller from the presence
    of an `even` header/footer story in any section — the AST stores no
    such boolean, exactly as `w:titlePg` is derived from a `first`
    story (DEC-003 §2.1).
    """
    if compatibility_mode != "office2013_plus":
        raise ValueError(f"build_settings: unsupported compatibility_mode {compatibility_mode!r}")
    root = qn_root("settings")
    # CT_Settings element order (wml.xsd xsd:sequence): evenAndOddHeaders
    # precedes updateFields, which precedes compat.
    if even_and_odd_headers:
        etree.SubElement(root, qn("w", "evenAndOddHeaders")).set(qn("w", "val"), "true")
    update_fields_el = etree.SubElement(root, qn("w", "updateFields"))
    update_fields_el.set(qn("w", "val"), "true" if update_fields else "false")
    compat = etree.SubElement(root, qn("w", "compat"))
    compat_setting = etree.SubElement(compat, qn("w", "compatSetting"))
    compat_setting.set(qn("w", "name"), "compatibilityMode")
    compat_setting.set(qn("w", "uri"), "http://schemas.microsoft.com/office/word")
    compat_setting.set(qn("w", "val"), "15")
    return serialize(root)


def has_mappable_core_properties(metadata: dict) -> bool:
    """`docProps/core.xml` is emitted only when the AST carries at least
    one of `title`/`subject`/`language` (RFC002 "Metadata" — conditional
    on mappable fields; no current-timestamp `created`/`modified` per
    the determinism rule).
    """
    return any(metadata.get(field) for field in ("title", "subject", "language"))


def build_core_properties(metadata: dict) -> bytes:
    """Build `/docProps/core.xml` from `metadata`.

    `dc:creator` is always the fixed literal `"unflatten"` (RFC002
    "Renderer may not ... add unobserved source semantics" — the
    renderer must not hallucinate a source author); `created`/
    `modified` are omitted entirely when absent from the AST (never a
    wall-clock timestamp).
    """
    root = etree.Element(
        f"{{{_CP_NS}}}coreProperties",
        nsmap={"cp": _CP_NS, "dc": _DC_NS},
    )
    # OPC core-properties schema order (CT_CoreProperties): creator,
    # language, subject, title, among the fields V0 maps.
    etree.SubElement(root, f"{{{_DC_NS}}}creator").text = _CORE_PROPERTIES_CREATOR
    if metadata.get("language"):
        etree.SubElement(root, f"{{{_DC_NS}}}language").text = metadata["language"]
    if metadata.get("subject"):
        etree.SubElement(root, f"{{{_DC_NS}}}subject").text = metadata["subject"]
    if metadata.get("title"):
        etree.SubElement(root, f"{{{_DC_NS}}}title").text = metadata["title"]
    return serialize(root)


def build_font_table(font_names: list[str]) -> bytes:
    """Build `/word/fontTable.xml` listing `font_names`, stably sorted."""
    root = qn_root("fonts")
    for name in sorted(set(font_names)):
        font = etree.SubElement(root, qn("w", "font"))
        font.set(qn("w", "name"), name)
    return serialize(root)
