"""`word/numbering.xml` builder + `NumberingRegistry` (RFC002 "Lists").

Each Canonical AST `list` node carries a *complete* marker/layout
definition (it is not a level within a shared multi-level numbering
definition) — nested lists are separate `list` AST nodes, each with
their own explicit `layout.left_indent_mpt`/`hanging_mpt`. V0 therefore
maps every `list` AST node to its own single-level (`ilvl=0`)
`w:abstractNum`/`w:num` pair rather than building 9-level abstractNum
definitions and varying `w:ilvl` by nesting depth: the AST does not
carry the notion of "same numbering sequence, deeper level", and
inventing one would be unobserved source semantics (RFC002 "Renderer
may not ... add unobserved source semantics").

Deduplication: `abstractNumId` is assigned per unique canonical
definition (identical marker+layout share one `w:abstractNum`) so two
lists with the same style don't duplicate the definition. `numId` is
assigned per `list` AST node (never deduplicated across nodes), because
sharing a `numId` would make Word continue the running counter across
two logically independent lists instead of restarting each at its own
`start` value — a visible-content bug, not just a technical-ID
optimization. Ordered instances also carry an explicit `startOverride`:
LibreOffice continues counters across distinct numIds sharing an
abstractNum unless the instance explicitly restarts its level.
Both ID spaces are assigned from a stable sort key over
the canonical AST (never insertion/traversal order): `abstractNumId`
by sorting the set of distinct definition keys, `numId` by sorting
`(abstractNumId, list_node_id)` pairs — `list_node_id` is already a
stable canonical `n<k>` id post-canonicalization renumbering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lxml import etree

from unflatten.renderer.units import mpt_to_twip
from unflatten.renderer.xml_writer import NSMAP, qn, serialize

_ORDERED_FORMAT_MAP: dict[str, str] = {
    "decimal": "decimal",
    "lower_alpha": "lowerLetter",
    "upper_alpha": "upperLetter",
    "lower_roman": "lowerRoman",
    "upper_roman": "upperRoman",
}

# Standard Word bullet-gallery glyph+font pairs (RFC002 "pick the
# standard deterministic trio and document it"): `bullet` is the
# Symbol-font private-use codepoint Word itself emits for a solid dot,
# `circle` is a plain "o" (no special font needed), `square` is the
# Wingdings private-use codepoint for a solid square.
_UNORDERED_GLYPH: dict[str, tuple[str, str]] = {
    "bullet": ("", "Symbol"),
    "circle": ("o", "Courier New"),
    "square": ("", "Wingdings"),
}

_ORDERED_FORMATS = frozenset(_ORDERED_FORMAT_MAP)
_UNORDERED_GLYPH_FORMATS = frozenset(_UNORDERED_GLYPH)

# Canonical definition key:
# (list_kind, format, text, suffix, start, left_indent_mpt, hanging_mpt).
NumberingKey = tuple[str, str, str, str, int, int, int]


@dataclass(frozen=True)
class NumberingDefinition:
    key: NumberingKey
    list_kind: str
    marker_format: str
    text: str
    suffix: str
    start: int
    left_indent_mpt: int
    hanging_mpt: int


@dataclass
class NumberingRegistry:
    """Deterministic `abstractNumId`/`numId` allocator for `list` nodes."""

    abstract_num_id_by_key: dict[NumberingKey, int] = field(default_factory=dict)
    definitions_by_abstract_id: dict[int, NumberingDefinition] = field(default_factory=dict)
    num_id_by_node_id: dict[str, int] = field(default_factory=dict)
    abstract_id_by_num_id: dict[int, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.num_id_by_node_id


def _list_key(node: dict[str, Any]) -> NumberingKey:
    marker = node["marker"]
    layout = node["layout"]
    marker_format = marker["format"]
    text = marker.get("text") or ""
    suffix = marker.get("suffix") or ""
    start = marker.get("start") if marker.get("start") is not None else 1
    return (
        node["list_kind"],
        marker_format,
        text,
        suffix,
        int(start),
        layout["left_indent_mpt"],
        layout["hanging_mpt"],
    )


def collect_list_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect every `list` AST node under `nodes`, in preorder.

    Walks the same block/inline containers the document builder walks
    (section children, list items, table cells) so nested lists inside
    list items or table cells are found too.
    """
    found: list[dict[str, Any]] = []
    for node in nodes:
        node_type = node.get("type")
        if node_type == "list":
            found.append(node)
            for item in node.get("items", []):
                found.extend(collect_list_nodes(item.get("children", [])))
        elif node_type == "table":
            for row in node.get("rows", []):
                for cell in row.get("cells", []):
                    found.extend(collect_list_nodes(cell.get("children", [])))
        # paragraph/page_break/column_break carry no nested block_node children.
    return found


def build_numbering_registry(nodes: list[dict[str, Any]]) -> NumberingRegistry:
    """Build a `NumberingRegistry` for every `list` node reachable from `nodes`."""
    list_nodes = collect_list_nodes(nodes)
    registry = NumberingRegistry()

    distinct_keys = sorted({_list_key(n) for n in list_nodes})
    for abstract_id, key in enumerate(distinct_keys):
        registry.abstract_num_id_by_key[key] = abstract_id
        registry.definitions_by_abstract_id[abstract_id] = NumberingDefinition(key, *key)

    num_id_entries = sorted(
        (registry.abstract_num_id_by_key[_list_key(n)], n["id"]) for n in list_nodes
    )
    for num_id, (abstract_id, _node_id) in enumerate(num_id_entries, start=1):
        registry.abstract_id_by_num_id[num_id] = abstract_id

    for num_id, (_abstract_id, node_id) in enumerate(num_id_entries, start=1):
        registry.num_id_by_node_id[node_id] = num_id

    return registry


def _level_text(definition: NumberingDefinition) -> tuple[str, str]:
    """Return `(w:lvlText, w:numFmt)` for `definition`."""
    fmt = definition.marker_format
    if fmt in _ORDERED_FORMATS:
        return f"%1{definition.suffix}", _ORDERED_FORMAT_MAP[fmt]
    if fmt in _UNORDERED_GLYPH_FORMATS:
        glyph, _font = _UNORDERED_GLYPH[fmt]
        return glyph, "bullet"
    # custom_text (RFC001 ERRATA-002 E6): `marker.text` carries the
    # literal marker text; `suffix` keeps its trailing-punctuation role,
    # appended exactly as for ordered formats.
    return f"{definition.text}{definition.suffix}", "bullet"


def build_numbering_xml(registry: NumberingRegistry) -> bytes:
    """Build `/word/numbering.xml` for every definition in `registry`."""
    root = etree.Element(qn("w", "numbering"), nsmap={"w": NSMAP["w"]})
    for abstract_id in sorted(registry.definitions_by_abstract_id):
        _append_abstract_num(root, abstract_id, registry.definitions_by_abstract_id[abstract_id])
    for num_id in sorted(registry.abstract_id_by_num_id):
        num = etree.SubElement(root, qn("w", "num"))
        num.set(qn("w", "numId"), str(num_id))
        abstract_num_id = etree.SubElement(num, qn("w", "abstractNumId"))
        abstract_id = registry.abstract_id_by_num_id[num_id]
        abstract_num_id.set(qn("w", "val"), str(abstract_id))
        definition = registry.definitions_by_abstract_id[abstract_id]
        if definition.list_kind == "ordered":
            override = etree.SubElement(num, qn("w", "lvlOverride"))
            override.set(qn("w", "ilvl"), "0")
            start = etree.SubElement(override, qn("w", "startOverride"))
            start.set(qn("w", "val"), str(definition.start))
    return serialize(root)


def _append_abstract_num(
    root: etree._Element, abstract_id: int, definition: NumberingDefinition
) -> None:
    abstract_num = etree.SubElement(root, qn("w", "abstractNum"))
    abstract_num.set(qn("w", "abstractNumId"), str(abstract_id))
    lvl = etree.SubElement(abstract_num, qn("w", "lvl"))
    lvl.set(qn("w", "ilvl"), "0")
    start = etree.SubElement(lvl, qn("w", "start"))
    start.set(qn("w", "val"), str(definition.start if definition.list_kind == "ordered" else 1))
    lvl_text, num_fmt = _level_text(definition)
    num_fmt_el = etree.SubElement(lvl, qn("w", "numFmt"))
    num_fmt_el.set(qn("w", "val"), num_fmt)
    # CT_Lvl element order (wml.xsd xsd:sequence): start, numFmt, suff,
    # lvlText, ..., pPr, rPr.
    suff = etree.SubElement(lvl, qn("w", "suff"))
    suff.set(qn("w", "val"), "tab")
    lvl_text_el = etree.SubElement(lvl, qn("w", "lvlText"))
    lvl_text_el.set(qn("w", "val"), lvl_text)
    p_pr = etree.SubElement(lvl, qn("w", "pPr"))
    ind = etree.SubElement(p_pr, qn("w", "ind"))
    ind.set(qn("w", "left"), str(mpt_to_twip(definition.left_indent_mpt)))
    ind.set(qn("w", "hanging"), str(mpt_to_twip(definition.hanging_mpt)))
    if definition.marker_format in _UNORDERED_GLYPH_FORMATS:
        _, font = _UNORDERED_GLYPH[definition.marker_format]
        r_pr = etree.SubElement(lvl, qn("w", "rPr"))
        r_fonts = etree.SubElement(r_pr, qn("w", "rFonts"))
        r_fonts.set(qn("w", "ascii"), font)
        r_fonts.set(qn("w", "hAnsi"), font)
        r_fonts.set(qn("w", "hint"), "default")
