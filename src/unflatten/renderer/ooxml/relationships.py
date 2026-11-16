"""Per-part OPC relationships (RFC002 "Package structure").

A relationship id is scoped to the *part that declares it*: `rId4` in
`word/_rels/header1.xml.rels` and `rId4` in
`word/_rels/document.xml.rels` name entirely different targets. Before
`word-visual-v1` every image lived in `document.xml`, so a single
document-wide sha256 -> rel-id map happened to be correct; a logo in a
header breaks that assumption (24.7 % of the corpus has a header part
whose own rels reference an image, DEC-003 §1.2). This module therefore
allocates ids per part and there is deliberately no document-wide id
map.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lxml import etree

from unflatten.renderer.xml_writer import serialize

_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_DOCUMENT = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

RELTYPE_OFFICE_DOCUMENT = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
)
RELTYPE_CORE_PROPERTIES = (
    "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties"
)
RELTYPE_STYLES = f"{_OFFICE_DOCUMENT}/styles"
RELTYPE_SETTINGS = f"{_OFFICE_DOCUMENT}/settings"
RELTYPE_FONT_TABLE = f"{_OFFICE_DOCUMENT}/fontTable"
RELTYPE_NUMBERING = f"{_OFFICE_DOCUMENT}/numbering"
RELTYPE_IMAGE = f"{_OFFICE_DOCUMENT}/image"
RELTYPE_HEADER = f"{_OFFICE_DOCUMENT}/header"
RELTYPE_FOOTER = f"{_OFFICE_DOCUMENT}/footer"
RELTYPE_FOOTNOTES = f"{_OFFICE_DOCUMENT}/footnotes"
RELTYPE_ENDNOTES = f"{_OFFICE_DOCUMENT}/endnotes"
RELTYPE_HYPERLINK = f"{_OFFICE_DOCUMENT}/hyperlink"


@dataclass(frozen=True)
class Relationship:
    rel_id: str
    rel_type: str
    target: str
    external: bool


class PartRelationships:
    """The relationship table of exactly one package part.

    Ids are allocated `rId1`, `rId2`, ... in the order `add` is called,
    so callers are responsible for calling in a deterministic order.
    Re-adding an identical `(rel_type, target, external)` triple returns
    the id already assigned rather than a second, duplicate entry —
    two hyperlinks to the same URI, or two images with the same
    sha256, share one relationship exactly as Word writes them.
    """

    def __init__(self, part_name: str, seed: Sequence[tuple[str, str]] = ()) -> None:
        self.part_name = part_name
        self._entries: list[Relationship] = []
        self._by_key: dict[tuple[str, str, bool], str] = {}
        for rel_type, target in seed:
            self.add(rel_type, target)

    def add(self, rel_type: str, target: str, *, external: bool = False) -> str:
        key = (rel_type, target, external)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        rel_id = f"rId{len(self._entries) + 1}"
        self._entries.append(Relationship(rel_id, rel_type, target, external))
        self._by_key[key] = rel_id
        return rel_id

    @property
    def entries(self) -> list[Relationship]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def rels_part_name(self) -> str:
        """Package name of the `.rels` part describing `part_name`."""
        head, _, tail = self.part_name.rpartition("/")
        prefix = f"{head}/" if head else ""
        return f"{prefix}_rels/{tail}.rels"

    def to_bytes(self) -> bytes:
        root = etree.Element(f"{{{_REL_NS}}}Relationships", nsmap={None: _REL_NS})
        for entry in self._entries:
            rel = etree.SubElement(root, f"{{{_REL_NS}}}Relationship")
            rel.set("Id", entry.rel_id)
            rel.set("Type", entry.rel_type)
            rel.set("Target", entry.target)
            if entry.external:
                rel.set("TargetMode", "External")
        return serialize(root)
