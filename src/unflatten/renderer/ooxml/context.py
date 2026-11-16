"""Cross-cutting build state threaded through the R2/R3/R4 dispatch
branches of `document_builder` (lists, tables, inline images) and, from
`word-visual-v1` on, through the header/footer/footnote story builders
too.

Built once per compile, before `document.xml` generation starts,
because numbering ids, image relationship ids, `wp:docPr` ids,
hyperlink relationship ids and footnote/endnote numeric ids all require
a whole-document pre-scan (RFC002 stage S6 "build registries").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lxml import etree

from unflatten.renderer.api import FontPolicy
from unflatten.renderer.ooxml.drawing_builder import ImageRef
from unflatten.renderer.ooxml.numbering import NumberingRegistry

DOCUMENT_PART = "word/document.xml"


@dataclass
class BuildContext:
    font_policy: FontPolicy
    numbering: NumberingRegistry = field(default_factory=NumberingRegistry)
    images: dict[str, ImageRef] = field(default_factory=dict)
    #: text_frame node id -> package-global `wp:docPr/@id`. Text frames
    #: carry no resource, but their anchored DrawingML shape still shares
    #: the same non-visual drawing id space as pictures.
    text_frame_docpr_ids: dict[str, int] = field(default_factory=dict)
    #: text_frame node id -> the usable width (page minus margins, EMU)
    #: of the section/story it was found in. Only consulted for
    #: alignment-positioned anchors (`drawing_builder.append_text_frame_
    #: drawing`) — see that function's docstring for why width matters
    #: there and nowhere else.
    text_frame_width_emu: dict[str, int] = field(default_factory=dict)
    #: hyperlink node id -> `r:id` of its external relationship in the
    #: part it lives in. An `internal_anchor` hyperlink has no entry:
    #: it renders as `w:anchor`, which needs no relationship at all.
    hyperlink_rel_ids: dict[str, str] = field(default_factory=dict)
    #: footnote/endnote AST node id -> the `w:id` it gets in
    #: `footnotes.xml` / `endnotes.xml`.
    note_numeric_ids: dict[str, int] = field(default_factory=dict)
    #: footnote/endnote AST node id -> "footnote" | "endnote", so a
    #: `note_ref` knows which of the two reference elements to emit.
    note_kinds: dict[str, str] = field(default_factory=dict)
    #: section index -> ordered `(element_local_name, w:type, r:id)`
    #: triples for that section's `w:headerReference`/`w:footerReference`.
    story_refs_by_section: dict[int, list[tuple[str, str, str]]] = field(default_factory=dict)
    #: node ids that are the target of an `internal_anchor` hyperlink
    #: and therefore need a `w:bookmarkStart`/`w:bookmarkEnd` pair.
    bookmark_targets: dict[str, int] = field(default_factory=dict)
    technical_cells: int = 0
    technical_paragraphs: int = 0
    rendered_node_ids: set[str] = field(default_factory=set)
    technical_boundary_paragraph_positions: set[int] = field(default_factory=set)
    #: Package part currently being built; every `mark_technical_paragraph`
    #: is recorded against it.
    current_part: str = DOCUMENT_PART
    #: part name -> child-index paths (from that part's root element) of
    #: every synthesized paragraph that carries no AST node. Explicit,
    #: never re-derived from XML shape by the text-conservation checker
    #: — "empty `w:p`" also describes a genuinely empty AST paragraph.
    technical_paragraph_paths: dict[str, set[tuple[int, ...]]] = field(default_factory=dict)

    def mark_technical_paragraph(self, paragraph: etree._Element) -> None:
        """Record `paragraph` as synthesized in the part being built."""
        self.technical_paragraphs += 1
        self.technical_paragraph_paths.setdefault(self.current_part, set()).add(
            _index_path(paragraph)
        )


def _index_path(element: etree._Element) -> tuple[int, ...]:
    """Child-index path from the part's root element down to `element`.

    Valid only while the tree is append-only below `element`'s
    ancestors, which is how every builder here works: children are
    appended, never inserted before an already-recorded paragraph.
    """
    path: list[int] = []
    node = element
    parent = node.getparent()
    while parent is not None:
        path.append(parent.index(node))
        node = parent
        parent = node.getparent()
    return tuple(reversed(path))
