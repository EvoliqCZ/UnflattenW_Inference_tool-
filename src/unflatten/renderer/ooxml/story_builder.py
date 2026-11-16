"""Header/footer and footnote/endnote parts (`word-visual-v1`, DEC-003).

A story is a *container*, never a text property: over 1 500 originals
51 % of content-bearing headers carry an image and 17 % a table
(DEC-003 §1.2). Every story is therefore rendered through
`document_builder.append_block`, the same mapping the body uses.

Each story becomes its own package part (`word/header1.xml`,
`word/footnotes.xml`, ...) with its own `_rels`, its own
`[Content_Types].xml` override, and — the part that bites — its own
relationship id space. Part numbering is fixed here (`plan_story_parts`
emits all headers, then all footers, each in section order) so that
part names never depend on dict iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lxml import etree

from unflatten.ast.issues import Issue
from unflatten.renderer.api import FontPolicy
from unflatten.renderer.ooxml.context import BuildContext
from unflatten.renderer.ooxml.document_builder import append_block, part_prefixes
from unflatten.renderer.ooxml.package_parts import (
    CONTENT_TYPE_ENDNOTES,
    CONTENT_TYPE_FOOTER,
    CONTENT_TYPE_FOOTNOTES,
    CONTENT_TYPE_HEADER,
)
from unflatten.renderer.ooxml.relationships import (
    RELTYPE_ENDNOTES,
    RELTYPE_FOOTER,
    RELTYPE_FOOTNOTES,
    RELTYPE_HEADER,
)
from unflatten.renderer.xml_writer import make_root, qn

#: ECMA-376 reserves `w:id` -1 and 0 of `footnotes.xml`/`endnotes.xml`
#: for the separator and continuation-separator notes; real notes start
#: at 1.
SEPARATOR_ID = -1
CONTINUATION_SEPARATOR_ID = 0
FIRST_NOTE_ID = 1


@dataclass(frozen=True)
class StoryPart:
    """One `word/header{n}.xml` or `word/footer{n}.xml`."""

    part_name: str
    root_local: str  # "hdr" | "ftr"
    reference_local: str  # "headerReference" | "footerReference"
    rel_type: str
    content_type: str
    section_index: int
    kind: str  # "default" | "first" | "even"
    node: dict[str, Any]

    @property
    def target(self) -> str:
        """Relationship target, relative to `word/_rels/document.xml.rels`."""
        return self.part_name.removeprefix("word/")


@dataclass(frozen=True)
class NotesPart:
    """`word/footnotes.xml` or `word/endnotes.xml`."""

    part_name: str
    root_local: str  # "footnotes" | "endnotes"
    item_local: str  # "footnote" | "endnote"
    rel_type: str
    content_type: str
    notes: list[dict[str, Any]]

    @property
    def target(self) -> str:
        return self.part_name.removeprefix("word/")


def plan_story_parts(sections: list[dict[str, Any]]) -> list[StoryPart]:
    """Assign part names to every header/footer story, deterministically.

    All headers are numbered first (`header1.xml`, `header2.xml`, ...)
    in section order and, within a section, in the canonical
    `default, first, even` order the canonicalizer already imposed;
    then all footers the same way.
    """
    parts: list[StoryPart] = []
    for key, root_local, reference_local, rel_type, content_type, prefix in (
        ("headers", "hdr", "headerReference", RELTYPE_HEADER, CONTENT_TYPE_HEADER, "header"),
        ("footers", "ftr", "footerReference", RELTYPE_FOOTER, CONTENT_TYPE_FOOTER, "footer"),
    ):
        index = 0
        for section_index, section in enumerate(sections):
            for story in section.get(key) or []:
                index += 1
                parts.append(
                    StoryPart(
                        part_name=f"word/{prefix}{index}.xml",
                        root_local=root_local,
                        reference_local=reference_local,
                        rel_type=rel_type,
                        content_type=content_type,
                        section_index=section_index,
                        kind=story["kind"],
                        node=story,
                    )
                )
    return parts


def plan_notes_parts(document: dict[str, Any]) -> list[NotesPart]:
    """`footnotes.xml` before `endnotes.xml`; a part is planned only
    when the document actually carries notes of that kind.
    """
    parts: list[NotesPart] = []
    for key, item_local, rel_type, content_type in (
        ("footnotes", "footnote", RELTYPE_FOOTNOTES, CONTENT_TYPE_FOOTNOTES),
        ("endnotes", "endnote", RELTYPE_ENDNOTES, CONTENT_TYPE_ENDNOTES),
    ):
        notes = document.get(key) or []
        if not notes:
            continue
        parts.append(
            NotesPart(
                part_name=f"word/{key}.xml",
                root_local=key,
                item_local=item_local,
                rel_type=rel_type,
                content_type=content_type,
                notes=list(notes),
            )
        )
    return parts


def build_story_root(
    story: StoryPart, font_policy: FontPolicy, used_fonts: set[str], ctx: BuildContext
) -> tuple[etree._Element, list[Issue]]:
    """Build the `w:hdr`/`w:ftr` root for `story` (ECMA-376 CT_HdrFtr)."""
    node = story.node
    children = node.get("children", [])
    root = make_root("w", story.root_local, used_prefixes=part_prefixes(children))
    if node.get("id") is not None:
        ctx.rendered_node_ids.add(node["id"])
    path = f"/sections/{story.section_index}/{story.reference_local}"
    issues = _append_story_blocks(root, children, path, font_policy, used_fonts, ctx)
    return root, issues


def build_notes_root(
    notes_part: NotesPart, font_policy: FontPolicy, used_fonts: set[str], ctx: BuildContext
) -> tuple[etree._Element, list[Issue]]:
    """Build the `w:footnotes`/`w:endnotes` root (ECMA-376 CT_Footnotes).

    The separator and continuation-separator notes are emitted first:
    without them Word and LibreOffice draw no rule above the note area.
    They carry no AST node, so their paragraphs are marked technical.
    """
    blocks = [child for note in notes_part.notes for child in note.get("children", [])]
    root = make_root("w", notes_part.root_local, used_prefixes=part_prefixes(blocks))
    issues: list[Issue] = []
    for numeric_id, separator_local in (
        (SEPARATOR_ID, "separator"),
        (CONTINUATION_SEPARATOR_ID, "continuationSeparator"),
    ):
        _append_separator_note(root, notes_part.item_local, numeric_id, separator_local, ctx)
    for index, note in enumerate(notes_part.notes):
        note_el = etree.SubElement(root, qn("w", notes_part.item_local))
        note_el.set(qn("w", "id"), str(ctx.note_numeric_ids[note["id"]]))
        ctx.rendered_node_ids.add(note["id"])
        issues.extend(
            _append_story_blocks(
                note_el,
                note.get("children", []),
                f"/{notes_part.root_local}/{index}",
                font_policy,
                used_fonts,
                ctx,
            )
        )
    return root, issues


def _append_separator_note(
    root: etree._Element, item_local: str, numeric_id: int, separator_local: str, ctx: BuildContext
) -> None:
    note = etree.SubElement(root, qn("w", item_local))
    note.set(qn("w", "type"), separator_local)
    note.set(qn("w", "id"), str(numeric_id))
    paragraph = etree.SubElement(note, qn("w", "p"))
    run = etree.SubElement(paragraph, qn("w", "r"))
    etree.SubElement(run, qn("w", separator_local))
    ctx.mark_technical_paragraph(paragraph)


def _append_story_blocks(
    parent: etree._Element,
    children: list[dict[str, Any]],
    path: str,
    font_policy: FontPolicy,
    used_fonts: set[str],
    ctx: BuildContext,
) -> list[Issue]:
    """Render a story body, guaranteeing the one block CT_HdrFtr and
    CT_FtnEdn both require (`EG_BlockLevelElts minOccurs="1"`).

    An empty story is legal in the AST and means "this section has this
    story, and it is blank"; the synthesized paragraph carries no AST
    node and is recorded as technical so text conservation does not
    count it as content.
    """
    issues: list[Issue] = []
    for index, child in enumerate(children):
        child_issues, _fatal = append_block(
            parent, child, f"{path}/children/{index}", font_policy, used_fonts, ctx
        )
        issues.extend(child_issues)
    if len(parent) == 0:
        ctx.mark_technical_paragraph(etree.SubElement(parent, qn("w", "p")))
    return issues
