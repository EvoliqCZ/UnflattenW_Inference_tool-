"""Text conservation validator (RFC002 "Text conservation acceptance").

Extracts a visible-text event stream independently from the source
Canonical AST and from the compiled package, then compares them. A
mismatch means the renderer altered, reordered, or dropped visible
content — the acceptance criterion for V0 is 100% conservation after
normalization (NFC + a fixed control-token representation), and
normalization must never mask a real content difference.

`word-visual-v1` moves visible text out of `document.xml` and into
`header{n}.xml`, `footer{n}.xml`, `footnotes.xml` and `endnotes.xml`,
so the comparison spans every one of those parts. An empty header that
opened in Word but rendered nothing would otherwise pass silently.
"""

from __future__ import annotations

import unicodedata
import zipfile
from collections.abc import Iterable
from io import BytesIO
from typing import Any

from lxml import etree

from unflatten.ast.issues import Issue
from unflatten.renderer.ooxml.context import DOCUMENT_PART
from unflatten.renderer.ooxml.story_builder import plan_notes_parts, plan_story_parts
from unflatten.renderer.xml_writer import NSMAP

_W_NS = NSMAP["w"]

#: Child-index path of `w:body` inside `w:document`. Story and notes
#: parts have no such wrapper: their blocks hang off the root.
_BODY_PATH: tuple[int, ...] = (0,)


def _ast_events(ast: dict[str, Any]) -> list[str]:
    events: list[str] = []
    for section in ast.get("sections", []):
        for child in section.get("children", []):
            events.extend(_ast_block_events(child))
    return events


def _ast_part_events(ast: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Per-part event streams for the header/footer/notes parts, in the
    same order `story_builder` names them.
    """
    streams: list[tuple[str, list[str]]] = []
    for story in plan_story_parts(ast.get("sections", [])):
        events: list[str] = []
        for child in story.node.get("children", []):
            events.extend(_ast_block_events(child))
        streams.append((story.part_name, events))
    for notes in plan_notes_parts(ast):
        events = []
        for note in notes.notes:
            for child in note.get("children", []):
                events.extend(_ast_block_events(child))
        streams.append((notes.part_name, events))
    return streams


def _ast_block_events(node: dict[str, Any]) -> list[str]:
    node_type = node.get("type")
    if node_type == "paragraph":
        events: list[str] = []
        for inline in node.get("children", []):
            events.extend(_ast_inline_events(inline))
        events.append("PARAGRAPH_END")
        return events
    if node_type == "page_break":
        return ["PAGE_BREAK_EXPLICIT"]
    if node_type == "column_break":
        return ["COLUMN_BREAK_EXPLICIT"]
    if node_type == "list":
        events = []
        for item in node.get("items", []):
            for child in item.get("children", []):
                events.extend(_ast_block_events(child))
        return events
    if node_type == "table":
        events = []
        for row in node.get("rows", []):
            for cell in row.get("cells", []):
                for child in cell.get("children", []):
                    events.extend(_ast_block_events(child))
                events.append("CELL_END")
            events.append("ROW_END")
        return events
    return []


def _ast_inline_events(node: dict[str, Any]) -> list[str]:
    node_type = node.get("type")
    if node_type == "run":
        text = unicodedata.normalize("NFC", node["text"])
        return [f"TEXT:{text}"] if text else []
    if node_type == "tab":
        return ["TAB"]
    if node_type == "line_break":
        return ["LINE_BREAK"]
    if node_type == "note_ref":
        # The mark is visible text on the page and is emitted as an
        # explicit custom mark, so it conserves like run text.
        return [f"TEXT:{unicodedata.normalize('NFC', node['marker_text'])}"]
    if node_type == "hyperlink":
        events: list[str] = []
        for child in node.get("children", []):
            events.extend(_ast_inline_events(child))
        return events
    # `inline_image`, `floating_image` and `field` contribute no text:
    # an image has none, and a field deliberately carries no cached
    # result (the page number differs on every page).
    return []


def _docx_part_events(
    docx_bytes: bytes,
    technical_paths_by_part: dict[str, list[tuple[int, ...]]],
    part_names: list[str] | None = None,
) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """`part_names=None` walks the story parts by `_story_part_names`'s
    contiguous `header1..N`/`footer1..N` convention (the renderer's own
    naming). Pass an explicit `part_names` list to bypass that
    convention — e.g. a source package whose story parts are not
    contiguous (`header3.xml` with no `header1.xml`), where the default
    walk would find nothing.
    """
    with zipfile.ZipFile(BytesIO(docx_bytes)) as zf:
        names = set(zf.namelist())
        root = etree.fromstring(zf.read(DOCUMENT_PART))
        body = root.find(f"{{{_W_NS}}}body")
        body_events = _docx_container_events(
            body, _BODY_PATH, frozenset(technical_paths_by_part.get(DOCUMENT_PART, ()))
        )
        streams: list[tuple[str, list[str]]] = []
        for part_name in _story_part_names(names) if part_names is None else part_names:
            technical = frozenset(technical_paths_by_part.get(part_name, ()))
            part_root = etree.fromstring(zf.read(part_name))
            if part_name.endswith(("footnotes.xml", "endnotes.xml")):
                events: list[str] = []
                for index, note in enumerate(part_root):
                    events.extend(_docx_container_events(note, (index,), technical))
                streams.append((part_name, events))
            else:
                streams.append((part_name, _docx_container_events(part_root, (), technical)))
    return body_events, streams


def _story_part_names(names: set[str]) -> list[str]:
    """Header/footer/notes part names in `story_builder`'s order:
    `header1..N`, then `footer1..N`, then footnotes, then endnotes.
    """
    ordered: list[str] = []
    for prefix in ("header", "footer"):
        index = 1
        while f"word/{prefix}{index}.xml" in names:
            ordered.append(f"word/{prefix}{index}.xml")
            index += 1
    ordered.extend(n for n in ("word/footnotes.xml", "word/endnotes.xml") if n in names)
    return ordered


def _docx_container_events(
    container: etree._Element,
    base_path: tuple[int, ...],
    technical_paragraph_paths: frozenset[tuple[int, ...]] = frozenset(),
) -> list[str]:
    """Walk `container`'s direct children in document order — `w:p` and
    `w:tbl` at body/story level, or those inside a `w:tc`. `w:sectPr` is
    skipped: it carries no visible content.

    `technical_paragraph_paths` is the *explicit* set of child-index
    paths (from the part's root element) of paragraphs the renderer
    synthesized with no AST node behind them: section-boundary
    paragraphs, the paragraph that closes a cell ending in a nested
    table, the single paragraph an empty story needs, and the
    footnote separators. It is supplied by the renderer
    (`BuildContext.technical_paragraph_paths`), never inferred from XML
    shape here — shape inference ("empty `w:p`") would also match a
    genuinely empty AST paragraph and silently drop its real
    `PARAGRAPH_END`.
    """
    events: list[str] = []
    for index, child in enumerate(container):
        path = (*base_path, index)
        tag = etree.QName(child).localname
        if tag == "p":
            if path in technical_paragraph_paths:
                continue
            events.extend(_docx_paragraph_events(child))
        elif tag == "tbl":
            events.extend(_docx_table_events(child, path, technical_paragraph_paths))
    return events


def _docx_table_events(
    tbl: etree._Element,
    base_path: tuple[int, ...],
    technical_paragraph_paths: frozenset[tuple[int, ...]],
) -> list[str]:
    events: list[str] = []
    for row_index, child in enumerate(tbl):
        if etree.QName(child).localname != "tr":
            continue
        row_path = (*base_path, row_index)
        for cell_index, cell in enumerate(child):
            if etree.QName(cell).localname != "tc":
                continue
            if _is_technical_cell(cell):
                continue
            events.extend(
                _docx_container_events(
                    cell, (*row_path, cell_index), technical_paragraph_paths
                )
            )
            events.append("CELL_END")
        events.append("ROW_END")
    return events


def _is_technical_cell(tc: etree._Element) -> bool:
    """A synthesized `w:vMerge` continuation cell (RFC002 "Spans") — no
    source AST node, contributes zero text-conservation events.
    """
    tc_pr = tc.find(f"{{{_W_NS}}}tcPr")
    if tc_pr is None:
        return False
    v_merge = tc_pr.find(f"{{{_W_NS}}}vMerge")
    return v_merge is not None and v_merge.get(f"{{{_W_NS}}}val") == "continue"


def _docx_paragraph_events(p: etree._Element) -> list[str]:
    para_events: list[str] = []
    for child in p:
        tag = etree.QName(child).localname
        if tag == "r":
            para_events.extend(_docx_run_events(child))
        elif tag == "hyperlink":
            for run in child.findall(f"{{{_W_NS}}}r"):
                para_events.extend(_docx_run_events(run))
    if para_events in (["PAGE_BREAK_EXPLICIT"], ["COLUMN_BREAK_EXPLICIT"]):
        return para_events
    return [*para_events, "PARAGRAPH_END"]


def _docx_run_events(r: etree._Element) -> list[str]:
    events: list[str] = []
    for child in r:
        tag = etree.QName(child).localname
        if tag == "t":
            text = unicodedata.normalize("NFC", child.text or "")
            if text:
                events.append(f"TEXT:{text}")
        elif tag == "tab":
            events.append("TAB")
        elif tag == "br":
            break_type = child.get(f"{{{_W_NS}}}type")
            if break_type == "page":
                events.append("PAGE_BREAK_EXPLICIT")
            elif break_type == "column":
                events.append("COLUMN_BREAK_EXPLICIT")
            else:
                events.append("LINE_BREAK")
    return events


def verify_text_conservation(
    ast: dict[str, Any],
    docx_bytes: bytes,
    technical_paragraph_positions: Iterable[int] | None = None,
    technical_paragraph_paths: dict[str, Iterable[tuple[int, ...]]] | None = None,
) -> list[Issue]:
    """Compare the AST-derived and package-derived text event streams;
    return `[]` if identical, else one diagnostic Issue per part that
    differs.

    Both technical-paragraph arguments come from `RendererResult
    .metrics`. `technical_paragraph_positions` is the legacy
    body-level form (`w:body` child indices); `technical_paragraph_paths`
    is the per-part form the story parts need. Pass whichever the
    caller has — the two are merged, so an older caller keeps working.
    """
    paths_by_part: dict[str, list[tuple[int, ...]]] = {
        part: [tuple(path) for path in paths]
        for part, paths in (technical_paragraph_paths or {}).items()
    }
    body_paths = paths_by_part.setdefault(DOCUMENT_PART, [])
    body_paths.extend((*_BODY_PATH, index) for index in technical_paragraph_positions or ())

    expected_body = _ast_events(ast)
    actual_body, actual_streams = _docx_part_events(docx_bytes, paths_by_part)

    issues: list[Issue] = []
    if expected_body != actual_body:
        issues.append(_mismatch(DOCUMENT_PART, expected_body, actual_body))

    expected_streams = dict(_ast_part_events(ast))
    for part_name, actual in actual_streams:
        expected = expected_streams.pop(part_name, None)
        if expected != actual:
            issues.append(_mismatch(part_name, expected or [], actual))
    for part_name, expected in expected_streams.items():
        issues.append(_mismatch(part_name, expected, []))
    return issues


def _mismatch(part_name: str, expected: list[str], actual: list[str]) -> Issue:
    return Issue(
        code="TEXT_CONSERVATION_MISMATCH",
        severity="error",
        node_id=None,
        path=f"/{part_name}",
        message=f"visible text event stream differs between AST and compiled {part_name}.",
        details={"part": part_name, "expected": expected, "actual": actual},
    )
