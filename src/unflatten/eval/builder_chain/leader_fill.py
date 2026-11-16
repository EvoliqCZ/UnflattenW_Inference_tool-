"""Opt-in chain policy `leader-fill.v0` (RATIFIED 2026-10-05, RFC004-ERRATA-010, DEC-059).

DEC-036 canonicalizes every form fill-leader run (>= 4 identical `_`, `.`,
`-` or `…` in one text range) to exactly 4 characters at cache build: the
model cannot count thousands of identical tokens and loops (DEC-036). The
model therefore emits `Date: ____` for a page that shows a writing line
running to the margin, and a Word document built from it has a 4-character
stub where the form had a line.

This policy restores the line deterministically, without the model ever
counting: in a body paragraph, each leader run becomes a `tab` node and the
paragraph's tab stops are replaced by one left stop per leader carrying the
matching `leader` (`_` underscore, `.`/`…` dots, `-` hyphen), so Word and
LibreOffice draw the line themselves up to the stop. The free width of the
line (container width minus indents minus the estimated width of the
paragraph's text) is shared equally between its leaders: a single trailing
leader runs to the right edge, `Student: ____ Grade: ____` splits the line.

Text width is an ESTIMATE (`AVG_CHAR_EM` x font size per character, no font
metrics): a wider real text only shortens the last leader, it never moves
text onto a new line by itself (a tab past its stop advances to the next
one). Container width: section text width (page width minus margins, split
equally for a multi-column section); in a table cell, the cell width minus
its left/right padding.

Refused (left unchanged, counted under `leader_fill:skipped_<reason>`):
* a paragraph that already holds a `tab` -- its existing stops are live;
* a paragraph holding anything but runs (break, image, field, hyperlink ...);
* alignment other than left/justify -- stops on a centred line move;
* a container whose width is unknown (missing geometry, text frame, note,
  header/footer story -- v0 scope is the body and its tables/lists);
* free width below `MIN_SHARE_MPT` per leader (the line would wrap);
* a paragraph that is only hyphen leaders (`----`): in tables it is a
  literal "no value" placeholder far more often than a writing line
  (GC i6 page 214), and the canonical form cannot tell the two apart.
A paragraph with no leader is never counted. Without a leader left after a
rewrite, a second `apply` changes and counts nothing (idempotent). A tab stop
list only acts on tab characters, so replacing the stops of a paragraph that
had no tab before changes nothing else in it.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from unflatten.assembly.assemble import AssembledDocumentHypothesis, AssembledNode

POLICY_ID = "leader-fill.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-010.md"
COUNT_PREFIX = "leader_fill:"
#: Average advance of one character in em (proportional body fonts ~0.45-0.55).
AVG_CHAR_EM = 0.5
#: Smallest line a leader may get (18 pt); below it the paragraph is refused.
MIN_SHARE_MPT = 18_000
#: A stop exactly on the right edge wraps the tab onto a new line in
#: LibreOffice (measured on GC i6 page 070); the last stop ends this far inside.
EDGE_SAFETY_MPT = 6_000
#: Font size when a run carries none (11 pt).
DEFAULT_FONT_MPT = 11_000
LEADER_KIND = {"_": "underscore", ".": "dots", "…": "dots", "-": "hyphen"}
_LEADER = re.compile(r"([_.\-…])\1{3,}")
_ALIGNMENTS = frozenset({"left", "justify"})


def policy_record() -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "leader_pattern": _LEADER.pattern,
        "leader_kind": dict(sorted(LEADER_KIND.items())),
        "avg_char_em": AVG_CHAR_EM,
        "min_share_mpt": MIN_SHARE_MPT,
        "edge_safety_mpt": EDGE_SAFETY_MPT,
        "default_font_mpt": DEFAULT_FONT_MPT,
        "alignments": sorted(_ALIGNMENTS),
        "scope": "body sections, their tables and lists",
    }


def apply(
    document: AssembledDocumentHypothesis,
) -> tuple[AssembledDocumentHypothesis, Counter[str]]:
    """`document` with every eligible body leader paragraph rewritten, and one
    count per outcome."""
    counts: Counter[str] = Counter()
    sections = tuple(_section(section, counts) for section in document.sections)
    if all(new is old for new, old in zip(sections, document.sections, strict=True)):
        return document, counts
    return dataclasses.replace(document, sections=sections), counts


def _section(section: AssembledNode, counts: Counter[str]) -> AssembledNode:
    return _walk(section, _section_width(section.typed), counts)


def _section_width(typed: Mapping[str, Any]) -> int | None:
    page = typed.get("/page/width_mpt")
    left, right = typed.get("/margins/left_mpt"), typed.get("/margins/right_mpt")
    if not all(isinstance(v, int) for v in (page, left, right)):
        return None
    width = page - left - right
    columns = typed.get("/columns/count")
    if isinstance(columns, int) and columns > 1:
        gap = typed.get("/columns/gap_mpt")
        width = (width - (gap if isinstance(gap, int) else 0) * (columns - 1)) // columns
    return width if width > 0 else None


def _cell_width(typed: Mapping[str, Any]) -> int | None:
    width = typed.get("/layout/width_mpt")
    if not isinstance(width, int):
        return None
    pads = [typed.get(f"/layout/padding/{side}_mpt") for side in ("left", "right")]
    width -= sum(p for p in pads if isinstance(p, int))
    return width if width > 0 else None


def _walk(node: AssembledNode, width: int | None, counts: Counter[str]) -> AssembledNode:
    if node.node_type == "paragraph":
        return _paragraph(node, width, counts)
    if node.node_type == "cell":
        width = _cell_width(node.typed)
    elif node.node_type in ("text_frame", "footnote", "endnote", "header", "footer"):
        width = None
    children = tuple(_walk(child, width, counts) for child in node.children)
    if all(new is old for new, old in zip(children, node.children, strict=True)):
        return node
    return dataclasses.replace(node, children=children)


def _paragraph(node: AssembledNode, width: int | None, counts: Counter[str]) -> AssembledNode:
    runs = node.children
    if not any(c.node_type == "run" and _LEADER.search(c.text or "") for c in runs):
        return node
    reason = _refusal(node, width)
    if reason is not None:
        counts[COUNT_PREFIX + "skipped_" + reason] += 1
        return node
    assert width is not None
    typed = node.typed
    left = _int(typed, "/layout/indent/left_mpt")
    start = left + _int(typed, "/layout/indent/first_line_mpt") - _int(
        typed, "/layout/indent/hanging_mpt"
    )
    end = width - _int(typed, "/layout/indent/right_mpt") - EDGE_SAFETY_MPT
    # (text before each leader, leader char) per run, in order
    pieces: list[tuple[AssembledNode, str, str | None]] = []
    for run in runs:
        pos = 0
        text = run.text or ""
        for match in _LEADER.finditer(text):
            pieces.append((run, text[pos:match.start()], match.group(1)))
            pos = match.end()
        pieces.append((run, text[pos:], None))
    leaders = sum(1 for _, _, leader in pieces if leader is not None)
    text_width = sum(_text_width(run, text) for run, text, _ in pieces)
    free = end - start - text_width
    if free < leaders * MIN_SHARE_MPT:
        counts[COUNT_PREFIX + "skipped_no_room"] += 1
        return node
    share = free // leaders
    children: list[AssembledNode] = []
    stops: list[tuple[int, str]] = []
    advance = start
    counter = itertools.count()
    # each original run's `sources` go to the first TEXT piece made from it;
    # a tab never owns a run's piece (its node type would be foreign to the
    # piece), so a leader-only run's sources are dropped. The chain recovers
    # resources before post-assembly policies, so nothing downstream reads them.
    placed: set[str] = set()

    def sources_of(run: AssembledNode) -> tuple[Any, ...]:
        if run.assembled_id in placed:
            return ()
        placed.add(run.assembled_id)
        return run.sources

    for run, text, leader in pieces:
        if text:
            children.append(dataclasses.replace(
                run,
                assembled_id=f"{node.assembled_id}.lf{next(counter)}",
                text=text,
                sources=sources_of(run),
            ))
            advance += _text_width(run, text)
        if leader is not None:
            advance += share
            stops.append((advance, LEADER_KIND[leader]))
            children.append(AssembledNode(
                assembled_id=f"{node.assembled_id}.lf{next(counter)}",
                node_type="tab",
                children=(),
                text=None,
                typed={},
                sources=(),
            ))
    new_typed = {k: v for k, v in typed.items() if not k.startswith("/layout/tabs/")}
    new_typed["/layout/tabs"] = len(stops)
    for i, (position, kind) in enumerate(stops):
        new_typed[f"/layout/tabs/{i}/position_mpt"] = position
        new_typed[f"/layout/tabs/{i}/alignment"] = "left"
        new_typed[f"/layout/tabs/{i}/leader"] = kind
    counts[COUNT_PREFIX + "paragraphs_rewritten"] += 1
    counts[COUNT_PREFIX + "leaders_filled"] += leaders
    return dataclasses.replace(node, children=tuple(children), typed=new_typed)


def _refusal(node: AssembledNode, width: int | None) -> str | None:
    if any(child.node_type == "tab" for child in node.children):
        return "existing_tab"
    if any(child.node_type != "run" for child in node.children):
        return "non_run_child"
    if node.typed.get("/layout/alignment", "left") not in _ALIGNMENTS:
        return "alignment"
    texts = [child.text or "" for child in node.children]
    if not any(_LEADER.sub("", t).strip() for t in texts) and all(
        m.group(1) == "-" for t in texts for m in _LEADER.finditer(t)
    ):
        return "standalone_hyphen"
    if width is None:
        return "unknown_width"
    return None


def _int(typed: Mapping[str, Any], pointer: str) -> int:
    value = typed.get(pointer)
    return value if isinstance(value, int) else 0


def _text_width(run: AssembledNode, text: str) -> int:
    size = run.typed.get("/style/font_size_mpt")
    size = size if isinstance(size, int) and size > 0 else DEFAULT_FONT_MPT
    return round(len(text) * size * AVG_CHAR_EM)
