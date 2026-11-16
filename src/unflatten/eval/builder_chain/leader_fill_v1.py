"""Opt-in chain policy `leader-fill.v1` (RATIFIED 2026-10-16, DEC-077;
`docs/rfc/RFC004-ERRATA-011.md` E4). An ALTERNATIVE to the RATIFIED
`leader-fill.v0`, called at the same place of the chain and never together
with it; v0 and its hash are untouched, this module only imports v0's
shared constants and helpers.

Measured on the S4 v16 selection cohort (40 pages, checkpoint 178 975,
2026-10-16): v0 rewrote 18 paragraphs / 24 leaders and refused 14 paragraphs
for `existing_tab`, although the gold form `Text: TAB ____` is the common
shape of a form field (7 refusals on one page); and on two checkpoints the
model looped `…………..…………..` 150+ times inside one run, which v0 refused for
`no_room` (literal dots over two pages). Two rules more than v0:

1. a paragraph holding `tab` nodes is not refused: a leader run directly
   after a `tab` gives that tab a stop carrying the leader and the run is
   dropped; a tab followed by no leader gets a plain left stop on the next
   default tab stop (`DEFAULT_TAB_MPT`); a leader with no tab before it
   becomes a new tab exactly as in v0. The free width (line minus text minus
   `DEFAULT_TAB_MPT` per plain tab) is shared equally between the leader
   stops;
2. a chain of >= 3 leader groups separated only by filler (leader characters
   and spaces, any length) collapses to its first two groups and their
   separator (`leader_groups_collapsed`): the gold two-field line
   `………….…………` keeps both fields, `Date: ____ Dept: ____` is untouched
   (text between the fields is not filler);
3. every other refusal is v0's and counted the same way under this prefix;
   new counts: `paragraphs_with_existing_tab` (rule 1 fired),
   `skipped_leading_leader` -- a leader as the very first piece of the
   paragraph (no text and no tab before it) is refused, since a tab stop
   before any text draws no visible line in LibreOffice (OPEN in the draft).

Width of a plain tab (`DEFAULT_TAB_MPT`) is Word's default tab stop interval
(1.27 cm = 36 pt), an ESTIMATE of where a tab without a stop lands, not a
measurement of the model's paragraphs.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from collections import Counter
from typing import Any

from unflatten.assembly.assemble import AssembledDocumentHypothesis, AssembledNode
from unflatten.eval.builder_chain import leader_fill as lf

POLICY_ID = "leader-fill.v1"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-011.md"
COUNT_PREFIX = "leader_fill_v1:"
#: Word's default tab stop interval, 1.27 cm = 36 pt -- an estimate of where a
#: plain tab (no stop, no leader) advances to; not measured on model output.
DEFAULT_TAB_MPT = 36_000
#: Rule 2: between two leader groups of one chain only these characters.
_FILLER = re.compile(r"^[ _.\-…]*$")
RULES = (
    "existing_tab_takes_leader_stop.v1",
    "plain_tab_default_stop.v1",
    "leader_without_tab_new_tab_as_v0.v1",
    "leader_chain_collapsed_to_two.v1",
    "leading_leader_refused.v1",
    "v0_refusals_otherwise.v1",
)


def policy_record() -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "alternative_to": lf.POLICY_ID,
        "leader_pattern": lf._LEADER.pattern,
        "leader_kind": dict(sorted(lf.LEADER_KIND.items())),
        "filler_pattern": _FILLER.pattern,
        "avg_char_milli_em": round(lf.AVG_CHAR_EM * 1000),  # canonical JSON carries no floats
        "min_share_mpt": lf.MIN_SHARE_MPT,
        "edge_safety_mpt": lf.EDGE_SAFETY_MPT,
        "default_font_mpt": lf.DEFAULT_FONT_MPT,
        "default_tab_mpt": DEFAULT_TAB_MPT,
        "alignments": sorted(lf._ALIGNMENTS),
        "scope": "body sections, their tables and lists",
        "rules": list(RULES),
    }


def collapse(text: str) -> tuple[str, int]:
    """Rule 2: `(text, groups_dropped)`; a chain of >= 3 leader groups with
    only filler between them keeps its first two groups and their separator."""
    matches = list(lf._LEADER.finditer(text))
    if len(matches) < 3:
        return text, 0
    chains: list[list[re.Match[str]]] = [[matches[0]]]
    for prev, cur in zip(matches, matches[1:], strict=False):
        if _FILLER.match(text[prev.end():cur.start()]):
            chains[-1].append(cur)
        else:
            chains.append([cur])
    out: list[str] = []
    pos = dropped = 0
    for chain in chains:
        if len(chain) < 3:
            continue
        out.append(text[pos:chain[0].start()])
        out.append(text[chain[0].start():chain[1].end()])
        pos = chain[-1].end()
        dropped += len(chain) - 2
    out.append(text[pos:])
    return "".join(out), dropped


def apply(
    document: AssembledDocumentHypothesis,
) -> tuple[AssembledDocumentHypothesis, Counter[str]]:
    """`document` with every eligible body leader paragraph rewritten, and one
    count per outcome -- same shape as `leader_fill.apply`."""
    counts: Counter[str] = Counter()
    sections = tuple(_walk(s, lf._section_width(s.typed), counts) for s in document.sections)
    if all(new is old for new, old in zip(sections, document.sections, strict=True)):
        return document, counts
    return dataclasses.replace(document, sections=sections), counts


def _walk(node: AssembledNode, width: int | None, counts: Counter[str]) -> AssembledNode:
    if node.node_type == "paragraph":
        return _paragraph(node, width, counts)
    if node.node_type == "cell":
        width = lf._cell_width(node.typed)
    elif node.node_type in ("text_frame", "footnote", "endnote", "header", "footer"):
        width = None
    children = tuple(_walk(child, width, counts) for child in node.children)
    if all(new is old for new, old in zip(children, node.children, strict=True)):
        return node
    return dataclasses.replace(node, children=children)


def _refusal(node: AssembledNode, width: int | None) -> str | None:
    """v0's refusals minus `existing_tab` (rule 1)."""
    if any(child.node_type not in ("run", "tab") for child in node.children):
        return "non_run_child"
    if node.typed.get("/layout/alignment", "left") not in lf._ALIGNMENTS:
        return "alignment"
    texts = [child.text or "" for child in node.children if child.node_type == "run"]
    if not any(lf._LEADER.sub("", t).strip() for t in texts) and all(
        m.group(1) == "-" for t in texts for m in lf._LEADER.finditer(t)
    ):
        return "standalone_hyphen"
    if width is None:
        return "unknown_width"
    return None


# a piece is ("text", run, text) | ("tab", tab_node, leader | None) | ("newtab", run, leader)
_Piece = tuple[str, AssembledNode, str | None]


def _pieces(node: AssembledNode) -> tuple[list[_Piece], int]:
    """The paragraph cut into text / tab / leader pieces (rule 2 applied), with
    rule 1 merged: a leader right after a tab rides on that tab."""
    pieces: list[_Piece] = []
    collapsed = 0
    for child in node.children:
        if child.node_type == "tab":
            pieces.append(("tab", child, None))
            continue
        text, dropped = collapse(child.text or "")
        collapsed += dropped
        pos = 0
        for match in lf._LEADER.finditer(text):
            if text[pos:match.start()]:
                pieces.append(("text", child, text[pos:match.start()]))
            pieces.append(("leader", child, match.group(1)))
            pos = match.end()
        if text[pos:]:
            pieces.append(("text", child, text[pos:]))
    merged: list[_Piece] = []
    for kind, owner, value in pieces:
        if kind == "leader" and merged and merged[-1][0] == "tab" and merged[-1][2] is None:
            merged[-1] = ("tab", merged[-1][1], value)
            continue
        merged.append(("newtab", owner, value) if kind == "leader" else (kind, owner, value))
    return merged, collapsed


def _paragraph(node: AssembledNode, width: int | None, counts: Counter[str]) -> AssembledNode:
    if not any(c.node_type == "run" and lf._LEADER.search(c.text or "") for c in node.children):
        return node
    reason = _refusal(node, width)
    if reason is not None:
        counts[COUNT_PREFIX + "skipped_" + reason] += 1
        return node
    assert width is not None
    pieces, collapsed = _pieces(node)
    if pieces and pieces[0][0] == "newtab":
        counts[COUNT_PREFIX + "skipped_leading_leader"] += 1
        return node
    typed = node.typed
    left = lf._int(typed, "/layout/indent/left_mpt")
    start = left + lf._int(typed, "/layout/indent/first_line_mpt") - lf._int(
        typed, "/layout/indent/hanging_mpt"
    )
    end = width - lf._int(typed, "/layout/indent/right_mpt") - lf.EDGE_SAFETY_MPT
    leaders = sum(1 for kind, _, value in pieces if kind in ("tab", "newtab") and value)
    plain = sum(1 for kind, _, value in pieces if kind == "tab" and value is None)
    text_width = sum(lf._text_width(owner, value or "") for kind, owner, value in pieces
                     if kind == "text")
    free = end - start - text_width - plain * DEFAULT_TAB_MPT
    if leaders and free < leaders * lf.MIN_SHARE_MPT:
        counts[COUNT_PREFIX + "skipped_no_room"] += 1
        return node
    share = free // leaders if leaders else 0
    children: list[AssembledNode] = []
    stops: list[tuple[int, str]] = []
    advance = start
    counter = itertools.count()
    # each original run's `sources` go to the first TEXT piece made from it (as in v0)
    placed: set[str] = set()

    def sources_of(run: AssembledNode) -> tuple[Any, ...]:
        if run.assembled_id in placed:
            return ()
        placed.add(run.assembled_id)
        return run.sources

    for kind, owner, value in pieces:
        if kind == "text":
            children.append(dataclasses.replace(
                owner,
                assembled_id=f"{node.assembled_id}.lf{next(counter)}",
                text=value,
                sources=sources_of(owner),
            ))
            advance += lf._text_width(owner, value or "")
        elif kind == "tab" and value is None:
            advance = (advance // DEFAULT_TAB_MPT + 1) * DEFAULT_TAB_MPT
            stops.append((advance, "none"))
            children.append(owner)
        else:
            assert value is not None
            advance += share
            stops.append((advance, lf.LEADER_KIND[value]))
            children.append(owner if kind == "tab" else AssembledNode(
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
    if any(kind == "tab" for kind, _, _ in pieces):
        counts[COUNT_PREFIX + "paragraphs_with_existing_tab"] += 1
    if collapsed:
        counts[COUNT_PREFIX + "leader_groups_collapsed"] += collapsed
    return dataclasses.replace(node, children=tuple(children), typed=new_typed)
