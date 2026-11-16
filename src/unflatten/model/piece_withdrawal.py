"""DEC-033 §10 page-piece withdrawal lists, dataset-free.

Moved verbatim from `typed_property_targets` (assembler design §3, option
(a)) so the page assembler can derive what it hides from its rules from
the one source of truth without importing `unflatten.dataset`.
`typed_property_targets` re-imports every name here under its old
private name.
"""

from __future__ import annotations

from unflatten.model.page_pieces import PieceKind

#: Closed list, paragraph `middle`/`last` (DEC-033 §10): unobservable on a
#: continuation piece because they describe how the paragraph BEGINS or
#: how it behaves as a pagination unit — neither is visible past its
#: first piece.
PARAGRAPH_UNOBSERVABLE_ON_CONTINUATION = frozenset(
    {
        "/layout/indent/first_line_mpt",
        "/layout/indent/hanging_mpt",
        "/layout/spacing/before_mpt",
        "/layout/pagination/keep_with_next",
        "/layout/pagination/keep_lines",
        "/layout/pagination/page_break_before",
        "/semantic/role",
        "/semantic/heading_level",
        "/semantic/outline_level",
    }
)

#: Closed list, paragraph `first`/`middle`: "space after" describes how
#: the paragraph ENDS, unobservable on any piece but its last.
PARAGRAPH_UNOBSERVABLE_SPACING_AFTER = frozenset({"/layout/spacing/after_mpt"})

#: Closed list, list `middle`/`last`: the marker start number is only
#: visible on the list's first piece.
LIST_UNOBSERVABLE_ON_CONTINUATION = frozenset({"/marker/start"})

#: Closed list, table `middle`/`last`: "repeat this row as a header" is a
#: judgement about the TABLE's continuation, recorded on its `row` nodes
#: (the registry has no table-level pointer for it).
ROW_UNOBSERVABLE_ON_CONTINUATION_TABLE = frozenset({"/layout/repeat_header"})

#: Closed list, section `middle`/`last` (DEC-049 point 7, item 9.2): how a
#: section STARTS -- a page break, a new column layout, an odd-page
#: restart -- is a Word `sectPr` fact recorded once, at the section's
#: start; a continuation piece cannot show it (design §0.5, M1 §5.3/§5.4:
#: this was 2 of 2 `ASSEMBLER_PROPERTY_CONFLICT`s on M1).
SECTION_UNOBSERVABLE_ON_CONTINUATION = frozenset({"/break_type"})

#: Closed list, cell `middle`/`last` (DEC-049 point 7, item 9.2): a
#: vertically merged cell's row span is recorded on the cell's first
#: (anchor) piece; a continuation piece is the merge's tail, not its
#: anchor, and cannot show it.
CELL_UNOBSERVABLE_ON_CONTINUATION = frozenset({"/row_span"})


#: Version of the table below. The typed-target materializer reads it, but no
#: cache identity hash covers it (changing the identity formula would make new
#: code refuse the existing v15b caches). Bump it on every edit of the table and
#: record it in the build manifest of the next cache; the pinned test in
#: tests/model/test_piece_withdrawal.py fails until you do.
#: v1: DEC-033 §10 (paragraph, list, row). v2 (DEC-049 bod 7, 2026-09-27):
#: + section /break_type and cell /row_span on continuation pieces.
PIECE_WITHDRAWAL_VERSION = "v2"


def withdrawn_pointers_for_piece(node_type: str, kind: PieceKind) -> frozenset[str]:
    """The closed set of pointers DEC-033 §10 withdraws on `node_type`'s
    `kind` piece. Empty for every node type/kind this table does not
    name — "other properties ... are learned on both pieces" (DEC-033
    §10)."""
    if node_type == "paragraph":
        pointers: set[str] = set()
        if kind in ("middle", "last"):
            pointers |= PARAGRAPH_UNOBSERVABLE_ON_CONTINUATION
        if kind in ("first", "middle"):
            pointers |= PARAGRAPH_UNOBSERVABLE_SPACING_AFTER
        return frozenset(pointers)
    if node_type == "list" and kind in ("middle", "last"):
        return LIST_UNOBSERVABLE_ON_CONTINUATION
    if node_type == "row" and kind in ("middle", "last"):
        return ROW_UNOBSERVABLE_ON_CONTINUATION_TABLE
    if node_type == "section" and kind in ("middle", "last"):
        return SECTION_UNOBSERVABLE_ON_CONTINUATION
    if node_type == "cell" and kind in ("middle", "last"):
        return CELL_UNOBSERVABLE_ON_CONTINUATION
    return frozenset()
