"""Per-page piece kind for a node under DEC-033 standalone-page encoding.

DEC-033 §4/§10: a node split across pages is two (or more) independently
closed pieces, and some of its typed properties are not observable on a
continuation piece (indent, spacing before, "keep with next", ...). §6/§7
(T6 page projection, T7 typed withdrawal) need to know, per node per page,
which piece it is — this module answers exactly that question and nothing
else: no projection, no withdrawal, no ledger.

`continuation` (envelope.py `_continuation`) already carries it: it is
computed per node from that node's GLOBAL resolved pages against the
pages visible in this window, which is precisely "is this node's content
whole here, or the first/middle/last piece of a node that crosses a page
boundary" — for any node type, not only runs, so no separate walk over
descendant run `char_ranges` is needed to answer the same question twice.
`present_supervised_event_indices`/the standalone encoder do not read
`continuation` any more (design R8); this module is a different consumer
and reads it for a different purpose (piece shape, not supervision).

One override: a node under a page-scoped story (`chain_scope: "page"`,
T1) is always `"whole"`. A repeated header/footer is placed independently
on every page it spells — each placement is a complete node, not a piece
of one split across pages — even though its `continuation` reads
`"middle"` on any page but the section's first/last (the header's
resolved pages span the whole section, but this page only sees one of
them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PieceKind = Literal["whole", "first", "middle", "last"]

_CONTINUATION_TO_PIECE: dict[str, PieceKind] = {
    "complete": "whole",
    "start": "first",
    "middle": "middle",
    "end": "last",
}

#: Node types placed independently on every page they spell (T1 stories),
#: never split across pages the way a body paragraph or table is.
_STORY_ROOT_NODE_TYPES = frozenset({"header", "footer"})

#: `repeat_header` (DEC-033 §10) is a judgement about a `row`'s enclosing
#: `table`, not the row itself -- see `_table_ancestor_id`.
_TABLE_NODE_TYPE = "table"


def _slice_by_node_id(fragment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["node_id"]: entry for entry in fragment.get("slices", ())}


@dataclass(frozen=True, slots=True)
class _StructuralIndex:
    """`structural`'s parallel `node_order`/`node_types`/`parent_index`
    arrays (RFC003 `targets/events.py::derive_event_streams`), plus the two
    lookups every ancestor walk needs, built ONCE per call site rather than
    once per node -- the previous per-node rebuild made a whole page's
    worth of ancestor walks quadratic in slice count."""

    node_order: list[str]
    node_types: list[str]
    parent_index: list[int]
    ordinal_by_open_event_index: dict[int, int]
    ordinal_by_node_id: dict[str, int]


def _build_index(structural: dict[str, Any]) -> _StructuralIndex:
    node_order: list[str] = list(structural.get("node_order", ()))
    node_types: list[str] = list(structural.get("node_types", ()))
    parent_index: list[int] = list(structural.get("parent_index", ()))
    ordinal_by_open_event_index: dict[int, int] = {}
    ordinal = 0
    for event in structural.get("events", ()):
        if event["op"] == "open":
            ordinal_by_open_event_index[event["index"]] = ordinal
            ordinal += 1
    return _StructuralIndex(
        node_order=node_order,
        node_types=node_types,
        parent_index=parent_index,
        ordinal_by_open_event_index=ordinal_by_open_event_index,
        ordinal_by_node_id={node_id: i for i, node_id in enumerate(node_order)},
    )


def _walk_ancestor_ordinals(node_id: str, index: _StructuralIndex):
    """Yield each ancestor's ordinal, root-ward, from `node_id`'s parent."""
    node_ordinal = index.ordinal_by_node_id.get(node_id)
    if node_ordinal is None:
        return
    parent_open_event = index.parent_index[node_ordinal]
    while parent_open_event != -1:
        parent_ordinal = index.ordinal_by_open_event_index[parent_open_event]
        yield parent_ordinal
        parent_open_event = index.parent_index[parent_ordinal]


def _ancestor_types_indexed(node_id: str, index: _StructuralIndex) -> set[str]:
    return {index.node_types[ordinal] for ordinal in _walk_ancestor_ordinals(node_id, index)}


def _table_ancestor_id(node_id: str, index: _StructuralIndex) -> str | None:
    """The nearest ancestor node_id of type `table`, or `None`.

    A row's OWN `continuation` always reads `"complete"` -- a row is never
    itself split across a page boundary -- so `_piece_kind_indexed` alone
    cannot answer DEC-033 §10's table-level `repeat_header` question;
    `piece_kinds_for_page` uses this to make a row follow its table.
    """
    for ordinal in _walk_ancestor_ordinals(node_id, index):
        if index.node_types[ordinal] == _TABLE_NODE_TYPE:
            return index.node_order[ordinal]
    return None


def _piece_kind_indexed(
    fragment_slices: dict[str, dict[str, Any]], node_id: str, index: _StructuralIndex
) -> PieceKind:
    slice_ = fragment_slices.get(node_id)
    if slice_ is None:
        raise KeyError(f"node {node_id!r} has no slice on this page's fragment")
    if slice_.get("node_type") in _STORY_ROOT_NODE_TYPES:
        return "whole"
    if _STORY_ROOT_NODE_TYPES & _ancestor_types_indexed(node_id, index):
        return "whole"
    return _CONTINUATION_TO_PIECE[slice_["continuation"]]


def piece_kind(fragment: dict[str, Any], node_id: str, structural: dict[str, Any]) -> PieceKind:
    """Whether `node_id`'s content on this page (`fragment`) is whole, or
    one piece of a node split across pages (DEC-033 §4/§10)."""
    return _piece_kind_indexed(_slice_by_node_id(fragment), node_id, _build_index(structural))


def piece_kinds_for_page(
    fragment: dict[str, Any], structural: dict[str, Any]
) -> dict[str, PieceKind]:
    """`node_id -> piece_kind` for every node with a slice on this page,
    in one pass over `structural`'s index rather than one rebuild per node
    (DEC-033 §10, T7's `withdraw_page_piece_typed`).

    One override on top of the per-node `piece_kind` rule: a `row`'s kind
    is its enclosing `table`'s kind, not its own (`_table_ancestor_id`).
    """
    slices = _slice_by_node_id(fragment)
    index = _build_index(structural)
    kinds: dict[str, PieceKind] = {
        node_id: _piece_kind_indexed(slices, node_id, index) for node_id in slices
    }
    for node_id, slice_ in slices.items():
        if slice_.get("node_type") != "row":
            continue
        table_id = _table_ancestor_id(node_id, index)
        if table_id is not None and table_id in kinds:
            kinds[node_id] = kinds[table_id]
    return kinds
