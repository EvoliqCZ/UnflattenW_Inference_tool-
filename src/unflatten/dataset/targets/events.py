"""Structural, text-span and visible-text event derivation (RFC003 §27–§28).

Streams A (structural events), B (text spans) and E (visible text events)
are derived in a single preorder walk over a *canonical* AST — the input
must already be `unflatten.ast.canonicalizer.canonicalize()`d, which is
asserted via the preorder n-id invariant (RFC001 §35 step 5). NFC text is
therefore guaranteed by the canonicalizer, not re-normalized here.

The derived streams are never hand-fixed independently of the AST — fix
the AST and re-derive (RFC003 §27).

Stream shapes follow the normative RFC003 §27 example:

- structural events: ``{"index", "op": "open"|"text"|"close", "node_id", ...}``
  with ``node_type`` on open, ``text_span_id`` on text and ``open_index``
  on close; a synthesized ``doc``/``document`` root wraps the sections.
- parallel per-node arrays ``node_order`` / ``node_types`` /
  ``parent_index`` (index of the parent's *open event*, ``-1`` for the
  root) / ``close_event_index`` (index of the matching close event).
- leaf controls (tab, line_break, page_break, column_break, inline_image)
  are open + immediate close.
- ``text_span_id = "txt_" + str(structural event index)`` (§27), one span
  per run text event with half-open Unicode-scalar offsets in the node.
- visible text events use the §28 enum; ``SECTION_END`` is dataset-side
  only and absent from the renderer's AST-side derivation
  (`unflatten.renderer.text_extract`), with which the common subset must
  agree.

HYPERLINK FLATTENING (DEC-043 §1, v15 rebuild): ``hyperlink`` is a
*transparent* inline container here — it emits no ``open``/``close``
structural event of its own and never occupies a slot in
``node_order``/``node_types``/``parent_index``, but its children (runs)
ARE visited, exactly as if they were direct children of the hyperlink's
own parent. This is a stream-derivation-time projection only: the input
AST is never mutated, node ids are never renumbered (a flattened-away
hyperlink's id simply has no event; every other id keeps meaning what it
already meant), and `preorder_nodes`/`document_roots` below (used by
`targets/masks.py` and everything keyed on the canonical AST) are
untouched — they still see the hyperlink. So the STORED bundle AST keeps
the hyperlink node and the renderer can still emit a real link from it;
only the model-facing event/token stream this module derives no longer
carries a HYPERLINK_OPEN/CLOSE pair, which is what lets a page with a
hyperlink train instead of being refused by
`composite_grammar._structural_domain` under a profile that excludes
`hyperlink` (DEC-037 §2 revised by DEC-043 §1: exclude the structural
token, but flatten first so excluding it never costs the page). The
`hyperlink /target/uri` pointer has no model channel either way
(DEC-003 §1.3, `masks.NO_CHANNEL_POINTERS`) — flattening does not change
that, it just stops the address from being defaulted or invented: after
flattening there is no hyperlink node in the stream for the model to
attach a target to at all.

Flattening runs strictly AFTER canonicalization's own
`_merge_adjacent_runs` (`ast/canonicalizer.py`), never before and never
again here: a run just before a hyperlink and the hyperlink's own run are
never merged even if they share language+style, because at
canonicalization time the hyperlink node still separates them. Structure
changes (the wrapper disappears from the stream); text never does.

KNOWN DIVERGENCE — hyperlink children (D4 should_fix 4): this module
DOES emit ``RUN_TEXT`` for runs inside a ``hyperlink`` node (RFC003
visible-text events cover all visible run text — correct for the dataset
layer), while the renderer's ``text_extract`` returns ``[]`` for
hyperlink nodes because hyperlinks are UNSUPPORTED in renderer
word-core-v0 (its AST-side derivation never sees them). The cross-check
test special-cases this and a guard test pins the valid corpus
hyperlink-free (word-core-v0 TRAIN_READY excludes hyperlinks), so the
divergence cannot silently bite.

KNOWN DIVERGENCE — note markers (`word-visual-v1`): the renderer's
AST-side derivation emits a ``note_ref``'s ``marker_text`` as visible
text (it is ink on the page and RFC002 conserves it); this module emits
NOTHING for a ``note_ref``. Stream E is a supervision surface, the §28
enum has no marker event, and RFC004 §18.3 confines
``TEXT_BEGIN…TEXT_END`` to Run — so emitting the marker as ``RUN_TEXT``
would claim a run that does not exist and supervise a string the model
has no channel for (DEC-003 §1.3; the same reason
`masks.NO_CHANNEL_POINTERS` withdraws `/marker_text`). Adding a §28
event is an RFC003 erratum, not a code change to make here. The v1
cross-check test (`test_visual_v1_targets`) strips markers from the
renderer side and says so, so the divergence stays named rather than
discovered.
"""

from __future__ import annotations

import copy
from typing import Any

# The preorder walk is imported from the canonicalizer (the module that
# DEFINES it) rather than re-derived here, so the two cannot diverge.
# `_child_lists` is the word-visual-v1 shape: a section owns
# `headers`/`footers`/`children`, and the note stories hang off the
# document root — a node-type -> single-children-key mapping cannot
# express either, which is why the stories used to produce no events.
from unflatten.ast.canonicalizer import _NOTE_KEYS as NOTE_KEYS
from unflatten.ast.canonicalizer import _child_lists as child_lists
from unflatten.ast.canonicalizer import canonicalize

VISIBLE_TEXT_EVENTS: frozenset[str] = frozenset(
    {
        "RUN_TEXT",
        "TAB",
        "LINE_BREAK",
        "PAGE_BREAK_EXPLICIT",
        "COLUMN_BREAK_EXPLICIT",
        "PARAGRAPH_END",
        "CELL_END",
        "ROW_END",
        "SECTION_END",
    }
)

_OPEN_VISIBLE_EVENT: dict[str, str] = {
    "tab": "TAB",
    "line_break": "LINE_BREAK",
    "page_break": "PAGE_BREAK_EXPLICIT",
    "column_break": "COLUMN_BREAK_EXPLICIT",
}

_CLOSE_VISIBLE_EVENT: dict[str, str] = {
    "paragraph": "PARAGRAPH_END",
    "cell": "CELL_END",
    "row": "ROW_END",
    "section": "SECTION_END",
}

#: DEC-043 §1: node types that are transparent to the derived event
#: streams -- visited (their children are walked) but never themselves
#: opened/closed/added to `node_order`. `hyperlink` is the only one today;
#: growing this set flattens another container type the same way.
_TRANSPARENT_CONTAINER_TYPES: frozenset[str] = frozenset({"hyperlink"})


def derive_event_streams(ast: dict[str, Any]) -> dict[str, Any]:
    """Derive streams A/B/E from a canonical AST in one preorder walk.

    Raises ``ValueError`` if the AST does not carry the canonical preorder
    ``n1..nN`` node ids (i.e. was not ``canonicalize()``d).
    """
    _assert_preorder_ids(ast)

    events: list[dict[str, Any]] = []
    node_order: list[str] = []
    node_types: list[str] = []
    parent_index: list[int] = []
    close_event_index: list[int] = []
    spans: list[dict[str, Any]] = []
    visible: list[dict[str, Any]] = []

    def open_node(node_id: str, node_type: str, parent_open_event: int) -> int:
        ordinal = len(node_order)
        node_order.append(node_id)
        node_types.append(node_type)
        parent_index.append(parent_open_event)
        close_event_index.append(-1)
        events.append(
            {"index": len(events), "op": "open", "node_id": node_id, "node_type": node_type}
        )
        return ordinal

    def close_node(ordinal: int, node_id: str, open_event: int) -> None:
        close_event_index[ordinal] = len(events)
        events.append(
            {"index": len(events), "op": "close", "node_id": node_id, "open_index": open_event}
        )

    def visit(node: dict[str, Any], parent_open_event: int) -> None:
        node_id: str = node["id"]
        node_type: str = node["type"]

        if node_type in _TRANSPARENT_CONTAINER_TYPES:
            # DEC-043 §1: flatten -- no open/close event, no node_order
            # slot, children reparent directly to this node's own parent.
            for children_key in child_lists(node):
                for child in node[children_key]:
                    visit(child, parent_open_event)
            return

        open_event = len(events)
        ordinal = open_node(node_id, node_type, parent_open_event)

        if node_type == "run" and node["text"]:
            text: str = node["text"]
            text_event_index = len(events)
            text_span_id = f"txt_{text_event_index}"
            events.append(
                {
                    "index": text_event_index,
                    "op": "text",
                    "node_id": node_id,
                    "text_span_id": text_span_id,
                }
            )
            spans.append(
                {
                    "text_span_id": text_span_id,
                    "node_id": node_id,
                    "text": text,
                    "start": 0,
                    "end": len(text),
                    "normalization": "NFC",
                }
            )
            visible.append(
                {
                    "index": len(visible),
                    "event": "RUN_TEXT",
                    "node_id": node_id,
                    "text_span_id": text_span_id,
                }
            )
        elif node_type in _OPEN_VISIBLE_EVENT:
            visible.append(
                {
                    "index": len(visible),
                    "event": _OPEN_VISIBLE_EVENT[node_type],
                    "node_id": node_id,
                }
            )

        for children_key in child_lists(node):
            for child in node[children_key]:
                visit(child, open_event)

        close_node(ordinal, node_id, open_event)

        if node_type in _CLOSE_VISIBLE_EVENT:
            visible.append(
                {
                    "index": len(visible),
                    "event": _CLOSE_VISIBLE_EVENT[node_type],
                    "node_id": node_id,
                }
            )

    doc_ordinal = open_node("doc", "document", -1)
    for root in document_roots(ast):
        visit(root, 0)
    close_node(doc_ordinal, "doc", 0)

    return {
        "structural": {
            "events": events,
            "node_order": node_order,
            "node_types": node_types,
            "parent_index": parent_index,
            "close_event_index": close_event_index,
        },
        "text": {"spans": spans, "visible_text_events": visible},
    }


def reconstruct_visible_events(structural: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-derive stream E purely from the structural stream (§28).

    Stream E is exactly derivable from stream A: RUN_TEXT at every text
    event, TAB/LINE_BREAK/*_EXPLICIT at leaf-control opens, and the END
    events at container closes. Used by the bundle validator to compare
    the stored stream element-wise — order included — against this
    reconstruction, so a reordered, duplicated or dropped visible event
    can never pass validation.
    """
    events = structural["events"]
    node_types_by_open_index: dict[int, str] = {}
    node_type_by_ordinal = structural["node_types"]
    # Map open-event index -> node type by replaying opens in order.
    ordinal = 0
    for event in events:
        if event["op"] == "open":
            node_types_by_open_index[event["index"]] = node_type_by_ordinal[ordinal]
            ordinal += 1
    visible: list[dict[str, Any]] = []
    for event in events:
        if event["op"] == "text":
            visible.append(
                {
                    "index": len(visible),
                    "event": "RUN_TEXT",
                    "node_id": event["node_id"],
                    "text_span_id": event["text_span_id"],
                }
            )
        elif event["op"] == "open":
            node_type = node_types_by_open_index[event["index"]]
            if node_type in _OPEN_VISIBLE_EVENT:
                visible.append(
                    {
                        "index": len(visible),
                        "event": _OPEN_VISIBLE_EVENT[node_type],
                        "node_id": event["node_id"],
                    }
                )
        elif event["op"] == "close":
            node_type = node_types_by_open_index.get(event["open_index"])
            if node_type in _CLOSE_VISIBLE_EVENT:
                visible.append(
                    {
                        "index": len(visible),
                        "event": _CLOSE_VISIBLE_EVENT[node_type],
                        "node_id": event["node_id"],
                    }
                )
    return visible


def document_roots(ast: dict[str, Any]) -> list[dict[str, Any]]:
    """The document's root nodes in canonical preorder (RFC001 §35 step 5).

    Sections first, then the note stories in `footnotes`, `endnotes`
    order — the same order the canonicalizer numbers ids in, which the
    RFC004 §18.4 grammar also enforces on the token stream.
    """
    roots: list[dict[str, Any]] = list(ast.get("sections", []))
    for key in NOTE_KEYS:
        roots += list(ast.get(key) or [])
    return roots


def preorder_nodes(ast: dict[str, Any]):
    """Every id-carrying node of `ast` in canonical preorder.

    Excludes the synthesized `doc` root, which is not an AST node.
    """
    yield from _preorder(document_roots(ast))


def _assert_preorder_ids(ast: dict[str, Any]) -> None:
    counter = 0
    for node in preorder_nodes(ast):
        counter += 1
        expected = f"n{counter}"
        if node.get("id") != expected:
            raise ValueError(
                "derive_event_streams requires a canonicalize()d AST with preorder "
                f"n-ids: expected id {expected!r}, found {node.get('id')!r} — "
                "canonicalize the AST first (RFC001 §35 step 5)."
            )


def _preorder(nodes: list[dict[str, Any]]):
    for node in nodes:
        yield node
        for children_key in child_lists(node):
            yield from _preorder(node[children_key])


def unwrap_hyperlinks(ast: dict[str, Any]) -> dict[str, Any]:
    """AST-level counterpart of the `derive_event_streams` hyperlink
    flatten, for GOLD COMPARISON only (design §3 "profile-projected gold",
    the same role `page_projection.project` plays for page splits) — never
    for the stored bundle, which keeps every hyperlink node untouched.

    Splices each `hyperlink` node's children into its parent's child list
    in its place, recursively, then re-`canonicalize()`s: a strict AST
    build from the flattened token stream produces a document with no
    hyperlink node at all, and canonicalization is exactly what assigns
    that rebuilt document its final `n1..nN` ids and merges any run that
    is now adjacent to an equal-style neighbour it wasn't adjacent to
    before (`ast/canonicalizer.py`'s own `_merge_adjacent_runs`) — so
    comparing `ast_hash(rebuilt)` against `ast_hash(unwrap_hyperlinks(gold))`
    is apples to apples. This is *not* how `derive_event_streams` itself
    flattens (that never re-canonicalizes or merges, see the module
    docstring) — the two flattenings serve different purposes and must not
    be confused for one another.
    """
    working = copy.deepcopy(ast)

    def unwrap(node: dict[str, Any]) -> None:
        for children_key in child_lists(node):
            flattened: list[dict[str, Any]] = []
            for child in node[children_key]:
                if child.get("type") == "hyperlink":
                    unwrap(child)
                    flattened.extend(child["children"])
                else:
                    unwrap(child)
                    flattened.append(child)
            node[children_key] = flattened

    for root in document_roots(working):
        unwrap(root)

    return canonicalize(working)
