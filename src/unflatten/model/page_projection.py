"""DEC-033 §5/§6 page-normalization projection: trailing drop, interior
empty-paragraph cap, TAB cap, container floor (design §4, R4, R9).

`s2-standalone-page-v1` gives every page a closed stream of its own
(`encode_fragment.encode_standalone_fragment_with_origins`). Left
unprojected, that stream still carries whatever runs of visually-empty
paragraphs and tab leaves the page happened to contain -- up to the 53
and 202 ceilings the old, page-of-owner encoding tolerated. This module
computes, from the page's own present nodes, which of them a DEC-033
build drops before a single token is emitted: dropped nodes never reach
`encode_standalone_fragment_with_origins`'s `present` set, so they carry
no tokens and no loss units -- withdrawn by absence, never by a masked
zero (`missing grant is not empty text`).

Order of operations, each pass working only on what the previous pass
left present:

1. trailing drop -- visually-empty paragraphs (and a trailing
   `page_break`/`column_break`) at the tail of each story (the body, and
   every `footnote`/`endnote` on its own), with no content event after
   them in that story (coordinator decision 2, 17.9.);
2. interior cap -- within one parent, a run of more than 3 consecutive
   visually-empty paragraph children keeps the first 3;
3. container floor -- a `cell`/`list_item`/`text_frame`/`header`/`footer`/
   `footnote`/`endnote`/`section` never loses its last present (direct) child (the first
   dropped child is restored); `cell` additionally never ends on a
   non-paragraph child (the LAST dropped paragraph child *after the
   cell's last surviving child* is restored -- only such a paragraph can
   end the cell, `grammar._close_allowed`'s `last_child_is_paragraph`;
   if that paragraph is empty and joins, across a break or a `text_frame`,
   a run the cap left full, the kept empty nearest to it is dropped instead);
4. TAB cap -- within one surviving `paragraph` or `hyperlink` (the AST
   schema's only two `tab` parents), a run of more than 3 consecutive
   `tab` leaf children keeps the first 3.

`project` then refuses (`PageProjectionInvariantViolation`) if either
cap was escaped -- the grammar's bounds are cap + 1, so an escape is
absorbed silently there rather than reported.

"Visually empty" (DEC-033 §9, R4) means no text and no image: a
paragraph holding only `tab`/`line_break` still counts as empty. Ink
comes from what THIS PAGE OWNS of `text_spans`, not a span's full
underlying text -- `_ink_node_ids` reuses `ScalarBoundaryProjector`
(`training_materializer._interleave_text`'s own source of scalar
ranges) so a span whose real characters live on another page contributes
no ink here, exactly as it contributes no token there (real-data defect,
prod dry-run 2026-09-18: 11/378 pages, a continuation piece's owned-
elsewhere text read as ink, so the projection never capped an interior
run the grammar counted as empty). A paragraph subtree with a present
`field`/`note_ref`/`hyperlink`/`inline_image`/`floating_image`/
`text_frame` descendant is content regardless of ink.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from unflatten.model.boundary_projector import ScalarBoundaryProjector, TextSpanAuthority

#: A descendant of one of these types makes a paragraph non-empty,
#: regardless of ink (DEC-033 §9, R4: `tab`/`line_break` do NOT).
_CONTENT_DESCENDANT_TYPES = frozenset(
    {"field", "note_ref", "hyperlink", "inline_image", "floating_image", "text_frame"}
)

#: Block-level node types: a page's content, read top to bottom, is a
#: sequence of these -- everything else is either a bare container
#: (skipped, its content surfaces through its own block descendants) or
#: lives inside one of these blocks already (a run/tab/line_break inside
#: a paragraph, a paragraph inside a list item).
_BLOCK_TYPES = frozenset({"paragraph", "table", "list", "page_break", "column_break"})

#: A drop may never empty one of these containers entirely (design §4).
_FLOOR_TYPES = frozenset(
    {"cell", "list_item", "text_frame", "header", "footer", "footnote", "endnote", "section"}
)

#: Note stories: each gets its own trailing drop (`_story_of`).
_NOTE_TYPES = frozenset({"footnote", "endnote"})

#: The interior empty-paragraph run cap (DEC-033 §5, R3): keep the first 3.
_EMPTY_RUN_CAP = 3

#: The TAB run cap (DEC-033 §6, R3): keep the first 3.
_TAB_RUN_CAP = 3


class PageProjectionInvariantViolation(ValueError):
    """A projected page still carries a run above one of the caps.

    The composite grammar's bounds are the caps + 1
    (`composite_grammar.MAX_CONSECUTIVE_EMPTY_PARAGRAPHS`), so an escape
    here does not refuse the page -- it silently consumes the whole
    headroom. Refused loudly at projection instead (RFC004-ERRATA-002 E1
    correction, 18.9.)."""


def _assert_run_bounds(*, max_empty_run_after: int, max_tab_run_after: int) -> None:
    """Postcondition of `project`: every surviving run is within its cap."""
    if max_empty_run_after > _EMPTY_RUN_CAP:
        raise PageProjectionInvariantViolation(
            f"page projection left max_empty_run_after={max_empty_run_after}, "
            f"above the interior empty-paragraph cap {_EMPTY_RUN_CAP}"
        )
    if max_tab_run_after > _TAB_RUN_CAP:
        raise PageProjectionInvariantViolation(
            f"page projection left max_tab_run_after={max_tab_run_after}, "
            f"above the TAB cap {_TAB_RUN_CAP}"
        )


class _Tree:
    """Parallel `node_order`/`node_types`/`parent_index` arrays, read into
    ordinal-indexed lookups once (same shape `page_pieces._ancestor_types`
    reads, generalized to full parent/children maps)."""

    def __init__(self, structural: dict[str, Any]):
        self.node_order: list[str] = list(structural.get("node_order", ()))
        self.node_types: list[str] = list(structural.get("node_types", ()))
        self.parent_index: list[int] = list(structural.get("parent_index", ()))
        self.ordinal_by_node_id: dict[str, int] = {
            node_id: i for i, node_id in enumerate(self.node_order)
        }

        ordinal_by_open_event_index: dict[int, int] = {}
        ordinal = 0
        for event in structural.get("events", ()):
            if event["op"] == "open":
                ordinal_by_open_event_index[event["index"]] = ordinal
                ordinal += 1

        self.parent_ordinal: list[int] = [
            ordinal_by_open_event_index[p] if p != -1 else -1 for p in self.parent_index
        ]
        self.children: dict[int, list[int]] = {}
        for i, parent in enumerate(self.parent_ordinal):
            self.children.setdefault(parent, []).append(i)

    def type_of(self, ordinal: int) -> str:
        return self.node_types[ordinal]

    def node_id_of(self, ordinal: int) -> str:
        return self.node_order[ordinal]


def _ink_node_ids(text_spans: Sequence[TextSpanAuthority]) -> frozenset[str]:
    """Node ids with non-whitespace ink OWNED on this page.

    "Owned" means exactly what `training_materializer._interleave_text`
    emits: the `ScalarPolicyRange`s `ScalarBoundaryProjector` derives from
    each span's `grants` (this page's `char_ranges`), never the span's
    full `text`. `_project_span` yields no range at all for a stretch no
    grant covers -- "not emitted, not guessed" -- so a span whose real
    characters live on another page contributes no ink here, agreeing by
    construction with what the encoded stream actually contains (see the
    module docstring's real-data defect note).
    """
    span_by_id = {span.text_span_id: span for span in text_spans}
    ink: set[str] = set()
    for policy_range in ScalarBoundaryProjector().project(text_spans):
        span = span_by_id.get(policy_range.text_span_id)
        if span is None:
            continue
        owned_text = span.text[policy_range.scalar_start : policy_range.scalar_end]
        if owned_text.strip():
            ink.add(policy_range.node_id)
    return frozenset(ink)


def _has_content_descendant(
    tree: _Tree, ordinal: int, present: set[int], ink_node_ids: frozenset[str]
) -> bool:
    stack = list(tree.children.get(ordinal, ()))
    while stack:
        child = stack.pop()
        if child not in present:
            continue
        if tree.node_id_of(child) in ink_node_ids:
            return True
        if tree.type_of(child) in _CONTENT_DESCENDANT_TYPES:
            return True
        stack.extend(tree.children.get(child, ()))
    return False


def _subtree(tree: _Tree, ordinal: int, present: set[int]) -> set[int]:
    """`ordinal` plus every present descendant -- the full set of node
    ordinals a drop of `ordinal` removes."""
    out = {ordinal}
    stack = list(tree.children.get(ordinal, ()))
    while stack:
        child = stack.pop()
        if child not in present:
            continue
        out.add(child)
        stack.extend(tree.children.get(child, ()))
    return out


def _keep_run_within_cap(
    tree: _Tree,
    children: list[int],
    restored: int,
    dropped_blocks: set[int],
    visually_empty: set[int],
) -> None:
    """A cell-floor restore of an empty paragraph can still land on a run
    the cap left full: breaks are neutral in the run, so `[e1, e2, e3,
    column_break, e4, e5]` restores `e5` right after the kept `e1..e3`
    (the only shape left in which a restore after the last survivor
    joins a kept run; the cell must end on `e5`, so the restore cannot be
    skipped). Drop the kept empty
    nearest to `restored` instead: the run stays at the cap, the break
    stays where it was, and the cell still ends with a paragraph.

    The walk skips exactly what the run count skips: breaks AND every
    non-block child (a `text_frame` directly in the cell is not in
    `_BLOCK_TYPES`, so the count never sees it -- stopping at it here
    let `[e1, e2, e3, text_frame, e4]` restore `e4` into a run of 4,
    v13b train build)."""
    run: list[int] = []
    for child in reversed(children[: children.index(restored)]):
        if (
            child in dropped_blocks
            or tree.type_of(child) not in _BLOCK_TYPES
            or tree.type_of(child) in ("page_break", "column_break")
        ):
            continue
        if child not in visually_empty:
            break
        run.append(child)
    if len(run) >= _EMPTY_RUN_CAP:
        dropped_blocks.add(run[0])


def _blocks(tree: _Tree, present: set[int]) -> list[int]:
    """Every present block-level ordinal, in document order -- a `table`
    or `list` node is one block among the page's top-level content, and
    each of ITS descendant paragraphs is a separate block of its own
    (grouped, below, by its real parent -- a cell's or list item's own
    run). One flat list serves both: trailing drop walks each story's
    share of it (`_story_of`) back to front regardless of container, so a
    story's true tail paragraph is
    found whether it sits directly under a section or inside the last
    cell of the last table; interior cap then groups what trailing drop
    left by real parent, which separates a section's own children from a
    cell's without a second pass."""
    return [o for o in sorted(present) if tree.type_of(o) in _BLOCK_TYPES]


def _story_of(tree: _Tree, ordinal: int) -> int:
    """The trailing-drop story `ordinal` belongs to: its nearest
    `footnote`/`endnote` ancestor, else -1 (the section body, headers,
    footers and text frames -- one story, exactly as before notes were
    split out). Notes are document children ordered AFTER every section,
    so one walk over the whole page stopped inside the last note and
    never reached the body's own tail (bug-dec033-cohort-projection)."""
    parent = tree.parent_ordinal[ordinal]
    while parent != -1:
        if tree.type_of(parent) in _NOTE_TYPES:
            return parent
        parent = tree.parent_ordinal[parent]
    return -1


def _drop_withdrawn_whitespace_runs(
    tree: _Tree,
    present: set[int],
    dropped_ordinals: set[int],
    text_spans: Sequence[TextSpanAuthority],
) -> int:
    """V1e: every surviving `run` whose text is whitespace only and which
    owns no character on this page joins `dropped_ordinals` (mutated). It
    has no ink, so there is nothing to supervise, and V1a already granted
    every whitespace run framed by owned words; what is left sits at a tab,
    a field or a paragraph edge, where its empty wrapper is the stray
    withdrawn run the data-quality gate refuses. A run whose removal would
    join the tabs on its two sides into a run above the TAB cap stays (the
    cap already ran; 6 pages of the iteration-2 sample). Returns the count."""
    span_by_node = {span.node_id: span for span in text_spans}

    def tabs_beside(siblings: list[int], position: int, step: int) -> int:
        count = 0
        position += step
        while 0 <= position < len(siblings) and tree.type_of(siblings[position]) == "tab":
            count += 1
            position += step
        return count

    dropped = 0
    for ordinal in sorted(present):
        if tree.type_of(ordinal) != "run" or ordinal in dropped_ordinals:
            continue
        span = span_by_node.get(tree.node_id_of(ordinal))
        if span is None or not span.text or span.text.strip():
            continue
        if any(end > start for grant in span.grants for start, end in grant.char_ranges):
            continue
        siblings = [
            c
            for c in tree.children.get(tree.parent_ordinal[ordinal], ())
            if c in present and c not in dropped_ordinals
        ]
        position = siblings.index(ordinal)
        left, right = tabs_beside(siblings, position, -1), tabs_beside(siblings, position, 1)
        if left and right and left + right > _TAB_RUN_CAP:
            continue
        dropped_ordinals.add(ordinal)
        dropped += 1
    return dropped


def _collapse_withdrawn_run_chains(
    tree: _Tree,
    present: set[int],
    dropped_ordinals: set[int],
    text_spans: Sequence[TextSpanAuthority],
) -> tuple[int, int]:
    """DEC-055 V1b (pass 5, v16 target policy only): within one surviving
    `paragraph`/`hyperlink`, a maximal run of >= 2 consecutive surviving
    `run` children none of which owns a character on this page keeps its
    FIRST run; the rest join `dropped_ordinals` (mutated) exactly like any
    other drop -- no token, no loss unit, their wrapper stays withdrawn on
    the kept run. Any other surviving child (a tab, a text run, a field)
    ends the chain. Returns `(chains collapsed, runs dropped)`."""
    owned = frozenset(
        span.node_id
        for span in text_spans
        if any(end > start for grant in span.grants for start, end in grant.char_ranges)
    )
    chains = 0
    collapsed = 0
    for ordinal in sorted(present):
        if tree.type_of(ordinal) not in ("paragraph", "hyperlink") or ordinal in dropped_ordinals:
            continue
        chain: list[int] = []
        children = [
            c for c in tree.children.get(ordinal, ()) if c in present and c not in dropped_ordinals
        ]
        for child in [*children, None]:
            if (
                child is not None
                and tree.type_of(child) == "run"
                and tree.node_id_of(child) not in owned
            ):
                chain.append(child)
                continue
            if len(chain) >= 2:
                chains += 1
                collapsed += len(chain) - 1
                dropped_ordinals.update(chain[1:])
            chain = []
    return chains, collapsed


def project(
    fragment: dict[str, Any],
    structural: dict[str, Any],
    text_spans: Sequence[TextSpanAuthority] = (),
    *,
    collapse_withdrawn_runs: bool = False,
    drop_withdrawn_whitespace_runs: bool = False,
) -> tuple[frozenset[str], dict[str, Any]]:
    """`(dropped_node_ids, ledger_row)` for one page under
    `s2-standalone-page-v1` (design §4, R4, R9).

    Runs BEFORE token emission: the caller removes `dropped_node_ids`
    from `present` (`encode_fragment.encode_standalone_fragment_with_origins`)
    and from supervised structure (`targets.present_supervised_event_indices`)
    so a dropped node carries no token and no loss unit at all.

    `collapse_withdrawn_runs` (DEC-055 V1b, `v16-targets.v1` only) adds
    pass 5 (:func:`_collapse_withdrawn_run_chains`) and two ledger keys;
    off, the result is exactly the pre-v16 one.

    `drop_withdrawn_whitespace_runs` (V1e, `v16-targets.v1` only) runs
    before pass 5: a surviving whitespace-only `run` that owns no character
    on this page is dropped (:func:`_drop_withdrawn_whitespace_runs`), plus
    one ledger key.
    """
    tree = _Tree(structural)
    present = {
        tree.ordinal_by_node_id[entry["node_id"]]
        for entry in fragment.get("slices", ())
        if entry["node_id"] in tree.ordinal_by_node_id
    }
    ink_node_ids = _ink_node_ids(text_spans)

    visually_empty = {
        ordinal
        for ordinal in present
        if tree.type_of(ordinal) == "paragraph"
        and not _has_content_descendant(tree, ordinal, present, ink_node_ids)
    }

    def kind(ordinal: int) -> str:
        node_type = tree.type_of(ordinal)
        if node_type == "paragraph":
            return "empty" if ordinal in visually_empty else "content"
        if node_type in ("page_break", "column_break"):
            return "break"
        return "content"

    blocks = _blocks(tree, present)
    block_kind = {ordinal: kind(ordinal) for ordinal in blocks}

    # 1. Trailing drop, per story: pop empties/breaks off each story's
    # tail; stop at content.
    trailing_dropped: set[int] = set()
    empty_trailing_dropped = 0
    trailing_breaks_dropped = 0
    by_story: dict[int, list[int]] = {}
    for ordinal in blocks:
        by_story.setdefault(_story_of(tree, ordinal), []).append(ordinal)
    for tail in by_story.values():
        while tail and block_kind[tail[-1]] in ("empty", "break"):
            ordinal = tail.pop()
            trailing_dropped.add(ordinal)
            if block_kind[ordinal] == "empty":
                empty_trailing_dropped += 1
            else:
                trailing_breaks_dropped += 1

    # 2. Interior cap: consecutive empty runs among one parent's surviving
    # block children (breaks are neutral -- skipped, not counted).
    remaining = [o for o in blocks if o not in trailing_dropped]
    by_parent: dict[int, list[int]] = {}
    for ordinal in remaining:
        by_parent.setdefault(tree.parent_ordinal[ordinal], []).append(ordinal)

    interior_dropped: set[int] = set()
    empty_interior_runs_capped = 0
    empty_interior_dropped = 0
    for siblings in by_parent.values():
        run: list[int] = []
        for ordinal in siblings:
            k = block_kind[ordinal]
            if k == "break":
                continue
            if k == "empty":
                run.append(ordinal)
                if len(run) == _EMPTY_RUN_CAP + 1:
                    empty_interior_runs_capped += 1
                if len(run) > _EMPTY_RUN_CAP:
                    interior_dropped.add(ordinal)
                    empty_interior_dropped += 1
            else:
                run = []

    dropped_blocks = trailing_dropped | interior_dropped

    # 3. Container floor: never drop every present direct child of one of
    # these containers -- restore the first dropped child.
    #
    # `cell` carries a second, stricter grammar requirement on top of that
    # (`grammar._close_allowed`, `last_child_is_paragraph`): CELL_CLOSE
    # needs its LAST child to be a paragraph, not merely SOME child to
    # survive. A cell tailed by `[table, empty paragraph]` or `[list,
    # empty paragraph]` loses only the trailing empty paragraph above --
    # the table/list survives, so the generic "all children dropped" floor
    # never fires, yet the cell's last surviving child is now the
    # table/list, not a paragraph. Restoring the LAST dropped paragraph
    # (not the first, and not an arbitrary child) is what fixes the tail
    # shape without reordering anything.
    #
    # Only a paragraph dropped AFTER the last surviving child can do that.
    # Restoring one from BEFORE it lands next to the 3 the cap kept (a run
    # of 4) and still leaves the cell's last child a non-paragraph -- a
    # restore that costs the grammar's whole `cap + 1` headroom and buys
    # nothing. A restore after the last survivor follows a non-paragraph
    # block and starts a run of 1 -- unless that block is a break or a
    # non-block (`text_frame`), which the run skips: `_keep_run_within_cap`
    # handles that case (the v13
    # train build measured 6 pages with a run of 4, every one in a `cell`;
    # the postcondition below refuses any page this reasoning missed).
    kept_for_container_floor = 0
    for ordinal in present:
        if tree.type_of(ordinal) not in _FLOOR_TYPES:
            continue
        children = [c for c in tree.children.get(ordinal, ()) if c in present]
        if not children:
            continue
        if tree.type_of(ordinal) == "cell":
            surviving = [c for c in children if c not in dropped_blocks]
            if surviving and tree.type_of(surviving[-1]) == "paragraph":
                continue
            last_surviving = surviving[-1] if surviving else -1
            dropped_paragraphs = [
                c
                for c in children
                if c in dropped_blocks
                and tree.type_of(c) == "paragraph"
                and c > last_surviving
            ]
            if dropped_paragraphs:
                restored = dropped_paragraphs[-1]
                dropped_blocks.discard(restored)
                kept_for_container_floor += 1
                if restored in visually_empty:
                    _keep_run_within_cap(tree, children, restored, dropped_blocks, visually_empty)
            continue
        dropped_children = [c for c in children if c in dropped_blocks]
        if dropped_children and len(dropped_children) == len(children):
            dropped_blocks.discard(dropped_children[0])
            kept_for_container_floor += 1

    dropped_ordinals: set[int] = set()
    for ordinal in dropped_blocks:
        dropped_ordinals |= _subtree(tree, ordinal, present)

    # 4. TAB cap: consecutive `tab` leaf children within one surviving
    # inline container (`paragraph` or `hyperlink` -- the AST schema's
    # only two `tab` parents, `inline_node`/`hyperlink_child`), in
    # event/child order. Each container's run is independent, matching
    # the composite grammar's `tab_run` (reset by ANY non-`TAB` token,
    # `composite_grammar.py:559`): a `HYPERLINK_OPEN`/`_CLOSE` already
    # resets the count there, so a hyperlink's own tabs are already a
    # separate run from its enclosing paragraph's -- this mirrors that
    # exactly rather than flattening across the boundary.
    tab_runs_capped = 0
    tabs_dropped = 0
    max_tab_run_after = 0
    for ordinal in present:
        if tree.type_of(ordinal) not in ("paragraph", "hyperlink") or ordinal in dropped_ordinals:
            continue
        children = [c for c in tree.children.get(ordinal, ()) if c in present]
        run = []
        surviving_run = 0
        for child in children:
            if tree.type_of(child) == "tab":
                run.append(child)
                if len(run) == _TAB_RUN_CAP + 1:
                    tab_runs_capped += 1
                if len(run) > _TAB_RUN_CAP:
                    dropped_ordinals.add(child)
                    tabs_dropped += 1
                # counted from what SURVIVES, not from `min(len(run), cap)`:
                # a cap that stopped dropping must show up here, not be
                # clamped into looking correct (the postcondition below is
                # only an instrument if it can read a violation).
                if child not in dropped_ordinals:
                    surviving_run += 1
                    max_tab_run_after = max(max_tab_run_after, surviving_run)
            else:
                run = []
                surviving_run = 0

    # `max_empty_run_after`: the longest surviving consecutive-empty run,
    # per parent, after every drop above -- must be <= 3.
    max_empty_run_after = 0
    for siblings in by_parent.values():
        run_len = 0
        for ordinal in siblings:
            if ordinal in dropped_blocks:
                continue
            if block_kind[ordinal] == "break":
                continue
            if block_kind[ordinal] == "empty":
                run_len += 1
                max_empty_run_after = max(max_empty_run_after, run_len)
            else:
                run_len = 0

    _assert_run_bounds(
        max_empty_run_after=max_empty_run_after, max_tab_run_after=max_tab_run_after
    )

    if drop_withdrawn_whitespace_runs:
        whitespace_dropped = _drop_withdrawn_whitespace_runs(
            tree, present, dropped_ordinals, text_spans
        )
    if collapse_withdrawn_runs:
        withdrawn_chains, withdrawn_collapsed = _collapse_withdrawn_run_chains(
            tree, present, dropped_ordinals, text_spans
        )

    dropped_node_ids = frozenset(tree.node_id_of(o) for o in dropped_ordinals)

    ledger_row = {
        "empty_trailing_dropped": empty_trailing_dropped,
        "empty_interior_runs_capped": empty_interior_runs_capped,
        "empty_interior_dropped": empty_interior_dropped,
        "tab_runs_capped": tab_runs_capped,
        "tabs_dropped": tabs_dropped,
        "kept_for_container_floor": kept_for_container_floor,
        "trailing_breaks_dropped": trailing_breaks_dropped,
        "max_empty_run_after": max_empty_run_after,
        "max_tab_run_after": max_tab_run_after,
        "dropped_node_ids": sorted(dropped_node_ids),
    }
    if collapse_withdrawn_runs:
        ledger_row["withdrawn_run_chains_collapsed"] = withdrawn_chains
        ledger_row["withdrawn_runs_collapsed"] = withdrawn_collapsed
    if drop_withdrawn_whitespace_runs:
        ledger_row["withdrawn_whitespace_runs_dropped"] = whitespace_dropped
    return dropped_node_ids, ledger_row
