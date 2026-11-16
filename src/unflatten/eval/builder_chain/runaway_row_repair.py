"""Opt-in chain policy `runaway-row-repair.v0` (PROPOSED, owner 2026-09-27;
draft `docs/rfc/RFC004-ERRATA-008.md`).

`table_geometry.v0`'s own docstring names the failure this repairs: a row
whose cell count is far beyond every other row's (a footer table with one
row of ~130 mostly-empty cells, measured on prod) exceeds `MAX_COLUMN_SLOTS`
(118), gets clamped onto 118 slots, and two clamped cells then land on the
same slot -- `TABLE_GRID_OVERLAP`, a shape the column head cannot address at
all. This policy removes the shape before `table-geometry.v0` ever sees it:
a runaway row's own extra cells, when every one of them is EMPTY (no run,
no nested content at all in the decoded tree), are dropped so the row falls
back to the table's modal cell count.

Scope: `table_geometry` runs on the pre-assembly `sealed.PageDecodeResult`
(the section's typed predictions carry no assembled tree yet), so this
policy runs on the same pre-assembly shape, in the same chain slot -- after
`headless-defaults.v0` (whose `_table_grids` rule already assigned every
cell's `/grid_column`, so the surviving cells' columns stay 0..modal-1,
already contiguous and correct with no further adjustment) and strictly
BEFORE `table-geometry.v0` (module docstring's own ordering requirement).

A cell is not a token-annotation, it is tokens: `PageDecodeResult.
accepted_tokens` is what `hy.from_token_ids` actually replays into the
assembled tree (`sealed_adapter.page_hypothesis`); `node_states`/
`typed_predictions` are only per-anchor overlays on top of it. Dropping a
cell from the built document therefore means splicing its own token span
(`[open_event_index, close_event_index]`, or the single `leaf_event_index`
for a childless leaf cell) out of `accepted_tokens`, dropping its own
`NodeStateProjection` and any `typed_predictions`/`region_predictions`
naming it, and shifting every other node's open/leaf/resume/close event
index down by the number of tokens removed before it. Legal only because
the removed cells are proven empty: no other node's anchor can fall inside
an empty cell's own span.

Rule, per table (`open_event_index` order, `table_geometry`'s own rule 0),
per row with >= 2 rows in the table:

1. `modal` = the most common cell count among the table's OTHER rows (ties
   broken toward the smallest count, deterministic); a row is a candidate
   when its own cell count is `>= max(min_factor * modal, modal +
   min_offset)` (policy constants, default `min_factor=2`, `min_offset=4`);
2. `extra` = the candidate row's cells from index `modal` onward (in row
   order, i.e. the TRAILING cells); when every one of them has zero
   children in `node_states` (no run, no nested content whatsoever) they
   are all removed (`rows_repaired` once, `cells_removed` += len(extra));
   when any one of them has a child, the row is left untouched
   (`refused_nonempty`);
3. a table with exactly one row is refused outright (`refused_no_modal`,
   once per such table): there are no "other rows" to compute a modal
   from. A table with zero rows is skipped silently (nothing to repair).

Idempotent: a second `apply` finds no row past the (now corrected) modal
count and counts nothing. Applies to a table anywhere on the page (body,
header, footer stories alike) -- the rule reads `node_states` structurally,
never the story it sits under.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any

from unflatten.eval.s2_gate.sealed import NodeStateProjection, PageDecodeResult

POLICY_ID = "runaway-row-repair.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-008.md"
COUNT_PREFIX = "runaway_row_repair:"
DEFAULT_MIN_FACTOR = 2
DEFAULT_MIN_OFFSET = 4
RULES = (
    "modal_row_cell_count_from_other_rows.v0",
    "trailing_empty_cells_removed.v0",
    "nonempty_extra_refused.v0",
    "single_row_table_refused.v0",
)


def policy_record(
    *, min_factor: int = DEFAULT_MIN_FACTOR, min_offset: int = DEFAULT_MIN_OFFSET
) -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "min_factor": min_factor,
        "min_offset": min_offset,
        "rules": list(RULES),
    }


def apply(
    page: PageDecodeResult,
    *,
    min_factor: int = DEFAULT_MIN_FACTOR,
    min_offset: int = DEFAULT_MIN_OFFSET,
) -> tuple[PageDecodeResult, Counter[str]]:
    """`page` with every runaway row's trailing empty cells removed
    (tokens, node states and predictions kept consistent), and one count
    per outcome."""
    counts: Counter[str] = Counter()
    if page.node_states is None or page.accepted_tokens is None:
        return page, counts
    states = sorted(page.node_states, key=lambda s: s.open_event_index or 0)
    kind = {s.local_node_id: s.node_type for s in states}
    children: dict[str, list[str]] = {}
    for s in states:
        if s.parent_local_node_id is not None:
            children.setdefault(s.parent_local_node_id, []).append(s.local_node_id)

    to_remove: list[str] = []
    for table in (s.local_node_id for s in states if s.node_type == "table"):
        rows = [r for r in children.get(table, ()) if kind.get(r) == "row"]
        if len(rows) < 2:
            if rows:
                counts[COUNT_PREFIX + "refused_no_modal"] += 1
            continue
        row_cells = {r: [c for c in children.get(r, ()) if kind.get(c) == "cell"] for r in rows}
        for row in rows:
            others = [len(row_cells[r]) for r in rows if r != row]
            modal = _mode(others)
            cells = row_cells[row]
            if len(cells) < max(min_factor * modal, modal + min_offset):
                continue
            extra = cells[modal:]
            if all(not children.get(cell) for cell in extra):
                to_remove.extend(extra)
                counts[COUNT_PREFIX + "rows_repaired"] += 1
                counts[COUNT_PREFIX + "cells_removed"] += len(extra)
            else:
                counts[COUNT_PREFIX + "refused_nonempty"] += 1
    if not to_remove:
        return page, counts
    return _remove_nodes(page, to_remove), counts


def _mode(values: list[int]) -> int:
    best_count, best_value = -1, 0
    for value in sorted(set(values)):
        count = values.count(value)
        if count > best_count:
            best_count, best_value = count, value
    return best_value


def _span(state: NodeStateProjection) -> tuple[int, int]:
    positions = [p for p in (state.open_event_index, state.leaf_event_index,
                              state.resume_event_index, state.close_event_index)
                 if p is not None]
    return min(positions), max(positions)


def _remove_nodes(page: PageDecodeResult, node_ids: list[str]) -> PageDecodeResult:
    remove_set = set(node_ids)
    by_id = {s.local_node_id: s for s in page.node_states or ()}
    spans = sorted({_span(by_id[node_id]) for node_id in node_ids}, reverse=True)
    tokens = list(page.accepted_tokens or ())
    for start, end in spans:
        del tokens[start:end + 1]

    def shift(position: int | None) -> int | None:
        if position is None:
            return None
        removed_before = sum(end - start + 1 for start, end in spans if end < position)
        return position - removed_before

    kept_states = tuple(
        dataclasses.replace(
            state,
            open_event_index=shift(state.open_event_index),
            leaf_event_index=shift(state.leaf_event_index),
            resume_event_index=shift(state.resume_event_index),
            close_event_index=shift(state.close_event_index),
        )
        for state in page.node_states or ()
        if state.local_node_id not in remove_set
    )
    kept_predictions = tuple(
        p for p in page.typed_predictions if p["local_node_id"] not in remove_set
    )
    kept_regions = tuple(
        r for r in page.region_predictions if r.local_node_id not in remove_set
    )
    return dataclasses.replace(
        page,
        accepted_tokens=tuple(tokens),
        node_states=kept_states,
        typed_predictions=kept_predictions,
        region_predictions=kept_regions,
    )
