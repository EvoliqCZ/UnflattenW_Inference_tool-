"""Opt-in chain policy `table-geometry.v0` (RATIFIED 2026-09-26, RFC004-ERRATA-004
draft; measured in `docs/runs/v15b-layout-diagnostic-handoff-2026-09-26.md`
§5).

A table's width is predicted on three independent surfaces -- the preferred
width `/layout/width/*` (typed_heads.table), the column grid
`/columns/*/width_mpt` (table_plan) and every cell's `/layout/width_mpt`
(typed_heads.row_cell) -- and nothing makes them agree: on the sealed v15b
cohort 28 of 31 absolute tables carry a preferred width under 0.8 x their
grid sum, only 12 % of cells match their spanned columns within +-10 %, and
21 of 56 grids are wider than the predicted text width. Both LibreOffice
(measured) and, by hypothesis, Word then render whichever surface wins in
their engine. This policy derives ONE geometry from the grid (or, opt-in,
from the cells), writes it to all three surfaces, and counts every edit so
the chain report can audit it. Precedent: `headless_defaults` (RFC004-
ERRATA-003 E3, "the grid is derived, not predicted").

Precondition: `headless_defaults.apply` already ran on the page (cells carry
a derived `/grid_column`, `/col_span`, `/row_span`; integers are clamped).
The policy never changes `/row_span` nor the width mode; since RFC004-
ERRATA-004 decision 7 (2026-09-26, "cap at 118 slots with a counted
refusal") it does clamp `/grid_column`/`/col_span` for a cell that would
otherwise need a `/columns/i/width_mpt` slot beyond `MAX_COLUMN_SLOTS`.

Rules, in order, per table (tables in `open_event_index` order):

1. columns: `ncols = max(grid_column + col_span)` over the table's cells (a
   table without cells is left alone). Column `i` is `/columns/i/width_mpt`
   when that is a positive integer; with `column_source == "cell_median"` it
   is instead the median `/layout/width_mpt` of the span-1 cells in that
   column whenever one positive sample exists. A missing or nonpositive
   column takes (a) that cell median, else (b) an equal share of
   `max(0, available - sum(known))`, floored at 50 mpt, where `available` =
   section page width - left margin - right margin - table indent. A column
   at a slot index >= `MAX_COLUMN_SLOTS` has no head slot (RFC004-ERRATA-004
   decision 7, "cap at 118 slots with a counted refusal", 2026-09-26): every
   cell's `/grid_column`/`/col_span` is first clamped so no cell needs a
   slot >= `MAX_COLUMN_SLOTS` -- a cell starting before the limit has its
   `col_span` truncated to end exactly at the last slot, a cell starting at
   or past the limit is moved onto the last slot with `col_span := 1` --
   and counted (`columns_capped` once per capped table,
   `columns_capped_slots_dropped` for the number of slots the cap removed).
   This is a repair of a shape the column head cannot address at all, not a
   guarantee of grid validity for every input: a row with more genuinely
   distinct cells than head slots can still land two clamped cells on the
   same slot, which the semantic validator still catches as
   `TABLE_GRID_OVERLAP` exactly as it would for any other overlapping grid
   -- this policy only stops the column head's slot limit itself from being
   the reason every such table is refused outright. Without a section page
   width (or when margins and indent leave nothing) the equal-share and
   overflow rules are skipped, so a column that would have needed the
   equal share leaves the table unresolved too.
2. overflow: when `sum(cols) > available + OVERFLOW_TOLERANCE_MPT`, every
   column is scaled by `(available + tolerance) / sum` (round half up, the
   rounding remainder lands on the widest column, every column stays >= 1
   mpt) so the sum equals the target exactly. Never scaled up. A percent
   table is instead rescaled (up or down) to `round(value_ppm / 1e6 *
   available)`, the share its own preferred width names.
3. the derived columns are written back to `/columns/i/width_mpt`.
4. preferred width: `absolute` -> `/layout/width/value_mpt := sum(cols)`
   (a predicted 0 -- Word's "no preferred width", legal and common in gold
   -- becomes the grid sum too, counted separately as `tblw_zero_to_grid`);
   `auto` and `percent` keep their value; a missing mode is counted and left.
5. every cell `/layout/width_mpt := sum(cols[grid_column : grid_column +
   col_span])`.
6. `autofit`: a table the policy edited in rules 1-5 is forced to fixed
   layout (`/layout/autofit := False`) -- the only layout mode whose result
   is the grid by contract (RFC001 §22.4, RFC002 §51); an untouched table
   keeps its predicted value.

Idempotent: a second `apply` changes nothing and counts nothing. Counts are
`table_geometry:<kind>` per edit, except `table_geometry:max_overflow_ppm`,
which is NOT a count: it holds the worst overflow `sum(cols) / (available +
tolerance)` seen on this page as an integer ppm (>= 1 000 000 whenever a grid
was scaled down; a page-level max, so a report that merges pages by summing
must take the max for keys ending in `_ppm`).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any

from unflatten.eval.s2_gate import sealed

POLICY_ID = "table-geometry.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-004.md"
COLUMN_SOURCES = ("grid", "cell_median")
#: `cell_median` since 2026-09-26 (owner, after the 100-page measurement in
#: docs/runs/column-source-2026-09-26: mean column-share error 0.047 vs 0.103).
DEFAULT_COLUMN_SOURCE = "cell_median"
#: 2 x Word's default 0.08" left/right cell margin (115.2 twips x 50 mpt x 2):
#: a gold table may legitimately protrude into the page margins by its cell
#: margins (the v15b gold Nitra table does, by 1.36 %). HYPOTHESIS about
#: Word's default; only LibreOffice was measured.
OVERFLOW_TOLERANCE_MPT = 11_520
#: `typed_head_config.K_TABLE_COLUMNS`, the table_plan head's `/columns/i/
#: width_mpt` slot count (pinned here so this module stays torch-free; a test
#: asserts equality with the registry constant).
MAX_COLUMN_SLOTS = 118
COUNT_PREFIX = "table_geometry:"
HEAD_ID = "policy." + POLICY_ID
RULES = (
    "columns_capped_to_slots.v0",
    "columns_from_grid_or_cells.v0",
    "grid_scaled_to_available.v0",
    "column_write_back.v0",
    "tblw_from_grid.v0",
    "cell_width_from_grid.v0",
    "autofit_forced_fixed.v0",
)

_MIN_EQUAL_SHARE_MPT = 50
_MAX_OVERFLOW_KEY = COUNT_PREFIX + "max_overflow_ppm"
#: Count keys that are page-level extrema, not counts: a report merging pages
#: takes their maximum, never their sum (`report._merged_counts`).
EXTREMUM_KEYS = frozenset({_MAX_OVERFLOW_KEY})


def policy_record(column_source: str = DEFAULT_COLUMN_SOURCE) -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    _check_column_source(column_source)
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "column_source": column_source,
        "overflow_tolerance_mpt": OVERFLOW_TOLERANCE_MPT,
        "max_column_slots": MAX_COLUMN_SLOTS,
        "slot_overflow": "cap_and_count",
        "rules": list(RULES),
    }


def apply(
    page: sealed.PageDecodeResult, *, column_source: str = DEFAULT_COLUMN_SOURCE
) -> tuple[sealed.PageDecodeResult, Counter[str]]:
    """`page` with the policy applied, and one count per edit kind (plus the
    page-level `max_overflow_ppm`, see the module docstring)."""
    _check_column_source(column_source)
    counts: Counter[str] = Counter()
    if page.node_states is None:
        return page, counts
    preds = _Predictions(page.typed_predictions)
    states = sorted(page.node_states, key=lambda s: s.open_event_index or 0)
    children: dict[str, list[str]] = {}
    for st in states:
        if st.parent_local_node_id is not None:
            children.setdefault(st.parent_local_node_id, []).append(st.local_node_id)
    kind = {st.local_node_id: st.node_type for st in states}
    parent = {st.local_node_id: st.parent_local_node_id for st in states}
    for table in (st.local_node_id for st in states if st.node_type == "table"):
        rows = [r for r in children.get(table, ()) if kind.get(r) == "row"]
        cells = [c for r in rows for c in children.get(r, ()) if kind.get(c) == "cell"]
        if not cells:
            counts[COUNT_PREFIX + "table_without_cells"] += 1
            continue
        available = _available(preds, table, parent, kind)
        if available is None:
            counts[COUNT_PREFIX + "section_geometry_missing"] += 1
        _fix_table(preds, table, cells, available, column_source, counts)
    edited = dataclasses.replace(page, typed_predictions=type(page.typed_predictions)(preds.out))
    return edited, counts


def _check_column_source(column_source: str) -> None:
    if column_source not in COLUMN_SOURCES:
        raise ValueError(f"column_source must be one of {COLUMN_SOURCES}, got {column_source!r}")


class _Predictions:
    """The page's typed predictions with `(node, pointer)` access; `set`
    reports whether it changed anything and adds a missing prediction under
    this policy's head id."""

    def __init__(self, predictions: tuple[Any, ...]) -> None:
        self.out: list[dict[str, Any]] = list(predictions)
        self._index = {(p["local_node_id"], p["pointer"]): i for i, p in enumerate(self.out)}

    def get(self, node: str, pointer: str) -> Any:
        i = self._index.get((node, pointer))
        return self.out[i]["value"] if i is not None else None

    def get_int(self, node: str, pointer: str) -> int | None:
        value = self.get(node, pointer)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def set(self, node: str, node_type: str, pointer: str, value: Any) -> bool:
        i = self._index.get((node, pointer))
        if i is None:
            self._index[(node, pointer)] = len(self.out)
            self.out.append({"head_id": HEAD_ID, "local_node_id": node, "node_type": node_type,
                             "pointer": pointer, "value": value})
            return True
        old = self.out[i]["value"]
        if type(old) is type(value) and old == value:
            return False
        self.out[i] = {**self.out[i], "value": value}
        return True


def _available(
    preds: _Predictions, table: str, parent: dict[str, str | None], kind: dict[str, str]
) -> int | None:
    """Text width left for `table`: its ancestor section's page width minus
    both margins minus the table indent; `None` without a positive page
    width (or without a section at all)."""
    node: str | None = parent.get(table)
    while node is not None and kind.get(node) != "section":
        node = parent.get(node)
    if node is None:
        return None
    page_width = preds.get_int(node, "/page/width_mpt")
    if page_width is None or page_width <= 0:
        return None
    available = (page_width - (preds.get_int(node, "/margins/left_mpt") or 0)
                 - (preds.get_int(node, "/margins/right_mpt") or 0)
                 - (preds.get_int(table, "/layout/indent_mpt") or 0))
    return available if available > 0 else None


def _cap_spans(
    preds: _Predictions,
    cells: list[str],
    spans: dict[str, tuple[int, int]],
    counts: Counter[str],
) -> dict[str, tuple[int, int]]:
    """RFC004-ERRATA-004 decision 7: clamp every cell so none needs a
    `/columns/i/width_mpt` slot the column head does not have. A cell
    starting before `MAX_COLUMN_SLOTS` has its `col_span` truncated to end
    exactly at the last slot; a cell starting at or past the limit is moved
    onto the last slot with `col_span := 1`. `spans` unchanged (same object)
    when nothing needs it."""
    if max(column + span for column, span in spans.values()) <= MAX_COLUMN_SLOTS:
        return spans
    last = MAX_COLUMN_SLOTS - 1
    dropped = max(column + span for column, span in spans.values()) - MAX_COLUMN_SLOTS
    capped: dict[str, tuple[int, int]] = {}
    for cell in cells:
        column, span = spans[cell]
        new_column = min(column, last)
        new_span = min(span, MAX_COLUMN_SLOTS - new_column)
        capped[cell] = (new_column, new_span)
        if (new_column, new_span) != (column, span):
            preds.set(cell, "cell", "/grid_column", new_column)
            preds.set(cell, "cell", "/col_span", new_span)
    counts[COUNT_PREFIX + "columns_capped"] += 1
    counts[COUNT_PREFIX + "columns_capped_slots_dropped"] += dropped
    return capped


def _fix_table(
    preds: _Predictions, table: str, cells: list[str], available: int | None,
    column_source: str, counts: Counter[str],
) -> None:
    spans = {c: (max(0, preds.get_int(c, "/grid_column") or 0),
                 max(1, preds.get_int(c, "/col_span") or 1)) for c in cells}
    spans = _cap_spans(preds, cells, spans, counts)
    ncols = max(column + span for column, span in spans.values())
    samples: dict[int, list[int]] = {}
    for cell, (column, span) in spans.items():
        width = preds.get_int(cell, "/layout/width_mpt")
        if span == 1 and width is not None and width > 0:
            samples.setdefault(column, []).append(width)

    # rule 1 -- one column width per applicable column
    cols: list[int | None] = []
    from_cells = 0
    for i in range(ncols):
        grid = preds.get_int(table, f"/columns/{i}/width_mpt")
        known = grid if grid is not None and grid > 0 else None
        if column_source == "cell_median" and i in samples:
            cols.append(_median(samples[i]))
        elif known is not None:
            cols.append(known)
        elif i in samples:
            cols.append(_median(samples[i]))
            from_cells += 1
        else:
            cols.append(None)
    unknown = [i for i, value in enumerate(cols) if value is None]
    if unknown and available is None:
        # Nothing is written to this table, so nothing else is counted for it
        # (a 665-column row otherwise reported 585 "columns from cells").
        counts[COUNT_PREFIX + "column_unresolved"] += 1
        return
    counts[COUNT_PREFIX + "column_from_cells"] += from_cells
    if unknown:
        known_sum = sum(value for value in cols if value is not None)
        share = max(_MIN_EQUAL_SHARE_MPT, max(0, available - known_sum) // len(unknown))
        for i in unknown:
            cols[i] = share
            counts[COUNT_PREFIX + "column_equal_share"] += 1
    widths: list[int] = [value for value in cols if value is not None]

    # rule 2 -- overflow (or the percent share)
    mode = preds.get(table, "/layout/width/mode")
    ppm = preds.get_int(table, "/layout/width/value_ppm")
    total = sum(widths)
    if mode == "percent" and available is not None and ppm is not None and ppm > 0:
        target = _round_div(ppm * available, 1_000_000)
        if target < len(widths):
            # A share too small to give every column its 1 mpt: nothing is
            # rescaled (the model's own grid stays), only counted.
            counts[COUNT_PREFIX + "percent_share_degenerate"] += 1
        elif total != target:
            widths = _rescale(widths, target)
            counts[COUNT_PREFIX + "grid_scaled_to_percent"] += 1
    elif available is not None and total > available + OVERFLOW_TOLERANCE_MPT:
        target = available + OVERFLOW_TOLERANCE_MPT
        widths = _rescale(widths, target)
        counts[COUNT_PREFIX + "grid_scaled_to_available"] += 1
        counts[_MAX_OVERFLOW_KEY] = max(counts[_MAX_OVERFLOW_KEY],
                                        _round_div(total * 1_000_000, target))
    total = sum(widths)

    # rule 3 -- write the columns back
    edited = False
    for i, width in enumerate(widths):
        if preds.set(table, "table", f"/columns/{i}/width_mpt", width):
            counts[COUNT_PREFIX + "column_width_changed"] += 1
            edited = True

    # rule 4 -- preferred width
    if mode == "absolute":
        old = preds.get_int(table, "/layout/width/value_mpt")
        if preds.set(table, "table", "/layout/width/value_mpt", total):
            what = ("tblw_from_grid" if old else
                    "tblw_zero_to_grid" if old is not None else "tblw_missing_to_grid")
            counts[COUNT_PREFIX + what] += 1
            edited = True
    elif mode not in ("auto", "percent"):
        counts[COUNT_PREFIX + "width_mode_missing"] += 1

    # rule 5 -- cells
    for cell, (column, span) in spans.items():
        if preds.set(cell, "cell", "/layout/width_mpt", sum(widths[column:column + span])):
            counts[COUNT_PREFIX + "cell_width_from_grid"] += 1
            edited = True

    # rule 6 -- fixed layout for an edited table
    if edited and preds.set(table, "table", "/layout/autofit", False):
        counts[COUNT_PREFIX + "autofit_forced_fixed"] += 1


def _median(values: list[int]) -> int:
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) // 2


def _round_div(numerator: int, denominator: int) -> int:
    """`numerator / denominator` rounded half up, in integers."""
    return (2 * numerator + denominator) // (2 * denominator)


def _rescale(widths: list[int], target: int) -> list[int]:
    """`widths` scaled so they sum to `target` exactly: round half up per
    column, never below 1 mpt, the remainder on the (first) widest column."""
    if target < len(widths):
        raise ValueError(f"target {target} cannot give {len(widths)} columns 1 mpt each")
    total = sum(widths)
    scaled = [max(1, _round_div(width * target, total)) for width in widths]
    widest = widths.index(max(widths))
    scaled[widest] += target - sum(scaled)
    return scaled
