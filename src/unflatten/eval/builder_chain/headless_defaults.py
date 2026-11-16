"""Opt-in chain policy `headless-defaults.v0` (RATIFIED, RFC004-ERRATA-003,
DEC-044; owner 2026-09-25 "cesta 1").

`ast_builder.build_strict` never imputes (RFC004 §33.6) and the derivation
registry is closed (DEC-037). Without this policy no v15b page reaches a
strict build: every run lacks `/style/font_family` and `/language`, every
image `/alt_text` -- properties with no typed head -- and every table's
per-cell `grid_column`/`col_span`/`row_span` are independent head outputs
that do not form one grid. This policy edits a decoded page's typed
predictions BEFORE the chain assembles and builds it, only when the caller
asks for it, and counts every edit; `report.build_report` records the policy
and the counts in the hashed report. Nothing here is silent.

Rules, in order:

1. decode projection (`typed_projection`, the rules `typed_decode` applies to
   every new decode; here for sealed artefacts decoded before them): the
   string "null" at a colour pointer -> JSON null, integers clamped into the
   schema range, first-line/hanging kept as one net offset, section
   `/line_numbering`/`/page_borders` dropped (no presence head);
2. list `/marker/format` "custom_text" -> "bullet" (its glyph `/marker/text`
   has no head; the schema requires it for custom_text);
3. defaults where the model gave nothing: run `/style/font_family` =
   "Calibri", run `/language` = null, list `/marker/suffix` = "." for an
   ordered list else null, image `/alt_text` = null;
4. table grid: each cell's `grid_column` is DERIVED from its row order and the
   spans (first free column), a `row_span` that would overlap a later cell is
   shortened, and a short row's last cell is widened to the table width.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any

from unflatten.eval.s2_gate import sealed
from unflatten.model import typed_projection as tp

POLICY_ID = "headless-defaults.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-003.md"

DEFAULT_FONT_FAMILY = "Calibri"
_HEAD_ID = "policy." + POLICY_ID
_COLOUR_SUFFIXES = ("/color", "/background", "/highlight")
_IMAGE_TYPES = ("inline_image", "floating_image")


def policy_record() -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "defaults": {
            "run:/style/font_family": DEFAULT_FONT_FAMILY,
            "run:/language": None,
            "list:/marker/suffix": {"ordered": ".", "unordered": None},
            "image:/alt_text": None,
            "list:/marker/format": {"custom_text": "bullet"},
        },
        "rules": [
            "decode_projection.v0", "marker_custom_text_to_bullet.v0",
            "headless_defaults.v0", "table_grid_first_free_column.v0",
        ],
    }


def apply(page: sealed.PageDecodeResult) -> tuple[sealed.PageDecodeResult, Counter[str]]:
    """`page` with the policy applied, and one count per edit kind."""
    counts: Counter[str] = Counter()
    if page.node_states is None:
        return page, counts
    predictions = _project(list(page.typed_predictions), counts)
    predictions = _defaults(page, predictions, counts)
    predictions = _table_grids(page, predictions, counts)
    edited = dataclasses.replace(page, typed_predictions=type(page.typed_predictions)(predictions))
    return edited, counts


def _project(predictions: list[dict[str, Any]], counts: Counter[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in predictions:
        pointer, value = p["pointer"], p["value"]
        if tp.optional_objects_without_presence_head(p["node_type"], (pointer,)):
            counts["projection:optional_object_dropped"] += 1
            continue
        if value == tp.NULL_COLOUR_LABEL and pointer.endswith(_COLOUR_SUFFIXES):
            p = {**p, "value": None}
            counts["projection:null_colour"] += 1
        elif isinstance(value, int) and not isinstance(value, bool):
            clamped = tp.clamp_integer(p["node_type"], pointer, value)
            if clamped != value:
                p = {**p, "value": clamped}
                counts["projection:integer_clamp"] += 1
        elif pointer == "/marker/format" and value == "custom_text":
            p = {**p, "value": "bullet"}
            counts["marker:custom_text_to_bullet"] += 1
        out.append(p)
    by_node: dict[str, dict[str, int]] = {}
    for i, p in enumerate(out):
        if p["pointer"] in (tp.FIRST_LINE, tp.HANGING):
            by_node.setdefault(p["local_node_id"], {})[p["pointer"]] = i
    for positions in by_node.values():
        values = {pointer: out[i]["value"] for pointer, i in positions.items()}
        if tp.one_first_line_offset(values):
            for pointer, i in positions.items():
                out[i] = {**out[i], "value": values[pointer]}
            counts["projection:first_line_net"] += 1
    return out


def _defaults(
    page: sealed.PageDecodeResult, predictions: list[dict[str, Any]], counts: Counter[str]
) -> list[dict[str, Any]]:
    given = {(p["local_node_id"], p["pointer"]) for p in predictions}
    list_kind = {
        p["local_node_id"]: p["value"]
        for p in predictions
        if p["node_type"] == "list" and p["pointer"] == "/list_kind"
    }
    added: list[dict[str, Any]] = []

    def add(node_id: str, node_type: str, pointer: str, value: Any) -> None:
        if (node_id, pointer) not in given:
            added.append({"head_id": _HEAD_ID, "local_node_id": node_id,
                          "node_type": node_type, "pointer": pointer, "value": value})
            counts[f"default:{node_type}:{pointer}"] += 1

    for state in page.node_states or ():
        if state.node_type == "run":
            add(state.local_node_id, "run", "/style/font_family", DEFAULT_FONT_FAMILY)
            add(state.local_node_id, "run", "/language", None)
        elif state.node_type == "list":
            suffix = "." if list_kind.get(state.local_node_id) == "ordered" else None
            add(state.local_node_id, "list", "/marker/suffix", suffix)
        elif state.node_type in _IMAGE_TYPES:
            add(state.local_node_id, state.node_type, "/alt_text", None)
    return [*predictions, *added]


def _table_grids(
    page: sealed.PageDecodeResult, predictions: list[dict[str, Any]], counts: Counter[str]
) -> list[dict[str, Any]]:
    preds = list(predictions)
    index = {(p["local_node_id"], p["pointer"]): i for i, p in enumerate(preds)}
    states = sorted(page.node_states or (), key=lambda s: s.open_event_index or 0)
    children: dict[str, list[str]] = {}
    for st in states:
        if st.parent_local_node_id is not None:
            children.setdefault(st.parent_local_node_id, []).append(st.local_node_id)
    kind = {st.local_node_id: st.node_type for st in states}

    def get(node: str, pointer: str, default: int) -> int:
        i = index.get((node, pointer))
        value = preds[i]["value"] if i is not None else default
        return value if isinstance(value, int) and not isinstance(value, bool) else default

    for table in (st.local_node_id for st in states if st.node_type == "table"):
        rows = [r for r in children.get(table, []) if kind.get(r) == "row"]
        owner: dict[tuple[int, int], str] = {}  # (row, column) -> cell
        spans: dict[str, list[int]] = {}  # cell -> [column, col_span, row_span, start_row]
        last_cell: dict[int, str] = {}
        for r, row in enumerate(rows):
            column = 0
            for cell in (c for c in children.get(row, []) if kind.get(c) == "cell"):
                while (r, column) in owner:
                    column += 1
                span = max(1, get(cell, "/col_span", 1))
                # This row's cell keeps its width: an earlier row_span reaching
                # into these columns is cut to end before this row.
                for c in range(column, column + span):
                    earlier = owner.get((r, c))
                    if earlier is not None:
                        col0, width0, down0, start0 = spans[earlier]
                        for k in range(r - start0, down0):
                            for cc in range(col0, col0 + width0):
                                owner.pop((start0 + k, cc), None)
                        spans[earlier][2] = r - start0
                down = max(1, min(get(cell, "/row_span", 1), len(rows) - r))
                for k in range(down):
                    for c in range(column, column + span):
                        owner[(r + k, c)] = cell
                spans[cell] = [column, span, down, r]
                last_cell[r] = cell
                column += span
        width = max((c + 1 for _, c in owner), default=0)
        for r, cell in last_cell.items():
            col0, width0, down0, start0 = spans[cell]
            extra = 0
            while col0 + width0 + extra < width and (r, col0 + width0 + extra) not in owner:
                extra += 1
            if extra and down0 == 1:
                for c in range(col0 + width0, col0 + width0 + extra):
                    owner[(r, c)] = cell
                spans[cell][1] = width0 + extra
                counts["table_grid:short_row_widened"] += 1
        for cell, (column, span, down, _start) in spans.items():
            for pointer, value, what in (
                ("/grid_column", column, "grid_column_derived"),
                ("/col_span", span, "col_span_changed"),
                ("/row_span", down, "row_span_changed"),
            ):
                i = index.get((cell, pointer))
                if i is not None and preds[i]["value"] != value:
                    preds[i] = {**preds[i], "value": value}
                    counts[f"table_grid:{what}"] += 1
    return preds
