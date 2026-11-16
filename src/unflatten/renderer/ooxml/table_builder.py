"""`w:tbl` builder (RFC002 "Tables"/"Rows"/"Cells"/"Spans").

Building a table happens in two steps: `_expand_grid` turns the AST's
authored `row.cells` (which list only real, authored cells — a
`row_span > 1` cell is authored once, at its origin row) into, per
row, the *full* left-to-right sequence of grid positions including
synthesized `w:vMerge` continuation cells; `_append_table` then walks
that expanded grid to emit XML. The AST semantic validator already
guarantees the authored grid has no overlaps/gaps and all spans are in
range — this module does not re-validate that, only expands it.

Cell content is rendered through the caller-supplied `append_block_fn`
(`document_builder._append_block`) so paragraphs/lists/breaks — and,
since `word-visual-v1` (DEC-003 §1: 495 documents were discarded whole
at import for want of it), nested tables — inside a cell reuse the
exact same mapping as top-level blocks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lxml import etree

from unflatten.ast.issues import Issue
from unflatten.renderer.api import FontPolicy
from unflatten.renderer.units import mpt_to_eighth_point, mpt_to_twip, ppm_to_word_pct
from unflatten.renderer.xml_writer import qn

_BORDER_STYLE_MAP = {
    "none": "none",
    "single": "single",
    "double": "double",
    "dotted": "dotted",
    "dashed": "dashed",
}
_ALIGNMENT_MAP = {"left": "left", "center": "center", "right": "right"}
_VALIGN_MAP = {"top": "top", "center": "center", "bottom": "bottom"}

AppendBlockFn = Callable[
    [etree._Element, dict[str, Any], str, FontPolicy, set[str], Any], tuple[list[Issue], bool]
]


@dataclass(frozen=True)
class _GridCell:
    grid_column: int
    col_span: int
    is_continuation: bool
    source: dict[str, Any] | None
    origin_id: str | None
    # Index of `source` within its row's *authored* `cells` list (only
    # meaningful when `not is_continuation`) — carried explicitly so
    # `_append_row` never has to re-derive it with an O(n) `.index()`
    # lookup per cell (O(n^2) per row for a wide table).
    authored_index: int | None = None


def _expand_grid(rows: list[dict[str, Any]]) -> list[list[_GridCell]]:
    """Build the full occupancy grid, one `list[_GridCell]` per row,
    sorted left-to-right by `grid_column`, including synthesized
    `w:vMerge` continuation positions (RFC002 "Spans").
    """
    pending: dict[int, tuple[int, int, str]] = {}  # grid_column -> (rows_left, col_span, origin_id)
    expanded: list[list[_GridCell]] = []
    for row in rows:
        row_cells: list[_GridCell] = []
        still_pending: dict[int, tuple[int, int, str]] = {}
        for grid_column, (rows_left, col_span, origin_id) in pending.items():
            row_cells.append(_GridCell(grid_column, col_span, True, None, origin_id))
            if rows_left > 1:
                still_pending[grid_column] = (rows_left - 1, col_span, origin_id)
        pending = still_pending

        for authored_index, cell in enumerate(row["cells"]):
            row_cells.append(
                _GridCell(
                    cell["grid_column"],
                    cell["col_span"],
                    False,
                    cell,
                    None,
                    authored_index=authored_index,
                )
            )
            if cell["row_span"] > 1:
                pending[cell["grid_column"]] = (cell["row_span"] - 1, cell["col_span"], cell["id"])

        row_cells.sort(key=lambda c: c.grid_column)
        expanded.append(row_cells)
    return expanded


def append_table(
    body: etree._Element,
    node: dict[str, Any],
    path: str,
    font_policy: FontPolicy,
    used_fonts: set[str],
    ctx: Any,
    append_block_fn: AppendBlockFn,
) -> list[Issue]:
    """Append `w:tbl` for AST `table` node `node` to `body`."""
    issues: list[Issue] = []
    layout = node["layout"]
    tbl = etree.SubElement(body, qn("w", "tbl"))
    tbl_pr = etree.SubElement(tbl, qn("w", "tblPr"))
    # CT_TblPrBase element order (wml.xsd xsd:sequence): tblW, jc,
    # tblInd, tblBorders, shd, tblLayout.
    _append_table_width(tbl_pr, layout["width"])
    jc = etree.SubElement(tbl_pr, qn("w", "jc"))
    jc.set(qn("w", "val"), _ALIGNMENT_MAP[layout["alignment"]])
    if layout["indent_mpt"]:
        tbl_ind = etree.SubElement(tbl_pr, qn("w", "tblInd"))
        tbl_ind.set(qn("w", "w"), str(mpt_to_twip(layout["indent_mpt"])))
        tbl_ind.set(qn("w", "type"), "dxa")
    _append_table_borders(tbl_pr, node["style"].get("borders"))
    _append_shading(tbl_pr, node["style"].get("background"))
    tbl_layout = etree.SubElement(tbl_pr, qn("w", "tblLayout"))
    tbl_layout.set(qn("w", "type"), "autofit" if layout["autofit"] else "fixed")

    grid = etree.SubElement(tbl, qn("w", "tblGrid"))
    for column in node["columns"]:
        grid_col = etree.SubElement(grid, qn("w", "gridCol"))
        grid_col.set(qn("w", "w"), str(mpt_to_twip(column["width_mpt"])))

    expanded_rows = _expand_grid(node["rows"])
    for row_index, (row, grid_row) in enumerate(zip(node["rows"], expanded_rows, strict=True)):
        row_path = f"{path}/rows/{row_index}"
        issues.extend(
            _append_row(tbl, row, grid_row, row_path, font_policy, used_fonts, ctx, append_block_fn)
        )
    return issues


def _append_table_width(tbl_pr: etree._Element, width: dict[str, Any]) -> None:
    tbl_w = etree.SubElement(tbl_pr, qn("w", "tblW"))
    mode = width["mode"]
    if mode == "absolute":
        tbl_w.set(qn("w", "type"), "dxa")
        tbl_w.set(qn("w", "w"), str(mpt_to_twip(width["value_mpt"])))
    elif mode == "percent":
        # RFC001 ERRATA-002 E7: percent mode carries the ppm ratio
        # (0..1_000_000) in the dedicated `value_ppm` field.
        tbl_w.set(qn("w", "type"), "pct")
        tbl_w.set(qn("w", "w"), str(ppm_to_word_pct(width["value_ppm"])))
    else:
        tbl_w.set(qn("w", "type"), "auto")
        tbl_w.set(qn("w", "w"), "0")


def _append_shading(parent: etree._Element, background: str | None) -> None:
    if background is None:
        return
    shd = etree.SubElement(parent, qn("w", "shd"))
    shd.set(qn("w", "val"), "clear")
    shd.set(qn("w", "color"), "auto")
    shd.set(qn("w", "fill"), background.lstrip("#") if background != "auto" else "auto")


def _append_table_borders(tbl_pr: etree._Element, borders: dict[str, Any] | None) -> None:
    if borders is None:
        return
    tbl_borders = etree.SubElement(tbl_pr, qn("w", "tblBorders"))
    for side in ("top", "left", "bottom", "right"):
        border = borders.get(side)
        if not border:
            continue
        _append_border_side(tbl_borders, side, border)


def _append_border_side(parent: etree._Element, side: str, border: dict[str, Any]) -> None:
    side_el = etree.SubElement(parent, qn("w", side))
    side_el.set(qn("w", "val"), _BORDER_STYLE_MAP[border["style"]])
    side_el.set(qn("w", "sz"), str(mpt_to_eighth_point(border["width_mpt"])))
    side_el.set(qn("w", "space"), "0")
    color = border["color"]
    side_el.set(qn("w", "color"), color.lstrip("#") if color != "auto" else "auto")


def _append_row(
    tbl: etree._Element,
    row: dict[str, Any],
    grid_row: list[_GridCell],
    path: str,
    font_policy: FontPolicy,
    used_fonts: set[str],
    ctx: Any,
    append_block_fn: AppendBlockFn,
) -> list[Issue]:
    issues: list[Issue] = []
    if row.get("id") is not None:
        ctx.rendered_node_ids.add(row["id"])
    tr = etree.SubElement(tbl, qn("w", "tr"))
    tr_pr = etree.SubElement(tr, qn("w", "trPr"))
    # CT_TrPrBase element order (wml.xsd xsd:sequence): cantSplit,
    # trHeight, tblHeader.
    if row["layout"]["cant_split"]:
        etree.SubElement(tr_pr, qn("w", "cantSplit"))
    _append_row_height(tr_pr, row["layout"]["height"])
    if row["layout"]["repeat_header"]:
        etree.SubElement(tr_pr, qn("w", "tblHeader"))

    for grid_cell in grid_row:
        if grid_cell.is_continuation:
            _append_continuation_cell(tr, grid_cell, ctx)
        else:
            cell_path = f"{path}/cells/{grid_cell.authored_index}"
            issues.extend(
                _append_cell(
                    tr, grid_cell.source, cell_path, font_policy, used_fonts, ctx, append_block_fn
                )
            )
    return issues


def _append_row_height(tr_pr: etree._Element, height: dict[str, Any]) -> None:
    mode = height["mode"]
    if mode == "auto":
        return
    tr_height = etree.SubElement(tr_pr, qn("w", "trHeight"))
    tr_height.set(qn("w", "val"), str(mpt_to_twip(height["value_mpt"])))
    tr_height.set(qn("w", "hRule"), "exact" if mode == "exact" else "atLeast")


def _append_continuation_cell(tr: etree._Element, grid_cell: _GridCell, ctx: Any) -> None:
    """A `w:vMerge` continuation cell: no AST node, minimal content
    (RFC002 "Spans" — must be counted as a technical node, never content).
    """
    tc = etree.SubElement(tr, qn("w", "tc"))
    tc_pr = etree.SubElement(tc, qn("w", "tcPr"))
    if grid_cell.col_span > 1:
        grid_span = etree.SubElement(tc_pr, qn("w", "gridSpan"))
        grid_span.set(qn("w", "val"), str(grid_cell.col_span))
    v_merge = etree.SubElement(tc_pr, qn("w", "vMerge"))
    v_merge.set(qn("w", "val"), "continue")
    etree.SubElement(tc, qn("w", "p"))
    ctx.technical_cells += 1


def _append_cell(
    tr: etree._Element,
    cell: dict[str, Any],
    path: str,
    font_policy: FontPolicy,
    used_fonts: set[str],
    ctx: Any,
    append_block_fn: AppendBlockFn,
) -> list[Issue]:
    issues: list[Issue] = []
    if cell.get("id") is not None:
        ctx.rendered_node_ids.add(cell["id"])
    tc = etree.SubElement(tr, qn("w", "tc"))
    tc_pr = etree.SubElement(tc, qn("w", "tcPr"))
    # CT_TcPrBase element order (wml.xsd xsd:sequence): tcW, gridSpan,
    # vMerge, tcBorders, shd, tcMar, vAlign.
    tc_w = etree.SubElement(tc_pr, qn("w", "tcW"))
    tc_w.set(qn("w", "type"), "dxa")
    tc_w.set(qn("w", "w"), str(mpt_to_twip(cell["layout"]["width_mpt"])))
    if cell["col_span"] > 1:
        grid_span = etree.SubElement(tc_pr, qn("w", "gridSpan"))
        grid_span.set(qn("w", "val"), str(cell["col_span"]))
    if cell["row_span"] > 1:
        v_merge = etree.SubElement(tc_pr, qn("w", "vMerge"))
        v_merge.set(qn("w", "val"), "restart")
    _append_cell_borders(tc_pr, cell["style"].get("borders"))
    _append_shading(tc_pr, cell["style"].get("background"))
    _append_cell_margins(tc_pr, cell["layout"]["padding"])
    v_align = etree.SubElement(tc_pr, qn("w", "vAlign"))
    v_align.set(qn("w", "val"), _VALIGN_MAP[cell["layout"]["vertical_alignment"]])

    children = cell.get("children", [])
    for i, child in enumerate(children):
        child_issues, _child_fatal = append_block_fn(
            tc, child, f"{path}/children/{i}", font_policy, used_fonts, ctx
        )
        issues.extend(child_issues)
    _close_cell(tc, children, ctx)
    return issues


def _close_cell(tc: etree._Element, children: list[dict[str, Any]], ctx: Any) -> None:
    """A `w:tc` must not end with a `w:tbl`.

    ECMA-376's CT_Tc only demands `EG_BlockLevelElts minOccurs="1"`, so
    a cell ending in a nested table is XSD-valid, but Word treats the
    trailing paragraph as structural and repairs the file without it.
    The paragraph carries no AST node, so it is recorded as technical
    and contributes nothing to text conservation.
    """
    if not children or children[-1].get("type") != "table":
        return
    ctx.mark_technical_paragraph(etree.SubElement(tc, qn("w", "p")))


def _append_cell_margins(tc_pr: etree._Element, padding: dict[str, int]) -> None:
    tc_mar = etree.SubElement(tc_pr, qn("w", "tcMar"))
    for side in ("top", "left", "bottom", "right"):
        side_el = etree.SubElement(tc_mar, qn("w", side))
        side_el.set(qn("w", "w"), str(mpt_to_twip(padding[f"{side}_mpt"])))
        side_el.set(qn("w", "type"), "dxa")


def _append_cell_borders(tc_pr: etree._Element, borders: dict[str, Any] | None) -> None:
    if borders is None:
        return
    tc_borders = etree.SubElement(tc_pr, qn("w", "tcBorders"))
    for side in ("top", "left", "bottom", "right"):
        border = borders.get(side)
        if not border:
            continue
        _append_border_side(tc_borders, side, border)
