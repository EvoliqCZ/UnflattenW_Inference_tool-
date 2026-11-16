import copy

from unflatten.ast.semantic_validator import validate_semantics


def _run(id_="r1", text="x", color="#000000"):
    return {
        "id": id_,
        "type": "run",
        "text": text,
        "language": "en-US",
        "style": {
            "font_family": "Arial",
            "font_size_mpt": 11000,
            "bold": False,
            "italic": False,
            "underline": "none",
            "strike": False,
            "color": color,
            "highlight": None,
            "superscript": False,
            "subscript": False,
            "character_spacing_mpt": 0,
            "small_caps": False,
            "all_caps": False,
        },
    }


def _paragraph(id_="p1", children=None, role="body", heading_level=None):
    return {
        "id": id_,
        "type": "paragraph",
        "semantic": {"role": role, "heading_level": heading_level, "outline_level": None},
        "layout": {
            "alignment": "left",
            "direction": "ltr",
            "indent": {"left_mpt": 0, "right_mpt": 0, "first_line_mpt": 0, "hanging_mpt": 0},
            "spacing": {"before_mpt": 0, "after_mpt": 0, "line": {"mode": "single"}},
            "pagination": {
                "keep_with_next": False,
                "keep_lines": False,
                "page_break_before": False,
                "widow_control": True,
            },
            "tabs": [],
        },
        "style": {"background": None, "borders": None},
        "children": children if children is not None else [],
    }


def test_validate_semantics_minimal_valid_doc_returns_no_issues(minimal_valid_doc_copy):
    issues = validate_semantics(minimal_valid_doc_copy)
    assert issues == []


def test_validate_semantics_duplicate_node_ids_returns_error(minimal_valid_doc_copy):
    minimal_valid_doc_copy["sections"][0]["children"][0]["id"] = "n1"
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "DUPLICATE_NODE_ID" in codes
    assert all(i.severity == "error" for i in issues if i.code == "DUPLICATE_NODE_ID")


def test_validate_semantics_duplicate_resource_ids_returns_error(minimal_valid_doc_copy):
    minimal_valid_doc_copy["resources"] = [
        {
            "id": "res_aaaa",
            "kind": "image",
            "mime_type": "image/png",
            "sha256": "a" * 64,
            "path": "assets/a.png",
            "pixel_width": 10,
            "pixel_height": 10,
        },
        {
            "id": "res_aaaa",
            "kind": "image",
            "mime_type": "image/png",
            "sha256": "b" * 64,
            "path": "assets/b.png",
            "pixel_width": 10,
            "pixel_height": 10,
        },
    ]
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "DUPLICATE_RESOURCE_ID" in codes


def test_validate_semantics_paragraph_containing_table_returns_error(minimal_valid_doc_copy):
    table = _table_skeleton()
    minimal_valid_doc_copy["sections"][0]["children"][0]["children"].append(table)
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "INVALID_PARENT_CHILD" in codes


def test_validate_semantics_cell_without_paragraph_returns_error(minimal_valid_doc_copy):
    table = _table_skeleton(cell_children=[])
    minimal_valid_doc_copy["sections"][0]["children"].append(table)
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "CELL_MISSING_PARAGRAPH" in codes


def test_validate_semantics_cell_last_child_not_paragraph_returns_error(minimal_valid_doc_copy):
    # RFC004 §18.5 cell rule: >= 1 paragraph AND last child is a paragraph;
    # the grammar enforces both, so the AST validator must too.
    table = _table_skeleton(
        cell_children=[
            _paragraph("tp1", children=[_run("tr1")]),
            {"id": "pb1", "type": "page_break"},
        ]
    )
    minimal_valid_doc_copy["sections"][0]["children"].append(table)
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "CELL_LAST_CHILD_NOT_PARAGRAPH" in codes
    assert "CELL_MISSING_PARAGRAPH" not in codes


def test_validate_semantics_cell_last_child_paragraph_ok(minimal_valid_doc_copy):
    table = _table_skeleton(
        cell_children=[
            {"id": "pb1", "type": "page_break"},
            _paragraph("tp1", children=[_run("tr1")]),
        ]
    )
    minimal_valid_doc_copy["sections"][0]["children"].append(table)
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "CELL_LAST_CHILD_NOT_PARAGRAPH" not in codes
    assert "CELL_MISSING_PARAGRAPH" not in codes


def test_validate_semantics_broken_resource_ref_returns_error(minimal_valid_doc_copy):
    img = {
        "id": "img1",
        "type": "inline_image",
        "resource_ref": "res_doesnotexist",
        "size": {"width_mpt": 1000, "height_mpt": 1000},
        "crop": {"left_ppm": 0, "top_ppm": 0, "right_ppm": 0, "bottom_ppm": 0},
        "rotation_mdeg": 0,
        "alt_text": None,
    }
    minimal_valid_doc_copy["sections"][0]["children"][0]["children"].append(img)
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "BROKEN_RESOURCE_REF" in codes


def test_validate_semantics_valid_resource_ref_returns_no_issues(minimal_valid_doc_copy):
    minimal_valid_doc_copy["resources"] = [
        {
            "id": "res_aaaa",
            "kind": "image",
            "mime_type": "image/png",
            "sha256": "a" * 64,
            "path": "assets/a.png",
            "pixel_width": 10,
            "pixel_height": 10,
        }
    ]
    img = {
        "id": "img1",
        "type": "inline_image",
        "resource_ref": "res_aaaa",
        "size": {"width_mpt": 1000, "height_mpt": 1000},
        "crop": {"left_ppm": 0, "top_ppm": 0, "right_ppm": 0, "bottom_ppm": 0},
        "rotation_mdeg": 0,
        "alt_text": None,
    }
    minimal_valid_doc_copy["sections"][0]["children"][0]["children"].append(img)
    issues = validate_semantics(minimal_valid_doc_copy)
    assert issues == []


def test_validate_semantics_table_grid_overlap_returns_error(minimal_valid_doc_copy):
    table = _table_skeleton()
    # Second row cell overlaps because col_span extends beyond the grid and
    # a subsequent cell claims the same grid_column.
    table["rows"][0]["cells"][0]["col_span"] = 2
    table["columns"].append({"width_mpt": 100000})
    overlapping_cell = copy.deepcopy(table["rows"][0]["cells"][0])
    overlapping_cell["id"] = "c_overlap"
    overlapping_cell["grid_column"] = 1
    overlapping_cell["col_span"] = 1
    table["rows"][0]["cells"].append(overlapping_cell)
    minimal_valid_doc_copy["sections"][0]["children"].append(table)
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "TABLE_GRID_OVERLAP" in codes


def test_validate_semantics_table_grid_gap_returns_error(minimal_valid_doc_copy):
    table = _table_skeleton()
    table["columns"].append({"width_mpt": 100000})
    # only one cell defined covering grid_column 0; grid_column 1 has a gap
    minimal_valid_doc_copy["sections"][0]["children"].append(table)
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "TABLE_GRID_GAP" in codes


def _cell(id_prefix, grid_column, col_span=1, row_span=1):
    return {
        "id": f"{id_prefix}_cell",
        "type": "cell",
        "grid_column": grid_column,
        "col_span": col_span,
        "row_span": row_span,
        "layout": {
            "width_mpt": 100000,
            "padding": {"top_mpt": 0, "right_mpt": 0, "bottom_mpt": 0, "left_mpt": 0},
            "vertical_alignment": "top",
            "text_direction": "horizontal",
        },
        "style": {"background": None, "borders": None},
        "children": [_paragraph(f"{id_prefix}_p", children=[_run(f"{id_prefix}_r")])],
    }


def _row(id_, cells):
    return {
        "id": id_,
        "type": "row",
        "layout": {
            "height": {"mode": "auto", "value_mpt": 0},
            "repeat_header": False,
            "cant_split": False,
        },
        "cells": cells,
    }


def test_validate_semantics_table_grid_gap_fires_on_non_last_row(minimal_valid_doc_copy):
    # 2-column table: row 0 only covers grid_column 0 (gap at column 1),
    # row 1 covers both columns fully. TABLE_GRID_GAP must be reported for
    # row 0 even though it is not the last row of the table.
    table = _table_skeleton()
    table["columns"].append({"width_mpt": 100000})
    table["rows"] = [
        _row("row1", [_cell("r1c0", grid_column=0)]),
        _row("row2", [_cell("r2c0", grid_column=0), _cell("r2c1", grid_column=1)]),
    ]
    minimal_valid_doc_copy["sections"][0]["children"].append(table)
    issues = validate_semantics(minimal_valid_doc_copy)
    gap_issues = [i for i in issues if i.code == "TABLE_GRID_GAP"]
    assert any(i.path == "/sections/0/children/1/rows/0" for i in gap_issues), [
        (i.code, i.path) for i in issues
    ]


def test_validate_semantics_broken_internal_anchor_returns_error(minimal_valid_doc_copy):
    hyperlink_broken = {
        "id": "hl_broken",
        "type": "hyperlink",
        "target": {"kind": "internal_anchor", "uri": "#does_not_exist"},
        "children": [_run("r_broken")],
    }
    minimal_valid_doc_copy["sections"][0]["children"][0]["children"].append(hyperlink_broken)
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "BROKEN_INTERNAL_ANCHOR" in codes


def test_validate_semantics_valid_internal_anchor_returns_no_issues(minimal_valid_doc_copy):
    hyperlink_ok = {
        "id": "hl_ok",
        "type": "hyperlink",
        "target": {"kind": "internal_anchor", "uri": "#n2"},
        "children": [_run("r_ok")],
    }
    minimal_valid_doc_copy["profile"] = "word-visual-v1"  # hyperlink is a V1 node
    minimal_valid_doc_copy["sections"][0]["children"][0]["children"].append(hyperlink_ok)
    issues = validate_semantics(minimal_valid_doc_copy)
    assert issues == []


def test_validate_semantics_row_span_past_final_row_returns_error(minimal_valid_doc_copy):
    table = _table_skeleton()
    table["rows"][0]["cells"][0]["row_span"] = 2
    minimal_valid_doc_copy["sections"][0]["children"].append(table)
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "TABLE_ROW_SPAN_OUT_OF_RANGE" in codes


def test_validate_semantics_non_section_at_document_root_returns_error(minimal_valid_doc_copy):
    minimal_valid_doc_copy["sections"][0]["type"] = "paragraph"
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "INVALID_PARENT_CHILD" in codes


def test_validate_semantics_indent_first_line_and_hanging_both_nonzero_returns_error(
    minimal_valid_doc_copy,
):
    minimal_valid_doc_copy["sections"][0]["children"][0]["layout"]["indent"]["first_line_mpt"] = 100
    minimal_valid_doc_copy["sections"][0]["children"][0]["layout"]["indent"]["hanging_mpt"] = 100
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "INDENT_MUTUAL_EXCLUSION" in codes


def test_validate_semantics_superscript_and_subscript_both_true_returns_error(
    minimal_valid_doc_copy,
):
    run = minimal_valid_doc_copy["sections"][0]["children"][0]["children"][0]
    run["style"]["superscript"] = True
    run["style"]["subscript"] = True
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "RUN_SCRIPT_MUTUAL_EXCLUSION" in codes


def test_validate_semantics_section_unequal_columns_wrong_width_count_returns_error(
    minimal_valid_doc_copy,
):
    minimal_valid_doc_copy["sections"][0]["columns"] = {
        "count": 2,
        "equal_width": False,
        "gap_mpt": 0,
        "widths_mpt": [100000],
    }
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "COLUMN_WIDTH_COUNT_MISMATCH" in codes


def test_validate_semantics_heading_role_without_level_returns_error(minimal_valid_doc_copy):
    minimal_valid_doc_copy["sections"][0]["children"][0]["semantic"]["role"] = "heading"
    minimal_valid_doc_copy["sections"][0]["children"][0]["semantic"]["heading_level"] = None
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "HEADING_LEVEL_MISSING" in codes


def test_validate_semantics_heading_role_with_valid_level_returns_no_issues(
    minimal_valid_doc_copy,
):
    minimal_valid_doc_copy["sections"][0]["children"][0]["semantic"]["role"] = "heading"
    minimal_valid_doc_copy["sections"][0]["children"][0]["semantic"]["heading_level"] = 1
    issues = validate_semantics(minimal_valid_doc_copy)
    assert issues == []


def test_validate_semantics_font_size_zero_returns_error(minimal_valid_doc_copy):
    run = minimal_valid_doc_copy["sections"][0]["children"][0]["children"][0]
    run["style"]["font_size_mpt"] = 0
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "INVALID_FONT_SIZE" in codes


def test_validate_semantics_cell_col_span_zero_returns_error(minimal_valid_doc_copy):
    table = _table_skeleton()
    table["rows"][0]["cells"][0]["col_span"] = 0
    minimal_valid_doc_copy["sections"][0]["children"].append(table)
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "INVALID_CELL_SPAN" in codes


def test_validate_semantics_reference_cycle_returns_error(minimal_valid_doc_copy):
    hyperlink_a = {
        "id": "hl_a",
        "type": "hyperlink",
        "target": {"kind": "internal_anchor", "uri": "#hl_b"},
        "children": [_run("r_a")],
    }
    minimal_valid_doc_copy["sections"][0]["children"][0]["children"].append(hyperlink_a)
    hyperlink_b = {
        "id": "hl_b",
        "type": "hyperlink",
        "target": {"kind": "internal_anchor", "uri": "#hl_a"},
        "children": [_run("r_b")],
    }
    minimal_valid_doc_copy["sections"][0]["children"][0]["children"].append(hyperlink_b)
    issues = validate_semantics(minimal_valid_doc_copy)
    codes = [i.code for i in issues]
    assert "REFERENCE_CYCLE" in codes


def _table_skeleton(cell_children=None):
    if cell_children is None:
        cell_children = [_paragraph("tp1", children=[_run("tr1")])]
    return {
        "id": "t1",
        "type": "table",
        "layout": {
            "alignment": "left",
            "indent_mpt": 0,
            "width": {"mode": "auto", "value_mpt": 0},
            "autofit": True,
        },
        "columns": [{"width_mpt": 150000}],
        "style": {"background": None, "borders": None},
        "rows": [
            {
                "id": "row1",
                "type": "row",
                "layout": {
                    "height": {"mode": "auto", "value_mpt": 0},
                    "repeat_header": False,
                    "cant_split": False,
                },
                "cells": [
                    {
                        "id": "cell1",
                        "type": "cell",
                        "grid_column": 0,
                        "col_span": 1,
                        "row_span": 1,
                        "layout": {
                            "width_mpt": 150000,
                            "padding": {
                                "top_mpt": 0,
                                "right_mpt": 0,
                                "bottom_mpt": 0,
                                "left_mpt": 0,
                            },
                            "vertical_alignment": "top",
                            "text_direction": "horizontal",
                        },
                        "style": {"background": None, "borders": None},
                        "children": cell_children,
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------- ERRATA-002 E6 --


def _list_skeleton(marker):
    return {
        "id": "l1",
        "type": "list",
        "list_kind": "unordered",
        "marker": marker,
        "layout": {"left_indent_mpt": 36000, "hanging_mpt": 18000},
        "items": [
            {
                "id": "li1",
                "type": "list_item",
                "children": [_paragraph("lp1", children=[_run("lr1")])],
            }
        ],
    }


def test_validate_semantics_custom_text_marker_with_text_returns_no_issues(
    minimal_valid_doc_copy,
):
    marker = {"format": "custom_text", "start": None, "suffix": None, "text": ">>"}
    minimal_valid_doc_copy["sections"][0]["children"].append(_list_skeleton(marker))
    assert validate_semantics(minimal_valid_doc_copy) == []


def test_validate_semantics_custom_text_marker_missing_text_returns_error(
    minimal_valid_doc_copy,
):
    marker = {"format": "custom_text", "start": None, "suffix": None}
    minimal_valid_doc_copy["sections"][0]["children"].append(_list_skeleton(marker))
    issues = validate_semantics(minimal_valid_doc_copy)
    assert [i.code for i in issues] == ["MARKER_TEXT_REQUIRED"]
    assert issues[0].severity == "error"
    assert issues[0].path.endswith("/marker/text")


def test_validate_semantics_custom_text_marker_empty_text_returns_error(
    minimal_valid_doc_copy,
):
    marker = {"format": "custom_text", "start": None, "suffix": None, "text": ""}
    minimal_valid_doc_copy["sections"][0]["children"].append(_list_skeleton(marker))
    issues = validate_semantics(minimal_valid_doc_copy)
    assert [i.code for i in issues] == ["MARKER_TEXT_REQUIRED"]


def test_validate_semantics_non_custom_marker_with_text_returns_error(
    minimal_valid_doc_copy,
):
    marker = {"format": "bullet", "start": None, "suffix": None, "text": ">>"}
    minimal_valid_doc_copy["sections"][0]["children"].append(_list_skeleton(marker))
    issues = validate_semantics(minimal_valid_doc_copy)
    assert [i.code for i in issues] == ["MARKER_TEXT_FORBIDDEN"]
    assert issues[0].severity == "error"


def test_validate_semantics_non_custom_marker_without_text_returns_no_issues(
    minimal_valid_doc_copy,
):
    marker = {"format": "decimal", "start": 1, "suffix": "."}
    minimal_valid_doc_copy["sections"][0]["children"].append(_list_skeleton(marker))
    assert validate_semantics(minimal_valid_doc_copy) == []


# ---------------------------------------------------------- ERRATA-002 E7 --


def _doc_with_table_width(doc, width):
    table = _table_skeleton()
    table["layout"]["width"] = width
    doc["sections"][0]["children"].append(table)
    return doc


def test_validate_semantics_percent_width_with_value_ppm_returns_no_issues(
    minimal_valid_doc_copy,
):
    doc = _doc_with_table_width(minimal_valid_doc_copy, {"mode": "percent", "value_ppm": 500000})
    assert validate_semantics(doc) == []


def test_validate_semantics_percent_width_with_value_mpt_returns_error(
    minimal_valid_doc_copy,
):
    doc = _doc_with_table_width(minimal_valid_doc_copy, {"mode": "percent", "value_mpt": 500000})
    issues = validate_semantics(doc)
    assert [i.code for i in issues] == ["TABLE_WIDTH_FIELD_MISMATCH"]
    assert issues[0].severity == "error"
    assert issues[0].path.endswith("/layout/width")


def test_validate_semantics_absolute_width_with_value_ppm_returns_error(
    minimal_valid_doc_copy,
):
    doc = _doc_with_table_width(
        minimal_valid_doc_copy, {"mode": "absolute", "value_mpt": 1000, "value_ppm": 1}
    )
    issues = validate_semantics(doc)
    assert [i.code for i in issues] == ["TABLE_WIDTH_FIELD_MISMATCH"]


def test_validate_semantics_absolute_width_missing_value_mpt_returns_error(
    minimal_valid_doc_copy,
):
    doc = _doc_with_table_width(minimal_valid_doc_copy, {"mode": "absolute"})
    issues = validate_semantics(doc)
    assert [i.code for i in issues] == ["TABLE_WIDTH_FIELD_MISMATCH"]


def test_validate_semantics_auto_width_bare_returns_no_issues(minimal_valid_doc_copy):
    doc = _doc_with_table_width(minimal_valid_doc_copy, {"mode": "auto"})
    assert validate_semantics(doc) == []


def test_validate_semantics_auto_width_legacy_zero_value_mpt_tolerated(
    minimal_valid_doc_copy,
):
    doc = _doc_with_table_width(minimal_valid_doc_copy, {"mode": "auto", "value_mpt": 0})
    assert validate_semantics(doc) == []


def test_validate_semantics_auto_width_nonzero_value_mpt_returns_error(
    minimal_valid_doc_copy,
):
    doc = _doc_with_table_width(minimal_valid_doc_copy, {"mode": "auto", "value_mpt": 100})
    issues = validate_semantics(doc)
    assert [i.code for i in issues] == ["TABLE_WIDTH_FIELD_MISMATCH"]


def test_validate_semantics_nested_table_returns_error():
    """Layering consistency: nested tables must fail semantic validation
    (grammar and renderer already forbid them in word-core-v0)."""
    import copy
    import json
    import pathlib

    from unflatten.ast.semantic_validator import validate_semantics

    corpus = pathlib.Path(__file__).parent.parent / "data" / "corpus" / "valid"
    doc = copy.deepcopy(json.loads((corpus / "simple_table_2x2.json").read_text()))

    def find_first_cell(node):
        if isinstance(node, dict):
            if node.get("type") == "cell":
                return node
            for key in ("children", "items", "rows", "cells"):
                for child in node.get(key, []) or []:
                    found = find_first_cell(child)
                    if found is not None:
                        return found
        return None

    outer_cell = None
    for section in doc["sections"]:
        outer_cell = find_first_cell(section)
        if outer_cell:
            break
    assert outer_cell is not None
    inner_table = None
    for section in doc["sections"]:
        for child in section["children"]:
            if child.get("type") == "table":
                inner_table = copy.deepcopy(child)
    assert inner_table is not None

    def bump_ids(node, suffix):
        if isinstance(node, dict):
            if isinstance(node.get("id"), str):
                node["id"] = node["id"] + suffix
            for key in ("children", "items", "rows", "cells"):
                for child in node.get(key, []) or []:
                    bump_ids(child, suffix)

    bump_ids(inner_table, "_nested")
    # keep the cell's last child a paragraph (insert table before it)
    outer_cell["children"].insert(0, inner_table)
    issues = validate_semantics(doc)
    assert any(i.code == "NESTED_TABLE_UNSUPPORTED" for i in issues)
