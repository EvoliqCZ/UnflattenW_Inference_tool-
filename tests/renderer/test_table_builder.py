import copy
import json
from pathlib import Path

from unflatten.renderer.api import FontPolicy
from unflatten.renderer.ooxml.context import BuildContext
from unflatten.renderer.ooxml.document_builder import build_document
from unflatten.renderer.xml_writer import serialize

VALID_DIR = Path(__file__).parent.parent / "data" / "corpus" / "valid"
_POLICY = FontPolicy(mode="allow_fallback")


def _load(name: str) -> dict:
    with open(VALID_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _table(ast: dict) -> dict:
    return ast["sections"][0]["children"][0]


def _xml(ast: dict, ctx: BuildContext | None = None) -> tuple[str, "object"]:
    result = build_document(ast, _POLICY, ctx)
    assert not result.fatal, result.issues
    return serialize(result.root).decode(), result


def test_simple_table_grid_and_cell_dimensions():
    ast = _load("simple_table_2x2")
    xml, _result = _xml(ast)
    assert '<w:gridCol w:w="2000"/>' in xml
    assert xml.count("<w:gridCol") == 2
    assert '<w:tcW w:type="dxa" w:w="2000"/>' in xml
    assert '<w:tblLayout w:type="autofit"/>' in xml
    assert '<w:tblW w:type="auto" w:w="0"/>' in xml


def test_colspan_rowspan_grid_expansion_produces_vmerge_continuations():
    ast = _load("table_with_colspan_rowspan")
    xml, _result = _xml(ast)
    assert '<w:gridSpan w:val="2"/>' in xml
    assert '<w:vMerge w:val="restart"/>' in xml
    assert '<w:vMerge w:val="continue"/>' in xml
    # c12 spans 2 rows -> exactly one synthesized continuation cell.
    assert xml.count('w:val="continue"') == 1


def test_technical_cells_counted_via_context():
    ast = _load("table_with_colspan_rowspan")
    ctx = BuildContext(font_policy=_POLICY)
    _xml(ast, ctx)
    assert ctx.technical_cells == 1


def test_absolute_table_width_uses_dxa():
    ast = _load("simple_table_2x2")
    _table(ast)["layout"]["width"] = {"mode": "absolute", "value_mpt": 200000}
    xml, _result = _xml(ast)
    assert '<w:tblW w:type="dxa" w:w="4000"/>' in xml


def test_percent_table_width_uses_pct_via_ppm():
    ast = _load("simple_table_2x2")
    _table(ast)["layout"]["width"] = {"mode": "percent", "value_ppm": 500000}
    xml, _result = _xml(ast)
    assert '<w:tblW w:type="pct" w:w="2500"/>' in xml


def test_table_alignment_and_indent():
    ast = _load("simple_table_2x2")
    _table(ast)["layout"]["alignment"] = "center"
    _table(ast)["layout"]["indent_mpt"] = 36000
    xml, _result = _xml(ast)
    assert '<w:jc w:val="center"/>' in xml
    assert '<w:tblInd w:w="720" w:type="dxa"/>' in xml


def test_table_layout_fixed_when_autofit_false():
    ast = _load("simple_table_2x2")
    _table(ast)["layout"]["autofit"] = False
    xml, _result = _xml(ast)
    assert '<w:tblLayout w:type="fixed"/>' in xml


def test_row_height_modes():
    ast = _load("simple_table_2x2")
    row = _table(ast)["rows"][0]
    row["layout"]["height"] = {"mode": "exact", "value_mpt": 100000}
    xml, _result = _xml(ast)
    assert '<w:trHeight w:val="2000" w:hRule="exact"/>' in xml


def test_row_repeat_header_and_cant_split():
    ast = _load("simple_table_2x2")
    row = _table(ast)["rows"][0]
    row["layout"]["repeat_header"] = True
    row["layout"]["cant_split"] = True
    xml, _result = _xml(ast)
    assert "<w:tblHeader/>" in xml
    assert "<w:cantSplit/>" in xml


def test_table_and_cell_borders_and_shading():
    ast = _load("simple_table_2x2")
    table = _table(ast)
    table["style"]["background"] = "#00FF00"
    table["style"]["borders"] = {
        "top": {"style": "single", "width_mpt": 1000, "color": "#000000"},
        "left": None,
        "bottom": None,
        "right": None,
    }
    cell = table["rows"][0]["cells"][0]
    cell["style"]["background"] = "#0000FF"
    xml, _result = _xml(ast)
    assert '<w:shd w:val="clear" w:color="auto" w:fill="00FF00"/>' in xml
    assert '<w:shd w:val="clear" w:color="auto" w:fill="0000FF"/>' in xml
    assert (
        '<w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        "</w:tblBorders>" in xml
    )


def test_cell_vertical_alignment_and_padding():
    ast = _load("simple_table_2x2")
    cell = _table(ast)["rows"][0]["cells"][0]
    cell["layout"]["vertical_alignment"] = "center"
    xml, _result = _xml(ast)
    assert '<w:vAlign w:val="center"/>' in xml
    assert "<w:tcMar>" in xml


def test_nested_table_in_cell_renders_and_cell_does_not_end_with_tbl():
    """DEC-003 §1: nested tables discarded 495 documents whole at
    import, so `word-visual-v1` renders them. A `w:tc` must not end
    with `w:tbl`, so a technical paragraph closes the cell.
    """
    ast = _load("simple_table_2x2")
    cell = _table(ast)["rows"][0]["cells"][0]
    nested = copy.deepcopy(_table(ast))
    nested["id"] = "t2"
    cell["children"].append(nested)
    ctx = BuildContext(font_policy=_POLICY)
    result = build_document(ast, _POLICY, ctx)
    assert not any(i.code == "UNSUPPORTED_NESTED_TABLE" for i in result.issues)
    xml = serialize(result.root).decode()
    assert xml.count("<w:tbl>") == 2
    assert "</w:tbl><w:p/></w:tc>" in xml
    assert ctx.technical_paragraph_paths


def test_cell_content_reuses_paragraph_block_dispatch():
    ast = _load("simple_table_2x2")
    xml, _result = _xml(ast)
    # Cell paragraphs use the same paragraph mapping as top-level blocks.
    assert "<w:r><w:rPr>" in xml


def test_border_style_drift_guard_and_round_trip(tmp_path):
    """E-1.7 (D-1 §1.3): `_INV_BORDER_STYLE` stays an EXACT derivation of
    the renderer's forward `_BORDER_STYLE_MAP` (so `_append_border_side`,
    which indexes `_BORDER_STYLE_MAP` directly, can never see a value the
    importer cannot produce); the `nil` input-only spelling in
    `_EXTRA_BORDER_STYLE_IN` is disjoint from it, so it can never shadow
    a forward-mapped value; and every AST border style round-trips
    AST -> OOXML -> AST.
    """
    from unflatten.dataset.import_ import import_docx
    from unflatten.dataset.import_.importer import (
        _BORDER_STYLE_MAP,
        _EXTRA_BORDER_STYLE_IN,
        _INV_BORDER_STYLE,
    )
    from unflatten.renderer.api import RendererOptions
    from unflatten.renderer.compiler import compile_docx

    assert set(_INV_BORDER_STYLE) == set(_BORDER_STYLE_MAP.values())
    assert set(_EXTRA_BORDER_STYLE_IN).isdisjoint(_INV_BORDER_STYLE)

    for style in _BORDER_STYLE_MAP:
        ast = _load("simple_table_2x2")
        table = _table(ast)
        table["style"]["borders"] = {
            "top": {"style": style, "width_mpt": 1000, "color": "#000000"},
        }
        output = tmp_path / f"border_{style}.docx"
        render_result = compile_docx(ast, output, RendererOptions())
        assert render_result.status in ("success", "success_with_warnings"), render_result.issues
        imported = import_docx(output.read_bytes())
        assert imported.status in ("success", "success_with_warnings"), imported.report
        assert _table(imported.ast)["style"]["borders"]["top"]["style"] == style
