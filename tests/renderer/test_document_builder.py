import copy
import json
from pathlib import Path

import pytest

from unflatten.renderer.api import FontPolicy
from unflatten.renderer.ooxml.document_builder import build_document
from unflatten.renderer.xml_writer import serialize

VALID_DIR = Path(__file__).parent.parent / "data" / "corpus" / "valid"
_POLICY = FontPolicy(mode="allow_fallback")


def _base_ast() -> dict:
    with open(VALID_DIR / "minimal_paragraph.json", encoding="utf-8") as f:
        return json.load(f)


def _paragraph(ast: dict) -> dict:
    return ast["sections"][0]["children"][0]


def _run(ast: dict) -> dict:
    return _paragraph(ast)["children"][0]


def _xml(ast: dict) -> str:
    result = build_document(ast, _POLICY)
    assert not result.fatal, result.issues
    return serialize(result.root).decode()


@pytest.mark.parametrize(
    "alignment,ooxml_val",
    [
        ("left", "left"),
        ("center", "center"),
        ("right", "right"),
        ("justify", "both"),
        ("distributed", "distribute"),
    ],
)
def test_alignment_mapping(alignment, ooxml_val):
    ast = _base_ast()
    _paragraph(ast)["layout"]["alignment"] = alignment
    assert f'<w:jc w:val="{ooxml_val}"/>' in _xml(ast)


def test_indent_negative_left_and_first_line():
    ast = _base_ast()
    indent = _paragraph(ast)["layout"]["indent"]
    indent["left_mpt"] = -36000
    indent["right_mpt"] = 18000
    indent["first_line_mpt"] = 12000
    xml = _xml(ast)
    assert 'w:left="-720"' in xml
    assert 'w:right="360"' in xml
    assert 'w:firstLine="240"' in xml
    assert "w:hanging" not in xml


def test_hanging_indent():
    ast = _base_ast()
    indent = _paragraph(ast)["layout"]["indent"]
    indent["hanging_mpt"] = 24000
    xml = _xml(ast)
    assert 'w:hanging="480"' in xml
    assert "w:firstLine" not in xml


def test_spacing_multiple_line_mode():
    ast = _base_ast()
    _paragraph(ast)["layout"]["spacing"]["line"] = {"mode": "multiple", "multiplier_ppm": 1_500_000}
    xml = _xml(ast)
    assert 'w:line="360" w:lineRule="auto"' in xml


def test_spacing_exact_line_mode():
    ast = _base_ast()
    _paragraph(ast)["layout"]["spacing"]["line"] = {"mode": "exact", "value_mpt": 288000}
    xml = _xml(ast)
    assert 'w:line="5760" w:lineRule="exact"' in xml


def test_spacing_at_least_line_mode():
    ast = _base_ast()
    _paragraph(ast)["layout"]["spacing"]["line"] = {"mode": "at_least", "value_mpt": 144000}
    xml = _xml(ast)
    assert 'w:line="2880" w:lineRule="atLeast"' in xml


def test_heading_role_emits_outline_level():
    ast = _base_ast()
    _paragraph(ast)["semantic"] = {"role": "heading", "heading_level": 2, "outline_level": None}
    xml = _xml(ast)
    assert '<w:outlineLvl w:val="1"/>' in xml


def test_body_role_omits_outline_level():
    ast = _base_ast()
    xml = _xml(ast)
    assert "outlineLvl" not in xml


def test_tabs_with_leader_and_no_leader():
    ast = _base_ast()
    _paragraph(ast)["layout"]["tabs"] = [
        {"position_mpt": 100000, "alignment": "center", "leader": "hyphen"},
        {"position_mpt": 200000, "alignment": "decimal", "leader": "none"},
    ]
    xml = _xml(ast)
    assert '<w:tab w:val="center" w:pos="2000" w:leader="hyphen"/>' in xml
    assert '<w:tab w:val="decimal" w:pos="4000"/>' in xml


def test_shading_and_borders():
    ast = _base_ast()
    style = _paragraph(ast)["style"]
    style["background"] = "#00FF00"
    style["borders"] = {
        "top": {"style": "single", "width_mpt": 1000, "color": "#000000"},
        "left": None,
        "bottom": None,
        "right": None,
    }
    xml = _xml(ast)
    assert '<w:shd w:val="clear" w:color="auto" w:fill="00FF00"/>' in xml
    assert '<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>' in xml


def test_underline_dashed_maps_to_dash_ooxml_token():
    ast = _base_ast()
    _run(ast)["style"]["underline"] = "dashed"
    xml = _xml(ast)
    assert '<w:u w:val="dash"/>' in xml


def test_color_auto_preserved():
    ast = _base_ast()
    _run(ast)["style"]["color"] = "auto"
    xml = _xml(ast)
    assert '<w:color w:val="auto"/>' in xml


def test_superscript_and_subscript_vert_align():
    ast = _base_ast()
    run = _run(ast)
    run["style"]["superscript"] = True
    assert '<w:vertAlign w:val="superscript"/>' in _xml(ast)

    ast2 = _base_ast()
    _run(ast2)["style"]["subscript"] = True
    assert '<w:vertAlign w:val="subscript"/>' in _xml(ast2)


def test_highlight_emits_shading_in_run():
    ast = _base_ast()
    _run(ast)["style"]["highlight"] = "#FFFF00"
    xml = _xml(ast)
    assert '<w:shd w:val="clear" w:color="auto" w:fill="FFFF00"/>' in xml


def test_run_text_preserve_leading_trailing_and_double_space():
    ast = _base_ast()
    _run(ast)["text"] = "  double  space  "
    xml = _xml(ast)
    assert 'xml:space="preserve"' in xml


def test_run_text_no_preserve_for_single_spaced_text():
    ast = _base_ast()
    _run(ast)["text"] = "no extra spaces here"
    xml = _xml(ast)
    assert 'xml:space="preserve"' not in xml


def test_run_text_preserve_for_interior_tab_without_leading_trailing_whitespace():
    ast = _base_ast()
    _run(ast)["text"] = "a\tb"
    xml = _xml(ast)
    assert 'xml:space="preserve"' in xml


def test_run_text_preserve_for_interior_newline_without_leading_trailing_whitespace():
    ast = _base_ast()
    _run(ast)["text"] = "a\nb"
    xml = _xml(ast)
    assert 'xml:space="preserve"' in xml


def test_control_char_in_run_text_fails_loud_at_the_renderer_boundary():
    """Defect B guard: the AST is defined as deterministically renderable
    (RFC001) — an XML-1.0-illegal control character reaching the
    renderer is a contract breach the importer should have caught, and
    must fail with a message naming the node and the offending
    codepoint rather than an opaque lxml error deep in serialization."""
    from unflatten.renderer.ooxml.document_builder import RendererContractError

    ast = _base_ast()
    run = _run(ast)
    run["text"] = "bad\x01text"
    with pytest.raises(RendererContractError) as exc_info:
        build_document(ast, _POLICY)
    message = str(exc_info.value)
    assert run["id"] in message
    assert "U+0001" in message


def test_tab_newline_cr_survive_illegal_char_guard():
    """`\\t`/`\\n`/`\\r` are legal XML whitespace and must never be
    treated as the illegal control characters the renderer guard
    rejects — the text round-trips through the built document byte for
    byte."""
    from lxml import etree

    ast = _base_ast()
    _run(ast)["text"] = "a\tb\nc\rd"
    result = build_document(ast, _POLICY)
    assert not result.fatal, result.issues
    [t_el] = result.root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
    assert t_el.text == "a\tb\nc\rd"
    assert etree.tostring(result.root)  # serializes without raising


def test_page_orientation_landscape():
    ast = _base_ast()
    ast["sections"][0]["page"]["orientation"] = "landscape"
    xml = _xml(ast)
    assert 'w:orient="landscape"' in xml


def test_column_break_block_node_renders_technical_paragraph():
    ast = _base_ast()
    section = ast["sections"][0]
    section["children"].append({"id": "cb1", "type": "column_break"})
    xml = _xml(ast)
    assert '<w:br w:type="column"/>' in xml


def test_unknown_inline_node_type_is_reported_never_dropped():
    ast = _base_ast()
    _paragraph(ast)["children"].append({"id": "x1", "type": "smart_tag"})
    result = build_document(ast, _POLICY)
    assert any(i.code == "UNSUPPORTED_NODE_R0" for i in result.issues)


def test_internal_anchor_hyperlink_renders_w_anchor_without_relationship():
    ast = _base_ast()
    paragraph = _paragraph(ast)
    paragraph["children"].append(
        {
            "id": "h2",
            "type": "hyperlink",
            "target": {"kind": "internal_anchor", "uri": "#p1"},
            "children": [
                {
                    "id": "hr2",
                    "type": "run",
                    "text": "jump",
                    "language": "en-US",
                    "style": _run(ast)["style"],
                }
            ],
        }
    )
    result = build_document(ast, _POLICY)
    assert not any(i.code.startswith("UNSUPPORTED") for i in result.issues)
    assert '<w:hyperlink w:anchor="p1">' in serialize(result.root).decode()


def test_external_hyperlink_without_registered_relationship_is_reported():
    """The relationship is allocated by `package_plan`, per part. A
    hyperlink reaching the builder without one is a renderer bug and
    must surface as an Issue, never as a silently anchor-less link.
    """
    ast = _base_ast()
    _paragraph(ast)["children"].append(
        {
            "id": "h1",
            "type": "hyperlink",
            "target": {"kind": "external", "uri": "https://example.com"},
            "children": [
                {
                    "id": "hr1",
                    "type": "run",
                    "text": "link",
                    "language": "en-US",
                    "style": _run(ast)["style"],
                }
            ],
        }
    )
    result = build_document(ast, _POLICY)
    assert any(i.code == "HYPERLINK_REL_UNRESOLVED" for i in result.issues)


def test_multi_section_compiles_with_one_sectpr_per_section():
    ast = _base_ast()
    ast["sections"] = ast["sections"] + [copy.deepcopy(ast["sections"][0])]
    result = build_document(ast, _POLICY)
    assert not result.fatal
    xml = serialize(result.root).decode()
    assert xml.count("<w:sectPr>") == 2
    # The first section's sectPr lives inside its own last paragraph's
    # pPr (non-final section); the second (final) section's sectPr is
    # the last child of w:body.
    assert "</w:sectPr></w:pPr>" in xml
    assert xml.rstrip().endswith("</w:sectPr></w:body></w:document>")


def test_section_ending_in_table_gets_technical_boundary_paragraph():
    ast = json.load(open(VALID_DIR / "simple_table_2x2.json", encoding="utf-8"))
    ast["sections"] = ast["sections"] + [copy.deepcopy(ast["sections"][0])]
    from unflatten.renderer.ooxml.context import BuildContext

    ctx = BuildContext(font_policy=_POLICY)
    result = build_document(ast, _POLICY, ctx)
    assert not result.fatal
    assert ctx.technical_paragraphs == 1
    xml = serialize(result.root).decode()
    assert xml.count("<w:sectPr>") == 2
