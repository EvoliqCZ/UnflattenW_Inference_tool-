import json
from pathlib import Path

from unflatten.renderer.api import FontPolicy
from unflatten.renderer.ooxml.context import BuildContext
from unflatten.renderer.ooxml.document_builder import build_document
from unflatten.renderer.ooxml.numbering import build_numbering_registry
from unflatten.renderer.xml_writer import serialize

VALID_DIR = Path(__file__).parent.parent / "data" / "corpus" / "valid"
_POLICY = FontPolicy(mode="allow_fallback")


def _load(name: str) -> dict:
    with open(VALID_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _ctx_for(ast: dict) -> BuildContext:
    registry = build_numbering_registry(ast["sections"][0]["children"])
    return BuildContext(font_policy=_POLICY, numbering=registry)


def _xml(ast: dict) -> str:
    result = build_document(ast, _POLICY, _ctx_for(ast))
    assert not result.fatal, result.issues
    return serialize(result.root).decode()


def test_first_item_paragraph_gets_num_pr():
    ast = _load("ordered_unordered_nested_list")
    xml = _xml(ast)
    assert '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>' in xml


def test_nested_list_uses_its_own_num_id():
    ast = _load("ordered_unordered_nested_list")
    xml = _xml(ast)
    assert '<w:numId w:val="2"/>' in xml


def test_second_paragraph_in_same_item_gets_continuation_indent_no_marker():
    ast = _load("ordered_unordered_nested_list")
    item = ast["sections"][0]["children"][0]["items"][0]
    extra_paragraph = json.loads(json.dumps(item["children"][0]))
    extra_paragraph["id"] = "p1b"
    extra_paragraph["children"][0]["id"] = "r1b"
    extra_paragraph["children"][0]["text"] = "continuation"
    item["children"].append(extra_paragraph)

    xml = _xml(ast)
    # 3 markers total (list1 item1, list1 item2, list2 item1); the
    # appended continuation paragraph gets no 4th marker.
    assert xml.count("<w:numPr>") == 3
    assert 'w:left="720" w:right="0"/>' in xml  # list.left_indent_mpt continuation, no hanging


def test_list_item_not_starting_with_paragraph_is_unsupported():
    ast = _load("ordered_unordered_nested_list")
    item = ast["sections"][0]["children"][0]["items"][0]
    item["children"].insert(0, {"id": "cb1", "type": "column_break"})
    result = build_document(ast, _POLICY, _ctx_for(ast))
    assert any(i.code == "UNSUPPORTED_LIST_ITEM_NOT_PARAGRAPH" for i in result.issues)


def test_marker_paragraph_indent_folds_in_list_layout():
    ast = _load("ordered_unordered_nested_list")
    xml = _xml(ast)
    assert 'w:left="720" w:right="0" w:hanging="360"' in xml
