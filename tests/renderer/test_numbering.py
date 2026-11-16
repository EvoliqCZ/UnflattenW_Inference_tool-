import copy
import json
import re
from pathlib import Path

import pytest
from lxml import etree

from unflatten.renderer.api import RendererOptions
from unflatten.renderer.compiler import compile_docx
from unflatten.renderer.ooxml.numbering import build_numbering_registry, build_numbering_xml
from unflatten.renderer.probe import libreoffice_pdf_text, probe_available
from unflatten.renderer.xml_writer import NSMAP
from unflatten.renderer.xsd_validation import validate_part_against_xsd

VALID_DIR = Path(__file__).parent.parent / "data" / "corpus" / "valid"


def _nested_list_ast() -> dict:
    with open(VALID_DIR / "ordered_unordered_nested_list.json", encoding="utf-8") as f:
        return json.load(f)


def _top_nodes(ast: dict) -> list[dict]:
    return ast["sections"][0]["children"]


def test_each_list_node_gets_its_own_numid():
    ast = _nested_list_ast()
    registry = build_numbering_registry(_top_nodes(ast))
    assert registry.num_id_by_node_id == {"list1": 1, "list2": 2}


def test_identical_definitions_share_abstract_num_id():
    ast = _nested_list_ast()
    top = _top_nodes(ast)
    # Make list2's marker/layout identical to list1's (still a separate
    # `list` node -> separate numId, but same abstractNumId).
    list2 = top[0]["items"][1]["children"][1]
    list2["list_kind"] = "ordered"
    list2["marker"] = copy.deepcopy(top[0]["marker"])

    registry = build_numbering_registry(top)
    assert registry.num_id_by_node_id["list1"] != registry.num_id_by_node_id["list2"]
    abstract1 = registry.abstract_id_by_num_id[registry.num_id_by_node_id["list1"]]
    abstract2 = registry.abstract_id_by_num_id[registry.num_id_by_node_id["list2"]]
    assert abstract1 == abstract2
    assert len(registry.definitions_by_abstract_id) == 1


@pytest.mark.parametrize("marker_format", [
    "decimal", "lower_alpha", "upper_alpha", "lower_roman", "upper_roman",
])
@pytest.mark.parametrize("start", [1, 7])
def test_ordered_lists_shared_definition_emit_explicit_counter_restart(marker_format, start):
    # Separate numIds alone do not restart a shared abstractNum counter in LibreOffice.
    nodes = [
        {
            "id": node_id, "type": "list", "list_kind": "ordered",
            "marker": {"format": marker_format, "start": start, "suffix": "."},
            "layout": {"left_indent_mpt": 36000, "hanging_mpt": 18000}, "items": [],
        }
        for node_id in ("l1", "l2")
    ]
    xml = build_numbering_xml(build_numbering_registry(nodes))
    assert validate_part_against_xsd("/word/numbering.xml", xml) == []
    root = etree.fromstring(xml)
    assert len(root.findall("w:abstractNum", NSMAP)) == 1
    nums = root.findall("w:num", NSMAP)
    assert len(nums) == 2
    for num in nums:
        override = num.find("w:lvlOverride", NSMAP)
        assert override is not None
        assert override.get(f"{{{NSMAP['w']}}}ilvl") == "0"
        assert override.find("w:startOverride", NSMAP).get(f"{{{NSMAP['w']}}}val") == str(start)


@pytest.mark.skipif(not probe_available(), reason="LibreOffice headless probe unavailable here")
@pytest.mark.parametrize("start,first,second", [(1, "1", "2"), (7, "7", "8")])
def test_compiled_independent_lists_restart_in_rendered_pdf(tmp_path, start, first, second):
    ast = _nested_list_ast()
    original = _top_nodes(ast)[0]
    original["marker"]["start"] = start
    original["items"][1]["children"] = original["items"][1]["children"][:1]
    other = copy.deepcopy(original)
    for list_node, prefix in ((original, "Alpha"), (other, "Beta")):
        list_node["id"] = prefix
        for index, item in enumerate(list_node["items"]):
            item["id"] = f"{prefix}_item{index}"
            paragraph = item["children"][0]
            paragraph["id"] = f"{prefix}_paragraph{index}"
            paragraph["children"][0]["id"] = f"{prefix}_run{index}"
            paragraph["children"][0]["text"] = f"{prefix} item {index}"
    separator = copy.deepcopy(original["items"][0]["children"][0])
    separator["id"] = "separator"
    separator["children"][0]["id"] = "separator_run"
    separator["children"][0]["text"] = "Separate logical list"
    ast["sections"][0]["children"] = [original, separator, other]
    out = tmp_path / "independent-lists.docx"
    result = compile_docx(ast, out, RendererOptions())
    assert result.status in ("success", "success_with_warnings"), result.issues
    text, issue = libreoffice_pdf_text(out)
    assert issue is None, issue
    assert text is not None
    for prefix in ("Alpha", "Beta"):
        assert re.search(rf"\b{first}\.\s+{prefix} item 0", text), text
        assert re.search(rf"\b{second}\.\s+{prefix} item 1", text), text


def test_id_assignment_is_stable_regardless_of_traversal_order():
    ast = _nested_list_ast()
    top = _top_nodes(ast)
    reversed_top = list(reversed(top))
    registry_forward = build_numbering_registry(top)
    registry_reversed = build_numbering_registry(reversed_top)
    assert registry_forward.num_id_by_node_id == registry_reversed.num_id_by_node_id
    assert registry_forward.abstract_id_by_num_id == registry_reversed.abstract_id_by_num_id


def test_empty_when_no_lists():
    registry = build_numbering_registry([{"id": "p1", "type": "paragraph", "children": []}])
    assert registry.is_empty


def test_ordered_decimal_lvl_text_and_num_fmt():
    ast = _nested_list_ast()
    registry = build_numbering_registry(_top_nodes(ast))
    xml = build_numbering_xml(registry).decode()
    assert '<w:numFmt w:val="decimal"/>' in xml
    assert '<w:lvlText w:val="%1."/>' in xml
    assert '<w:start w:val="1"/>' in xml


def test_unordered_bullet_emits_symbol_font():
    ast = _nested_list_ast()
    registry = build_numbering_registry(_top_nodes(ast))
    xml = build_numbering_xml(registry).decode()
    assert '<w:numFmt w:val="bullet"/>' in xml
    assert '<w:rFonts w:ascii="Symbol" w:hAnsi="Symbol" w:hint="default"/>' in xml


def test_ordered_format_map_all_variants():
    formats_and_ooxml = [
        ("decimal", "decimal"),
        ("lower_alpha", "lowerLetter"),
        ("upper_alpha", "upperLetter"),
        ("lower_roman", "lowerRoman"),
        ("upper_roman", "upperRoman"),
    ]
    for marker_format, expected in formats_and_ooxml:
        node = {
            "id": "l1",
            "type": "list",
            "list_kind": "ordered",
            "marker": {"format": marker_format, "start": 1, "suffix": "."},
            "layout": {"left_indent_mpt": 36000, "hanging_mpt": 18000},
            "items": [],
        }
        registry = build_numbering_registry([node])
        xml = build_numbering_xml(registry).decode()
        assert f'<w:numFmt w:val="{expected}"/>' in xml


def test_custom_text_marker_uses_text_field_as_literal_text():
    # RFC001 ERRATA-002 E6: `marker.text` is the literal lvlText; `suffix`
    # keeps its trailing-punctuation role and is appended after it.
    node = {
        "id": "l1",
        "type": "list",
        "list_kind": "unordered",
        "marker": {"format": "custom_text", "start": None, "suffix": None, "text": ">>"},
        "layout": {"left_indent_mpt": 36000, "hanging_mpt": 18000},
        "items": [],
    }
    registry = build_numbering_registry([node])
    xml = build_numbering_xml(registry).decode()
    assert '<w:lvlText w:val="&gt;&gt;"/>' in xml


def test_custom_text_marker_appends_suffix_after_literal_text():
    node = {
        "id": "l1",
        "type": "list",
        "list_kind": "unordered",
        "marker": {"format": "custom_text", "start": None, "suffix": ")", "text": "*"},
        "layout": {"left_indent_mpt": 36000, "hanging_mpt": 18000},
        "items": [],
    }
    registry = build_numbering_registry([node])
    xml = build_numbering_xml(registry).decode()
    assert '<w:lvlText w:val="*)"/>' in xml


def test_lists_nested_inside_table_cells_are_collected():
    node = {
        "id": "t1",
        "type": "table",
        "rows": [
            {
                "id": "r1",
                "type": "row",
                "cells": [
                    {
                        "id": "c1",
                        "type": "cell",
                        "children": [
                            {
                                "id": "l1",
                                "type": "list",
                                "list_kind": "ordered",
                                "marker": {"format": "decimal", "start": 1, "suffix": "."},
                                "layout": {"left_indent_mpt": 0, "hanging_mpt": 0},
                                "items": [],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    registry = build_numbering_registry([node])
    assert registry.num_id_by_node_id == {"l1": 1}
