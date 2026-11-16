from lxml import etree

from unflatten.renderer.ooxml.drawing_builder import (
    append_drawing,
    build_image_refs,
    collect_inline_image_nodes,
)
from unflatten.renderer.xml_writer import NSMAP, qn, serialize


def _image_node(**overrides) -> dict:
    base = {
        "id": "img1",
        "type": "inline_image",
        "resource_ref": "res_a",
        "size": {"width_mpt": 100000, "height_mpt": 50000},
        "crop": {"left_ppm": 0, "top_ppm": 0, "right_ppm": 0, "bottom_ppm": 0},
        "rotation_mdeg": 0,
        "alt_text": None,
    }
    base.update(overrides)
    return base


def _paragraph(*inline_children: dict) -> dict:
    return {"id": "p1", "type": "paragraph", "children": list(inline_children)}


def test_collect_inline_image_nodes_finds_paragraph_and_hyperlink_children():
    img = _image_node()
    hyperlink = {
        "id": "h1",
        "type": "hyperlink",
        "children": [_image_node(id="img2")],
    }
    section_nodes = [_paragraph(img, hyperlink)]
    found = collect_inline_image_nodes(section_nodes)
    assert [n["id"] for n in found] == ["img1", "img2"]


def test_collect_inline_image_nodes_finds_nested_list_and_table():
    img_in_list = _image_node(id="img_list")
    list_node = {
        "id": "l1",
        "type": "list",
        "items": [{"id": "li1", "type": "list_item", "children": [_paragraph(img_in_list)]}],
    }
    img_in_table = _image_node(id="img_table")
    table_node = {
        "id": "t1",
        "type": "table",
        "rows": [
            {
                "id": "row1",
                "type": "row",
                "cells": [{"id": "c1", "type": "cell", "children": [_paragraph(img_in_table)]}],
            }
        ],
    }
    found = collect_inline_image_nodes([list_node, table_node])
    assert [n["id"] for n in found] == ["img_list", "img_table"]


def test_build_image_refs_assigns_preorder_docpr_ids_and_rel_ids():
    nodes = [
        _image_node(id="img1", resource_ref="res_a"),
        _image_node(id="img2", resource_ref="res_b"),
    ]
    refs, issues = build_image_refs(
        nodes,
        rel_id_by_sha256={"sha_a": "rId4", "sha_b": "rId5"},
        sha256_by_resource_id={"res_a": "sha_a", "res_b": "sha_b"},
    )
    assert issues == []
    assert refs["img1"].docpr_id == 1
    assert refs["img1"].rel_id == "rId4"
    assert refs["img2"].docpr_id == 2
    assert refs["img2"].rel_id == "rId5"


def test_build_image_refs_converts_size_to_emu():
    nodes = [_image_node(size={"width_mpt": 100000, "height_mpt": 50000})]
    refs, _issues = build_image_refs(
        nodes, {"sha_a": "rId4"}, {"res_a": "sha_a"}
    )
    ref = refs["img1"]
    assert ref.cx_emu == 1_270_000
    assert ref.cy_emu == 635_000


def test_build_image_refs_reports_unresolved_resource_ref():
    nodes = [_image_node(resource_ref="res_missing")]
    refs, issues = build_image_refs(nodes, {}, {})
    assert refs == {}
    assert issues[0].code == "RESOURCE_REF_UNRESOLVED"


def test_build_image_refs_converts_rotation_to_sixty_thousandths_of_a_degree():
    nodes = [_image_node(rotation_mdeg=45000)]
    refs, issues = build_image_refs(nodes, {"sha_a": "rId4"}, {"res_a": "sha_a"})
    assert issues == []
    assert refs["img1"].rotation_ooxml == 2_700_000


def test_build_image_refs_zero_rotation_stays_zero_so_the_attribute_is_omitted():
    refs, _issues = build_image_refs(
        [_image_node()], {"sha_a": "rId4"}, {"res_a": "sha_a"}
    )
    assert refs["img1"].rotation_ooxml == 0


def test_build_image_refs_nonzero_crop_produces_src_rect():
    nodes = [
        _image_node(crop={"left_ppm": 100000, "top_ppm": 0, "right_ppm": 0, "bottom_ppm": 0})
    ]
    refs, _issues = build_image_refs(nodes, {"sha_a": "rId4"}, {"res_a": "sha_a"})
    assert refs["img1"].src_rect == (10000, 0, 0, 0)


def test_build_image_refs_zero_crop_omits_src_rect():
    nodes = [_image_node()]
    refs, _issues = build_image_refs(nodes, {"sha_a": "rId4"}, {"res_a": "sha_a"})
    assert refs["img1"].src_rect is None


def _drawing_xml(image_ref) -> str:
    root = etree.Element(qn("w", "r"), nsmap=NSMAP)
    append_drawing(root, image_ref)
    return serialize(root).decode()


def test_append_drawing_includes_extent_docpr_and_blip_embed():
    refs, _issues = build_image_refs(
        [_image_node(alt_text="A diagram")], {"sha_a": "rId4"}, {"res_a": "sha_a"}
    )
    xml = _drawing_xml(refs["img1"])
    assert '<wp:extent cx="1270000" cy="635000"/>' in xml
    assert '<wp:docPr id="1" name="Picture 1" descr="A diagram"/>' in xml
    assert '<a:blip r:embed="rId4"/>' in xml


def test_append_drawing_omits_descr_when_no_alt_text():
    refs, _issues = build_image_refs([_image_node()], {"sha_a": "rId4"}, {"res_a": "sha_a"})
    xml = _drawing_xml(refs["img1"])
    assert "descr" not in xml


def test_append_drawing_includes_src_rect_only_when_cropped():
    refs, _issues = build_image_refs(
        [_image_node(crop={"left_ppm": 100000, "top_ppm": 0, "right_ppm": 0, "bottom_ppm": 0})],
        {"sha_a": "rId4"},
        {"res_a": "sha_a"},
    )
    xml = _drawing_xml(refs["img1"])
    assert '<a:srcRect l="10000" t="0" r="0" b="0"/>' in xml
