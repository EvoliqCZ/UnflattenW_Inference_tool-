import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from unflatten.renderer.api import FontPolicy, RendererOptions
from unflatten.renderer.compiler import _validate_package, compile_docx
from unflatten.renderer.opc import PackagePart
from unflatten.renderer.resources import DirResourceProvider
from unflatten.renderer.text_extract import verify_text_conservation

VALID_DIR = Path(__file__).parent.parent / "data" / "corpus" / "valid"
INVALID_DIR = Path(__file__).parent.parent / "data" / "corpus" / "invalid"
GOLDEN_DIR = Path(__file__).parent.parent / "data" / "golden" / "r1"
GOLDEN_R2R3_DIR = Path(__file__).parent.parent / "data" / "golden" / "r2r3"
GOLDEN_R4_DIR = Path(__file__).parent.parent / "data" / "golden" / "r4"
ASSETS_DIR = Path(__file__).parent.parent / "data" / "assets"


def _load(directory: Path, name: str) -> dict:
    with open(directory / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _document_xml(docx_path: Path) -> bytes:
    with zipfile.ZipFile(docx_path) as zf:
        return zf.read("word/document.xml")


@pytest.mark.parametrize("name", ["minimal_paragraph", "mixed_run_styles"])
def test_golden_fixture_compiles_and_matches_pinned_xml(tmp_path, name):
    ast = _load(VALID_DIR, name)
    output_path = tmp_path / f"{name}.docx"

    result = compile_docx(ast, output_path, RendererOptions())

    assert result.status in ("success", "success_with_warnings")
    docx_bytes = output_path.read_bytes()
    assert verify_text_conservation(ast, docx_bytes) == []

    result2 = compile_docx(ast, tmp_path / f"{name}_2.docx", RendererOptions())
    assert result.package_hash == result2.package_hash

    expected_xml = (GOLDEN_DIR / f"{name}.xml").read_bytes()
    assert _document_xml(output_path) == expected_xml


@pytest.mark.parametrize(
    "name",
    [
        "minimal_paragraph",
        "mixed_run_styles",
        "paragraph_with_tabs_and_line_break",
        "multi_paragraph_with_page_break",
    ],
)
def test_compile_is_byte_deterministic(tmp_path, name):
    ast = _load(VALID_DIR, name)
    out1 = tmp_path / "a.docx"
    out2 = tmp_path / "b.docx"
    compile_docx(ast, out1, RendererOptions())
    compile_docx(ast, out2, RendererOptions())
    assert out1.read_bytes() == out2.read_bytes()


def test_invalid_ast_fails_without_writing_package(tmp_path):
    ast = _load(INVALID_DIR, "missing_required_field")
    output_path = tmp_path / "out.docx"

    result = compile_docx(ast, output_path, RendererOptions())

    assert result.status == "failed"
    assert result.output_path is None
    assert result.package_hash is None
    assert not output_path.exists()
    assert any(i.severity in ("error", "fatal") for i in result.issues)


def test_multi_section_two_column_document_compiles(tmp_path):
    ast = _load(VALID_DIR, "two_section_two_column")
    output_path = tmp_path / "out.docx"
    result = compile_docx(ast, output_path, RendererOptions())
    assert result.status in ("success", "success_with_warnings")
    docx_bytes = output_path.read_bytes()
    assert verify_text_conservation(ast, docx_bytes) == []


def test_external_hyperlink_compiles_to_w_hyperlink_and_external_relationship(tmp_path):
    """DEC-003 §1: hyperlinks were flattened to plain text in 28.7 % of
    documents. They now render, so the old `UNSUPPORTED_NODE_R0`
    expectation is gone.
    """
    ast = _load(VALID_DIR, "minimal_paragraph")
    ast = copy.deepcopy(ast)
    ast["profile"] = "word-visual-v1"
    run = ast["sections"][0]["children"][0]["children"][0]
    ast["sections"][0]["children"][0]["children"].append(
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
                    "style": run["style"],
                }
            ],
        }
    )
    output_path = tmp_path / "out.docx"
    result = compile_docx(ast, output_path, RendererOptions())
    assert result.status in ("success", "success_with_warnings"), result.issues

    with zipfile.ZipFile(output_path) as zf:
        document = zf.read("word/document.xml").decode()
        rels = zf.read("word/_rels/document.xml.rels").decode()
    assert "<w:hyperlink r:id=" in document
    assert 'Target="https://example.com" TargetMode="External"' in rels


def test_require_exact_font_policy_rejects_font_outside_pack(tmp_path):
    ast = _load(VALID_DIR, "minimal_paragraph")
    options = RendererOptions(font_policy=FontPolicy(mode="require_exact"))
    result = compile_docx(ast, tmp_path / "out.docx", options)
    assert result.status == "failed"
    assert any(i.code == "FONT_NOT_IN_PACK" for i in result.issues)


def test_allow_fallback_reports_approximation_and_succeeds(tmp_path):
    ast = _load(VALID_DIR, "minimal_paragraph")
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    assert result.status == "success_with_warnings"
    assert len(result.approximations) == 1
    assert result.approximations[0].code == "FONT_FALLBACK"
    assert result.approximations[0].details == {
        "requested": "Arial", "used": "Liberation Sans", "via": "map",
    }


def test_font_in_pilot_pack_directly_has_no_issues(tmp_path):
    ast = _load(VALID_DIR, "minimal_paragraph")
    ast = copy.deepcopy(ast)
    ast["sections"][0]["children"][0]["children"][0]["style"]["font_family"] = "Liberation Sans"
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    assert result.status == "success"
    assert result.issues == []


def test_ast_hash_stable_and_ignores_metadata(tmp_path):
    ast = _load(VALID_DIR, "minimal_paragraph")
    result1 = compile_docx(ast, tmp_path / "a.docx", RendererOptions())

    ast_with_different_metadata = copy.deepcopy(ast)
    ast_with_different_metadata["metadata"]["title"] = "Different title"
    result2 = compile_docx(ast_with_different_metadata, tmp_path / "b.docx", RendererOptions())

    # ast_hash is a semantic-content identity that excludes metadata
    # (RFC001 ERRATA-001 E1) even though `title` is now rendered
    # visibly into docProps/core.xml, so the *package* legitimately
    # differs while the *AST identity* does not.
    assert result1.ast_hash == result2.ast_hash
    assert result1.package_hash != result2.package_hash


def test_validate_package_detects_malformed_xml_part():
    parts = [PackagePart("word/document.xml", b"<w:document><unclosed>")]
    issue = _validate_package(parts)
    assert issue is not None
    assert issue.code == "PACKAGE_INVALID"
    assert issue.severity == "fatal"


def test_validate_package_accepts_well_formed_parts():
    parts = [PackagePart("word/document.xml", b"<document/>")]
    assert _validate_package(parts) is None


def test_compile_docx_fails_and_writes_nothing_on_package_validation_failure(
    tmp_path, monkeypatch
):
    import unflatten.renderer.compiler as compiler_module

    monkeypatch.setattr(
        compiler_module,
        "_validate_package",
        lambda parts: compiler_module.Issue(
            code="PACKAGE_INVALID", severity="fatal", node_id=None, path="/", message="forced"
        ),
    )
    ast = _load(VALID_DIR, "minimal_paragraph")
    output_path = tmp_path / "out.docx"

    result = compile_docx(ast, output_path, RendererOptions())

    assert result.status == "failed"
    assert result.output_path is None
    assert not output_path.exists()
    assert any(i.code == "PACKAGE_INVALID" for i in result.issues)


def test_parts_dict_has_sha256_for_every_part(tmp_path):
    ast = _load(VALID_DIR, "minimal_paragraph")
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    expected_parts = {
        "[Content_Types].xml",
        "_rels/.rels",
        "word/document.xml",
        "word/_rels/document.xml.rels",
        "word/styles.xml",
        "word/settings.xml",
        "word/fontTable.xml",
        # minimal_paragraph's metadata carries a non-null `language`,
        # so docProps/core.xml is emitted (RFC002 "Metadata").
        "docProps/core.xml",
    }
    assert set(result.parts) == expected_parts
    for digest in result.parts.values():
        assert digest.startswith("sha256:")
        assert len(digest) == len("sha256:") + 64


# --- R2/R3 (lists, tables) ---------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["ordered_unordered_nested_list", "simple_table_2x2", "table_with_colspan_rowspan"],
)
def test_r2r3_golden_fixture_compiles_and_matches_pinned_xml(tmp_path, name):
    ast = _load(VALID_DIR, name)
    output_path = tmp_path / f"{name}.docx"

    result = compile_docx(ast, output_path, RendererOptions())

    assert result.status in ("success", "success_with_warnings")
    docx_bytes = output_path.read_bytes()
    assert verify_text_conservation(ast, docx_bytes) == []

    expected_xml = (GOLDEN_R2R3_DIR / f"{name}.xml").read_bytes()
    assert _document_xml(output_path) == expected_xml


@pytest.mark.parametrize(
    "name",
    ["ordered_unordered_nested_list", "simple_table_2x2", "table_with_colspan_rowspan"],
)
def test_r2r3_compile_is_byte_deterministic(tmp_path, name):
    ast = _load(VALID_DIR, name)
    out1 = tmp_path / "a.docx"
    out2 = tmp_path / "b.docx"
    compile_docx(ast, out1, RendererOptions())
    compile_docx(ast, out2, RendererOptions())
    assert out1.read_bytes() == out2.read_bytes()


def test_numbering_part_only_emitted_when_lists_present(tmp_path):
    ast = _load(VALID_DIR, "minimal_paragraph")
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    assert "word/numbering.xml" not in result.parts

    ast_with_list = _load(VALID_DIR, "ordered_unordered_nested_list")
    result2 = compile_docx(ast_with_list, tmp_path / "out2.docx", RendererOptions())
    assert "word/numbering.xml" in result2.parts
    with zipfile.ZipFile(tmp_path / "out2.docx") as zf:
        assert "word/numbering.xml" in zf.namelist()
        assert "word/_rels/document.xml.rels" in zf.namelist()
        rels = zf.read("word/_rels/document.xml.rels").decode()
        assert "numbering.xml" in rels


def test_list_item_not_starting_with_paragraph_fails_strict_compile(tmp_path):
    ast = _load(VALID_DIR, "ordered_unordered_nested_list")
    ast = copy.deepcopy(ast)
    item = ast["sections"][0]["children"][0]["items"][0]
    item["children"].insert(0, {"id": "cb1", "type": "column_break"})
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    assert result.status == "failed"
    assert any(i.code == "UNSUPPORTED_LIST_ITEM_NOT_PARAGRAPH" for i in result.issues)


def _rename_ids(node, prefix):
    if isinstance(node, dict):
        if "id" in node and isinstance(node["id"], str):
            node["id"] = f"{prefix}_{node['id']}"
        for value in node.values():
            _rename_ids(value, prefix)
    elif isinstance(node, list):
        for item in node:
            _rename_ids(item, prefix)


def test_nested_table_fails_strict_compile(tmp_path):
    ast = _load(VALID_DIR, "simple_table_2x2")
    ast = copy.deepcopy(ast)
    table = ast["sections"][0]["children"][0]
    cell = table["rows"][0]["cells"][0]
    nested = copy.deepcopy(table)
    _rename_ids(nested, "nested")
    cell["children"].append(nested)
    # keep the cell's last child a paragraph so the (now enforced)
    # CELL_LAST_CHILD_NOT_PARAGRAPH rule does not mask the nested-table check
    trailing = copy.deepcopy(cell["children"][0])
    _rename_ids(trailing, "trailing")
    cell["children"].append(trailing)
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    assert result.status == "failed"
    assert any(
        i.code in ("UNSUPPORTED_NESTED_TABLE", "NESTED_TABLE_UNSUPPORTED")
        for i in result.issues
    )  # semantic validation (S1) now rejects nested tables before the renderer's own check


def test_technical_cells_reported_in_metrics(tmp_path):
    ast = _load(VALID_DIR, "table_with_colspan_rowspan")
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    assert result.metrics["technical_cells"] == 1


# --- R4 (inline images) -------------------------------------------------------


def _image_ast_with_local_asset() -> tuple[dict, str]:
    ast = _load(VALID_DIR, "inline_image_with_resource")
    ast = copy.deepcopy(ast)
    data = (ASSETS_DIR / "tiny_2x2.png").read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    resource = ast["resources"][0]
    resource["sha256"] = sha256
    resource["path"] = "tiny_2x2.png"
    resource["pixel_width"] = 2
    resource["pixel_height"] = 2
    return ast, sha256


def test_inline_image_golden_fixture_compiles_and_matches_pinned_xml(tmp_path):
    ast, _sha256 = _image_ast_with_local_asset()
    output_path = tmp_path / "image.docx"

    result = compile_docx(
        ast, output_path, RendererOptions(), resources=DirResourceProvider(ASSETS_DIR)
    )

    assert result.status in ("success", "success_with_warnings")
    docx_bytes = output_path.read_bytes()
    assert verify_text_conservation(ast, docx_bytes) == []
    expected_xml = (GOLDEN_R4_DIR / "inline_image_with_resource.xml").read_bytes()
    assert _document_xml(output_path) == expected_xml


def test_inline_image_compile_is_byte_deterministic(tmp_path):
    ast, _sha256 = _image_ast_with_local_asset()
    provider = DirResourceProvider(ASSETS_DIR)
    out1 = tmp_path / "a.docx"
    out2 = tmp_path / "b.docx"
    compile_docx(ast, out1, RendererOptions(), resources=provider)
    compile_docx(ast, out2, RendererOptions(), resources=provider)
    assert out1.read_bytes() == out2.read_bytes()


def test_inline_image_media_part_is_content_addressed(tmp_path):
    ast, sha256 = _image_ast_with_local_asset()
    output_path = tmp_path / "image.docx"
    compile_docx(ast, output_path, RendererOptions(), resources=DirResourceProvider(ASSETS_DIR))
    with zipfile.ZipFile(output_path) as zf:
        assert f"word/media/{sha256[:16]}.png" in zf.namelist()


def test_missing_resource_provider_fails_strict_compile(tmp_path):
    ast, _sha256 = _image_ast_with_local_asset()
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    assert result.status == "failed"
    assert any(i.code == "RESOURCE_PROVIDER_MISSING" for i in result.issues)


def test_resource_hash_mismatch_fails_strict_compile(tmp_path):
    ast, _sha256 = _image_ast_with_local_asset()
    ast["resources"][0]["sha256"] = "0" * 64
    result = compile_docx(
        ast, tmp_path / "out.docx", RendererOptions(), resources=DirResourceProvider(ASSETS_DIR)
    )
    assert result.status == "failed"
    assert any(i.code == "RESOURCE_HASH_MISMATCH" for i in result.issues)


def test_image_rotation_renders_a_xfrm_rot_instead_of_failing(tmp_path):
    """Rotation used to abort the whole compile. The schema permits it
    on `inline_image` and `floating_image` alike, so one rotated logo
    must not cost a document.
    """
    ast, _sha256 = _image_ast_with_local_asset()
    image_node = ast["sections"][0]["children"][0]["children"][1]
    image_node["rotation_mdeg"] = 45000
    output_path = tmp_path / "out.docx"
    result = compile_docx(
        ast, output_path, RendererOptions(), resources=DirResourceProvider(ASSETS_DIR)
    )
    assert result.status in ("success", "success_with_warnings"), result.issues
    assert not any(i.code == "UNSUPPORTED_IMAGE_ROTATION" for i in result.issues)
    with zipfile.ZipFile(output_path) as zf:
        document = zf.read("word/document.xml").decode()
    # 45 000 mdeg == 45 deg == 2 700 000 sixty-thousandths of a degree.
    assert '<a:xfrm rot="2700000">' in document


def test_two_images_sharing_resource_dedupe_media_part(tmp_path):
    ast, sha256 = _image_ast_with_local_asset()
    paragraph = ast["sections"][0]["children"][0]
    second_image = copy.deepcopy(paragraph["children"][1])
    second_image["id"] = "img2"
    paragraph["children"].append(second_image)

    output_path = tmp_path / "out.docx"
    result = compile_docx(
        ast, output_path, RendererOptions(), resources=DirResourceProvider(ASSETS_DIR)
    )
    assert result.status in ("success", "success_with_warnings")
    with zipfile.ZipFile(output_path) as zf:
        media_files = [n for n in zf.namelist() if n.startswith("word/media/")]
    assert media_files == [f"word/media/{sha256[:16]}.png"]


# --- ERRATA-002 (custom_text marker, percent table width) --------------------

GOLDEN_ERRATA002_DIR = Path(__file__).parent.parent / "data" / "golden" / "errata002"


@pytest.mark.parametrize("name", ["custom_text_list", "percent_width_table"])
def test_errata002_golden_fixture_compiles_and_matches_pinned_xml(tmp_path, name):
    ast = _load(VALID_DIR, name)
    output_path = tmp_path / f"{name}.docx"

    result = compile_docx(ast, output_path, RendererOptions())

    assert result.status in ("success", "success_with_warnings")
    docx_bytes = output_path.read_bytes()
    assert verify_text_conservation(ast, docx_bytes) == []

    result2 = compile_docx(ast, tmp_path / f"{name}_2.docx", RendererOptions())
    assert result.package_hash == result2.package_hash

    expected_xml = (GOLDEN_ERRATA002_DIR / f"{name}.xml").read_bytes()
    assert _document_xml(output_path) == expected_xml


def test_errata002_custom_text_numbering_matches_pinned_xml(tmp_path):
    # E6: marker.text is the literal w:lvlText, marker.suffix appended after.
    ast = _load(VALID_DIR, "custom_text_list")
    output_path = tmp_path / "custom_text_list.docx"
    result = compile_docx(ast, output_path, RendererOptions())
    assert result.status in ("success", "success_with_warnings")

    with zipfile.ZipFile(output_path) as zf:
        numbering_xml = zf.read("word/numbering.xml")
    expected = (GOLDEN_ERRATA002_DIR / "custom_text_list.numbering.xml").read_bytes()
    assert numbering_xml == expected
    assert b'w:lvlText w:val="&gt;&gt;"' in numbering_xml


def test_errata002_percent_width_table_emits_pct_tblw(tmp_path):
    # E7: percent mode reads value_ppm (500000 ppm -> 50% -> w:w="2500").
    ast = _load(VALID_DIR, "percent_width_table")
    output_path = tmp_path / "percent_width_table.docx"
    result = compile_docx(ast, output_path, RendererOptions())
    assert result.status in ("success", "success_with_warnings")
    assert b'<w:tblW w:type="pct" w:w="2500"/>' in _document_xml(output_path)
