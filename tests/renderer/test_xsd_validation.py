"""RFC002 validation layer V4 — every generated wml.xsd-rooted part of
every compiled corpus fixture (plus a couple of pinned negative
examples) must validate against the vendored ECMA-376 Transitional
schemas.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from unflatten.renderer.api import RendererOptions
from unflatten.renderer.compiler import compile_docx
from unflatten.renderer.resources import DirResourceProvider
from unflatten.renderer.xsd_validation import validate_part_against_xsd

VALID_DIR = Path(__file__).parent.parent / "data" / "corpus" / "valid"
ASSETS_DIR = Path(__file__).parent.parent / "data" / "assets"

_WML_ROOTED_PARTS = (
    "word/document.xml",
    "word/numbering.xml",
    "word/styles.xml",
    "word/settings.xml",
    "word/fontTable.xml",
)


def _load(name: str) -> dict:
    with open(VALID_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _compile(name: str, tmp_path: Path):
    ast = _load(name)
    resources = None
    if name == "inline_image_with_resource":
        data = (ASSETS_DIR / "tiny_2x2.png").read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        resource = ast["resources"][0]
        resource["sha256"] = sha256
        resource["path"] = "tiny_2x2.png"
        resource["pixel_width"] = 2
        resource["pixel_height"] = 2
        resources = DirResourceProvider(ASSETS_DIR)
    output_path = tmp_path / f"{name}.docx"
    result = compile_docx(ast, output_path, RendererOptions(), resources=resources)
    assert result.status in ("success", "success_with_warnings"), result.issues
    return output_path


@pytest.mark.parametrize(
    "name",
    sorted(p.stem for p in VALID_DIR.glob("*.json")),
)
def test_every_wml_rooted_part_of_every_corpus_fixture_is_xsd_valid(tmp_path, name):
    output_path = _compile(name, tmp_path)
    with zipfile.ZipFile(output_path) as zf:
        names = set(zf.namelist())
        for part_name in _WML_ROOTED_PARTS:
            if part_name not in names:
                continue
            data = zf.read(part_name)
            issues = validate_part_against_xsd(f"/{part_name}", data)
            assert issues == [], [i.message for i in issues]


def test_validate_part_against_xsd_ignores_non_wml_parts():
    assert validate_part_against_xsd("/[Content_Types].xml", b"<not-even-checked/>") == []


def test_validate_part_against_xsd_reports_malformed_xml():
    issues = validate_part_against_xsd("/word/document.xml", b"<w:document><unclosed>")
    assert len(issues) == 1
    assert issues[0].code == "XSD_VALIDATION_ERROR"


def test_validate_part_against_xsd_reports_schema_violation():
    # A well-formed but schema-invalid document: unknown child of w:body.
    xml = (
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:body><w:bogus/></w:body></w:document>"
    )
    issues = validate_part_against_xsd("/word/document.xml", xml)
    assert len(issues) >= 1
    assert all(i.code == "XSD_VALIDATION_ERROR" for i in issues)


def test_compile_docx_fails_strict_when_xsd_invalid(tmp_path, monkeypatch):
    import unflatten.renderer.compiler as compiler_module
    from unflatten.ast.issues import Issue

    monkeypatch.setattr(
        compiler_module,
        "validate_all_parts",
        lambda parts: [
            Issue(
                code="XSD_VALIDATION_ERROR",
                severity="error",
                node_id=None,
                path="/word/document.xml",
                message="forced",
                details={},
            )
        ],
    )
    ast = _load("minimal_paragraph")
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    assert result.status == "failed"
    assert any(i.code == "XSD_VALIDATION_ERROR" for i in result.issues)


def test_validate_xsd_can_be_disabled(tmp_path, monkeypatch):
    import unflatten.renderer.compiler as compiler_module
    from unflatten.ast.issues import Issue

    monkeypatch.setattr(
        compiler_module,
        "validate_all_parts",
        lambda parts: [
            Issue(
                code="XSD_VALIDATION_ERROR",
                severity="error",
                node_id=None,
                path="/word/document.xml",
                message="forced",
                details={},
            )
        ],
    )
    ast = _load("minimal_paragraph")
    result = compile_docx(
        ast, tmp_path / "out.docx", RendererOptions(validate_xsd=False)
    )
    assert result.status in ("success", "success_with_warnings")
