import json
import zipfile
from io import BytesIO
from pathlib import Path

from unflatten.renderer.api import RendererOptions
from unflatten.renderer.compiler import compile_docx
from unflatten.renderer.text_extract import verify_text_conservation

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus" / "valid"


def _load(name: str) -> dict:
    with open(CORPUS_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _fake_docx(document_xml: str) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


def test_verify_text_conservation_clean_for_minimal_paragraph(tmp_path):
    ast = _load("minimal_paragraph")
    compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    docx_bytes = (tmp_path / "out.docx").read_bytes()
    assert verify_text_conservation(ast, docx_bytes) == []


def test_verify_text_conservation_clean_for_tabs_and_breaks(tmp_path):
    ast = _load("paragraph_with_tabs_and_line_break")
    compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    docx_bytes = (tmp_path / "out.docx").read_bytes()
    assert verify_text_conservation(ast, docx_bytes) == []


def test_verify_text_conservation_clean_for_page_break(tmp_path):
    ast = _load("multi_paragraph_with_page_break")
    compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    docx_bytes = (tmp_path / "out.docx").read_bytes()
    assert verify_text_conservation(ast, docx_bytes) == []


def test_verify_text_conservation_detects_dropped_text():
    ast = _load("minimal_paragraph")
    tampered = _fake_docx(
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Hello.</w:t></w:r></w:p></w:body></w:document>"
    )
    issues = verify_text_conservation(ast, tampered)
    assert len(issues) == 1
    assert issues[0].code == "TEXT_CONSERVATION_MISMATCH"
    assert issues[0].severity == "error"


def test_verify_text_conservation_clean_for_column_break(tmp_path):
    ast = _load("minimal_paragraph")
    ast["sections"][0]["children"].append({"id": "cb1", "type": "column_break"})
    compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    docx_bytes = (tmp_path / "out.docx").read_bytes()
    assert verify_text_conservation(ast, docx_bytes) == []


def test_verify_text_conservation_clean_for_empty_visual_paragraph(tmp_path):
    ast = _load("empty_visual_paragraph")
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    docx_bytes = (tmp_path / "out.docx").read_bytes()
    positions = result.metrics["technical_boundary_paragraph_positions"]
    assert positions == []
    assert verify_text_conservation(ast, docx_bytes, positions) == []


def test_genuinely_empty_paragraph_ending_non_final_section_is_not_misclassified(tmp_path):
    """Regression: an empty AST paragraph ending a non-final section
    carries the same XML shape (sectPr in pPr, no runs) that a
    synthesized technical boundary paragraph would — the renderer must
    distinguish them by explicit position, not by inferring "technical"
    from that shape (RFC002 review finding: false
    TEXT_CONSERVATION_MISMATCH / R7 gate failure).
    """
    ast = _load("two_section_first_ends_empty_paragraph")
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    assert result.status in ("success", "success_with_warnings")
    docx_bytes = (tmp_path / "out.docx").read_bytes()

    # No technical paragraph was synthesized — the empty paragraph
    # *is* the AST's own last paragraph of section 1, it just happens
    # to carry no runs.
    positions = result.metrics["technical_boundary_paragraph_positions"]
    assert positions == []

    issues = verify_text_conservation(ast, docx_bytes, positions)
    assert issues == []

    # Explicitly assert the empty paragraph's PARAGRAPH_END survived:
    # two PARAGRAPH_END events (one per section), not one.
    from unflatten.renderer.ooxml.context import DOCUMENT_PART
    from unflatten.renderer.text_extract import _ast_events, _docx_part_events

    assert _ast_events(ast).count("PARAGRAPH_END") == 2
    body_events, _streams = _docx_part_events(
        docx_bytes, {DOCUMENT_PART: [(0, index) for index in positions]}
    )
    assert body_events.count("PARAGRAPH_END") == 2


def test_verify_text_conservation_detects_dropped_paragraph():
    ast = _load("minimal_paragraph")
    tampered = _fake_docx(
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body></w:body></w:document>"
    )
    issues = verify_text_conservation(ast, tampered)
    assert len(issues) == 1
    assert issues[0].details["expected"] != issues[0].details["actual"]


def _fake_docx_with_parts(document_xml: str, extra_parts: dict[str, str]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
        for name, xml in extra_parts.items():
            zf.writestr(name, xml)
    return buffer.getvalue()


def test_docx_part_events_default_part_names_matches_story_part_names():
    """`part_names=None` must be byte-identical to the pre-existing
    behaviour: it discovers story parts via `_story_part_names`'s
    contiguous `header1..N` convention.
    """
    from unflatten.renderer.text_extract import _docx_part_events, _story_part_names

    header_xml = (
        '<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:p><w:r><w:t>Header one.</w:t></w:r></w:p></w:hdr>"
    )
    docx_bytes = _fake_docx_with_parts(
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Body.</w:t></w:r></w:p></w:body></w:document>",
        {"word/header1.xml": header_xml},
    )

    body_default, streams_default = _docx_part_events(docx_bytes, {})
    body_explicit, streams_explicit = _docx_part_events(
        docx_bytes, {}, part_names=None
    )

    with zipfile.ZipFile(BytesIO(docx_bytes)) as zf:
        names = set(zf.namelist())
    assert _story_part_names(names) == ["word/header1.xml"]
    assert body_default == body_explicit
    assert streams_default == streams_explicit
    assert streams_default == [
        ("word/header1.xml", ["TEXT:Header one.", "PARAGRAPH_END"])
    ]


def test_docx_part_events_explicit_part_names_finds_noncontiguous_header():
    """A source package can have `header3.xml` with no `header1.xml` or
    `header2.xml` — `_story_part_names`'s contiguous walk finds
    nothing, but an explicit `part_names` list finds the part.
    """
    from unflatten.renderer.text_extract import _docx_part_events, _story_part_names

    header_xml = (
        '<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:p><w:r><w:t>Header three.</w:t></w:r></w:p></w:hdr>"
    )
    docx_bytes = _fake_docx_with_parts(
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Body.</w:t></w:r></w:p></w:body></w:document>",
        {"word/header3.xml": header_xml},
    )

    with zipfile.ZipFile(BytesIO(docx_bytes)) as zf:
        names = set(zf.namelist())
    assert _story_part_names(names) == []

    _body_default, streams_default = _docx_part_events(docx_bytes, {})
    assert streams_default == []

    _body_explicit, streams_explicit = _docx_part_events(
        docx_bytes, {}, part_names=["word/header3.xml"]
    )
    assert streams_explicit == [
        ("word/header3.xml", ["TEXT:Header three.", "PARAGRAPH_END"])
    ]
