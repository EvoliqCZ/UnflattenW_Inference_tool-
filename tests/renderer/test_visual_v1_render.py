"""F2 — rendering the `word-visual-v1` profile (DEC-003).

Covers the four surfaces the profile adds to the package: header/footer
story parts, `footnotes.xml`/`endnotes.xml`, anchored images and
hyperlinks — plus the advanced section settings and whitelisted fields
that landed with them.

The load-bearing invariant here is that **relationship ids are scoped
to a part**: `rId4` in `header1.xml.rels` and `rId4` in
`document.xml.rels` are different targets, and 24.7 % of the corpus has
a header part whose own rels reference an image (DEC-003 §1.2).

XSD validity is necessary but not sufficient: a file LibreOffice opens
without rendering the header is a failure, so the probe reads the text
back out of the produced PDF.
"""

import copy
import hashlib
import io
import json
import re
import zipfile
import zlib
from collections import Counter
from pathlib import Path

import pytest

from unflatten.ast.canonicalizer import canonicalize
from unflatten.dataset.import_ import import_docx
from unflatten.renderer.api import RendererOptions
from unflatten.renderer.compiler import compile_docx
from unflatten.renderer.probe import (
    libreoffice_open_probe,
    libreoffice_pdf_bytes,
    libreoffice_pdf_text,
    probe_available,
)
from unflatten.renderer.resources import DirResourceProvider
from unflatten.renderer.text_extract import _docx_part_events, verify_text_conservation
from unflatten.renderer.units import mpt_to_emu
from unflatten.renderer.xsd_validation import validate_part_against_xsd, wml_root_element

VALID_DIR = Path(__file__).parent.parent / "data" / "corpus" / "visual_v1" / "valid"
ASSETS_DIR = Path(__file__).parent.parent / "data" / "assets"

_FIXTURE_NAMES = sorted(p.stem for p in VALID_DIR.glob("*.json"))
_OPTIONS = RendererOptions(profile="word-visual-v1")


def _prepare(name: str) -> tuple[dict, DirResourceProvider | None]:
    """Load a fixture, pointing any declared resource at the real test
    asset so `compile_docx` can resolve and hash it.
    """
    with open(VALID_DIR / f"{name}.json", encoding="utf-8") as f:
        ast = json.load(f)
    if not ast["resources"]:
        return ast, None
    data = (ASSETS_DIR / "tiny_2x2.png").read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    for resource in ast["resources"]:
        resource["sha256"] = sha256
        resource["path"] = "tiny_2x2.png"
        resource["pixel_width"] = 2
        resource["pixel_height"] = 2
    return ast, DirResourceProvider(ASSETS_DIR)


def _compile(name: str, tmp_path: Path, suffix: str = ""):
    ast, resources = _prepare(name)
    out = tmp_path / f"{name}{suffix}.docx"
    result = compile_docx(ast, out, _OPTIONS, resources=resources)
    assert result.status in ("success", "success_with_warnings"), result.issues
    return ast, out, result


def _parts(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _text_frame_ast() -> dict:
    ast, _resources = _prepare("header_footer_default")
    ast["profile"] = "word-visual-v1.1"
    paragraph = copy.deepcopy(ast["sections"][0]["children"][0])
    paragraph["id"] = "tfp1"
    paragraph["children"][0]["id"] = "tfr1"
    paragraph["children"][0]["text"] = "FRAME-ONLY-TEXT"
    ast["sections"][0]["children"].append(
        {
            "id": "tf1",
            "type": "text_frame",
            "anchor": {
                "horizontal": {"relative_from": "margin", "mode": "align", "align": "right"},
                "vertical": {
                    "relative_from": "paragraph",
                    "mode": "offset",
                    "offset_mpt": 0,
                },
                "wrap": {"mode": "none"},
                "behind_text": False,
                "allow_overlap": True,
                "z_order": 1,
            },
            "children": [paragraph],
        }
    )
    return ast


# --- corpus gate ------------------------------------------------------------


@pytest.mark.parametrize("name", _FIXTURE_NAMES)
def test_every_visual_v1_fixture_compiles_deterministically_and_validates(tmp_path, name):
    ast, out1, result1 = _compile(name, tmp_path, "_1")
    _ast2, out2, result2 = _compile(name, tmp_path, "_2")

    assert canonicalize(ast)["profile"] == "word-visual-v1"
    assert not any(i.severity in ("error", "fatal") for i in result1.issues)
    assert out1.read_bytes() == out2.read_bytes()
    assert result1.package_hash == result2.package_hash

    for part_name, data in _parts(out1).items():
        if wml_root_element(f"/{part_name}") is None:
            continue
        issues = validate_part_against_xsd(f"/{part_name}", data)
        assert issues == [], [i.message for i in issues]


@pytest.mark.parametrize("name", _FIXTURE_NAMES)
def test_every_visual_v1_fixture_conserves_text_across_every_part(tmp_path, name):
    ast, out, result = _compile(name, tmp_path)
    issues = verify_text_conservation(
        ast,
        out.read_bytes(),
        result.metrics["technical_boundary_paragraph_positions"],
        result.metrics["technical_paragraph_paths"],
    )
    assert issues == [], [i.details for i in issues]


# --- per-part relationships -------------------------------------------------


def test_header_image_relationship_lives_in_the_header_parts_own_rels(tmp_path):
    _ast, out, _result = _compile("header_with_image_and_table", tmp_path)
    parts = _parts(out)

    header_rels = parts["word/_rels/header1.xml.rels"].decode()
    assert 'Id="rId1"' in header_rels
    assert "/image" in header_rels
    # The header's own id space starts at rId1 while document.xml.rels
    # is still using rId1-rId3 for styles/settings/fontTable.
    assert '<a:blip r:embed="rId1"/>' in parts["word/header1.xml"].decode()
    document_rels = parts["word/_rels/document.xml.rels"].decode()
    assert 'Id="rId1"' in document_rels and "styles.xml" in document_rels
    # The body references no image, so document.xml.rels must not carry
    # the header's image relationship.
    assert "/image" not in document_rels


def test_logo_shared_by_header_and_body_stores_once_with_two_relationship_ids(tmp_path):
    """`resources` is document-root and deduped by sha256, so the media
    part is stored once; only the *emission* of relationship ids is
    per part.
    """
    ast, resources = _prepare("header_with_image_and_table")
    image = copy.deepcopy(ast["sections"][0]["headers"][0]["children"][0]["children"][1])
    image["id"] = "bi1"
    ast["sections"][0]["children"][0]["children"].append(image)

    out = tmp_path / "shared.docx"
    result = compile_docx(ast, out, _OPTIONS, resources=resources)
    assert result.status in ("success", "success_with_warnings"), result.issues

    parts = _parts(out)
    media = [name for name in parts if name.startswith("word/media/")]
    assert len(media) == 1

    document_rels = parts["word/_rels/document.xml.rels"].decode()
    header_rels = parts["word/_rels/header1.xml.rels"].decode()
    # One media part, two relationship ids: `rId1` inside the header's
    # own table, `rId4` inside document.xml's (rId1-rId3 are the
    # mandatory styles/settings/fontTable parts). A single global
    # sha256 -> rel-id map cannot express this.
    assert 'Id="rId1"' in header_rels and media[0].removeprefix("word/") in header_rels
    assert 'Id="rId4" Type="' in document_rels and "/image" in document_rels
    assert '<a:blip r:embed="rId1"/>' in parts["word/header1.xml"].decode()
    assert '<a:blip r:embed="rId4"/>' in parts["word/document.xml"].decode()


def test_docpr_ids_are_unique_across_parts(tmp_path):
    ast, resources = _prepare("header_with_image_and_table")
    image = copy.deepcopy(ast["sections"][0]["headers"][0]["children"][0]["children"][1])
    image["id"] = "bi1"
    ast["sections"][0]["children"][0]["children"].append(image)
    out = tmp_path / "docpr.docx"
    compile_docx(ast, out, _OPTIONS, resources=resources)

    parts = _parts(out)
    assert '<wp:docPr id="1" ' in parts["word/document.xml"].decode()
    assert '<wp:docPr id="2" ' in parts["word/header1.xml"].decode()


# --- header/footer stories --------------------------------------------------


def test_section_references_each_story_part_by_kind(tmp_path):
    _ast, out, _result = _compile("header_footer_default", tmp_path)
    parts = _parts(out)
    document = parts["word/document.xml"].decode()

    assert '<w:headerReference w:type="default" r:id=' in document
    assert '<w:footerReference w:type="default" r:id=' in document
    assert parts["word/header1.xml"].decode().startswith('<?xml version="1.0"')
    assert "<w:hdr " in parts["word/header1.xml"].decode()
    assert "<w:ftr " in parts["word/footer1.xml"].decode()
    content_types = parts["[Content_Types].xml"].decode()
    assert "wordprocessingml.header+xml" in content_types
    assert "wordprocessingml.footer+xml" in content_types


def test_title_page_and_even_headers_are_derived_from_story_presence(tmp_path):
    _ast, out, _result = _compile("title_page_and_even_headers", tmp_path)
    parts = _parts(out)
    document = parts["word/document.xml"].decode()

    assert '<w:headerReference w:type="first" ' in document
    assert '<w:headerReference w:type="even" ' in document
    assert '<w:titlePg w:val="true"/>' in document
    assert '<w:evenAndOddHeaders w:val="true"/>' in parts["word/settings.xml"].decode()


def test_no_first_story_means_no_titlepg_and_no_evenandoddheaders(tmp_path):
    _ast, out, _result = _compile("header_footer_default", tmp_path)
    parts = _parts(out)
    assert "<w:titlePg" not in parts["word/document.xml"].decode()
    assert "<w:evenAndOddHeaders" not in parts["word/settings.xml"].decode()


def test_header_and_footer_distance_come_from_section_margins(tmp_path):
    ast, resources = _prepare("header_footer_default")
    ast["sections"][0]["margins"]["header_mpt"] = 21600  # 432 twip
    ast["sections"][0]["margins"]["footer_mpt"] = 14400  # 288 twip
    out = tmp_path / "margins.docx"
    compile_docx(ast, out, _OPTIONS, resources=resources)
    assert 'w:header="432" w:footer="288"' in _parts(out)["word/document.xml"].decode()


def test_word_core_v0_keeps_the_default_header_footer_distance(tmp_path):
    """A `word-core-v0` document has no story to derive the distance
    from, so Word's 0.5in default is still the only defensible value.
    """
    core_dir = Path(__file__).parent.parent / "data" / "corpus" / "valid"
    with open(core_dir / "minimal_paragraph.json", encoding="utf-8") as f:
        ast = json.load(f)
    out = tmp_path / "core.docx"
    compile_docx(ast, out, RendererOptions())
    assert 'w:header="720" w:footer="720"' in _parts(out)["word/document.xml"].decode()


def test_empty_story_still_emits_the_one_block_ct_hdrftr_requires(tmp_path):
    ast, resources = _prepare("header_footer_default")
    ast["sections"][0]["headers"][0]["children"] = []
    out = tmp_path / "empty_story.docx"
    result = compile_docx(ast, out, _OPTIONS, resources=resources)
    assert result.status in ("success", "success_with_warnings"), result.issues

    parts = _parts(out)
    assert "<w:p/>" in parts["word/header1.xml"].decode()
    # The synthesized paragraph is technical, so it must not be counted
    # as a lost/extra piece of visible content.
    assert result.metrics["technical_paragraph_paths"]["word/header1.xml"] == [(0,)]
    assert (
        verify_text_conservation(
            ast,
            out.read_bytes(),
            result.metrics["technical_boundary_paragraph_positions"],
            result.metrics["technical_paragraph_paths"],
        )
        == []
    )


# --- footnotes / endnotes ---------------------------------------------------


def test_footnote_reference_uses_an_explicit_custom_mark(tmp_path):
    _ast, out, _result = _compile("footnote_with_note_ref", tmp_path)
    parts = _parts(out)
    document = parts["word/document.xml"].decode()
    footnotes = parts["word/footnotes.xml"].decode()

    assert '<w:footnoteReference w:customMarkFollows="1" w:id="1"/><w:t>1</w:t>' in document
    assert '<w:footnote w:type="separator" w:id="-1">' in footnotes
    assert '<w:footnote w:type="continuationSeparator" w:id="0">' in footnotes
    assert '<w:footnote w:id="1">' in footnotes
    assert "wordprocessingml.footnotes+xml" in parts["[Content_Types].xml"].decode()


def test_footnotes_and_endnotes_are_separate_parts_with_separate_id_spaces(tmp_path):
    _ast, out, _result = _compile("endnote_and_footnote", tmp_path)
    parts = _parts(out)
    document = parts["word/document.xml"].decode()

    assert '<w:footnoteReference w:customMarkFollows="1" w:id="1"/><w:t>1</w:t>' in document
    assert '<w:endnoteReference w:customMarkFollows="1" w:id="1"/><w:t>i</w:t>' in document
    assert '<w:footnote w:id="1">' in parts["word/footnotes.xml"].decode()
    assert '<w:endnote w:id="1">' in parts["word/endnotes.xml"].decode()


# --- floating images --------------------------------------------------------


def test_floating_image_renders_wp_anchor_with_align_and_offset(tmp_path):
    _ast, out, _result = _compile("floating_image_anchored", tmp_path)
    document = _parts(out)["word/document.xml"].decode()

    assert "<wp:anchor " in document
    assert '<wp:positionH relativeFrom="page"><wp:align>right</wp:align></wp:positionH>' in document
    # -12000 mpt == -152400 EMU, vertical, relative to the paragraph.
    assert (
        '<wp:positionV relativeFrom="paragraph"><wp:posOffset>-152400</wp:posOffset>'
        "</wp:positionV>" in document
    )
    assert '<wp:wrapSquare wrapText="bothSides"/>' in document
    assert "<wp:wrapNone/>" in document
    assert 'behindDoc="1"' in document and 'behindDoc="0"' in document
    assert 'allowOverlap="1"' in document and 'allowOverlap="0"' in document
    assert 'relativeHeight="251658240"' in document


def test_inline_image_still_renders_wp_inline_not_an_anchor(tmp_path):
    _ast, out, _result = _compile("header_with_image_and_table", tmp_path)
    header = _parts(out)["word/header1.xml"].decode()
    assert "<wp:inline " in header
    assert "<wp:anchor" not in header


# --- text frames -------------------------------------------------------------


def test_text_frame_emits_deterministic_xsd_valid_anchored_shape(tmp_path):
    ast = _text_frame_ast()
    first = tmp_path / "frame-1.docx"
    second = tmp_path / "frame-2.docx"
    options = RendererOptions(profile="word-visual-v1.1")
    result1 = compile_docx(ast, first, options)
    result2 = compile_docx(ast, second, options)

    assert result1.status in ("success", "success_with_warnings"), result1.issues
    assert result2.package_hash == result1.package_hash
    assert first.read_bytes() == second.read_bytes()
    document = _parts(first)["word/document.xml"]
    xml = document.decode()
    assert "<wp:anchor " in xml
    assert '<wp:positionH relativeFrom="margin"><wp:align>right</wp:align>' in xml
    assert "<wps:wsp>" in xml
    assert "<wps:txbx><w:txbxContent>" in xml
    assert 'mc:Ignorable="wps"' in xml
    assert "FRAME-ONLY-TEXT" in xml
    assert validate_part_against_xsd("/word/document.xml", document) == []


def _content_width_mpt(section: dict) -> int:
    page, margins = section["page"], section["margins"]
    return page["width_mpt"] - margins["left_mpt"] - margins["right_mpt"]


def test_text_frame_align_mode_seed_extent_is_the_section_content_width(tmp_path):
    """The defect this guards: `a:spAutoFit` only ever autofits a
    text-frame's HEIGHT, never its width (confirmed against a real
    document, `fix-import-fidelity-defects`) — so an `align="right"`
    frame keeps whatever `wp:extent/@cx` it is seeded with for its
    whole life. A width guess unrelated to the section's own geometry
    (the old flat 3in) makes a right-aligned frame collapse onto a
    narrow strip pinned to one edge instead of spanning the printable
    area, which is exactly what "shifted to the right side of the
    page" looks like for a disclaimer/footer-style frame.
    """
    ast = _text_frame_ast()
    out = tmp_path / "frame-width.docx"
    result = compile_docx(ast, out, RendererOptions(profile="word-visual-v1.1"))
    assert result.status in ("success", "success_with_warnings"), result.issues
    xml = _parts(out)["word/document.xml"].decode()

    expected_cx = mpt_to_emu(_content_width_mpt(ast["sections"][0]))
    assert f'cx="{expected_cx}"' in xml
    # The old flat seed must not survive for an alignment-mode anchor.
    assert 'cx="2743200"' not in xml


def test_text_frame_uses_the_same_emitter_inside_a_header(tmp_path):
    ast = _text_frame_ast()
    frame = ast["sections"][0]["children"].pop()
    ast["sections"][0]["headers"][0]["children"].append(frame)
    out = tmp_path / "header-frame.docx"
    result = compile_docx(ast, out, RendererOptions(profile="word-visual-v1.1"))

    assert result.status in ("success", "success_with_warnings"), result.issues
    header = _parts(out)["word/header1.xml"]
    assert b"FRAME-ONLY-TEXT" in header
    assert b'<wp:docPr id="1" name="Text Frame 1"/>' in header
    assert validate_part_against_xsd("/word/header1.xml", header) == []
    # A header/footer story is owned by its section exactly like the
    # body: the align-mode seed width follows the frame, not the part.
    expected_cx = mpt_to_emu(_content_width_mpt(ast["sections"][0]))
    assert f'cx="{expected_cx}"'.encode() in header


@pytest.mark.slow
@pytest.mark.skipif(not probe_available(), reason="LibreOffice headless probe unavailable here")
def test_text_frame_is_rendered_but_not_unwrapped_into_body_flow(tmp_path):
    ast = _text_frame_ast()
    out = tmp_path / "frame-render.docx"
    result = compile_docx(ast, out, RendererOptions(profile="word-visual-v1.1"))
    assert result.status in ("success", "success_with_warnings"), result.issues

    body_events, _story_events = _docx_part_events(
        out.read_bytes(), result.metrics["technical_paragraph_paths"]
    )
    assert "TEXT:Body text." in body_events
    assert "TEXT:FRAME-ONLY-TEXT" not in body_events

    rendered_text, issue = libreoffice_pdf_text(out)
    assert issue is None, issue
    assert rendered_text is not None
    assert "Body text." in rendered_text
    assert "FRAME-ONLY-TEXT" in rendered_text


_FRAME_CONTENT_RE = re.compile(rb"/Frame#20contents<<[^>]*>>BDC(.*?)EMC", re.DOTALL)
_TD_RE = re.compile(rb"([\-\d.]+)\s+([\-\d.]+)\s+Td")


def _frame_content_text_x(pdf: bytes) -> list[float]:
    """The x-coordinate (PDF points, page-left origin) of every `Td`
    text-positioning operator LibreOffice tags as belonging to a text
    frame's own content (`/Frame contents` marked content, distinct
    from the body/header/footer flow it is anchored beside). Reusing
    the gridline test's stdlib-`zlib` content-stream reader rather than
    rasterizing — see `_pdf_paint_histogram` above.
    """
    xs: list[float] = []
    for match in re.finditer(rb"\bstream\r?\n(.*?)endstream\b", pdf, re.DOTALL):
        try:
            stream = zlib.decompress(match.group(1))
        except zlib.error:
            continue
        for frame_match in _FRAME_CONTENT_RE.finditer(stream):
            td = _TD_RE.search(frame_match.group(1))
            if td is not None:
                xs.append(float(td.group(1)))
    return xs


@pytest.mark.slow
@pytest.mark.skipif(not probe_available(), reason="LibreOffice headless probe unavailable here")
def test_text_frame_anchored_right_of_margin_renders_at_the_left_margin_not_the_right(
    tmp_path,
):
    """E-2b.4 only proved the frame's text reaches SOME page — not
    where. This is the rendered-output proof the placement defect
    needed: `test_text_frame_stays_in_its_header_or_footer_story`
    (`tests/dataset/import_/test_import_visual_v1.py`) only asserted
    the frame lands in the right AST story, which is exactly the gap
    that let a page-wide disclaimer collapse onto a narrow strip
    pinned to the wrong edge survive human review undetected.

    `_text_frame_ast()`'s frame is `align="right"` relative to the
    section's own margins with NO explicit width — i.e. the family of
    anchor this defect broke. Its `word-visual-v1.1` A4 section has a
    72pt margin on both sides (595276 mpt page, 72000 mpt margins), so
    a frame that spans the full printable width must render with its
    left edge AT the left margin (~72pt + the shape's own text inset,
    empirically ~79pt) — not out past the page's horizontal midpoint,
    which is where a right-aligned box built from an undersized guess
    would land instead.
    """
    ast = _text_frame_ast()
    out = tmp_path / "frame-placement.docx"
    result = compile_docx(ast, out, RendererOptions(profile="word-visual-v1.1"))
    assert result.status in ("success", "success_with_warnings"), result.issues

    pdf, issue = libreoffice_pdf_bytes(out)
    assert issue is None, issue.message if issue else None
    assert pdf is not None

    xs = _frame_content_text_x(pdf)
    assert xs, "no /Frame contents marked-content text found in the rendered PDF"
    page_width_pt = ast["sections"][0]["page"]["width_mpt"] / 1000
    for x in xs:
        assert x < page_width_pt / 2, (
            f"frame text at x={x}pt sits past the page midpoint "
            f"({page_width_pt / 2}pt) — shifted toward the right edge, "
            "the exact defect this test guards"
        )
        assert x < 100, f"frame text at x={x}pt is not at the left margin"


# --- hyperlinks -------------------------------------------------------------


def test_external_hyperlink_gets_an_external_relationship(tmp_path):
    _ast, out, _result = _compile("hyperlinks_internal_and_external", tmp_path)
    parts = _parts(out)
    assert "<w:hyperlink r:id=" in parts["word/document.xml"].decode()
    assert (
        'Target="https://example.org/" TargetMode="External"'
        in parts["word/_rels/document.xml.rels"].decode()
    )


def test_internal_anchor_hyperlink_gets_a_bookmark_at_its_target(tmp_path):
    _ast, out, _result = _compile("hyperlinks_internal_and_external", tmp_path)
    document = _parts(out)["word/document.xml"].decode()
    # ERRATA-001 E5: the anchor is `#<node_id>`, so the bookmark name is
    # the canonical node id of the target paragraph.
    assert "<w:hyperlink w:anchor=" in document
    assert "<w:bookmarkStart " in document
    assert "<w:bookmarkEnd " in document
    anchor = document.split('<w:hyperlink w:anchor="')[1].split('"')[0]
    assert f'w:name="{anchor}"' in document


# --- fields and advanced section settings -----------------------------------


def test_page_field_renders_fldsimple_and_asks_for_a_field_update(tmp_path):
    _ast, out, _result = _compile("page_number_field_in_footer", tmp_path)
    parts = _parts(out)
    footer = parts["word/footer1.xml"].decode()
    assert '<w:fldSimple w:instr=" PAGE "/>' in footer
    assert '<w:fldSimple w:instr=" NUMPAGES "/>' in footer
    # The field carries no cached result on purpose, so the consumer
    # must be told to compute one.
    assert '<w:updateFields w:val="true"/>' in parts["word/settings.xml"].decode()


def test_document_without_fields_keeps_update_fields_false(tmp_path):
    _ast, out, _result = _compile("header_footer_default", tmp_path)
    assert '<w:updateFields w:val="false"/>' in _parts(out)["word/settings.xml"].decode()


def test_page_numbering_renders_pgnumtype(tmp_path):
    _ast, out, _result = _compile("page_numbering_roman_restart", tmp_path)
    document = _parts(out)["word/document.xml"].decode()
    assert '<w:pgNumType w:fmt="lowerRoman" w:start="1"/>' in document


def test_page_numbering_without_start_continues_the_previous_section(tmp_path):
    ast, resources = _prepare("page_numbering_roman_restart")
    for section in ast["sections"]:
        if "page_numbering" in section:
            section["page_numbering"]["start"] = None
    out = tmp_path / "continue.docx"
    compile_docx(ast, out, _OPTIONS, resources=resources)
    document = _parts(out)["word/document.xml"].decode()
    assert "<w:pgNumType " in document
    assert "w:start=" not in document


def test_advanced_section_settings_render_in_ct_sectpr_order(tmp_path):
    """NOTE: `w:lnNumType` has **no render-side proof**. LibreOffice
    does not draw line numbers from it, so the PDF-text probe cannot
    confirm it the way it confirms headers, notes and fields. The XML
    is asserted against CT_LineNumber and CT_SectPr here; treat the
    absence of a render probe as unproven, not as evidence it works.
    """
    _ast, out, _result = _compile("advanced_section_settings", tmp_path)
    document = _parts(out)["word/document.xml"].decode()
    assert '<w:pgBorders w:display="allPages" w:offsetFrom="text">' in document
    assert '<w:lnNumType w:countBy="5" w:start="1" w:distance="180" w:restart="newPage"/>' in (
        document
    )
    assert '<w:vAlign w:val="center"/>' in document
    # EG_SectPrContents order: pgMar, pgBorders, lnNumType, ..., cols,
    # vAlign — a strict xsd:sequence, not a bag of elements.
    assert document.index("<w:pgMar") < document.index("<w:pgBorders")
    assert document.index("<w:pgBorders") < document.index("<w:lnNumType")
    assert document.index("<w:lnNumType") < document.index("<w:vAlign")


# --- nested tables ----------------------------------------------------------


def test_nested_table_renders_and_conserves_all_three_texts(tmp_path):
    ast, out, result = _compile("nested_table", tmp_path)
    document = _parts(out)["word/document.xml"].decode()
    assert document.count("<w:tbl>") == 2
    assert (
        verify_text_conservation(
            ast,
            out.read_bytes(),
            result.metrics["technical_boundary_paragraph_positions"],
            result.metrics["technical_paragraph_paths"],
        )
        == []
    )


# --- LibreOffice render probe (V5/V6) ---------------------------------------


_WML = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_PAINT_OPERATOR = re.compile(rb"(?<!\S)(B\*|b\*|f\*|B|F|S|b|f|s)(?!\S)")


def _border_proof_docx(table_borders: str, *, styled: bool) -> bytes:
    """Build the E-1.1 source fixture; only the style variant has styles.xml."""
    style_ref = "<w:tblStyle w:val='Grid'/>" if styled else ""
    direct_borders = "" if styled else table_borders
    table = (
        f"<w:tbl><w:tblPr>{style_ref}{direct_borders}</w:tblPr>"
        "<w:tblGrid><w:gridCol w:w='2000'/><w:gridCol w:w='2000'/></w:tblGrid>"
        + "".join(
            "<w:tr>"
            + "".join(
                f"<w:tc><w:tcPr/><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc>"
                for text in row
            )
            + "</w:tr>"
            for row in (("A", "B"), ("C", "D"))
        )
        + "</w:tbl><w:p><w:r><w:t>after table</w:t></w:r></w:p>"
    )
    document = (
        f"<w:document xmlns:w='{_WML}'><w:body>{table}"
        "<w:sectPr><w:pgSz w:w='11906' w:h='16838'/>"
        "<w:pgMar w:top='1440' w:right='1440' w:bottom='1440' w:left='1440'/>"
        "</w:sectPr></w:body></w:document>"
    )
    entries = {
        "[Content_Types].xml": (
            "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
            "<Default Extension='rels' ContentType='application/vnd.openxmlformats-package."
            "relationships+xml'/><Default Extension='xml' ContentType='application/xml'/>"
            "<Override PartName='/word/document.xml' ContentType='application/vnd."
            "openxmlformats-officedocument.wordprocessingml.document.main+xml'/></Types>"
        ),
        "_rels/.rels": (
            "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
            "<Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/"
            "2006/relationships/officeDocument' Target='word/document.xml'/></Relationships>"
        ),
        "word/document.xml": document,
    }
    if styled:
        entries["word/styles.xml"] = (
            f"<w:styles xmlns:w='{_WML}'><w:style w:type='table' w:styleId='Grid'>"
            f"<w:tblPr>{table_borders}</w:tblPr></w:style></w:styles>"
        )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as package:
        for name, value in entries.items():
            package.writestr(name, value)
    return output.getvalue()


def _pdf_paint_histogram(pdf: bytes) -> Counter[str]:
    """Read Flate content streams back and count PDF path-paint operators."""
    histogram: Counter[str] = Counter()
    for match in re.finditer(rb"\bstream\r?\n(.*?)endstream\b", pdf, re.DOTALL):
        try:
            stream = zlib.decompress(match.group(1))
        except zlib.error:
            continue
        histogram.update(token.decode("ascii") for token in _PAINT_OPERATOR.findall(stream))
    return histogram


@pytest.mark.slow
@pytest.mark.skipif(not probe_available(), reason="LibreOffice headless probe unavailable here")
def test_style_only_table_gridlines_survive_import_compile_and_render(tmp_path):
    """E-1.1: prove D-1 from source OOXML through ink in LibreOffice's PDF."""
    borders = (
        "<w:tblBorders><w:top w:val='single'/><w:left w:val='single'/>"
        "<w:bottom w:val='single'/><w:right w:val='single'/>"
        "<w:insideH w:val='single'/><w:insideV w:val='single'/></w:tblBorders>"
    )
    sources = {
        "styled": _border_proof_docx(borders, styled=True),
        "direct": _border_proof_docx(borders, styled=False),
        "borderless": _border_proof_docx("", styled=False),
    }
    with zipfile.ZipFile(io.BytesIO(sources["styled"])) as source:
        document_xml = source.read("word/document.xml")
        styles_xml = source.read("word/styles.xml")
    assert b"<w:tblBorders" not in document_xml and b"<w:tcBorders" not in document_xml
    assert b"<w:insideH" in styles_xml and b"<w:insideV" in styles_xml

    histograms: dict[str, Counter[str]] = {}
    for name, source in sources.items():
        imported = import_docx(source, document_id="border-render-proof")
        assert imported.status in ("success", "success_with_warnings"), imported.report
        output = tmp_path / f"{name}.docx"
        rendered = compile_docx(imported.ast, output, RendererOptions())
        assert rendered.status in ("success", "success_with_warnings"), rendered.issues
        pdf, issue = libreoffice_pdf_bytes(output)
        assert issue is None, issue.message if issue else None
        assert pdf is not None
        histograms[name] = _pdf_paint_histogram(pdf)

    totals = {name: sum(histogram.values()) for name, histogram in histograms.items()}
    print(f"E-1.1 PDF path-paint histograms: {dict(histograms)}")
    assert totals["styled"] == totals["direct"], histograms
    assert totals["styled"] > totals["borderless"], histograms


@pytest.mark.slow
@pytest.mark.skipif(not probe_available(), reason="LibreOffice headless probe unavailable here")
@pytest.mark.parametrize("name", _FIXTURE_NAMES)
def test_libreoffice_opens_every_compiled_visual_v1_fixture(tmp_path, name):
    _ast, out, _result = _compile(name, tmp_path)
    issue = libreoffice_open_probe(out)
    assert issue is None, issue.message if issue else None


@pytest.mark.slow
@pytest.mark.skipif(not probe_available(), reason="LibreOffice headless probe unavailable here")
@pytest.mark.parametrize(
    ("name", "expected_texts"),
    [
        ("header_footer_default", ["Running header", "Page footer", "Body text."]),
        ("header_with_image_and_table", ["Logo:", "Cell", "Body text."]),
        ("title_page_and_even_headers", ["Title page", "Body text."]),
        ("footnote_with_note_ref", ["Claim", "The evidence."]),
        ("endnote_and_footnote", ["Page note.", "End note."]),
        ("nested_table", ["Outer", "Inner", "After"]),
        ("hyperlinks_internal_and_external", ["external", "internal"]),
        ("page_number_field_in_footer", ["Page 1 of 1"]),
    ],
)
def test_libreoffice_actually_renders_the_content_onto_the_page(tmp_path, name, expected_texts):
    """XSD validity proves nothing about what reaches the page.

    A header can parse, validate and still be dropped at layout time,
    which is indistinguishable from success at the open probe (a PDF is
    produced, exit code 0). Reading the text back out of the rendered
    PDF is the only evidence that counts.
    """
    _ast, out, _result = _compile(name, tmp_path)
    text, issue = libreoffice_pdf_text(out)
    assert issue is None, issue.message if issue else None
    for expected in expected_texts:
        assert expected in text, f"{expected!r} missing from rendered PDF: {text!r}"


def test_hyperlink_inside_a_footer_is_declared_by_the_footer_part(tmp_path):
    """The per-part rule is not header-specific: a link in a footer must
    land in `footer1.xml.rels`, not in `document.xml.rels`.
    """
    ast, resources = _prepare("page_number_field_in_footer")
    footer_paragraph = ast["sections"][0]["footers"][0]["children"][0]
    run = next(c for c in footer_paragraph["children"] if c["type"] == "run")
    footer_paragraph["children"].append(
        {
            "id": "fhl1",
            "type": "hyperlink",
            "target": {"kind": "external", "uri": "https://example.net/policy"},
            "children": [
                {
                    "id": "fhr1",
                    "type": "run",
                    "text": "policy",
                    "language": run.get("language"),
                    "style": run["style"],
                }
            ],
        }
    )
    out = tmp_path / "footer_link.docx"
    result = compile_docx(ast, out, _OPTIONS, resources=resources)
    assert result.status in ("success", "success_with_warnings"), result.issues

    parts = _parts(out)
    footer_rels = parts["word/_rels/footer1.xml.rels"].decode()
    assert 'Target="https://example.net/policy" TargetMode="External"' in footer_rels
    assert "https://example.net/policy" not in parts["word/_rels/document.xml.rels"].decode()
    assert "<w:hyperlink r:id=" in parts["word/footer1.xml"].decode()
    assert (
        verify_text_conservation(
            ast,
            out.read_bytes(),
            result.metrics["technical_boundary_paragraph_positions"],
            result.metrics["technical_paragraph_paths"],
        )
        == []
    )


def test_floating_image_inside_a_header_uses_the_headers_own_relationship(tmp_path):
    ast, resources = _prepare("header_with_image_and_table")
    inline = ast["sections"][0]["headers"][0]["children"][0]["children"][1]
    inline["type"] = "floating_image"
    inline["anchor"] = {
        "horizontal": {"relative_from": "margin", "mode": "align", "align": "left"},
        "vertical": {"relative_from": "paragraph", "mode": "offset", "offset_mpt": 0},
        "wrap": {"mode": "top_and_bottom"},
        "behind_text": False,
        "allow_overlap": True,
        "z_order": 1,
    }
    out = tmp_path / "header_float.docx"
    result = compile_docx(ast, out, _OPTIONS, resources=resources)
    assert result.status in ("success", "success_with_warnings"), result.issues

    header = _parts(out)["word/header1.xml"].decode()
    assert "<wp:anchor " in header
    assert "<wp:wrapTopAndBottom/>" in header
    assert '<a:blip r:embed="rId1"/>' in header


# --- schema/renderer enum coverage ------------------------------------------

_SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "src"
    / "unflatten"
    / "ast"
    / "schemas"
    / "canonical_ast.schema.json"
)


def _schema_defs() -> dict:
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)["$defs"]


def _enum(node: dict, *path: str) -> set[str]:
    for key in path:
        node = node[key]
    return set(node["enum"])


def test_every_visual_v1_enum_the_renderer_maps_is_covered_by_its_map():
    """A guard that only a test can reach is worthless; this is the
    mechanism instead.

    The profile schema is still growing. Whenever F1 adds a value to
    one of these enums, this fails immediately and names the enum,
    rather than the renderer emitting a guessed token or dying on a
    `KeyError` deep inside a corpus run.
    """
    from unflatten.renderer.ooxml.document_builder import (
        _FIELD_INSTRUCTION_MAP,
        _LINE_NUMBER_RESTART_MAP,
        _PAGE_BORDER_DISPLAY_MAP,
        _PAGE_NUMBER_FORMAT_MAP,
    )
    from unflatten.renderer.ooxml.drawing_builder import _RELATIVE_FROM_MAP, _WRAP_SIDE_MAP

    defs = _schema_defs()
    relative_from = _enum(
        defs["anchor_position_h"]["oneOf"][0], "properties", "relative_from"
    ) | _enum(defs["anchor_position_v"]["oneOf"][0], "properties", "relative_from")

    for name, schema_values, mapped in (
        ("field.field_type", _enum(defs["field"], "properties", "field_type"),
         set(_FIELD_INSTRUCTION_MAP)),
        ("line_numbering.restart", _enum(defs["line_numbering"], "properties", "restart"),
         set(_LINE_NUMBER_RESTART_MAP)),
        ("page_borders.display", _enum(defs["page_borders"], "properties", "display"),
         set(_PAGE_BORDER_DISPLAY_MAP)),
        ("page_numbering.format", _enum(defs["page_numbering"], "properties", "format"),
         set(_PAGE_NUMBER_FORMAT_MAP)),
        ("anchor.relative_from", relative_from, set(_RELATIVE_FROM_MAP)),
        ("anchor.wrap.side", _enum(defs["anchor_wrap"]["oneOf"][2], "properties", "side"),
         set(_WRAP_SIDE_MAP)),
    ):
        assert schema_values <= mapped, f"{name}: unmapped {sorted(schema_values - mapped)}"


def test_enums_passed_through_verbatim_match_the_ecma_tokens():
    """Some AST enums are emitted unmapped because their values are
    already the ECMA-376 tokens. That is only safe while the two
    agree, so it is asserted rather than assumed — the token sets are
    the ones quoted from `vendor/ecma376/transitional/wml.xsd` and
    `dml-wordprocessingDrawing.xsd`.
    """
    defs = _schema_defs()
    assert _enum(defs["header"], "properties", "kind") == {"default", "first", "even"}  # ST_HdrFtr
    assert _enum(defs["footer"], "properties", "kind") == {"default", "first", "even"}
    assert _enum(defs["section"], "properties", "vertical_alignment") == {
        "top",
        "center",
        "both",
        "bottom",
    }  # ST_VerticalJc
    assert _enum(defs["page_borders"], "properties", "offset_from") == {
        "page",
        "text",
    }  # ST_PageBorderOffset
    assert _enum(defs["anchor_position_h"]["oneOf"][0], "properties", "align") == {
        "left",
        "right",
        "center",
        "inside",
        "outside",
    }  # ST_AlignH
    assert _enum(defs["anchor_position_v"]["oneOf"][0], "properties", "align") == {
        "top",
        "bottom",
        "center",
        "inside",
        "outside",
    }  # ST_AlignV
    # EG_WrapType minus wrapTight/wrapThrough, which need a wrapPolygon.
    assert {variant["properties"]["mode"]["const"] for variant in defs["anchor_wrap"]["oneOf"]} == {
        "none",
        "top_and_bottom",
        "square",
    }


# --- guards -----------------------------------------------------------------


def test_image_referencing_an_undeclared_resource_is_reported_by_the_plan(tmp_path):
    """Through `compile_docx` the semantic validator rejects a dangling
    `resource_ref` first, so this branch is exercised at its own entry
    point, `plan_package`, rather than pretended to be reachable from
    the compile path.
    """
    from unflatten.renderer.package_plan import plan_package

    ast, resources = _prepare("header_with_image_and_table")
    ast["resources"] = []
    plan = plan_package(ast, _OPTIONS, resources)
    issue = next(i for i in plan.issues if i.code == "RESOURCE_REF_UNRESOLVED")
    assert issue.details["resource_ref"] == "res_deadbeef"

    # And the compile path still refuses the document, just earlier.
    result = compile_docx(ast, tmp_path / "missing.docx", _OPTIONS, resources=resources)
    assert result.status == "failed"
    assert not (tmp_path / "missing.docx").exists()


def test_note_ref_the_plan_never_registered_is_reported_not_dropped():
    """Reachable through `build_document`, which `story_builder` and the
    renderer tests both call directly; the semantic validator keeps it
    off the `compile_docx` path, so this is a defence in depth, not the
    primary check.
    """
    from unflatten.renderer.api import FontPolicy
    from unflatten.renderer.ooxml.document_builder import build_document

    with open(VALID_DIR / "footnote_with_note_ref.json", encoding="utf-8") as f:
        ast = json.load(f)
    result = build_document(ast, FontPolicy())
    assert any(i.code == "NOTE_REF_UNRESOLVED" for i in result.issues)


def test_field_type_with_no_mapped_instruction_is_reported_not_guessed():
    from unflatten.renderer.api import FontPolicy
    from unflatten.renderer.ooxml.document_builder import build_document

    with open(VALID_DIR / "page_number_field_in_footer.json", encoding="utf-8") as f:
        ast = json.load(f)
    paragraph = ast["sections"][0]["children"][0]
    paragraph["children"].append({"id": "xf1", "type": "field", "field_type": "toc"})
    result = build_document(ast, FontPolicy())
    issue = next(i for i in result.issues if i.code == "UNSUPPORTED_FIELD_TYPE")
    assert issue.details["field_type"] == "toc"


def test_internal_anchor_to_a_non_paragraph_target_is_reported_as_approximated(tmp_path):
    """OOXML anchors bookmarks to paragraphs. ERRATA-001 E5 lets the
    anchor name any node id, so a link to a table is emitted without a
    bookmark — and that loss is declared, not silent.
    """
    ast, resources = _prepare("hyperlinks_internal_and_external")
    section = ast["sections"][0]
    section["children"].append(copy.deepcopy(_load_table()))
    internal = section["children"][1]["children"][0]
    internal["target"]["uri"] = "#tbl1"

    out = tmp_path / "anchor_table.docx"
    result = compile_docx(ast, out, _OPTIONS, resources=resources)
    assert result.status == "success_with_warnings"
    issue = next(
        i for i in result.issues if i.code == "APPROXIMATED_ANCHOR_TARGET_NOT_BOOKMARKABLE"
    )
    assert issue.severity == "warning"
    # Canonicalization renumbers ids, so the anchor is the *canonical*
    # id of the table, not the authored one.
    document = _parts(out)["word/document.xml"].decode()
    anchor = issue.details["anchor"]
    assert f'<w:hyperlink w:anchor="{anchor}">' in document
    assert "<w:bookmarkStart" not in document


def _load_table() -> dict:
    core_dir = Path(__file__).parent.parent / "data" / "corpus" / "valid"
    with open(core_dir / "simple_table_2x2.json", encoding="utf-8") as f:
        table = json.load(f)["sections"][0]["children"][0]
    table["id"] = "tbl1"
    return table


# --- the two story conventions F3 imports against ---------------------------


def test_titlepg_is_emitted_exactly_when_the_section_has_a_first_story(tmp_path):
    """F3 imports `titlePg` <-> a `first` story and refuses to import a
    `first` reference Word never draws. The renderer must be the exact
    inverse or the round-trip diverges. 20.1 % of documents have a
    distinct first-page header, so this is a main path.
    """
    ast, resources = _prepare("header_footer_default")
    assert all(story["kind"] != "first" for story in ast["sections"][0]["headers"])
    out = tmp_path / "no_first.docx"
    compile_docx(ast, out, _OPTIONS, resources=resources)
    assert "<w:titlePg" not in _parts(out)["word/document.xml"].decode()

    ast["sections"][0]["headers"].append(
        {"id": "hfirst", "type": "header", "kind": "first", "children": []}
    )
    out = tmp_path / "with_first.docx"
    result = compile_docx(ast, out, _OPTIONS, resources=resources)
    assert result.status in ("success", "success_with_warnings"), result.issues
    document = _parts(out)["word/document.xml"].decode()
    assert '<w:titlePg w:val="true"/>' in document
    assert '<w:headerReference w:type="first" r:id=' in document


def test_deliberately_blank_first_page_header_is_a_real_part_not_an_omission(tmp_path):
    """`titlePg` with an empty `first` story means "this page has no
    header", which is different from having no story at all (inherit the
    default). Both must survive as distinct packages.
    """
    ast, resources = _prepare("header_footer_default")
    ast["sections"][0]["headers"].append(
        {"id": "hfirst", "type": "header", "kind": "first", "children": []}
    )
    out = tmp_path / "blank_first.docx"
    result = compile_docx(ast, out, _OPTIONS, resources=resources)
    parts = _parts(out)

    # The blank story is a part of its own, referenced by kind, holding
    # the single block CT_HdrFtr demands and nothing else.
    assert "word/header2.xml" in parts
    header2 = parts["word/header2.xml"].decode()
    assert "<w:p/>" in header2
    assert "<w:t" not in header2
    assert result.metrics["technical_paragraph_paths"]["word/header2.xml"] == [(0,)]
    assert (
        verify_text_conservation(
            ast,
            out.read_bytes(),
            result.metrics["technical_boundary_paragraph_positions"],
            result.metrics["technical_paragraph_paths"],
        )
        == []
    )


def test_even_and_odd_headers_is_emitted_exactly_when_an_even_story_exists(tmp_path):
    ast, resources = _prepare("header_footer_default")
    out = tmp_path / "no_even.docx"
    compile_docx(ast, out, _OPTIONS, resources=resources)
    assert "<w:evenAndOddHeaders" not in _parts(out)["word/settings.xml"].decode()

    # An `even` *footer* alone must set it too — the setting is
    # document-wide and not header-specific.
    ast["sections"][0]["footers"].append(
        {"id": "feven", "type": "footer", "kind": "even", "children": []}
    )
    out = tmp_path / "even_footer.docx"
    compile_docx(ast, out, _OPTIONS, resources=resources)
    assert '<w:evenAndOddHeaders w:val="true"/>' in _parts(out)["word/settings.xml"].decode()


def test_page_field_is_live_and_never_a_computed_string(tmp_path):
    """A `PAGE` flattened to text reads the same on every page — the
    defect this node exists to remove. The schema refuses a cached
    result, so the renderer must not invent one either.
    """
    _ast, out, _result = _compile("page_number_field_in_footer", tmp_path)
    footer = _parts(out)["word/footer1.xml"].decode()
    assert '<w:fldSimple w:instr=" PAGE "/>' in footer
    # Self-closing: no `EG_PContent` child, so no cached result of any
    # kind reached the package.
    assert '<w:fldSimple w:instr=" PAGE ">' not in footer
    assert '<w:fldSimple w:instr=" NUMPAGES ">' not in footer


# --- rotation ---------------------------------------------------------------


def test_rotated_floating_image_compiles_and_validates(tmp_path):
    """A rotated anchored logo used to abort the compile in strict mode.
    With body `wp:anchor` at 23.4 % of documents that would have cost
    whole documents during regeneration.
    """
    ast, resources = _prepare("floating_image_anchored")
    ast["sections"][0]["children"][0]["children"][1]["rotation_mdeg"] = 90000
    out = tmp_path / "rotated.docx"
    result = compile_docx(ast, out, _OPTIONS, resources=resources)

    assert result.status in ("success", "success_with_warnings"), result.issues
    assert not any(i.severity in ("error", "fatal") for i in result.issues)
    document = _parts(out)["word/document.xml"].decode()
    # 90 000 mdeg == 90 deg == 5 400 000 sixty-thousandths of a degree,
    # on the anchored picture only.
    assert '<a:xfrm rot="5400000">' in document
    assert document.count("<a:xfrm ") == 1
    assert validate_part_against_xsd("/word/document.xml", _parts(out)["word/document.xml"]) == []


def test_rotated_image_in_a_header_validates_against_ct_hdrftr(tmp_path):
    ast, resources = _prepare("header_with_image_and_table")
    ast["sections"][0]["headers"][0]["children"][0]["children"][1]["rotation_mdeg"] = -30000
    out = tmp_path / "rotated_header.docx"
    result = compile_docx(ast, out, _OPTIONS, resources=resources)
    assert result.status in ("success", "success_with_warnings"), result.issues

    header = _parts(out)["word/header1.xml"]
    assert '<a:xfrm rot="-1800000">' in header.decode()
    assert validate_part_against_xsd("/word/header1.xml", header) == []


def test_unrotated_image_omits_the_rot_attribute_entirely(tmp_path):
    """Keeps every already-compiled document byte-identical: a zero
    rotation must not start emitting `rot="0"`.
    """
    _ast, out, _result = _compile("header_with_image_and_table", tmp_path)
    assert "<a:xfrm>" in _parts(out)["word/header1.xml"].decode()
    assert "<a:xfrm " not in _parts(out)["word/header1.xml"].decode()
