from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from unflatten.renderer.probe import probe_available
from unflatten.service import preview


def _png(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestRenderSourcePreview:
    def test_downsamples_a_larger_source_to_the_max_long_side(self) -> None:
        preview_bytes = preview.render_source_preview(_png(2550, 3300), max_long_side=1400)

        with Image.open(io.BytesIO(preview_bytes)) as image:
            assert max(image.size) == 1400
            assert image.size[0] / image.size[1] == pytest.approx(2550 / 3300, rel=1e-2)

    def test_never_upsamples_a_smaller_source(self) -> None:
        preview_bytes = preview.render_source_preview(_png(400, 300), max_long_side=1400)

        with Image.open(io.BytesIO(preview_bytes)) as image:
            assert image.size == (400, 300)


def _minimal_docx_bytes(tmp_path: Path, page_count: int = 1) -> bytes:
    from unflatten.renderer.api import RendererOptions
    from unflatten.renderer.compiler import compile_docx

    sections = []
    for index in range(page_count):
        sections.append(
            {
                "id": f"s{index}",
                "type": "section",
                "break_type": "next_page",
                "page": {"width_mpt": 595276, "height_mpt": 841890, "orientation": "portrait"},
                "margins": {
                    "top_mpt": 72000, "right_mpt": 72000, "bottom_mpt": 72000,
                    "left_mpt": 72000, "gutter_mpt": 0,
                },
                "columns": {"count": 1, "equal_width": True, "gap_mpt": 18000, "widths_mpt": []},
                "children": [
                    {
                        "id": f"p{index}",
                        "type": "paragraph",
                        "semantic": {"role": "body", "heading_level": None, "outline_level": None},
                        "layout": {
                            "alignment": "left", "direction": "ltr",
                            "indent": {
                                "left_mpt": 0, "right_mpt": 0,
                                "first_line_mpt": 0, "hanging_mpt": 0,
                            },
                            "spacing": {
                                "before_mpt": 0, "after_mpt": 0, "line": {"mode": "single"},
                            },
                            "pagination": {
                                "keep_with_next": False, "keep_lines": False,
                                "page_break_before": False, "widow_control": True,
                            },
                            "tabs": [],
                        },
                        "style": {"background": None, "borders": None},
                        "children": [
                            {
                                "id": f"r{index}",
                                "type": "run",
                                "text": f"page {index}",
                                "language": "en-US",
                                "style": {
                                    "font_family": "Arial", "font_size_mpt": 11000,
                                    "bold": False, "italic": False, "underline": "none",
                                    "strike": False, "color": "#000000", "highlight": None,
                                    "superscript": False, "subscript": False,
                                    "character_spacing_mpt": 0, "small_caps": False,
                                    "all_caps": False,
                                },
                            }
                        ],
                    }
                ],
            }
        )
    ast = {
        "ast_version": "0.1.0",
        "profile": "word-core-v0",
        "document_id": "preview-test",
        "metadata": {"language": None, "title": None, "subject": None},
        "resources": [],
        "sections": sections,
        "extensions": {},
    }
    output_path = tmp_path / "source.docx"
    compile_docx(ast, output_path, RendererOptions())
    return output_path.read_bytes()


class TestRenderDocxPreviewsFailureHandling:
    def test_raises_preview_render_failed_when_libreoffice_returns_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as subprocess_module
        from types import SimpleNamespace

        monkeypatch.setattr(
            subprocess_module, "run",
            lambda *a, **kw: SimpleNamespace(returncode=1, stdout=b"", stderr=b""),
        )

        with pytest.raises(preview.PreviewRenderFailed, match="did not produce a PDF"):
            preview.render_docx_previews(b"anything")

    def test_raises_preview_render_failed_when_libreoffice_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as subprocess_module

        def _raise(*_a, **_kw):
            raise FileNotFoundError("libreoffice not found")

        monkeypatch.setattr(subprocess_module, "run", _raise)

        with pytest.raises(preview.PreviewRenderFailed, match="libreoffice convert-to pdf failed"):
            preview.render_docx_previews(b"anything")


@pytest.mark.slow
@pytest.mark.skipif(not probe_available(), reason="LibreOffice open probe unavailable here.")
class TestRenderDocxPreviews:
    def test_renders_one_png_per_docx_page(self, tmp_path: Path) -> None:
        docx_bytes = _minimal_docx_bytes(tmp_path, page_count=2)

        pages = preview.render_docx_previews(docx_bytes)

        assert len(pages) == 2
        for page_bytes in pages:
            with Image.open(io.BytesIO(page_bytes)) as image:
                assert max(image.size) == preview.DEFAULT_MAX_LONG_SIDE

