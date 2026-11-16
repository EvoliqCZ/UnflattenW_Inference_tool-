"""Digital-PDF text-layer geometry extraction (`pdftotext -bbox-layout`).

Runs poppler's ``pdftotext -bbox-layout`` (subprocess, never a shell) and
parses the XHTML word stream. Float PDF points exist only as *decimal
strings* handed to exact ``Fraction`` arithmetic — every published
coordinate is an integer ppm ``[0, 1_000_000]`` of the post-transform
page (RFC003 §24: floats may be internal computation, never canonical
identity values). Rounding is ``round_half_away_from_zero``, shared with
the renderer's unit conversions (RFC002).
"""

from __future__ import annotations

import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from lxml import etree

from unflatten.renderer.units import round_half_away_from_zero

_PDFTOTEXT_TIMEOUT_SECONDS = 120


class PdfTextExtractionError(RuntimeError):
    """pdftotext is unavailable, failed, or produced unparseable output."""


@dataclass(frozen=True)
class PdfWord:
    """One text-layer word: NFC text plus its integer-ppm page box."""

    page_index: int
    text: str
    bbox_ppm: tuple[int, int, int, int]


@dataclass(frozen=True)
class PdfPage:
    """One page of the text layer. Page dimensions are kept as the exact
    decimal-point strings pdftotext printed (internal only — converted
    to pixels/ppm via exact rational arithmetic, never floats)."""

    page_index: int
    width_pt_text: str
    height_pt_text: str
    words: tuple[PdfWord, ...]
    #: words pdftotext emitted with a degenerate box (``x1 >= x2`` or
    #: ``y1 >= y2`` — the same criterion sidecar.py's ``OBSERVATION_BBOX_INVALID``
    #: uses) and therefore dropped rather than published: a glyph with no
    #: area points at no pixels. Most are U+007F (DEL) rendered as a
    #: zero-width bullet glyph in Symbol-family fonts.
    zero_area_words_dropped: int = 0


def pt_text_to_ppm(coord_text: str | None, extent_text: str | None) -> int:
    """Convert a decimal point-coordinate string to integer ppm of the
    page extent, rounding half away from zero and clamping to
    ``[0, 1_000_000]``. Exact rational arithmetic — no float drift."""
    if coord_text is None or extent_text is None:
        raise PdfTextExtractionError("missing coordinate attribute in pdftotext bbox output")
    try:
        ratio = Fraction(coord_text) / Fraction(extent_text)
    except (ValueError, ZeroDivisionError) as exc:
        raise PdfTextExtractionError(
            f"malformed coordinate in pdftotext bbox output: {coord_text!r}/{extent_text!r}"
        ) from exc
    ppm = round_half_away_from_zero(ratio.numerator * 1_000_000, ratio.denominator)
    return min(max(ppm, 0), 1_000_000)


def pt_text_to_px(extent_text: str, dpi: int) -> int:
    """Convert a decimal point-extent string to pixels at ``dpi``
    (``px = pt * dpi / 72``), rounding half away from zero."""
    scaled = Fraction(extent_text) * dpi
    return round_half_away_from_zero(scaled.numerator, scaled.denominator * 72)


def _local_name(element: etree._Element) -> str:
    return etree.QName(element).localname


def parse_bbox_layout_xhtml(xhtml_bytes: bytes) -> list[PdfPage]:
    """Parse ``pdftotext -bbox-layout`` XHTML into pages of NFC words
    with integer-ppm boxes, in document (reading) order."""
    parser = etree.XMLParser(
        resolve_entities=False, no_network=True, load_dtd=False, recover=False
    )
    try:
        root = etree.fromstring(xhtml_bytes, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise PdfTextExtractionError(f"unparseable -bbox-layout output: {exc}") from exc

    pages: list[PdfPage] = []
    page_elements = [e for e in root.iter() if _local_name(e) == "page"]
    for page_index, page in enumerate(page_elements):
        width_text = page.get("width")
        height_text = page.get("height")
        if width_text is None or height_text is None:
            raise PdfTextExtractionError(f"page {page_index} lacks width/height attributes")
        words: list[PdfWord] = []
        zero_area_dropped = 0
        for word in page.iter():
            if _local_name(word) != "word":
                continue
            text = unicodedata.normalize("NFC", word.text or "")
            if not text:
                continue
            bbox_ppm = (
                pt_text_to_ppm(word.get("xMin"), width_text),
                pt_text_to_ppm(word.get("yMin"), height_text),
                pt_text_to_ppm(word.get("xMax"), width_text),
                pt_text_to_ppm(word.get("yMax"), height_text),
            )
            x1, y1, x2, y2 = bbox_ppm
            if x1 >= x2 or y1 >= y2:
                # a degenerate box points at no pixels — never published
                # as a word (sidecar.py's OBSERVATION_BBOX_INVALID is the
                # same criterion, applied downstream to what did get
                # through; dropping here is what keeps it from getting
                # through in the first place).
                zero_area_dropped += 1
                continue
            words.append(PdfWord(page_index=page_index, text=text, bbox_ppm=bbox_ppm))
        pages.append(
            PdfPage(
                page_index=page_index,
                width_pt_text=width_text,
                height_pt_text=height_text,
                words=tuple(words),
                zero_area_words_dropped=zero_area_dropped,
            )
        )
    return pages


#: pdftotext flags per reading order (DEC-026 addendum 2, pickR v2):
#: ``layout`` is today's ``-bbox-layout`` (poppler's column-aware reading
#: order); ``raw`` is ``-raw -bbox`` (content-stream order, same parser —
#: both emit the same bbox XHTML shape).
_ORDER_FLAGS = {"layout": ("-bbox-layout",), "raw": ("-raw", "-bbox")}


def extract_pdf_words(pdf_bytes: bytes, *, order: str = "layout") -> list[PdfPage]:
    """Run ``pdftotext`` over ``pdf_bytes`` and parse it.

    ``order`` selects the reading order: ``"layout"`` (default, today's
    ``-bbox-layout``) or ``"raw"`` (``-raw -bbox``, poppler's
    content-stream order) — both parsed by
    :func:`parse_bbox_layout_xhtml`, so the zero-area word drop applies
    identically to either. Raises :class:`PdfTextExtractionError` when
    the tool is missing or fails — callers decide whether that skips the
    sidecar."""
    if order not in _ORDER_FLAGS:
        raise ValueError(f"order must be one of {sorted(_ORDER_FLAGS)}, got {order!r}")
    with tempfile.TemporaryDirectory(prefix="unflatten-obs-") as tmp:
        pdf_path = Path(tmp) / "doc.pdf"
        pdf_path.write_bytes(pdf_bytes)
        try:
            proc = subprocess.run(
                ["pdftotext", *_ORDER_FLAGS[order], str(pdf_path), "-"],
                capture_output=True,
                timeout=_PDFTOTEXT_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise PdfTextExtractionError("pdftotext (poppler-utils) not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise PdfTextExtractionError("pdftotext timed out") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise PdfTextExtractionError(f"pdftotext failed (rc={proc.returncode}): {stderr}")
    return parse_bbox_layout_xhtml(proc.stdout)
