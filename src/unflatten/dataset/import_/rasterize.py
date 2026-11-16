"""Bounded DOCX graphic rasterization for charts and EMF/WMF media.

The converter receives a minimal DOCX package rather than a bare graphic: a
bare EMF is letterboxed on A4 and a bare chart XML can return success with the
wrong output.  Raster bytes, never OOXML, cross this module's importer-facing
boundary.
"""

from __future__ import annotations

import atexit
import shutil
import struct
import subprocess
import uuid
import zipfile
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from unflatten.dataset.identity import domain_hash
from unflatten.libreoffice import profile_dir

TARGET_DPI = 192
TIMEOUT_SECONDS = 60
# Bumped from "1": the minimal package now carries the source document's
# theme part, so a chart's `<a:schemeClr>` resolves to the document's real
# colours instead of silently rendering black.
PACKAGE_BUILDER_VERSION = "2"
_THEME_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme"
_THEME_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.theme+xml"
MAX_OUTPUT_BYTES = 20 * 1024 * 1024
MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 5_000_000
_ROOT = Path.home() / "unflatten-probe" / "import-rasterize"
_profile_created = False
_pin_cache: dict[str, str | int] | None = None


class RasterizerUnavailableError(RuntimeError):
    """LibreOffice is unavailable, so an identity-bearing run cannot start."""


@dataclass(frozen=True)
class RasterizedGraphic:
    png: bytes
    pixel_width: int
    pixel_height: int
    raster_pin_hash: str


class RasterizationError(RuntimeError):
    """A per-graphic conversion failure with a stable diagnostic reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def _libreoffice_version() -> str:
    if shutil.which("libreoffice") is None:
        raise RasterizerUnavailableError("libreoffice is not on PATH")
    try:
        proc = subprocess.run(
            ["libreoffice", "--version"], capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RasterizerUnavailableError(str(exc)) from exc
    version = proc.stdout.strip()
    if proc.returncode != 0 or not version:
        raise RasterizerUnavailableError(proc.stderr.strip() or "version query failed")
    return version


def rasterizer_pin() -> dict[str, str | int]:
    """Return the identity-bearing converter configuration."""
    global _pin_cache
    if _pin_cache is not None:
        return dict(_pin_cache)
    pin: dict[str, str | int] = {
        "tool": "libreoffice",
        "version": _libreoffice_version(),
        "target_dpi": TARGET_DPI,
        "filter": "writer_png_Export",
        "package_builder_version": PACKAGE_BUILDER_VERSION,
    }
    pin["raster_pin_hash"] = domain_hash("rfc003.graphic-rasterizer/v1", pin)
    _pin_cache = pin
    return dict(pin)


def _profile() -> Path:
    global _profile_created
    path = profile_dir("import_rasterize")
    path.mkdir(parents=True, exist_ok=True)
    if not _profile_created:
        atexit.register(shutil.rmtree, path, ignore_errors=True)
        _profile_created = True
    return path


def _document_xml(width_emu: int, height_emu: int, kind: str) -> bytes:
    graphic = (
        "<c:chart xmlns:c='http://schemas.openxmlformats.org/drawingml/2006/chart' "
        "r:id='rId1'/>" if kind == "chart" else
        "<pic:pic xmlns:pic='http://schemas.openxmlformats.org/drawingml/2006/picture'>"
        "<pic:nvPicPr><pic:cNvPr id='1' name='graphic'/><pic:cNvPicPr/></pic:nvPicPr>"
        "<pic:blipFill><a:blip r:embed='rId1'/><a:stretch><a:fillRect/></a:stretch>"
        "</pic:blipFill><pic:spPr><a:xfrm><a:off x='0' y='0'/><a:ext "
        f"cx='{width_emu}' cy='{height_emu}'/></a:xfrm><a:prstGeom prst='rect'>"
        "<a:avLst/></a:prstGeom></pic:spPr></pic:pic>"
    )
    uri = (
        "http://schemas.openxmlformats.org/drawingml/2006/chart" if kind == "chart"
        else "http://schemas.openxmlformats.org/drawingml/2006/picture"
    )
    width_twip = max(1, round(width_emu / 635))
    height_twip = max(1, round(height_emu / 635))
    return (
        "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main' "
        "xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships' "
        "xmlns:wp='http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing' "
        "xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main'><w:body><w:p><w:r>"
        f"<w:drawing><wp:inline><wp:extent cx='{width_emu}' cy='{height_emu}'/>"
        "<wp:docPr id='1' name='graphic'/><a:graphic>"
        f"<a:graphicData uri='{uri}'>{graphic}</a:graphicData></a:graphic></wp:inline>"
        f"</w:drawing></w:r></w:p><w:sectPr><w:pgSz w:w='{width_twip}' w:h='{height_twip}'/>"
        "<w:pgMar w:top='0' w:right='0' w:bottom='0' w:left='0' w:header='0' "
        "w:footer='0' w:gutter='0'/></w:sectPr></w:body></w:document>"
    ).encode()


def _package(source: bytes, source_name: str, kind: str,
             width_emu: int, height_emu: int, *, theme: bytes | None = None) -> bytes:
    """Build the minimal single-graphic DOCX the converter opens.

    `theme` is the source document's theme part (`word/theme/theme1.xml`):
    a DrawingML chart resolves `<a:schemeClr>` against it, so a chart
    package built without one renders every scheme-coloured series as
    black — wrong data, not just missing lines, and the two series
    become indistinguishable. It is meaningless for EMF/WMF media
    (EMF/WMF are literal-RGB GDI formats with no `schemeClr` concept).

    A chart can also declare a chart-level `themeOverride` relationship
    (`word/charts/_rels/chartN.xml.rels`) that is supposed to replace
    the document theme for that one chart. This is deliberately NOT
    wired here: the pinned LibreOffice (26.2.5.2) was verified to ignore
    it — rendering a real corpus document byte-identically across all
    23 pages with the relationship present and removed — so packaging
    it would be untested dead code (see rasterize.py module docstring
    history / RFC003-ERRATA import fidelity fix report).
    """
    target = "charts/chart1.xml" if kind == "chart" else f"media/image1.{source_name}"
    graphic_type = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart"
        if kind == "chart" else
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    )
    content = (
        "<Override PartName='/word/charts/chart1.xml' ContentType='application/vnd.openxmlformats-"
        "officedocument.drawingml.chart+xml'/>" if kind == "chart" else
        f"<Default Extension='{source_name}' ContentType='image/x-{source_name}'/>"
    )
    doc_rels = [("rId1", graphic_type, target)]
    if theme is not None:
        content += (
            f"<Override PartName='/word/theme/theme1.xml' ContentType='{_THEME_CONTENT_TYPE}'/>"
        )
        doc_rels.append(("rId2", _THEME_REL_TYPE, "theme/theme1.xml"))
    entries = {
        "[Content_Types].xml": (
            "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
            "<Default Extension='rels' ContentType='application/vnd.openxmlformats-package."
            "relationships+xml'/><Default Extension='xml' ContentType='application/xml'/>"
            "<Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-"
            f"officedocument.wordprocessingml.document.main+xml'/>{content}</Types>"
        ).encode(),
        "_rels/.rels": (
            b"<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
            b"<Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/"
            b"2006/relationships/officeDocument' Target='word/document.xml'/></Relationships>"
        ),
        "word/document.xml": _document_xml(width_emu, height_emu, kind),
        "word/_rels/document.xml.rels": (
            "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
            + "".join(
                f"<Relationship Id='{rid}' Type='{rtype}' Target='{rtarget}'/>"
                for rid, rtype, rtarget in doc_rels
            ) + "</Relationships>"
        ).encode(),
        f"word/{target}": source,
    }
    if theme is not None:
        entries["word/theme/theme1.xml"] = theme
    out = BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return out.getvalue()


def _run_convert(
    input_path: Path, outdir: Path, width: int, height: int
) -> subprocess.CompletedProcess:
    filter_options = (
        f'png:writer_png_Export:{{"PixelWidth":{{"type":"long","value":"{width}"}},'
        f'"PixelHeight":{{"type":"long","value":"{height}"}}}}'
    )
    return subprocess.run(
        ["libreoffice", "--headless", f"-env:UserInstallation=file://{_profile()}",
         "--convert-to", filter_options, "--outdir", str(outdir), str(input_path)],
        capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
    )


def _unfilter(raw: bytes, width: int, height: int, bpp: int) -> bytes:
    stride = width * bpp
    previous = bytearray(stride)
    pixels = bytearray()
    offset = 0
    for _ in range(height):
        filter_type, row = raw[offset], bytearray(raw[offset + 1:offset + stride + 1])
        offset += stride + 1
        for index in range(stride):
            left = row[index - bpp] if index >= bpp else 0
            above = previous[index]
            upper_left = previous[index - bpp] if index >= bpp else 0
            if filter_type == 1:
                row[index] = (row[index] + left) & 255
            elif filter_type == 2:
                row[index] = (row[index] + above) & 255
            elif filter_type == 3:
                row[index] = (row[index] + ((left + above) // 2)) & 255
            elif filter_type == 4:
                p = left + above - upper_left
                nearest = min((left, above, upper_left), key=lambda value: abs(p - value))
                row[index] = (row[index] + nearest) & 255
            elif filter_type != 0:
                raise RasterizationError("INVALID_PNG", f"unsupported PNG filter {filter_type}")
        pixels.extend(row)
        previous = row
    return bytes(pixels)


def validate_png(data: bytes, expected_width: int, expected_height: int) -> tuple[int, int]:
    """Validate type, requested dimensions, bounds and non-uniform pixels."""
    if (
        len(data) < 33
        or len(data) > MAX_OUTPUT_BYTES
        or not data.startswith(b"\x89PNG\r\n\x1a\n")
    ):
        raise RasterizationError("INVALID_PNG", "missing PNG signature or output too large")
    width, height, bit_depth, color_type, _comp, _filter, interlace = struct.unpack(
        ">IIBBBBB", data[16:29]
    )
    if abs(width - expected_width) > 1 or abs(height - expected_height) > 1:
        raise RasterizationError("DIMENSION_MISMATCH", f"got {width}x{height}")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if bit_depth != 8 or channels is None or interlace != 0:
        raise RasterizationError("INVALID_PNG", "unsupported PNG encoding")
    offset, compressed = 8, bytearray()
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind, payload = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + length]
        if kind == b"IDAT":
            compressed.extend(payload)
        offset += 12 + length
    expected_raw_size = height * (width * channels + 1)
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected_raw_size + 1)
        if len(raw) != expected_raw_size or decompressor.unconsumed_tail:
            raise RasterizationError("INVALID_PNG", "unexpected decompressed size")
        pixels = _unfilter(raw, width, height, channels)
    except (zlib.error, IndexError) as exc:
        raise RasterizationError("INVALID_PNG", str(exc)) from exc
    first = pixels[:channels]
    if not first or all(pixels[i:i + channels] == first for i in range(0, len(pixels), channels)):
        raise RasterizationError("DEGENERATE_OUTPUT", "PNG contains one uniform colour")
    return width, height


def rasterize_graphic(source: bytes, *, source_extension: str, kind: str,
                      width_emu: int, height_emu: int,
                      theme: bytes | None = None) -> RasterizedGraphic:
    """Rasterize one chart or EMF/WMF through an isolated minimal package.

    `theme` is the source document's theme part, needed so a chart's
    `<a:schemeClr>` series colours resolve correctly instead of
    rendering black (see `_package`). Callers pass `None` for EMF/WMF
    media, which has no scheme-colour concept.
    """
    width = round(width_emu / 914400 * TARGET_DPI)
    height = round(height_emu / 914400 * TARGET_DPI)
    if len(source) > MAX_SOURCE_BYTES:
        raise RasterizationError("RESOURCE_LIMIT", "source graphic is too large")
    if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
        raise RasterizationError("RESOURCE_LIMIT", f"requested dimensions {width}x{height}")
    pin = rasterizer_pin()
    call_dir = _ROOT / f"call-{uuid.uuid4().hex}"
    try:
        call_dir.mkdir(parents=True, exist_ok=False)
        package_path = call_dir / "graphic.docx"
        package_path.write_bytes(_package(
            source, source_extension, kind, width_emu, height_emu, theme=theme,
        ))
        try:
            proc = _run_convert(package_path, call_dir, width, height)
        except subprocess.TimeoutExpired as exc:
            raise RasterizationError("TIMEOUT", str(exc)) from exc
        except OSError as exc:
            raise RasterizationError("UNAVAILABLE", str(exc)) from exc
        output = call_dir / "graphic.png"
        if proc.returncode != 0 or not output.is_file():
            raise RasterizationError("CONVERSION_FAILED", proc.stderr.strip())
        png = output.read_bytes()
        actual_width, actual_height = validate_png(png, width, height)
        return RasterizedGraphic(png, actual_width, actual_height, str(pin["raster_pin_hash"]))
    finally:
        shutil.rmtree(call_dir, ignore_errors=True)
