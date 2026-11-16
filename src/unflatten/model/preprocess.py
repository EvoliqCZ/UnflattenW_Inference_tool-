"""Page preprocessing (RFC004 §12, A1 slice): PNG → PagePlane → tensor.

The factory stores page rasters as pdftoppm PNGs (8-bit RGB, non-interlaced),
so a minimal stdlib-only decoder covers exactly that subset plus 8-bit RGBA
(alpha dropped). No Pillow/numpy dependency; torch is imported lazily inside
:meth:`PagePreprocessor.to_tensor` so the base suite never requires it
(DEC-001).
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

from unflatten.model.input_contract import PagePlane

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Decompression-bomb guard: decoded pixel bytes per page may not exceed
#: this (≈ 64 A4 pages at 300 dpi RGB); §11.5 per-page pixel limit.
_MAX_DECODED_BYTES = 512 * 1024 * 1024


class PngDecodeError(ValueError):
    """The bytes are not a PNG in the supported subset (8-bit RGB/RGBA,
    non-interlaced) or are structurally invalid."""


class PagePreprocessor:
    """Decodes stored page images and converts planes to model tensors."""

    def load_png(self, source: Path | bytes, *, page_ordinal: int = 0) -> PagePlane:
        """Decode a PNG (8-bit RGB/RGBA, non-interlaced) to a raw RGB8 plane."""
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        width, height, channels, raw = _decode_png(data)
        stride = width * channels
        rows = _unfilter(raw, width, height, channels)
        if channels == 4:
            rgb = bytearray(width * height * 3)
            out = 0
            for row in rows:
                for x in range(0, stride, 4):
                    rgb[out : out + 3] = row[x : x + 3]
                    out += 3
            pixels = bytes(rgb)
        else:
            pixels = b"".join(rows)
        return PagePlane(
            page_ordinal=page_ordinal, width_px=width, height_px=height, pixels=pixels
        )

    def to_tensor(self, plane: PagePlane) -> Any:
        """Deterministic float32 CHW tensor in ``[0, 1]`` (pixels / 255).

        Imports torch lazily (DEC-001): the base suite must not require it.
        """
        import torch  # noqa: PLC0415 — lazy by policy

        flat = torch.frombuffer(bytearray(plane.pixels), dtype=torch.uint8)
        hwc = flat.reshape(plane.height_px, plane.width_px, 3)
        return hwc.permute(2, 0, 1).contiguous().to(torch.float32) / 255.0


def _decode_png(data: bytes) -> tuple[int, int, int, bytes]:
    """Parse chunks and inflate IDAT → (width, height, channels, filtered bytes)."""
    if not data.startswith(_PNG_SIGNATURE):
        raise PngDecodeError("missing PNG signature")
    offset = len(_PNG_SIGNATURE)
    width = height = channels = 0
    idat = bytearray()
    seen_ihdr = seen_iend = False
    while offset < len(data):
        if offset + 8 > len(data):
            raise PngDecodeError("truncated chunk header")
        (length,) = struct.unpack_from(">I", data, offset)
        tag = data[offset + 4 : offset + 8]
        body_start = offset + 8
        body_end = body_start + length
        if body_end + 4 > len(data):
            raise PngDecodeError(f"truncated {tag!r} chunk")
        body = data[body_start:body_end]
        if tag == b"IHDR":
            width, height, channels = _parse_ihdr(body)
            seen_ihdr = True
        elif tag == b"IDAT":
            if not seen_ihdr:
                raise PngDecodeError("IDAT before IHDR")
            idat += body
        elif tag == b"IEND":
            seen_iend = True
            break
        offset = body_end + 4  # skip CRC
    if not seen_ihdr or not seen_iend or not idat:
        raise PngDecodeError("missing IHDR, IDAT or IEND chunk")
    expected = height * (1 + width * channels)
    if expected > _MAX_DECODED_BYTES:
        raise PngDecodeError(f"decoded size {expected} exceeds the per-page pixel limit")
    try:
        # Streaming inflate with a HARD output cap: zlib.decompress's
        # bufsize is only an initial hint — the whole stream would inflate
        # into memory before any length check (decompression bomb).
        decomp = zlib.decompressobj()
        raw = decomp.decompress(bytes(idat), expected + 1)
    except zlib.error as exc:
        raise PngDecodeError(f"IDAT inflate failed: {exc}") from exc
    if len(raw) > expected or not decomp.eof or decomp.unconsumed_tail:
        raise PngDecodeError(
            f"IDAT stream inflates beyond the declared size ({expected} bytes)"
        )
    if len(raw) != expected:
        raise PngDecodeError(f"decoded {len(raw)} bytes, expected {expected}")
    return width, height, channels, raw


def _parse_ihdr(body: bytes) -> tuple[int, int, int]:
    if len(body) != 13:
        raise PngDecodeError(f"IHDR must be 13 bytes, got {len(body)}")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", body
    )
    if width <= 0 or height <= 0:
        raise PngDecodeError(f"dimensions must be > 0, got {width}x{height}")
    if bit_depth != 8:
        raise PngDecodeError(f"unsupported bit depth {bit_depth} (only 8)")
    if color_type not in (2, 6):
        raise PngDecodeError(f"unsupported color type {color_type} (only RGB/RGBA)")
    if compression != 0 or filter_method != 0:
        raise PngDecodeError("unsupported compression/filter method")
    if interlace != 0:
        raise PngDecodeError("interlaced PNGs are not supported")
    return width, height, 3 if color_type == 2 else 4


def _unfilter(raw: bytes, width: int, height: int, channels: int) -> list[bytes]:
    """Undo per-row PNG filters (types 0–4) → list of raw pixel rows."""
    stride = width * channels
    bpp = channels
    rows: list[bytes] = []
    prev = bytes(stride)
    pos = 0
    for _ in range(height):
        ftype = raw[pos]
        row = bytearray(raw[pos + 1 : pos + 1 + stride])
        pos += 1 + stride
        if ftype == 1:  # Sub
            for i in range(bpp, stride):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + (left + prev[i]) // 2) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                up = prev[i]
                up_left = prev[i - bpp] if i >= bpp else 0
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                if pa <= pb and pa <= pc:
                    predictor = left
                elif pb <= pc:
                    predictor = up
                else:
                    predictor = up_left
                row[i] = (row[i] + predictor) & 0xFF
        elif ftype != 0:
            raise PngDecodeError(f"unknown filter type {ftype}")
        prev = bytes(row)
        rows.append(prev)
    return rows
