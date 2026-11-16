"""DEC-044 Dodatok D §2: normalize raw image page(s) into one synthetic PDF.

Mirrors `scripts/s5_control_eval.py::image_to_pdf`: each image is fit (never
cropped) and centred on a white `page_px` canvas at 300 DPI, so every capture
class reaches the service's existing PDF admission/rasterization pipeline
with the same page geometry a real PDF page would have. Isolation choice
(RUNBOOK §10, design D §2): "sandbox preferred; else Pillow in worker with
MAX_IMAGE_PIXELS guard + EXIF strip + re-encode" -- this module is the
"else" branch. Pasting the decoded pixels onto a FRESH white canvas (rather
than re-saving the original image) discards EXIF as a side effect; the
pixel-count guard runs on the image's header size, before Pillow decodes
the full pixel buffer, so a decompression-bomb-shaped file is refused
before it can allocate anything large.
"""

from __future__ import annotations

import io
import struct

from PIL import Image

#: US Letter at 300 DPI -- the project's standing convention for a
#: normalized 300 DPI page (matches the sandbox's own raster geometry
#: fixtures and `scripts/s5_control_eval.py`'s own reference page size).
DEFAULT_PAGE_PX = (2550, 3300)
RASTER_DPI = 300
#: DEC-044 Dodatok D §2's own guard, independent of Pillow's global
#: `Image.MAX_IMAGE_PIXELS` (which this module does not rely on, so a
#: caller that has changed the global default elsewhere is not affected) --
#: matches the runtime policy's own `max_raster_pixels` ceiling.
MAX_IMAGE_PIXELS = 36_000_000
_ACCEPTED_IMAGE_FORMATS = frozenset({"PNG", "JPEG"})


class ImageInputInvalid(ValueError):
    """A submitted image page is malformed, oversized, or an unsupported format."""


def _checked_dimensions(width: int, height: int) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ImageInputInvalid("image page could not be parsed: invalid dimensions")
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageInputInvalid(f"image page exceeds the {MAX_IMAGE_PIXELS} pixel ceiling")
    return width, height


def _header_dimensions(data: bytes) -> tuple[int, int]:
    """Bound dimensions using container headers, before invoking any decoder.

    PNG plugins may allocate an APNG disposal canvas inside Image.open(), so
    checking the Pillow image's size after opening is already too late.
    This is admission only; Pillow still verifies and decodes the full file.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(data) < 33 or data[8:16] != b"\0\0\0\rIHDR":
            raise ImageInputInvalid("image page could not be parsed: invalid PNG header")
        dimensions = _checked_dimensions(*struct.unpack("!II", data[16:24]))
        offset = 33
        while offset + 12 <= len(data):
            length = int.from_bytes(data[offset : offset + 4], "big")
            kind = data[offset + 4 : offset + 8]
            offset += length + 12
            if offset > len(data) or kind == b"IHDR":
                raise ImageInputInvalid("image page could not be parsed: invalid PNG chunks")
            if kind == b"IEND":
                return dimensions
        raise ImageInputInvalid("image page could not be parsed: incomplete PNG")
    if data.startswith(b"\xff\xd8"):
        offset = 2
        dimensions = None
        sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while offset < len(data):
            if data[offset] != 0xFF:
                break
            while offset < len(data) and data[offset] == 0xFF:
                offset += 1
            if offset >= len(data):
                break
            marker = data[offset]
            offset += 1
            if marker in {0xDA, 0xD9}:
                if dimensions is not None:
                    return dimensions
                break
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue
            if offset + 2 > len(data):
                break
            length = int.from_bytes(data[offset : offset + 2], "big")
            if length < 2 or offset + length > len(data):
                break
            if marker in sof_markers:
                if length < 8:
                    break
                height, width = struct.unpack("!HH", data[offset + 3 : offset + 7])
                current = _checked_dimensions(width, height)
                if dimensions is not None and dimensions != current:
                    raise ImageInputInvalid(
                        "image page could not be parsed: conflicting JPEG sizes"
                    )
                dimensions = current
            offset += length
        raise ImageInputInvalid("image page could not be parsed: invalid JPEG header")
    if data.startswith((b"BM", b"GIF8", b"II", b"MM", b"RIFF")):
        raise ImageInputInvalid("unsupported image format")
    raise ImageInputInvalid("image page could not be parsed: expected PNG or JPEG")


def _normalized_page(image_bytes: bytes, *, page_px: tuple[int, int]) -> Image.Image:
    width, height = page_px
    dimensions = _header_dimensions(image_bytes)
    try:
        with Image.open(io.BytesIO(image_bytes), formats=("PNG", "JPEG")) as opened:
            opened.verify()
    except Exception as exc:
        # Untrusted-input boundary (error-handling.md: "map internal errors to
        # user-facing messages"): Pillow raises many distinct exception types
        # for a malformed image, including `DecompressionBombError`, which
        # subclasses `Exception` directly, not `OSError`/`ValueError`.
        raise ImageInputInvalid(f"image page could not be parsed: {exc}") from exc
    # `verify()` closes the file object; a second `Image.open` is required to
    # actually decode pixels (this is Pillow's own documented pattern).
    try:
        with Image.open(io.BytesIO(image_bytes), formats=("PNG", "JPEG")) as reopened:
            if reopened.format not in _ACCEPTED_IMAGE_FORMATS:
                raise ImageInputInvalid(f"unsupported image format: {reopened.format}")
            pixel_count = reopened.width * reopened.height
            if pixel_count > MAX_IMAGE_PIXELS:
                raise ImageInputInvalid(
                    f"image page exceeds the {MAX_IMAGE_PIXELS} pixel ceiling: {pixel_count}"
                )
            if reopened.size != dimensions:
                raise ImageInputInvalid("image page could not be parsed: inconsistent dimensions")
            capture = reopened.convert("RGB")
    except ImageInputInvalid:
        raise
    except Exception as exc:
        raise ImageInputInvalid(f"image page could not be decoded: {exc}") from exc
    scale = min(width / capture.width, height / capture.height, 1.0) or 1.0
    fitted = capture.resize(
        (max(1, round(capture.width * scale)), max(1, round(capture.height * scale)))
    )
    page = Image.new("RGB", (width, height), "white")
    page.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return page


def build_pdf_from_images(
    pages: list[bytes], *, page_px: tuple[int, int] = DEFAULT_PAGE_PX
) -> bytes:
    """Normalize every page image and save them as ONE multi-page PDF."""
    if not pages:
        raise ImageInputInvalid("at least one image page is required")
    normalized = [_normalized_page(page, page_px=page_px) for page in pages]
    buffer = io.BytesIO()
    normalized[0].save(
        buffer,
        format="PDF",
        save_all=True,
        append_images=normalized[1:],
        resolution=float(RASTER_DPI),
        # Idempotency hashes these bytes. Volatile PDF timestamps must not
        # change an identical ordered image request into a different job.
        creationDate=None,
        modDate=None,
    )
    return buffer.getvalue()
