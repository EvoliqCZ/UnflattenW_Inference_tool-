from __future__ import annotations

import io

import pytest
from PIL import Image

from unflatten.service import image_input


def _png(width: int, height: int, color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), (200, 100, 50))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_build_pdf_from_images_produces_one_page_at_the_target_size() -> None:
    pdf_bytes = image_input.build_pdf_from_images([_png(600, 800)], page_px=(1200, 1600))

    assert pdf_bytes.startswith(b"%PDF-")


def test_build_pdf_from_images_produces_a_multi_page_pdf(tmp_path) -> None:
    import subprocess

    pdf_bytes = image_input.build_pdf_from_images(
        [_png(600, 800), _jpeg(400, 300)], page_px=(1200, 1600)
    )
    pdf_path = tmp_path / "pages.pdf"
    pdf_path.write_bytes(pdf_bytes)

    info = subprocess.run(
        ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True
    ).stdout
    pages_line = next(line for line in info.splitlines() if line.startswith("Pages:"))
    assert int(pages_line.split(":")[1].strip()) == 2


def test_build_pdf_from_images_requires_at_least_one_page() -> None:
    with pytest.raises(image_input.ImageInputInvalid, match="at least one"):
        image_input.build_pdf_from_images([])


def test_build_pdf_from_images_accepts_png_and_jpeg() -> None:
    image_input.build_pdf_from_images([_png(100, 100)])
    image_input.build_pdf_from_images([_jpeg(100, 100)])


def test_build_pdf_from_images_rejects_a_non_image_payload() -> None:
    with pytest.raises(image_input.ImageInputInvalid, match="could not be parsed"):
        image_input.build_pdf_from_images([b"not an image at all"])


def test_build_pdf_from_images_rejects_an_unsupported_format() -> None:
    image = Image.new("RGB", (10, 10), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="BMP")

    with pytest.raises(image_input.ImageInputInvalid, match="unsupported image format"):
        image_input.build_pdf_from_images([buffer.getvalue()])


def test_build_pdf_from_images_rejects_over_the_pixel_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(image_input, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(image_input.ImageInputInvalid, match="pixel ceiling"):
        image_input.build_pdf_from_images([_png(50, 50)])


def test_normalized_page_fits_without_cropping_and_centres_on_white() -> None:
    page = image_input._normalized_page(_png(2000, 1000), page_px=(1000, 1000))

    assert page.size == (1000, 1000)
    # A wide source image fitted into a square page leaves white bars top
    # and bottom; the corner pixel must be the white canvas, not the image.
    assert page.getpixel((0, 0)) == (255, 255, 255)


def test_normalized_page_never_upsamples_a_smaller_source() -> None:
    page = image_input._normalized_page(_png(100, 100), page_px=(2550, 3300))

    assert page.size == (2550, 3300)
    assert page.getpixel((0, 0)) == (255, 255, 255)


def test_oversized_apng_is_rejected_before_entering_pillow(monkeypatch):
    import struct
    import zlib

    def chunk(kind, data):
        return (
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!IIBBBBB", 22000, 22000, 8, 6, 0, 0, 0))
        + chunk(b"acTL", struct.pack("!II", 2, 0))
        + chunk(b"fcTL", struct.pack("!IIIIIHHBB", 0, 1, 1, 0, 0, 1, 10, 1, 0))
        + chunk(b"IDAT", zlib.compress(b"\0\0\0\0\0"))
        + chunk(b"IEND", b"")
    )
    entered = []

    def blocked_decoder(*args, **kwargs):
        entered.append(True)
        raise RuntimeError("audit stopped decoder before allocation")

    monkeypatch.setattr(Image, "open", blocked_decoder)
    with pytest.raises(image_input.ImageInputInvalid, match="pixel ceiling"):
        image_input.build_pdf_from_images([payload])
    assert entered == []


def test_oversized_jpeg_is_rejected_before_entering_pillow(monkeypatch):
    # SOF0 header: 22000 x 22000, three components; no pixel allocation.
    import struct

    payload = b"\xff\xd8\xff\xc0" + struct.pack("!HBHHB", 17, 8, 22000, 22000, 3)
    payload += b"\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9"
    entered = []

    def blocked_decoder(*args, **kwargs):
        entered.append(True)
        raise RuntimeError("audit stopped decoder before allocation")

    monkeypatch.setattr(Image, "open", blocked_decoder)
    with pytest.raises(image_input.ImageInputInvalid, match="pixel ceiling"):
        image_input.build_pdf_from_images([payload])
    assert entered == []


def test_unsupported_image_never_enters_a_decoder(monkeypatch):
    image = Image.new("RGB", (1, 1), "white")
    output = io.BytesIO()
    image.save(output, format="TIFF")
    entered = []

    def blocked_decoder(*args, **kwargs):
        entered.append(True)
        raise RuntimeError("unsupported decoder entered")

    monkeypatch.setattr(Image, "open", blocked_decoder)
    with pytest.raises(image_input.ImageInputInvalid, match="unsupported image format"):
        image_input.build_pdf_from_images([output.getvalue()])
    assert entered == []
