from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from tests.service.test_api_endpoints import _client

AUTH = {"Authorization": f"Bearer {'a' * 32}"}


def _png_bytes(width: int = 200, height: int = 300) -> bytes:
    image = Image.new("RGB", (width, height), (10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(width: int = 200, height: int = 300) -> bytes:
    image = Image.new("RGB", (width, height), (200, 100, 50))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _submit_raw(client, body: bytes, content_type: str, **headers: str):
    return client.post(
        "/v1/conversions",
        content=body,
        headers={
            **AUTH,
            "Content-Type": content_type,
            "Idempotency-Key": "request-1",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
            **headers,
        },
    )


def test_submit_raw_png_body_is_accepted_and_becomes_one_page(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit_raw(client, _png_bytes(), "image/png")

    assert response.status_code == 202
    staged_pdfs = list((tmp_path / "artifacts").rglob("*.staged"))
    assert staged_pdfs, "expected the normalized PDF to have been staged"


def test_submit_raw_jpeg_body_is_accepted(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit_raw(client, _jpeg_bytes(), "image/jpeg")

    assert response.status_code == 202


def test_submit_garbage_image_body_is_refused(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit_raw(client, b"not an image", "image/png")

    assert response.status_code == 400
    assert response.json()["code"] == "IMAGE_INPUT_INVALID"


def test_submit_unsupported_media_type_is_refused(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit_raw(client, b"whatever", "text/plain")

    assert response.status_code == 415
    assert response.json()["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_readiness_accepted_media_types_are_all_actually_accepted(tmp_path: Path) -> None:
    client = _client(tmp_path)
    readiness = client.get("/v1/readiness", headers=AUTH)
    accepted = readiness.json()["limits"]["accepted_media_types"]

    for index, media_type in enumerate(accepted):
        response = _submit_raw(
            client, b"x", media_type, **{"Idempotency-Key": f"media-check-{index}"}
        )

        assert response.status_code != 415, media_type
        assert response.json().get("code") != "UNSUPPORTED_MEDIA_TYPE"


def test_submit_multipart_multiple_image_pages_is_accepted(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/v1/conversions",
        files=[
            ("page", ("p0.png", _png_bytes(), "image/png")),
            ("page", ("p1.jpg", _jpeg_bytes(), "image/jpeg")),
        ],
        headers={
            **AUTH,
            "Idempotency-Key": "request-multi",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
        },
    )

    assert response.status_code == 202


def test_submit_multipart_single_pdf_page_is_accepted_as_a_pdf(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/v1/conversions",
        files=[("page", ("doc.pdf", b"%PDF-1.7\nminimal", "application/pdf"))],
        headers={
            **AUTH,
            "Idempotency-Key": "request-pdf-multipart",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
        },
    )

    assert response.status_code == 202


def test_submit_multipart_mixed_pdf_and_image_is_refused(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/v1/conversions",
        files=[
            ("page", ("doc.pdf", b"%PDF-1.7\nminimal", "application/pdf")),
            ("page", ("p1.png", _png_bytes(), "image/png")),
        ],
        headers={
            **AUTH,
            "Idempotency-Key": "request-mixed",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
        },
    )

    assert response.status_code == 415
    assert response.json()["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_submit_multipart_no_page_parts_is_refused(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/v1/conversions",
        files=[("not_page", ("p0.png", _png_bytes(), "image/png"))],
        headers={
            **AUTH,
            "Idempotency-Key": "request-no-page",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
        },
    )

    assert response.status_code == 400
    assert response.json()["code"] == "REQUEST_INVALID"


def test_submit_multipart_too_many_pages_is_refused_with_max_pages(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/v1/conversions",
        files=[("page", (f"p{i}.png", _png_bytes(50, 50), "image/png")) for i in range(101)],
        headers={
            **AUTH,
            "Idempotency-Key": "request-too-many",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "TOO_MANY_PAGES"
    assert response.json()["max_pages"] == 100
