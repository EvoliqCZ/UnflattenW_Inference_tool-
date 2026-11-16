"""T25 (design §3.2, A10): `X-Unflatten-Label` decoding, `input_kind`
classification, and idempotent resubmission keeping the original label."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from tests.service.test_api_endpoints import AUTH, PDF, _client, _submit
from tests.service.test_api_image_input import _jpeg_bytes, _png_bytes


def _label_header(value: str) -> dict[str, str]:
    return {"X-Unflatten-Label": urllib.parse.quote(value, safe="")}


def _status(client, job_id: str) -> dict:
    return client.get(f"/v1/conversions/{job_id}", headers=AUTH).json()


def test_label_header_percent_decoded_and_reaches_status(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client, **_label_header("board report Q3.pdf"))

    assert response.status_code == 202
    assert _status(client, response.json()["job_id"])["label"] == "board report Q3.pdf"


def test_label_header_absent_is_null_in_status(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client)

    assert response.status_code == 202
    assert _status(client, response.json()["job_id"])["label"] is None


def test_label_header_nfc_normalizes_decomposed_unicode(tmp_path: Path) -> None:
    client = _client(tmp_path)
    decomposed = "résumé.pdf"  # "résumé.pdf" as combining accents
    precomposed = "résumé.pdf"

    response = _submit(client, **_label_header(decomposed))

    assert response.status_code == 202
    assert _status(client, response.json()["job_id"])["label"] == precomposed


def test_label_header_strips_surrounding_whitespace(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client, **_label_header("  quarterly.pdf  "))

    assert response.status_code == 202
    assert _status(client, response.json()["job_id"])["label"] == "quarterly.pdf"


def test_label_header_invalid_percent_encoding_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client, **{"X-Unflatten-Label": "%FF%FE"})

    assert response.status_code == 400
    assert response.json()["code"] == "REQUEST_INVALID"


def test_label_header_forbidden_filesystem_character_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client, **_label_header('board/report.pdf'))

    assert response.status_code == 400
    assert response.json()["code"] == "REQUEST_INVALID"


def test_label_header_forbidden_control_category_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client, **_label_header("board\x07report.pdf"))

    assert response.status_code == 400
    assert response.json()["code"] == "REQUEST_INVALID"


def test_label_header_empty_after_strip_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client, **_label_header("   "))

    assert response.status_code == 400
    assert response.json()["code"] == "REQUEST_INVALID"


def test_label_header_duplicate_headers_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/v1/conversions",
        content=PDF,
        headers=[
            ("Authorization", AUTH["Authorization"]),
            ("Content-Type", "application/pdf"),
            ("Idempotency-Key", "request-1"),
            ("X-Unflatten-Profile", "word-core-v0"),
            ("X-Unflatten-Output", "prediction_ir"),
            ("X-Unflatten-Label", "a.pdf"),
            ("X-Unflatten-Label", "b.pdf"),
        ],
    )

    assert response.status_code == 400
    assert response.json()["code"] == "REQUEST_INVALID"


def test_label_header_at_120_code_points_is_accepted(tmp_path: Path) -> None:
    client = _client(tmp_path)
    label = "a" * 120

    response = _submit(client, **_label_header(label))

    assert response.status_code == 202
    assert _status(client, response.json()["job_id"])["label"] == label


def test_label_header_over_120_code_points_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client, **_label_header("a" * 121))

    assert response.status_code == 400
    assert response.json()["code"] == "REQUEST_INVALID"


def test_idempotent_resubmission_keeps_original_label(tmp_path: Path) -> None:
    client = _client(tmp_path)

    first = _submit(client, **_label_header("original.pdf"))
    replay = _submit(client, **_label_header("renamed.pdf"))

    assert first.status_code == 202
    assert replay.status_code == 202
    assert first.json()["job_id"] == replay.json()["job_id"]
    assert _status(client, first.json()["job_id"])["label"] == "original.pdf"


def test_input_kind_is_pdf_for_raw_pdf_body(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client)

    assert response.status_code == 202
    assert _status(client, response.json()["job_id"])["input_kind"] == "pdf"


def test_input_kind_is_image_for_raw_image_body(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/v1/conversions",
        content=_png_bytes(),
        headers={
            **AUTH,
            "Content-Type": "image/png",
            "Idempotency-Key": "request-image",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
        },
    )

    assert response.status_code == 202
    assert _status(client, response.json()["job_id"])["input_kind"] == "image"


def test_input_kind_is_pdf_for_multipart_single_pdf_part(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/v1/conversions",
        files=[("page", ("doc.pdf", b"%PDF-1.7\nminimal", "application/pdf"))],
        headers={
            **AUTH,
            "Idempotency-Key": "request-multipart-pdf",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
        },
    )

    assert response.status_code == 202
    assert _status(client, response.json()["job_id"])["input_kind"] == "pdf"


def test_input_kind_is_image_for_multipart_image_parts(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/v1/conversions",
        files=[
            ("page", ("p0.png", _png_bytes(), "image/png")),
            ("page", ("p1.jpg", _jpeg_bytes(), "image/jpeg")),
        ],
        headers={
            **AUTH,
            "Idempotency-Key": "request-multipart-image",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
        },
    )

    assert response.status_code == 202
    assert _status(client, response.json()["job_id"])["input_kind"] == "image"
