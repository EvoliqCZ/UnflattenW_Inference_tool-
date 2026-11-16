"""Tests for `artifacts_zip.py` — pure ZIP + MANIFEST.json builder (T04, design §3.6)
and its API wiring, `GET /v1/conversions/{job_id}/artifacts.zip` (T33)."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

import pytest
from tests.service.test_api_endpoints import AUTH, TOKEN, _client, _submit

from unflatten.service.artifacts_zip import (
    NOT_PAGE_SCOPED,
    ArtifactZipRecord,
    build_artifacts_zip,
)
from unflatten.service.auth import Principal, PrincipalStore
from unflatten.service.job_store import ArtifactMetadata


@dataclass(frozen=True)
class _Record:
    role: str
    page_index: int
    ref: str
    sha256: str
    size_bytes: int
    mime_type: str


def _record(role: str, page_index: int = NOT_PAGE_SCOPED, *, sha: str = "a" * 64) -> _Record:
    mime = {
        "prediction_ir": "application/json",
        "diagnostic_docx": (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        "source_preview": "image/png",
        "docx_preview": "image/png",
    }[role]
    return _Record(
        role=role,
        page_index=page_index,
        ref=f"principals/p/jobs/j/artifacts/{role}/x",
        sha256=f"sha256:{sha}",
        size_bytes=4,
        mime_type=mime,
    )


def _reader_returning(payload: bytes) -> ArtifactZipRecord:
    def _read(_record: ArtifactZipRecord) -> bytes:
        return payload

    return _read


def test_zip_roles_follow_capabilities() -> None:
    records = [
        _record("prediction_ir"),
        _record("diagnostic_docx"),
        _record("source_preview", 0),
    ]
    result = build_artifacts_zip(
        job_id="job_1",
        generated_at="2026-09-27T00:00:00Z",
        capabilities=frozenset({"download_docx"}),
        records=records,
        read_downloadable=_reader_returning(b"content"),
    )
    with zipfile.ZipFile(BytesIO(result.content)) as archive:
        names = set(archive.namelist())
    assert "diagnostic.docx" in names
    assert "source_preview/page-000.png" in names
    assert "prediction_ir.json" not in names


def test_manifest_lists_sha256_size_mime_for_every_file() -> None:
    records = [_record("prediction_ir"), _record("diagnostic_docx")]
    result = build_artifacts_zip(
        job_id="job_1",
        generated_at="2026-09-27T00:00:00Z",
        capabilities=frozenset({"download_prediction_ir", "download_docx"}),
        records=records,
        read_downloadable=_reader_returning(b"content"),
    )
    files = {entry["path"]: entry for entry in result.manifest["files"]}
    assert files["prediction_ir.json"] == {
        "path": "prediction_ir.json",
        "role": "prediction_ir",
        "page_index": NOT_PAGE_SCOPED,
        "sha256": "sha256:" + "a" * 64,
        "size_bytes": 4,
        "mime_type": "application/json",
    }
    assert files["diagnostic.docx"]["mime_type"] == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    with zipfile.ZipFile(BytesIO(result.content)) as archive:
        manifest_in_zip = json.loads(archive.read("MANIFEST.json"))
    assert manifest_in_zip == result.manifest


def test_zip_uses_stored_compression() -> None:
    records = [_record("prediction_ir")]
    result = build_artifacts_zip(
        job_id="job_1",
        generated_at="2026-09-27T00:00:00Z",
        capabilities=frozenset({"download_prediction_ir"}),
        records=records,
        read_downloadable=_reader_returning(b"content"),
    )
    with zipfile.ZipFile(BytesIO(result.content)) as archive:
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED


def test_member_names_server_generated_no_user_text_no_traversal() -> None:
    records = [
        _record("prediction_ir", 0),
        _record("source_preview", 0),
        _record("docx_preview", 0),
        _record("diagnostic_docx"),
    ]
    result = build_artifacts_zip(
        job_id="job_1",
        generated_at="2026-09-27T00:00:00Z",
        capabilities=frozenset({"download_prediction_ir", "download_docx"}),
        records=records,
        read_downloadable=_reader_returning(b"content"),
    )
    with zipfile.ZipFile(BytesIO(result.content)) as archive:
        names = set(archive.namelist())
    assert names == {
        "prediction_ir/page-000.json",
        "source_preview/page-000.png",
        "docx_preview/page-000.png",
        "diagnostic.docx",
        "MANIFEST.json",
    }
    for name in names:
        assert ".." not in name
        assert not name.startswith("/")


def test_omitted_roles_reported_capability_denied() -> None:
    records = [_record("prediction_ir"), _record("diagnostic_docx")]
    result = build_artifacts_zip(
        job_id="job_1",
        generated_at="2026-09-27T00:00:00Z",
        capabilities=frozenset({"download_prediction_ir"}),
        records=records,
        read_downloadable=_reader_returning(b"content"),
    )
    assert result.manifest["omitted_roles"] == [
        {"role": "diagnostic_docx", "reason": "capability_denied"},
        {"role": "docx_preview", "reason": "capability_denied"},
        {"role": "source_preview", "reason": "capability_denied"},
    ]


def test_sha_mismatch_from_reader_propagates() -> None:
    class _ShaMismatch(Exception):
        pass

    def _read(_record: ArtifactZipRecord) -> bytes:
        raise _ShaMismatch("ARTIFACT_HASH_MISMATCH")

    with pytest.raises(_ShaMismatch):
        build_artifacts_zip(
            job_id="job_1",
            generated_at="2026-09-27T00:00:00Z",
            capabilities=frozenset({"download_prediction_ir"}),
            records=[_record("prediction_ir")],
            read_downloadable=_read,
        )


# --- API wiring: GET /v1/conversions/{job_id}/artifacts.zip (T33, design §3.6) ---


def _finalize_with_prediction_ir(client, job_id: str, *, payload: bytes = b'{"sealed":true}'):
    dependencies = client.app.state.dependencies
    lease = dependencies.job_store.claim_next(worker_id="worker_a", now=100.0)
    assert lease is not None
    staged = dependencies.artifact_store.stage(
        principal_id="principal_a",
        job_id=job_id,
        execution_id=lease.execution_id,
        role="prediction_ir",
        mime_type="application/json",
        chunks=[payload],
    )
    published = dependencies.artifact_store.publish(
        staged, lambda _artifact, publish: (publish(), True)[1]
    )
    dependencies.job_store.finalize_success(
        lease,
        artifacts=[
            ArtifactMetadata(
                role="prediction_ir",
                ref=published.ref,
                sha256=published.sha256.removeprefix("sha256:"),
                size_bytes=published.size_bytes,
                mime_type=published.mime_type,
            )
        ],
        diagnostic_outcome={
            "prediction_ir_status": "sealed",
            "claim_scope": "diagnostic",
            "canonical_ast_produced": False,
            "docx_produced": False,
            "reason_codes": ["AST_PROMOTION_UNAVAILABLE"],
        },
        retention_seconds=100,
        now=101.0,
    )


def test_artifacts_zip_returns_zip_with_manifest_and_correct_filename(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    _finalize_with_prediction_ir(client, job_id)

    response = client.get(f"/v1/conversions/{job_id}/artifacts.zip", headers=AUTH)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == f'attachment; filename="{job_id}.zip"'
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("MANIFEST.json"))
    assert "prediction_ir.json" in names
    assert manifest["job_id"] == job_id
    assert manifest["schema_version"] == "1.0"


def test_artifacts_zip_403_without_either_capability(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    _finalize_with_prediction_ir(client, job_id)
    original = client.app.state.dependencies
    digest = hashlib.sha256(TOKEN.encode()).digest()
    client.app.state.dependencies = replace(
        original,
        principal_store=PrincipalStore(
            [(digest, Principal("principal_a", frozenset({"read_status"})))]
        ),
    )

    response = client.get(f"/v1/conversions/{job_id}/artifacts.zip", headers=AUTH)

    assert response.status_code == 403


def test_artifacts_zip_404_outside_succeeded(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]

    response = client.get(f"/v1/conversions/{job_id}/artifacts.zip", headers=AUTH)

    assert response.status_code == 404


def test_artifacts_zip_404_for_unknown_job(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/conversions/job_missing/artifacts.zip", headers=AUTH)

    assert response.status_code == 404


def test_artifacts_zip_410_after_expiry(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    _finalize_with_prediction_ir(client, job_id)
    original = client.app.state.dependencies
    client.app.state.dependencies = replace(original, clock=lambda: 201.0)

    response = client.get(f"/v1/conversions/{job_id}/artifacts.zip", headers=AUTH)

    assert response.status_code == 410


def test_artifacts_zip_410_after_expiry_even_when_no_allowed_role_has_a_record(
    tmp_path: Path,
) -> None:
    """No artifact record matches an allowed role -> `build_artifacts_zip`
    never calls `read_downloadable` (whose own expiry check would otherwise
    catch this), so the route's own expiry check is the only guard."""
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    _finalize_with_prediction_ir(client, job_id)
    original = client.app.state.dependencies
    digest = hashlib.sha256(TOKEN.encode()).digest()
    client.app.state.dependencies = replace(
        original,
        principal_store=PrincipalStore(
            [(digest, Principal("principal_a", frozenset({"download_docx"})))]
        ),
        clock=lambda: 201.0,
    )

    response = client.get(f"/v1/conversions/{job_id}/artifacts.zip", headers=AUTH)

    assert response.status_code == 410


def test_artifacts_zip_cache_control_is_no_store(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    _finalize_with_prediction_ir(client, job_id)

    response = client.get(f"/v1/conversions/{job_id}/artifacts.zip", headers=AUTH)

    assert response.headers["cache-control"] == "no-store"
