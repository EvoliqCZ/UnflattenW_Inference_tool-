from __future__ import annotations

import asyncio
import hashlib
import stat
import tempfile
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
from tests.service.test_api_endpoints import AUTH, PDF, TOKEN, WebSettings, _client, _submit

from unflatten.service.auth import Principal, PrincipalStore
from unflatten.service.job_store import ArtifactMetadata


def test_submit_authenticates_and_authorizes_before_reading_body(tmp_path: Path) -> None:
    client = _client(tmp_path)

    unauthenticated = client.post(
        "/v1/conversions",
        content=PDF,
        headers={"Content-Type": "application/pdf"},
    )
    forbidden = _client(tmp_path / "denied", capabilities=frozenset({"read_status"})).post(
        "/v1/conversions",
        content=PDF,
        headers={
            **AUTH,
            "Content-Type": "application/pdf",
            "Idempotency-Key": "request-1",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
        },
    )

    assert unauthenticated.status_code == 401
    assert forbidden.status_code == 403


def test_missing_auth_returns_without_calling_asgi_receive(tmp_path: Path) -> None:
    app = _client(tmp_path).app
    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        raise AssertionError("request body was read before authentication")

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/conversions",
        "raw_path": b"/v1/conversions",
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1")],
        "client": ("test", 1),
        "server": ("test", 80),
        "root_path": "",
    }

    asyncio.run(app(scope, receive, send))

    assert messages[0]["status"] == 401


def test_session_get_never_sends_cors_headers_for_cross_origin_request(tmp_path: Path) -> None:
    """Design §11 bullet 3: this service sends no `Access-Control-*` header,
    ever -- an `Origin` header from a different site must not change that."""
    client = _client(tmp_path, web=WebSettings(web_auth_path=tmp_path / "web-auth.json"))

    response = client.get("/v1/session", headers={"Origin": "https://evil.example"})

    assert response.status_code == 200
    assert not any(name.lower().startswith("access-control-") for name in response.headers)


def test_submit_rejects_docx_before_persistence_or_job_creation(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client, **{"X-Unflatten-Output": "docx"})

    assert response.status_code == 503
    assert response.json()["code"] == "CAPABILITY_NOT_READY"
    assert list((tmp_path / "artifacts").rglob("*.staged")) == []
    assert client.get("/v1/conversions/job_missing", headers=AUTH).status_code == 404


def test_submit_rejects_unknown_service_header_but_allows_ordinary_headers(tmp_path: Path) -> None:
    client = _client(tmp_path)

    rejected = _submit(client, **{"X-Unflatten-Force": "true"})
    accepted = _submit(client, **{"X-Request-ID": "proxy-id"})

    assert rejected.status_code == 400
    assert accepted.status_code == 202


def test_submit_rejects_wrong_mime_bad_signature_and_idempotency_conflict(tmp_path: Path) -> None:
    client = _client(tmp_path)

    wrong_mime = _submit(client, **{"Content-Type": "application/octet-stream"})
    bad_signature = _submit(client, body=b"not a pdf")
    assert _submit(client).status_code == 202
    conflict = _submit(client, body=PDF + b"changed")

    assert wrong_mime.status_code == 415
    assert bad_signature.status_code == 400
    assert conflict.status_code == 409


def test_submit_stops_oversized_stream_without_leaving_persisted_bytes(tmp_path: Path) -> None:
    client = _client(tmp_path)
    limit = 52_428_800

    response = _submit(client, body=b"%PDF-" + b"x" * (limit + 1))

    assert response.status_code == 413
    assert list((tmp_path / "artifacts").rglob("*.staged")) == []


def test_large_upload_never_spills_outside_private_artifact_root(
    tmp_path: Path, monkeypatch
) -> None:
    forbidden_tmp = tmp_path / "forbidden-tmp"
    forbidden_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(forbidden_tmp))
    monkeypatch.setattr(
        tempfile,
        "SpooledTemporaryFile",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("plaintext tempfile used")),
    )
    client = _client(tmp_path / "service")
    body = b"%PDF-1.7\n" + b"x" * (1024 * 1024 + 17)

    response = _submit(client, body=body)

    assert response.status_code == 202
    staged = list((tmp_path / "service" / "artifacts").rglob("*.staged"))
    assert len(staged) == 1
    assert staged[0].read_bytes() == body
    assert stat.S_IMODE(staged[0].stat().st_mode) == 0o600
    assert list(forbidden_tmp.iterdir()) == []


def test_prediction_ir_download_reverifies_published_bytes(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    dependencies = client.app.state.dependencies
    lease = dependencies.job_store.claim_next(worker_id="worker_a", now=100.0)
    assert lease is not None
    staged = dependencies.artifact_store.stage(
        principal_id="principal_a",
        job_id=job_id,
        execution_id=lease.execution_id,
        role="prediction_ir",
        mime_type="application/json",
        chunks=[b'{"sealed":true}'],
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

    downloaded = client.get(
        f"/v1/conversions/{job_id}/artifacts/prediction_ir", headers=AUTH
    )
    assert downloaded.status_code == 200
    assert downloaded.content == b'{"sealed":true}'

    original = client.app.state.dependencies
    client.app.state.dependencies = replace(original, clock=lambda: 201.0)
    expired = client.get(
        f"/v1/conversions/{job_id}/artifacts/prediction_ir", headers=AUTH
    )
    assert expired.status_code == 410
    client.app.state.dependencies = original

    (dependencies.artifact_store.root / published.ref).write_bytes(b'{"sealed":fake}')
    corrupted = client.get(
        f"/v1/conversions/{job_id}/artifacts/prediction_ir", headers=AUTH
    )
    assert corrupted.status_code == 500


def test_diagnostic_docx_download_reverifies_published_bytes_and_capability(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path, diagnostic_docx_status=True)
    job_id = _submit(client, **{"X-Unflatten-Output": "diagnostic_docx"}).json()["job_id"]
    dependencies = client.app.state.dependencies
    lease = dependencies.job_store.claim_next(worker_id="worker_a", now=100.0)
    assert lease is not None
    docx_bytes = b"PK\x03\x04fake-docx"
    staged = dependencies.artifact_store.stage(
        principal_id="principal_a",
        job_id=job_id,
        execution_id=lease.execution_id,
        role="diagnostic_docx",
        mime_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        chunks=[docx_bytes],
    )
    published = dependencies.artifact_store.publish(
        staged, lambda _artifact, publish: (publish(), True)[1]
    )
    dependencies.job_store.finalize_success(
        lease,
        artifacts=[
            ArtifactMetadata(
                role="diagnostic_docx",
                ref=published.ref,
                sha256=published.sha256.removeprefix("sha256:"),
                size_bytes=published.size_bytes,
                mime_type=published.mime_type,
            )
        ],
        diagnostic_outcome={
            "prediction_ir_status": "not_produced",
            "claim_scope": "diagnostic",
            "canonical_ast_produced": True,
            "docx_produced": True,
            "reason_codes": ["HEADLESS_DEFAULTS_V0_APPLIED"],
            "docx_chain": {
                "policy_id": "headless-defaults.v0",
                "policy_status": "RATIFIED",
                "page_outcome": "terminated",
                "ast_status": "strict_valid",
                "renderer_status": "render_ready",
            },
        },
        retention_seconds=100,
        now=101.0,
    )

    downloaded = client.get(
        f"/v1/conversions/{job_id}/artifacts/diagnostic_docx", headers=AUTH
    )
    assert downloaded.status_code == 200
    assert downloaded.content == docx_bytes
    assert downloaded.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert downloaded.headers["content-disposition"] == 'attachment; filename="diagnostic.docx"'

    original = client.app.state.dependencies
    client.app.state.dependencies = replace(original, clock=lambda: 201.0)
    expired = client.get(
        f"/v1/conversions/{job_id}/artifacts/diagnostic_docx", headers=AUTH
    )
    assert expired.status_code == 410
    client.app.state.dependencies = original


def test_diagnostic_docx_download_requires_download_docx_capability(tmp_path: Path) -> None:
    client = _client(tmp_path, diagnostic_docx_status=True)
    job_id = _submit(client, **{"X-Unflatten-Output": "diagnostic_docx"}).json()["job_id"]
    original = client.app.state.dependencies
    digest = hashlib.sha256(("a" * 32).encode()).digest()
    client.app.state.dependencies = replace(
        original,
        principal_store=PrincipalStore(
            [
                (
                    digest,
                    Principal(
                        "principal_a", frozenset({"read_status", "download_prediction_ir"})
                    ),
                )
            ]
        ),
    )

    forbidden = client.get(
        f"/v1/conversions/{job_id}/artifacts/diagnostic_docx", headers=AUTH
    )

    assert forbidden.status_code == 403


def test_diagnostic_docx_download_404_when_not_produced_or_unknown_job(tmp_path: Path) -> None:
    client = _client(tmp_path, diagnostic_docx_status=True)
    job_id = _submit(client, **{"X-Unflatten-Output": "diagnostic_docx"}).json()["job_id"]

    not_produced = client.get(
        f"/v1/conversions/{job_id}/artifacts/diagnostic_docx", headers=AUTH
    )
    unknown_job = client.get(
        "/v1/conversions/job_missing/artifacts/diagnostic_docx", headers=AUTH
    )

    assert not_produced.status_code == 404
    assert unknown_job.status_code == 404


def test_prediction_ir_route_requires_download_prediction_ir_not_download_docx(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    original = client.app.state.dependencies
    digest = hashlib.sha256(("a" * 32).encode()).digest()
    client.app.state.dependencies = replace(
        original,
        principal_store=PrincipalStore(
            [(digest, Principal("principal_a", frozenset({"read_status", "download_docx"})))]
        ),
    )

    forbidden = client.get(
        f"/v1/conversions/{job_id}/artifacts/prediction_ir", headers=AUTH
    )

    assert forbidden.status_code == 403


def test_download_enforces_privilege_and_owner_hiding(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    original = client.app.state.dependencies
    digest = hashlib.sha256(("a" * 32).encode()).digest()

    client.app.state.dependencies = replace(
        original,
        principal_store=PrincipalStore(
            [(digest, Principal("principal_a", frozenset({"read_status"})))]
        ),
    )
    forbidden = client.get(
        f"/v1/conversions/{job_id}/artifacts/prediction_ir", headers=AUTH
    )
    assert forbidden.status_code == 403

    client.app.state.dependencies = replace(
        original,
        principal_store=PrincipalStore(
            [
                (
                    digest,
                    Principal(
                        "principal_b",
                        frozenset({"read_status", "download_prediction_ir", "download_docx"}),
                    ),
                )
            ]
        ),
    )
    hidden = client.get(f"/v1/conversions/{job_id}", headers=AUTH)
    assert hidden.status_code == 404


def test_source_and_raster_artifacts_have_no_http_route(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]

    for role in ("source_pdf", "page_png"):
        response = client.get(f"/v1/conversions/{job_id}/artifacts/{role}", headers=AUTH)
        assert response.status_code == 404


def _finalize_with_source_preview(
    client: TestClient, job_id: str, *, png_bytes: bytes = b"\x89PNGfake"
) -> None:
    dependencies = client.app.state.dependencies
    lease = dependencies.job_store.claim_next(worker_id="worker_a", now=100.0)
    assert lease is not None
    staged = dependencies.artifact_store.stage(
        principal_id="principal_a",
        job_id=job_id,
        execution_id=lease.execution_id,
        role="source_preview",
        mime_type="image/png",
        chunks=[png_bytes],
    )
    published = dependencies.artifact_store.publish(
        staged, lambda _artifact, publish: (publish(), True)[1]
    )
    dependencies.job_store.finalize_success(
        lease,
        artifacts=[
            ArtifactMetadata(
                role="source_preview",
                ref=published.ref,
                sha256=published.sha256.removeprefix("sha256:"),
                size_bytes=published.size_bytes,
                mime_type=published.mime_type,
                page_index=0,
            )
        ],
        diagnostic_outcome={
            "prediction_ir_status": "not_produced",
            "claim_scope": "diagnostic",
            "canonical_ast_produced": False,
            "docx_produced": False,
            "reason_codes": ["AST_PROMOTION_UNAVAILABLE"],
        },
        retention_seconds=100,
        now=101.0,
    )


def test_source_preview_download_is_inline_with_etag(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    _finalize_with_source_preview(client, job_id)

    response = client.get(
        f"/v1/conversions/{job_id}/artifacts/source_preview/0", headers=AUTH
    )

    assert response.status_code == 200
    assert response.content == b"\x89PNGfake"
    assert response.headers["content-type"] == "image/png"
    assert response.headers["content-disposition"] == (
        'inline; filename="source_preview-000.png"'
    )
    expected_sha = hashlib.sha256(b"\x89PNGfake").hexdigest()
    assert response.headers["etag"] == f'"sha256:{expected_sha}"'
    assert response.headers["cache-control"] == "private, no-cache"


def test_source_preview_if_none_match_returns_304_without_reading_bytes(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    _finalize_with_source_preview(client, job_id)
    url = f"/v1/conversions/{job_id}/artifacts/source_preview/0"
    first = client.get(url, headers=AUTH)
    etag = first.headers["etag"]

    revalidated = client.get(url, headers={**AUTH, "If-None-Match": etag})

    assert revalidated.status_code == 304
    assert revalidated.headers["etag"] == etag
    assert revalidated.headers["cache-control"] == "private, no-cache"
    assert revalidated.content == b""


def test_source_preview_if_none_match_never_304s_a_foreign_job(tmp_path: Path) -> None:
    """Design §3.7: `If-None-Match` matching gets a 304 only *after* the
    ownership check -- a foreign principal presenting the real ETag must
    still see the job as absent (404), never a 304 that would confirm the
    artifact's content is unchanged."""
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    _finalize_with_source_preview(client, job_id)
    url = f"/v1/conversions/{job_id}/artifacts/source_preview/0"
    real_etag = client.get(url, headers=AUTH).headers["etag"]

    original = client.app.state.dependencies
    digest = hashlib.sha256(TOKEN.encode()).digest()
    client.app.state.dependencies = replace(
        original,
        principal_store=PrincipalStore(
            [(digest, Principal("principal_b", frozenset({"read_status", "download_docx"})))]
        ),
    )

    foreign = client.get(url, headers={**AUTH, "If-None-Match": real_etag})

    assert foreign.status_code == 404


def test_source_preview_if_none_match_never_304s_an_expired_job(tmp_path: Path) -> None:
    """Design §3.7: expiry is checked before `If-None-Match` -- an expired
    artifact must return 410, never a stale 304."""
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    _finalize_with_source_preview(client, job_id)
    url = f"/v1/conversions/{job_id}/artifacts/source_preview/0"
    real_etag = client.get(url, headers=AUTH).headers["etag"]

    original = client.app.state.dependencies
    client.app.state.dependencies = replace(original, clock=lambda: 201.0)

    expired = client.get(url, headers={**AUTH, "If-None-Match": real_etag})

    assert expired.status_code == 410
