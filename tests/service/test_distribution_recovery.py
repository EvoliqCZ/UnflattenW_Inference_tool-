from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from tests.service.test_api_endpoints import AUTH, _client, _submit

from unflatten.service.artifact_store import _ROLE_MIME
from unflatten.service.job_store import ArtifactMetadata
from unflatten.service.policy import runtime_policy_hash
from unflatten.service.retention import JobStoreRetentionRepository, RetentionSweeper


async def _stream_request(app, chunks, media_type):
    read = []
    messages = []
    source = iter(chunks)
    scope = dict(
        type="http",
        asgi={"version": "3.0"},
        http_version="1.1",
        method="POST",
        scheme="http",
        path="/v1/conversions",
        raw_path=b"/v1/conversions",
        query_string=b"",
        root_path="",
        client=("test", 1),
        server=("127.0.0.1", 80),
        headers=[
            (b"host", b"127.0.0.1"),
            (b"authorization", AUTH["Authorization"].encode()),
            (b"content-type", media_type.encode()),
            (b"idempotency-key", b"bounded-upload"),
            (b"x-unflatten-profile", b"word-core-v0"),
            (b"x-unflatten-output", b"prediction_ir"),
        ],
    )

    async def receive():
        chunk = next(source, b"")
        if chunk:
            read.append(len(chunk))
        return {"type": "http.request", "body": chunk, "more_body": bool(chunk)}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    return status, sum(read), len(read)


def _limited_client(tmp_path):
    client = _client(tmp_path)
    deps = client.app.state.dependencies
    policy = {**deps.runtime_policy, "max_upload_bytes": 1024}
    client.app.state.dependencies = replace(
        deps, runtime_policy=policy, policy_hash=runtime_policy_hash(policy)
    )
    return client


def test_chunked_image_stops_receiving_at_upload_ceiling(tmp_path):
    client = _limited_client(tmp_path)
    status, size, reads = asyncio.run(_stream_request(client.app, [b"x" * 256] * 10, "image/png"))
    assert status == 413
    assert reads == 5 and size == 1280


def test_multipart_total_limit_counts_unknown_file_parts(tmp_path):
    client = _limited_client(tmp_path)
    body = (
        b'--audit\r\nContent-Disposition: form-data; name="ignored"; filename="junk"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n"
        + b"x"
        * 5000
        + b'\r\n--audit\r\nContent-Disposition: form-data; name="page"; filename="ok.pdf"\r\n'
        b"Content-Type: application/pdf\r\n\r\n%PDF-1.7\nminimal\r\n--audit--\r\n"
    )
    chunks = [body[i : i + 256] for i in range(0, len(body), 256)]
    status, size, _ = asyncio.run(
        _stream_request(client.app, chunks, "multipart/form-data; boundary=audit")
    )
    assert status == 413 and size <= 1280


def test_unknown_small_multipart_field_is_rejected(tmp_path):
    client = _client(tmp_path)
    response = client.post(
        "/v1/conversions",
        headers={
            **AUTH,
            "Idempotency-Key": "unknown-part",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
        },
        data={"unexpected": "value"},
        files={"page": ("ok.pdf", b"%PDF-1.7\nminimal", "application/pdf")},
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "role", ["prediction_ir", "diagnostic_docx", "source_preview", "docx_preview"]
)
def test_purge_discards_every_staged_output_role(tmp_path, role):
    client = _client(tmp_path)
    job = _submit(client).json()
    client.delete(job["status_url"], headers=AUTH)
    store = client.app.state.dependencies.artifact_store
    ref = store.stage(
        principal_id="principal_a",
        job_id=job["job_id"],
        execution_id="crashed_attempt",
        role=role,
        mime_type=_ROLE_MIME[role],
        chunks=(b"private synthetic document",),
    )
    assert client.delete(job["status_url"] + "?mode=purge", headers=AUTH).status_code == 204
    assert not (store.root / ref.ref).exists()


def _claimed_client(tmp_path):
    client = _client(tmp_path)
    client.app.state.dependencies = replace(
        client.app.state.dependencies, work_root=tmp_path / "work"
    )
    job = _submit(client).json()
    jobs = client.app.state.dependencies.job_store
    lease = jobs.claim_next(worker_id="worker", now=101)
    directory = tmp_path / "work" / lease.execution_id
    directory.mkdir(parents=True)
    (directory / "page_0.png").write_bytes(b"private synthetic raster")
    return client, job, lease, directory


def _sweeper(client):
    deps = client.app.state.dependencies
    return RetentionSweeper(
        store=deps.artifact_store,
        repository=JobStoreRetentionRepository(
            jobs=deps.job_store, store=deps.artifact_store, work_root=deps.work_root
        ),
        policy=dict(deps.runtime_policy),
    )


def test_purge_removes_crashed_attempt_work_before_deleting_ownership(tmp_path):
    client, job, lease, directory = _claimed_client(tmp_path)
    client.app.state.dependencies.job_store.fail_attempt(
        lease, failure_class="internal", retryable=False, error={"code": "INTERNAL_ERROR"}, now=102
    )
    assert client.delete(job["status_url"] + "?mode=purge", headers=AUTH).status_code == 204
    assert not directory.exists()


def test_sweeper_preserves_live_work_and_removes_it_after_lease_loss(tmp_path):
    client, _, lease, directory = _claimed_client(tmp_path)
    sweeper = _sweeper(client)
    sweeper.sweep(now=102)
    assert directory.exists()
    client.app.state.dependencies.job_store.requeue_expired(now=lease.lease_expires_at + 1)
    sweeper.sweep(now=lease.lease_expires_at + 2)
    assert not directory.exists()


def test_work_cleanup_does_not_follow_symlinks(tmp_path):
    client, _, lease, directory = _claimed_client(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "keep"
    secret.write_text("unrelated")
    (directory / "link").symlink_to(outside, target_is_directory=True)
    client.app.state.dependencies.job_store.requeue_expired(now=lease.lease_expires_at + 1)
    _sweeper(client).sweep(now=lease.lease_expires_at + 2)
    assert not directory.exists() and secret.read_text() == "unrelated"


def test_retention_marks_expired_job_and_filters_it(tmp_path):
    client = _client(tmp_path)
    job = _submit(client).json()
    deps = client.app.state.dependencies
    jobs, store = deps.job_store, deps.artifact_store
    lease = jobs.claim_next(worker_id="worker", now=101)
    staged = store.stage(
        principal_id="principal_a",
        job_id=job["job_id"],
        execution_id=lease.execution_id,
        role="prediction_ir",
        mime_type="application/json",
        chunks=(b"{}",),
    )

    def finalize(ref, publish):
        jobs.finalize_success(
            lease,
            artifacts=[
                ArtifactMetadata(
                    role="prediction_ir",
                    ref=ref.ref,
                    sha256=ref.sha256.removeprefix("sha256:"),
                    size_bytes=ref.size_bytes,
                    mime_type=ref.mime_type,
                )
            ],
            diagnostic_outcome={
                "prediction_ir_status": "sealed",
                "claim_scope": "diagnostic",
                "canonical_ast_produced": False,
                "docx_produced": False,
                "reason_codes": ["DIAGNOSTIC_DOCX_NOT_REQUESTED"],
                "docx_chain": None,
            },
            retention_seconds=deps.runtime_policy["prediction_ir_retention_seconds"],
            now=102,
            publish=publish,
        )
        return True

    store.publish(staged, finalize)
    sweeper = RetentionSweeper(
        store=store,
        repository=JobStoreRetentionRepository(jobs=jobs, store=store),
        policy=dict(deps.runtime_policy),
    )
    sweeper.sweep(now=102 + deps.runtime_policy["prediction_ir_retention_seconds"])
    status = client.get(job["status_url"], headers=AUTH).json()
    assert status["state"] == "expired"
    assert len(client.get("/v1/conversions?state=expired", headers=AUTH).json()["items"]) == 1


def test_failed_work_purge_is_reported_and_keeps_ownership_for_retry(tmp_path, monkeypatch):
    client, job, lease, directory = _claimed_client(tmp_path)
    client.app.state.dependencies.job_store.fail_attempt(
        lease, failure_class="internal", retryable=False, error={"code": "INTERNAL_ERROR"}, now=102
    )

    def unavailable(path):
        raise OSError("private internal directory")

    monkeypatch.setattr("unflatten.service.work_cleanup.shutil.rmtree", unavailable)
    response = client.delete(job["status_url"] + "?mode=purge", headers=AUTH)
    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert "private internal directory" not in response.text
    assert directory.exists()
    assert client.app.state.dependencies.job_store.get_job(job["job_id"], "principal_a") is not None
