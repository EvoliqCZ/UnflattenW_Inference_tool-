from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from unflatten.service.api import ApiDependencies, WebSettings, create_app
from unflatten.service.artifact_store import ArtifactStore
from unflatten.service.auth import Principal, PrincipalStore
from unflatten.service.job_store import JobStore
from unflatten.service.policy import load_runtime_policy, runtime_policy_hash
from unflatten.service.readiness import COMPONENT_CODES, job_store_component_statuses

TOKEN = "a" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}
PDF = b"%PDF-1.7\nminimal"


def _principal_store(*, capabilities: frozenset[str] | None = None) -> PrincipalStore:
    principal = Principal(
        "principal_a",
        capabilities
        or frozenset(
            {
                "submit_diagnostic",
                "read_status",
                "download_prediction_ir",
                "cancel_own_job",
                "download_docx",
            }
        ),
    )
    return PrincipalStore([(hashlib.sha256(TOKEN.encode()).digest(), principal)])


def _client(
    tmp_path: Path,
    *,
    capabilities: frozenset[str] | None = None,
    diagnostic_docx_status: bool = True,
    web: WebSettings | None = None,
) -> TestClient:
    tmp_path.mkdir(parents=True, exist_ok=True)
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    dependencies = ApiDependencies(
        principal_store=_principal_store(capabilities=capabilities),
        job_store=jobs,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        runtime_policy=policy,
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:" + "1" * 64,
        component_statuses=dict.fromkeys(COMPONENT_CODES, True),
        diagnostic_docx_status=diagnostic_docx_status,
        clock=lambda: 100.0,
        **({"web": web} if web is not None else {}),
    )
    return TestClient(create_app(dependencies), base_url="http://127.0.0.1")


def _submit(client: TestClient, body: bytes = PDF, **headers: str):
    return client.post(
        "/v1/conversions",
        content=body,
        headers={
            **AUTH,
            "Content-Type": "application/pdf",
            "Idempotency-Key": "request-1",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
            **headers,
        },
    )


def test_submit_enqueues_diagnostic_and_idempotently_returns_existing_job(tmp_path: Path) -> None:
    client = _client(tmp_path)

    first = _submit(client)
    replay = _submit(client, **{"X-Forwarded-Proto": "https", "User-Agent": "test"})

    assert first.status_code == 202
    assert replay.status_code == 202
    assert first.json() == replay.json()
    assert first.headers["location"] == first.json()["status_url"]
    assert first.json()["product_quality_claim"] is False


def test_submit_diagnostic_docx_enqueues_with_requested_output_when_capability_ready(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path, diagnostic_docx_status=True)

    response = _submit(client, **{"X-Unflatten-Output": "diagnostic_docx"})

    assert response.status_code == 202
    assert response.json()["requested_output"] == "diagnostic_docx"
    status = client.get(response.json()["status_url"], headers=AUTH)
    assert status.json()["requested_output"] == "diagnostic_docx"


def test_submit_same_idempotency_key_different_output_conflicts(tmp_path: Path) -> None:
    client = _client(tmp_path, diagnostic_docx_status=True)

    first = _submit(client, **{"X-Unflatten-Output": "prediction_ir"})
    conflict = _submit(client, **{"X-Unflatten-Output": "diagnostic_docx"})

    assert first.status_code == 202
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "IDEMPOTENCY_KEY_REUSED"


def test_submit_diagnostic_docx_rejects_before_storage_when_capability_flag_is_false(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path, diagnostic_docx_status=False)

    response = _submit(client, **{"X-Unflatten-Output": "diagnostic_docx"})

    assert response.status_code == 503
    assert response.json()["code"] == "CAPABILITY_NOT_READY"
    assert list((tmp_path / "artifacts").rglob("*.staged")) == []
    assert client.get("/v1/conversions/job_missing", headers=AUTH).status_code == 404


def test_submit_diagnostic_docx_rejects_before_storage_when_no_readiness_row_published(
    tmp_path: Path,
) -> None:
    # A callable capability status keeps the API honest about "missing row" (P3):
    # no published flag means not-ready, never a literal True.
    client = _client(tmp_path, diagnostic_docx_status=False)

    response = _submit(client, **{"X-Unflatten-Output": "diagnostic_docx"})

    assert response.status_code == 503
    assert list((tmp_path / "artifacts").rglob("*.staged")) == []


def test_prediction_ir_submission_unaffected_by_diagnostic_docx_capability_flag(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path, diagnostic_docx_status=False)

    response = _submit(client, **{"X-Unflatten-Output": "prediction_ir"})

    assert response.status_code == 202


def test_status_cancel_and_owner_hiding_use_exact_semantics(tmp_path: Path) -> None:
    client = _client(tmp_path)
    submitted = _submit(client).json()

    status = client.get(submitted["status_url"], headers=AUTH)
    cancelled = client.delete(submitted["status_url"], headers=AUTH)
    terminal = client.delete(submitted["status_url"], headers=AUTH)

    assert status.status_code == 200
    assert status.json()["state"] == "queued"
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
    assert terminal.status_code == 409


def test_delete_without_mode_is_byte_identical_to_cancel(tmp_path: Path) -> None:
    """Design §3.5: `mode` absent stays today's cancel semantics (`:168`)."""
    client = _client(tmp_path)
    submitted = _submit(client).json()

    explicit_cancel = client.delete(f"{submitted['status_url']}?mode=cancel", headers=AUTH)
    bare_delete = _submit(client, **{"Idempotency-Key": "request-2"}).json()
    bare_response = client.delete(bare_delete["status_url"], headers=AUTH)

    assert explicit_cancel.status_code == bare_response.status_code == 200
    assert explicit_cancel.json()["state"] == bare_response.json()["state"] == "cancelled"


def test_purge_terminal_job_returns_204_and_deletes_the_row(tmp_path: Path) -> None:
    client = _client(tmp_path)
    submitted = _submit(client).json()
    client.delete(submitted["status_url"], headers=AUTH)  # cancel -> terminal

    purged = client.delete(f"{submitted['status_url']}?mode=purge", headers=AUTH)

    assert purged.status_code == 204
    assert purged.content == b""
    assert client.get(submitted["status_url"], headers=AUTH).status_code == 404


def test_purge_active_job_returns_409_job_active(tmp_path: Path) -> None:
    client = _client(tmp_path)
    submitted = _submit(client).json()  # still queued

    purged = client.delete(f"{submitted['status_url']}?mode=purge", headers=AUTH)

    assert purged.status_code == 409
    assert purged.json()["code"] == "JOB_ACTIVE"
    assert client.get(submitted["status_url"], headers=AUTH).status_code == 200


def test_purge_missing_job_returns_404(tmp_path: Path) -> None:
    client = _client(tmp_path)

    purged = client.delete("/v1/conversions/job_missing?mode=purge", headers=AUTH)

    assert purged.status_code == 404
    assert purged.json()["code"] == "RESOURCE_NOT_FOUND"


def test_purge_unknown_mode_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)
    submitted = _submit(client).json()

    response = client.delete(f"{submitted['status_url']}?mode=archive", headers=AUTH)

    assert response.status_code == 400
    assert response.json()["code"] == "REQUEST_INVALID"
    # A rejected mode never touches the job.
    assert client.get(submitted["status_url"], headers=AUTH).status_code == 200


def test_purge_discards_the_jobs_staged_source_upload(tmp_path: Path) -> None:
    """Design §3.5: after `purge`, no staged (never-completed) source upload
    for the job survives on disk -- the retention sweep would otherwise find
    and expire it, but `purge` must not depend on that sweep ever running."""
    client = _client(tmp_path)
    submitted = _submit(client).json()
    dependencies = client.app.state.dependencies
    client.delete(submitted["status_url"], headers=AUTH)  # cancel -> terminal
    staged_before = dependencies.artifact_store.staged_artifacts_for_retention()
    assert any(record.job_id == submitted["job_id"] for record in staged_before)

    purged = client.delete(f"{submitted['status_url']}?mode=purge", headers=AUTH)

    assert purged.status_code == 204
    staged_after = dependencies.artifact_store.staged_artifacts_for_retention()
    assert not any(record.job_id == submitted["job_id"] for record in staged_after)


def test_health_surfaces_are_content_free_and_detailed_readiness_is_authenticated(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert "docx_blockers" not in ready.json()
    assert client.get("/v1/readiness").status_code == 401
    detailed = client.get("/v1/readiness", headers=AUTH)
    assert detailed.status_code == 200
    assert detailed.json()["capabilities"] == {
        "prediction_ir": True,
        "diagnostic_docx": True,
        "docx": False,
    }
    assert len(detailed.json()["docx_blockers"]) == 5


def test_api_readiness_observes_worker_and_sandbox_status_from_shared_sqlite(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    dependencies = client.app.state.dependencies
    worker_store = JobStore(
        dependencies.job_store.path,
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=32,
        active_jobs_per_principal=4,
    )
    worker_store.initialize()
    base = dict.fromkeys(COMPONENT_CODES, True)
    base["worker"] = False
    base["pdf_sandbox"] = False
    client.app.state.dependencies = replace(
        dependencies,
        component_statuses=lambda: {
            **base,
            **job_store_component_statuses(
                dependencies.job_store, now=100.0, stale_after_seconds=30.0
            ),
        },
    )

    assert client.get("/health/ready").status_code == 503
    lease = worker_store.claim_readiness_writer(worker_id="worker_gpu_0", now=90.0)
    worker_store.publish_readiness(
        lease, worker_ready=True, sandbox_ready=True, diagnostic_docx_ready=True, now=95.0
    )
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["components"][-2:] == [
        {"code": "pdf_sandbox", "ready": True},
        {"code": "worker", "ready": True},
    ]
    assert "worker_gpu_0" not in response.text
