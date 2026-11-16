from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from unflatten.service.api import ApiDependencies, create_app
from unflatten.service.artifact_store import ArtifactStore
from unflatten.service.auth import Principal, PrincipalStore
from unflatten.service.job_store import ArtifactMetadata, JobStore
from unflatten.service.policy import load_runtime_policy, runtime_policy_hash
from unflatten.service.readiness import COMPONENT_CODES

TOKEN = "a" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _principal_store() -> PrincipalStore:
    principal = Principal(
        "principal_a",
        frozenset({"submit_diagnostic", "read_status", "download_prediction_ir",
                   "cancel_own_job", "download_docx"}),
    )
    return PrincipalStore([(hashlib.sha256(TOKEN.encode()).digest(), principal)])


def _client(tmp_path: Path) -> TestClient:
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
        principal_store=_principal_store(),
        job_store=jobs,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        runtime_policy=policy,
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:" + "1" * 64,
        component_statuses=dict.fromkeys(COMPONENT_CODES, True),
        diagnostic_docx_status=True,
        clock=lambda: 200.0,
    )
    return TestClient(create_app(dependencies), base_url="http://127.0.0.1")


def _multi_page_job(client: TestClient, tmp_path: Path) -> str:
    """Drive a real 2-page job to 'succeeded' directly through JobStore's
    lease/finalize primitives -- the same calls `worker.py` makes."""
    jobs = client.app.state.dependencies.job_store
    policy = client.app.state.dependencies.runtime_policy
    job_id = "job_multi_api_test"
    jobs.create_job(
        job_id=job_id, principal_id="principal_a", idempotency_key_hash="key1",
        request_digest="digest1", requested_output="diagnostic_docx",
        policy_hash=runtime_policy_hash(policy), component_lock_hash="sha256:" + "1" * 64,
        now=100.0,
    )
    lease = jobs.claim_next(worker_id="worker_gpu_0", now=101.0)
    jobs.update_page_progress(
        lease, pages_total=2, pages_done=2, current_page=1,
        pages=[
            {"index": 0, "state": "succeeded", "error_code": None,
             "decode_steps": 5, "seconds": 1.0},
            {"index": 1, "state": "succeeded", "error_code": None,
             "decode_steps": 7, "seconds": 1.2},
        ],
        now=102.0,
    )
    artifacts = client.app.state.dependencies.artifact_store
    staged_ir0 = artifacts.stage(
        principal_id="principal_a", job_id=job_id, execution_id=lease.execution_id,
        role="prediction_ir", mime_type="application/json", chunks=(b'{"page":0}',),
    )
    staged_ir1 = artifacts.stage(
        principal_id="principal_a", job_id=job_id, execution_id=lease.execution_id,
        role="prediction_ir", mime_type="application/json", chunks=(b'{"page":1}',),
    )
    staged_docx = artifacts.stage(
        principal_id="principal_a", job_id=job_id, execution_id=lease.execution_id,
        role="diagnostic_docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        chunks=(b"docx-bytes",),
    )

    def fence(published, publish):
        jobs.finalize_success(
            lease,
            artifacts=[
                ArtifactMetadata(role="prediction_ir", ref=published[0].ref,
                                  sha256=published[0].sha256.removeprefix("sha256:"),
                                  size_bytes=published[0].size_bytes,
                                  mime_type=published[0].mime_type, page_index=0),
                ArtifactMetadata(role="prediction_ir", ref=published[1].ref,
                                  sha256=published[1].sha256.removeprefix("sha256:"),
                                  size_bytes=published[1].size_bytes,
                                  mime_type=published[1].mime_type, page_index=1),
                ArtifactMetadata(role="diagnostic_docx", ref=published[2].ref,
                                  sha256=published[2].sha256.removeprefix("sha256:"),
                                  size_bytes=published[2].size_bytes,
                                  mime_type=published[2].mime_type),
            ],
            diagnostic_outcome={
                "prediction_ir_status": "sealed", "claim_scope": "diagnostic",
                "canonical_ast_produced": True, "docx_produced": True,
                "reason_codes": ["HEADLESS_DEFAULTS_V0_APPLIED"],
                "docx_chain": {
                    "policy_id": "headless-defaults.v0", "policy_status": "RATIFIED",
                    "page_outcome": "terminated", "ast_status": "strict_valid",
                    "renderer_status": "render_ready",
                    "assembler": {"joins_accepted": 1, "joins_refused": 0, "refusal_codes": {}},
                },
            },
            retention_seconds=86400, now=103.0, publish=publish,
        )
        return True

    artifacts.publish_many(
        [staged_ir0, staged_ir1, staged_docx], fence, allow_repeated_roles=True
    )
    return job_id


def test_detailed_readiness_reports_limits(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/readiness", headers=AUTH)

    assert response.status_code == 200
    limits = response.json()["limits"]
    assert limits["max_pages"] == 100
    assert limits["max_upload_bytes"] == client.app.state.dependencies.runtime_policy[
        "max_upload_bytes"
    ]
    assert limits["accepted_media_types"] == [
        "application/pdf",
        "image/png",
        "image/jpeg",
        "multipart/form-data",
    ]


def test_multi_page_status_reports_pages_and_page_scoped_artifacts(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _multi_page_job(client, tmp_path)

    status = client.get(f"/v1/conversions/{job_id}", headers=AUTH)

    assert status.status_code == 200
    body = status.json()
    assert body["pages_total"] == 2
    assert body["pages_done"] == 2
    assert [p["state"] for p in body["pages"]] == ["succeeded", "succeeded"]
    by_key = {(a["role"], a["page_index"]): a["url"] for a in body["artifacts"]}
    assert set(by_key) == {("prediction_ir", 0), ("prediction_ir", 1), ("diagnostic_docx", None)}
    assert by_key[("prediction_ir", 0)] == f"/v1/conversions/{job_id}/artifacts/prediction_ir/0"
    assert by_key[("prediction_ir", 1)] == f"/v1/conversions/{job_id}/artifacts/prediction_ir/1"
    assert (
        by_key[("diagnostic_docx", None)]
        == f"/v1/conversions/{job_id}/artifacts/diagnostic_docx"
    )
    assert body["diagnostic_outcome"]["docx_chain"]["assembler"] == {
        "joins_accepted": 1, "joins_refused": 0, "refusal_codes": {},
    }


def test_page_scoped_download_serves_the_right_page(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _multi_page_job(client, tmp_path)

    page0 = client.get(f"/v1/conversions/{job_id}/artifacts/prediction_ir/0", headers=AUTH)
    page1 = client.get(f"/v1/conversions/{job_id}/artifacts/prediction_ir/1", headers=AUTH)
    legacy = client.get(f"/v1/conversions/{job_id}/artifacts/prediction_ir", headers=AUTH)

    assert page0.status_code == 200 and page0.content == b'{"page":0}'
    assert page1.status_code == 200 and page1.content == b'{"page":1}'
    # existing single-page route keeps working, resolving to page 0
    assert legacy.status_code == 200 and legacy.content == b'{"page":0}'


def test_page_scoped_download_out_of_range_is_not_found(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _multi_page_job(client, tmp_path)

    response = client.get(f"/v1/conversions/{job_id}/artifacts/prediction_ir/5", headers=AUTH)

    assert response.status_code == 404


def test_diagnostic_docx_download_still_works_for_multi_page_job(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _multi_page_job(client, tmp_path)

    response = client.get(f"/v1/conversions/{job_id}/artifacts/diagnostic_docx", headers=AUTH)

    assert response.status_code == 200
    assert response.content == b"docx-bytes"
