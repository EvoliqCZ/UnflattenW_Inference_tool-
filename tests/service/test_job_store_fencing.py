from __future__ import annotations

from pathlib import Path

import pytest

import unflatten.service.job_store as job_store


def make_claimed_store(tmp_path: Path) -> tuple[job_store.JobStore, object]:
    store = job_store.JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=10,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id="job_fenced",
        principal_id="principal_a",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output="prediction_ir",
        policy_hash="sha256:policy",
        component_lock_hash="sha256:components",
        now=100.0,
    )
    lease = store.claim_next(worker_id="worker_a", now=100.0)
    assert lease is not None
    return store, lease


def test_stale_worker_cannot_mutate_or_publish_after_reclaim(tmp_path: Path) -> None:
    assert hasattr(job_store, "ArtifactMetadata")
    assert hasattr(job_store.JobStore, "finalize_success")
    store, stale = make_claimed_store(tmp_path)
    assert store.requeue_expired(now=111.0) == 1
    current = store.claim_next(worker_id="worker_b", now=112.0)
    assert current is not None
    assert current.lease_epoch == stale.lease_epoch + 1
    artifact = job_store.ArtifactMetadata(
        role="prediction_ir",
        ref="principal_a/job_fenced/exec/result.json",
        sha256="a" * 64,
        size_bytes=12,
        mime_type="application/json",
    )

    with pytest.raises(job_store.FenceRejected):
        store.heartbeat(stale, now=113.0)
    with pytest.raises(job_store.FenceRejected):
        store.update_stage(stale, stage="model_decode", now=113.0)
    with pytest.raises(job_store.FenceRejected):
        store.fail_attempt(
            stale,
            failure_class="transient_infrastructure",
            retryable=True,
            error={"code": "INTERNAL_ERROR"},
            now=113.0,
        )
    with pytest.raises(job_store.FenceRejected):
        store.finalize_success(
            stale,
            artifacts=[artifact],
            diagnostic_outcome={"prediction_ir_status": "sealed"},
            retention_seconds=60,
            now=113.0,
        )
    assert store.list_artifacts("job_fenced") == []

    completed = store.finalize_success(
        current,
        artifacts=[artifact],
        diagnostic_outcome={"prediction_ir_status": "sealed"},
        retention_seconds=60,
        now=114.0,
    )
    assert completed.state == "succeeded"
    assert completed.expires_at == 174.0
    assert [item.execution_id for item in store.list_artifacts("job_fenced")] == [
        current.execution_id
    ]


def test_stale_worker_fence_rejection_leaves_zero_of_n_artifact_rows(tmp_path: Path) -> None:
    store, stale = make_claimed_store(tmp_path)
    assert store.requeue_expired(now=111.0) == 1
    current = store.claim_next(worker_id="worker_b", now=112.0)
    assert current is not None
    artifacts = [
        job_store.ArtifactMetadata(
            role="prediction_ir",
            ref="private/result.json",
            sha256="a" * 64,
            size_bytes=1,
            mime_type="application/json",
        ),
        job_store.ArtifactMetadata(
            role="diagnostic_docx",
            ref="private/result.docx",
            sha256="b" * 64,
            size_bytes=1,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
    ]

    with pytest.raises(job_store.FenceRejected):
        store.finalize_success(
            stale,
            artifacts=artifacts,
            diagnostic_outcome={"prediction_ir_status": "sealed"},
            retention_seconds=60,
            now=113.0,
        )

    assert store.list_artifacts("job_fenced") == []
    job = store.get_job("job_fenced", "principal_a")
    assert job is not None
    assert job.state == "running"


def test_expired_unreclaimed_lease_cannot_mutate_or_publish(tmp_path: Path) -> None:
    store, expired = make_claimed_store(tmp_path)
    artifact = job_store.ArtifactMetadata(
        role="prediction_ir",
        ref="private/result.json",
        sha256="c" * 64,
        size_bytes=12,
        mime_type="application/json",
    )

    with pytest.raises(job_store.FenceRejected):
        store.update_stage(expired, stage="model_decode", now=111.0)
    with pytest.raises(job_store.FenceRejected):
        store.finalize_success(
            expired,
            artifacts=[artifact],
            diagnostic_outcome={"prediction_ir_status": "sealed"},
            retention_seconds=60,
            now=111.0,
        )
    assert store.list_artifacts("job_fenced") == []
