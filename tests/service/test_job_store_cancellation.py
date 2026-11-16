from __future__ import annotations

from pathlib import Path

import pytest

import unflatten.service.job_store as job_store


def claimed_job(tmp_path: Path, job_id: str) -> tuple[job_store.JobStore, object]:
    store = job_store.JobStore(
        tmp_path / f"{job_id}.sqlite3",
        lease_duration_seconds=10,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id=job_id,
        principal_id="principal_a",
        idempotency_key_hash=f"key-{job_id}",
        request_digest=f"digest-{job_id}",
        requested_output="prediction_ir",
        policy_hash="sha256:policy",
        component_lock_hash="sha256:components",
        now=100.0,
    )
    lease = store.claim_next(worker_id="worker_a", now=101.0)
    assert lease is not None
    return store, lease


def artifact() -> job_store.ArtifactMetadata:
    return job_store.ArtifactMetadata(
        role="prediction_ir",
        ref="private/result.json",
        sha256="b" * 64,
        size_bytes=20,
        mime_type="application/json",
    )


def test_cancellation_winning_blocks_heartbeat_and_final_publication(tmp_path: Path) -> None:
    assert hasattr(job_store.JobStore, "complete_cancel")
    store, lease = claimed_job(tmp_path, "job_cancel_first")

    requested = store.request_cancel("job_cancel_first", "principal_a", now=102.0)
    assert requested.state == "running"
    assert requested.cancel_requested is True
    with pytest.raises(job_store.FenceRejected):
        store.heartbeat(lease, now=103.0)
    with pytest.raises(job_store.FenceRejected):
        store.finalize_success(
            lease,
            artifacts=[artifact()],
            diagnostic_outcome={"prediction_ir_status": "sealed"},
            retention_seconds=60,
            now=103.0,
        )
    assert store.list_artifacts("job_cancel_first") == []

    cancelled = store.complete_cancel(lease, now=104.0)
    assert cancelled.state == "cancelled"
    assert store.list_attempts("job_cancel_first")[0].outcome == "cancelled"


def test_finalization_winning_makes_later_cancellation_terminal(tmp_path: Path) -> None:
    store, lease = claimed_job(tmp_path, "job_finish_first")

    completed = store.finalize_success(
        lease,
        artifacts=[artifact()],
        diagnostic_outcome={"prediction_ir_status": "sealed"},
        retention_seconds=60,
        now=102.0,
    )
    assert completed.state == "succeeded"
    with pytest.raises(job_store.JobTerminal):
        store.request_cancel("job_finish_first", "principal_a", now=103.0)
    assert len(store.list_artifacts("job_finish_first")) == 1
