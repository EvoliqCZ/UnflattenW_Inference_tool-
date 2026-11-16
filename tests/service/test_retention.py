from __future__ import annotations

import os
from pathlib import Path

import pytest

from unflatten.dataset.artifact_ref import ArtifactRef
from unflatten.service.artifact_store import DOWNLOADABLE_ROLES, ArtifactStore
from unflatten.service.job_store import ArtifactMetadata, JobStore
from unflatten.service.policy import load_runtime_policy
from unflatten.service.retention import (
    JobStoreRetentionRepository,
    RetainedArtifact,
    RetainedJob,
    RetentionSweeper,
)


class FakeRepository:
    def reconcile(self, *, now, orphan_before):
        pass

    def __init__(
        self, artifacts: list[RetainedArtifact], jobs: list[RetainedJob] | None = None
    ) -> None:
        self.artifacts = artifacts
        self.jobs = jobs or []
        self.deleted_artifacts: list[str] = []
        self.deleted_jobs: list[str] = []

    def artifacts_for_retention(self):
        return list(self.artifacts)

    def jobs_for_retention(self):
        return list(self.jobs)

    def finalize_artifact_deletion(self, record, delete_bytes):
        delete_bytes()
        self.deleted_artifacts.append(record.role)
        self.artifacts.remove(record)
        return True

    def delete_job_metadata(self, record):
        self.deleted_jobs.append(record.job_id)
        self.jobs.remove(record)
        return True

    def delete_finished_commands(self, *, before):
        return 0


def _published(store: ArtifactStore, role: str):
    mime = {
        "source_pdf": "application/pdf",
        "page_png": "image/png",
        "prediction_ir": "application/json",
        "diagnostic_docx": (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        "source_preview": "image/png",
        "docx_preview": "image/png",
    }[role]
    staged = store.stage(
        principal_id="principal_alpha",
        job_id="job_1234567890abcdef",
        execution_id="execution_1234567890abcdef",
        role=role,
        mime_type=mime,
        chunks=[b"data"],
    )
    return store.publish(staged, lambda _artifact, publish: (publish(), True)[1])


@pytest.mark.parametrize("role", sorted(DOWNLOADABLE_ROLES))
def test_retention_every_downloadable_role_expires_at_its_own_ceiling_plus_one(
    tmp_path: Path, role: str
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    policy = load_runtime_policy()
    # DEC-044 Dodatok D (C): previews share diagnostic_docx's own retention
    # ceiling, not a dedicated policy field ("same retention" is literal).
    policy_role = "diagnostic_docx" if role in {"source_preview", "docx_preview"} else role
    retention_seconds = policy[f"{policy_role}_retention_seconds"]
    record = RetainedArtifact(
        "principal_alpha", "job_1234567890abcdef", role, _published(store, role), 100.0, 200.0
    )
    repository = FakeRepository([record])
    sweeper = RetentionSweeper(store=store, repository=repository, policy=policy)

    assert sweeper.sweep(now=100.0 + retention_seconds - 1) == (0, 0)
    assert sweeper.sweep(now=100.0 + retention_seconds + 1) == (1, 0)


def test_retention_unknown_artifact_role_raises_instead_of_never_expiring(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    policy = load_runtime_policy()
    record = RetainedArtifact(
        "principal_alpha",
        "job_1234567890abcdef",
        "unregistered_role",
        _published(store, "prediction_ir"),
        100.0,
        200.0,
    )
    repository = FakeRepository([record])
    sweeper = RetentionSweeper(store=store, repository=repository, policy=policy)

    with pytest.raises(ValueError, match="unregistered_role"):
        sweeper.sweep(now=1_000_000.0)


def test_retention_terminal_source_and_raster_delete_immediately(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    records = [
        RetainedArtifact(
            "principal_alpha", "job_1234567890abcdef", role, _published(store, role), 100.0, 200.0
        )
        for role in ("source_pdf", "page_png")
    ]
    repository = FakeRepository(records)

    assert RetentionSweeper(store=store, repository=repository, policy=load_runtime_policy()).sweep(
        now=200.0
    ) == (2, 0)
    assert repository.deleted_artifacts == ["source_pdf", "page_png"]


def test_retention_crash_ceiling_prediction_and_metadata_survive_restart(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    records = [
        RetainedArtifact(
            "principal_alpha",
            "job_1234567890abcdef",
            "source_pdf",
            _published(store, "source_pdf"),
            100.0,
            None,
            crash_orphan=True,
        ),
        RetainedArtifact(
            "principal_alpha",
            "job_1234567890abcdef",
            "prediction_ir",
            _published(store, "prediction_ir"),
            100.0,
            200.0,
        ),
    ]
    jobs = [RetainedJob("job_1234567890abcdef", 100.0, 200.0)]
    repository = FakeRepository(records, jobs)
    policy = load_runtime_policy()

    first_process = RetentionSweeper(store=store, repository=repository, policy=policy)
    assert first_process.sweep(now=3699.0) == (0, 0)
    restarted_process = RetentionSweeper(store=store, repository=repository, policy=policy)
    assert restarted_process.sweep(now=3700.0) == (1, 0)
    assert restarted_process.sweep(now=86500.0) == (1, 0)
    assert restarted_process.sweep(now=2592200.0) == (0, 1)


def test_retention_job_store_repository_deletes_terminal_ingress_and_metadata(
    tmp_path: Path,
) -> None:
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    jobs.create_job(
        job_id="job_1234567890abcdef",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output="prediction_ir",
        policy_hash="sha256:policy",
        component_lock_hash="sha256:components",
        now=100.0,
    )
    artifacts = ArtifactStore(tmp_path / "artifacts")
    source = artifacts.stage(
        principal_id="principal_alpha",
        job_id="job_1234567890abcdef",
        execution_id="ingress_1234567890abcdef",
        role="source_pdf",
        mime_type="application/pdf",
        chunks=(b"%PDF-1.7\n",),
    )
    jobs.request_cancel("job_1234567890abcdef", "principal_alpha", now=200.0)
    repository = JobStoreRetentionRepository(jobs=jobs, store=artifacts)

    first = RetentionSweeper(store=artifacts, repository=repository, policy=policy)
    assert first.sweep(now=200.0) == (1, 0)
    assert not (artifacts.root / source.ref).exists()

    restarted = RetentionSweeper(
        store=ArtifactStore(artifacts.root),
        repository=JobStoreRetentionRepository(
            jobs=JobStore(
                jobs.path,
                lease_duration_seconds=policy["lease_duration_seconds"],
                max_attempts=policy["max_attempts"],
                global_active_jobs=policy["global_active_jobs"],
                active_jobs_per_principal=policy["active_jobs_per_principal"],
            ),
            store=ArtifactStore(artifacts.root),
        ),
        policy=policy,
    )
    assert restarted.sweep(now=200.0 + policy["metadata_retention_seconds"]) == (0, 1)
    assert jobs.get_job("job_1234567890abcdef", "principal_alpha") is None


def test_retention_job_store_repository_enforces_crash_ceiling_for_staged_ingress(
    tmp_path: Path,
) -> None:
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    jobs.create_job(
        job_id="job_1234567890abcdef",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output="prediction_ir",
        policy_hash="sha256:policy",
        component_lock_hash="sha256:components",
        now=100.0,
    )
    artifacts = ArtifactStore(tmp_path / "artifacts")
    source = artifacts.stage(
        principal_id="principal_alpha",
        job_id="job_1234567890abcdef",
        execution_id="ingress_1234567890abcdef",
        role="source_pdf",
        mime_type="application/pdf",
        chunks=(b"%PDF-1.7\n",),
    )
    (artifacts.root / source.ref).touch()

    repository = JobStoreRetentionRepository(jobs=jobs, store=artifacts)
    records = list(repository.artifacts_for_retention())
    assert len(records) == 1
    cutoff = records[0].created_at + policy["crash_recovery_source_raster_seconds"]
    sweeper = RetentionSweeper(store=artifacts, repository=repository, policy=policy)
    assert sweeper.sweep(now=cutoff - 0.001) == (0, 0)
    assert sweeper.sweep(now=cutoff) == (0, 0)
    assert (artifacts.root / source.ref).exists()


def test_retention_preserves_queued_and_running_inputs_past_crash_ceiling(
    tmp_path: Path,
) -> None:
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    artifacts = ArtifactStore(tmp_path / "artifacts")
    for suffix in ("queued", "running"):
        job_id = f"job_1234567890_{suffix}"
        jobs.create_job(
            job_id=job_id,
            principal_id="principal_alpha",
            idempotency_key_hash=suffix,
            request_digest=suffix,
            requested_output="prediction_ir",
            policy_hash="sha256:policy",
            component_lock_hash="sha256:components",
            now=100.0,
        )
        artifact = artifacts.stage(
            principal_id="principal_alpha",
            job_id=job_id,
            execution_id=f"ingress_1234567890_{suffix}",
            role="source_pdf",
            mime_type="application/pdf",
            chunks=(b"%PDF-1.7\n",),
        )
        os.utime(artifacts.root / artifact.ref, (100.0, 100.0))
    assert jobs.claim_next(worker_id="worker_gpu_0", now=101.0) is not None
    sweeper = RetentionSweeper(
        store=artifacts,
        repository=JobStoreRetentionRepository(jobs=jobs, store=artifacts),
        policy=policy,
    )

    assert sweeper.sweep(now=100.0 + policy["crash_recovery_source_raster_seconds"]) == (0, 0)
    assert len(artifacts.staged_artifacts_for_retention()) == 2


def test_retention_job_store_repository_expires_published_prediction_ir(
    tmp_path: Path,
) -> None:
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    jobs.create_job(
        job_id="job_1234567890abcdef",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output="prediction_ir",
        policy_hash="sha256:policy",
        component_lock_hash="sha256:components",
        now=100.0,
    )
    lease = jobs.claim_next(worker_id="worker_gpu_0", now=101.0)
    assert lease is not None
    artifacts = ArtifactStore(tmp_path / "artifacts")
    staged = artifacts.stage(
        principal_id="principal_alpha",
        job_id="job_1234567890abcdef",
        execution_id=lease.execution_id,
        role="prediction_ir",
        mime_type="application/json",
        chunks=(b"{}",),
    )

    def fence(published, publish):
        jobs.finalize_success(
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
            diagnostic_outcome={"prediction_ir_status": "sealed"},
            retention_seconds=policy["prediction_ir_retention_seconds"],
            now=102.0,
            publish=publish,
        )
        return True

    published = artifacts.publish(staged, fence)
    sweeper = RetentionSweeper(
        store=artifacts,
        repository=JobStoreRetentionRepository(jobs=jobs, store=artifacts),
        policy=policy,
    )
    cutoff = 102.0 + policy["prediction_ir_retention_seconds"]
    assert sweeper.sweep(now=cutoff - 0.001) == (0, 0)
    assert sweeper.sweep(now=cutoff) == (1, 0)
    assert not (artifacts.root / published.ref).exists()
    assert jobs.list_artifacts("job_1234567890abcdef") == []


def test_retention_job_store_repository_deletes_crashed_precommit_ingress(
    tmp_path: Path,
) -> None:
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    artifacts = ArtifactStore(tmp_path / "artifacts")
    source = artifacts.stage(
        principal_id="principal_alpha",
        job_id="job_1234567890abcdef",
        execution_id="ingress_1234567890abcdef",
        role="source_pdf",
        mime_type="application/pdf",
        chunks=(b"%PDF-1.7\n",),
    )
    repository = JobStoreRetentionRepository(jobs=jobs, store=artifacts)
    records = list(repository.artifacts_for_retention())
    assert len(records) == 1
    cutoff = records[0].created_at + policy["crash_recovery_source_raster_seconds"]

    assert RetentionSweeper(
        store=artifacts, repository=repository, policy=policy
    ).sweep(now=cutoff) == (1, 0)
    assert not (artifacts.root / source.ref).exists()


def test_retention_reconciles_unregistered_publication_after_crash_with_attempt_fence(
    tmp_path: Path,
) -> None:
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    jobs.create_job(
        job_id="job_1234567890abcdef",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output="prediction_ir",
        policy_hash="sha256:policy",
        component_lock_hash="sha256:components",
        now=100.0,
    )
    lease = jobs.claim_next(worker_id="worker_gpu_0", now=101.0)
    assert lease is not None
    artifacts = ArtifactStore(tmp_path / "artifacts")
    staged = artifacts.stage(
        principal_id="principal_alpha",
        job_id=lease.job_id,
        execution_id=lease.execution_id,
        role="prediction_ir",
        mime_type="application/json",
        chunks=(b"{}",),
    )

    def crash_after_publication(published, publish):
        def publish_then_crash() -> None:
            publish()
            raise KeyboardInterrupt

        jobs.finalize_success(
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
            diagnostic_outcome={"prediction_ir_status": "sealed"},
            retention_seconds=policy["prediction_ir_retention_seconds"],
            now=102.0,
            publish=publish_then_crash,
        )
        return True

    with pytest.raises(KeyboardInterrupt):
        artifacts.publish(staged, crash_after_publication)
    orphan = next((artifacts.root / "principals").glob("**/artifacts/**/*.bin"))
    os.utime(orphan, (100.0, 100.0))
    sweeper = RetentionSweeper(
        store=artifacts,
        repository=JobStoreRetentionRepository(jobs=jobs, store=artifacts),
        policy=policy,
    )
    cutoff = 100.0 + policy["crash_recovery_source_raster_seconds"]

    assert sweeper.sweep(now=cutoff) == (0, 0)
    assert orphan.exists()
    assert jobs.requeue_expired(now=100.0 + policy["lease_duration_seconds"] + 2.0) == 1
    assert sweeper.sweep(now=cutoff) == (1, 0)
    assert not orphan.exists()


def _delete_via_artifact_store(artifacts: ArtifactStore, *, principal_id: str, job_id: str):
    def delete_bytes(record: ArtifactMetadata) -> None:
        artifacts.delete(
            principal_id=principal_id,
            job_id=job_id,
            artifact=ArtifactRef(
                ref=record.ref,
                sha256=f"sha256:{record.sha256}",
                size_bytes=record.size_bytes,
                mime_type=record.mime_type,
            ),
        )

    return delete_bytes


def test_purge_and_sweep_share_artifact_store_delete(tmp_path: Path) -> None:
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    jobs.create_job(
        job_id="job_purge_share",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output="diagnostic_docx",
        policy_hash="sha256:policy",
        component_lock_hash="sha256:components",
        now=100.0,
    )
    artifacts = ArtifactStore(tmp_path / "artifacts")
    staged = artifacts.stage(
        principal_id="principal_alpha",
        job_id="job_purge_share",
        execution_id="exec_purge_share",
        role="diagnostic_docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        chunks=[b"docx-bytes"],
    )
    published = artifacts.publish(staged, lambda _artifact, publish: (publish(), True)[1])
    lease = jobs.claim_next(worker_id="worker_a", now=101.0)
    jobs.finalize_success(
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
        diagnostic_outcome={"a": 1},
        retention_seconds=60,
        now=102.0,
    )

    assert (artifacts.root / published.ref).exists()
    assert jobs.purge_job(
        "job_purge_share",
        "principal_alpha",
        delete_bytes=_delete_via_artifact_store(
            artifacts, principal_id="principal_alpha", job_id="job_purge_share"
        ),
    )

    assert not (artifacts.root / published.ref).exists()
    assert jobs.get_job("job_purge_share", "principal_alpha") is None


def test_sweep_deletes_terminal_commands_after_metadata_retention(tmp_path: Path) -> None:
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    _, created = jobs.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_alpha", now=100.0
    )
    assert created
    claimed = jobs.claim_next_model_command(worker_id="worker_gpu_0", now=100.0)
    assert claimed is not None
    jobs.finish_model_command(
        claimed.command_id, worker_id="worker_gpu_0", state="succeeded", now=200.0
    )
    artifacts = ArtifactStore(tmp_path / "artifacts")
    sweeper = RetentionSweeper(
        store=artifacts,
        repository=JobStoreRetentionRepository(jobs=jobs, store=artifacts),
        policy=policy,
    )
    cutoff = 200.0 + policy["metadata_retention_seconds"]

    sweeper.sweep(now=cutoff - 0.001)
    assert jobs.last_model_command("variant_a", "download") is not None

    sweeper.sweep(now=cutoff)
    assert jobs.last_model_command("variant_a", "download") is None


def test_sweep_preserves_active_model_commands_past_metadata_retention(tmp_path: Path) -> None:
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    _, created = jobs.enqueue_model_command(
        kind="download", variant_id="variant_b", requested_by="principal_alpha", now=100.0
    )
    assert created
    artifacts = ArtifactStore(tmp_path / "artifacts")
    sweeper = RetentionSweeper(
        store=artifacts,
        repository=JobStoreRetentionRepository(jobs=jobs, store=artifacts),
        policy=policy,
    )
    far_future = 100.0 + policy["metadata_retention_seconds"] * 100

    sweeper.sweep(now=far_future)

    assert jobs.active_model_command("variant_b", "download") is not None


def test_sweep_after_purge_finds_nothing(tmp_path: Path) -> None:
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    jobs.create_job(
        job_id="job_sweep_after_purge",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output="prediction_ir",
        policy_hash="sha256:policy",
        component_lock_hash="sha256:components",
        now=100.0,
    )
    artifacts = ArtifactStore(tmp_path / "artifacts")
    staged = artifacts.stage(
        principal_id="principal_alpha",
        job_id="job_sweep_after_purge",
        execution_id="exec_sweep_after_purge",
        role="prediction_ir",
        mime_type="application/json",
        chunks=[b"{}"],
    )
    published = artifacts.publish(staged, lambda _artifact, publish: (publish(), True)[1])
    lease = jobs.claim_next(worker_id="worker_a", now=101.0)
    jobs.finalize_success(
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
        diagnostic_outcome={"a": 1},
        retention_seconds=60,
        now=102.0,
    )

    jobs.purge_job(
        "job_sweep_after_purge",
        "principal_alpha",
        delete_bytes=_delete_via_artifact_store(
            artifacts, principal_id="principal_alpha", job_id="job_sweep_after_purge"
        ),
    )

    sweeper = RetentionSweeper(
        store=artifacts,
        repository=JobStoreRetentionRepository(jobs=jobs, store=artifacts),
        policy=policy,
    )
    assert sweeper.sweep(now=1_000_000.0) == (0, 0)
