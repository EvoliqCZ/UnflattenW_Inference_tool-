"""Restart-safe artifact and metadata retention coordination."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from unflatten.dataset.artifact_ref import ArtifactRef
from unflatten.service.artifact_store import ArtifactStore
from unflatten.service.job_store import (
    JobStore,
    RetentionArtifactRecord,
    RetentionJobRecord,
    UnregisteredArtifactFence,
)
from unflatten.service.policy import validate_runtime_policy


@dataclass(frozen=True)
class RetainedArtifact:
    principal_id: str
    job_id: str
    role: str
    artifact: ArtifactRef
    created_at: float
    terminal_at: float | None
    staged: bool = False
    crash_orphan: bool = False
    fence_token: object | None = None


@dataclass(frozen=True)
class RetainedJob:
    job_id: str
    created_at: float
    terminal_at: float | None
    fence_token: object | None = None


class RetentionRepository(Protocol):
    def reconcile(self, *, now: float, orphan_before: float) -> None: ...

    def artifacts_for_retention(self) -> Iterable[RetainedArtifact]: ...

    def jobs_for_retention(self) -> Iterable[RetainedJob]: ...

    def finalize_artifact_deletion(
        self, record: RetainedArtifact, delete_bytes: Callable[[], None]
    ) -> bool: ...

    def delete_job_metadata(self, record: RetainedJob) -> bool: ...

    def delete_finished_commands(self, *, before: float) -> int: ...


class JobStoreRetentionRepository:
    """Production retention repository backed by the shared SQLite/filesystem stores."""

    def __init__(
        self, *, jobs: JobStore, store: ArtifactStore, work_root: Path | None = None
    ) -> None:
        self._jobs = jobs
        self._store = store
        self._work_root = work_root

    def reconcile(self, *, now: float, orphan_before: float) -> None:
        from unflatten.service.work_cleanup import cleanup_work_executions

        self._jobs.expire_jobs(now=now)
        if self._work_root is not None:
            cleanup_work_executions(
                self._work_root, jobs=self._jobs, now=now, orphan_before=orphan_before
            )

    def artifacts_for_retention(self) -> Iterable[RetainedArtifact]:
        records = [self._published_record(record) for record in self._jobs.retention_artifacts()]
        for staged in self._store.staged_artifacts_for_retention():
            job = self._jobs.get_job(staged.job_id, staged.principal_id)
            if job is None:
                records.append(
                    RetainedArtifact(
                        principal_id=staged.principal_id,
                        job_id=staged.job_id,
                        role=staged.role,
                        artifact=staged.artifact,
                        created_at=staged.created_at,
                        terminal_at=None,
                        staged=True,
                        crash_orphan=True,
                        fence_token=None,
                    )
                )
                continue
            records.append(
                RetainedArtifact(
                    principal_id=staged.principal_id,
                    job_id=staged.job_id,
                    role=staged.role,
                    artifact=staged.artifact,
                    created_at=staged.created_at,
                    terminal_at=job.terminal_at,
                    staged=True,
                    crash_orphan=False,
                    fence_token=(job.updated_at, job.state),
                )
            )
        for published in self._store.published_artifacts_for_retention():
            fence = self._jobs.unregistered_artifact_fence(
                job_id=published.job_id,
                execution_id=published.execution_id,
                ref=published.artifact.ref,
            )
            if fence is None:
                continue
            records.append(
                RetainedArtifact(
                    principal_id=published.principal_id,
                    job_id=published.job_id,
                    role=published.role,
                    artifact=published.artifact,
                    created_at=published.created_at,
                    terminal_at=None,
                    crash_orphan=True,
                    fence_token=fence,
                )
            )
        return records

    def jobs_for_retention(self) -> Iterable[RetainedJob]:
        return [
            RetainedJob(
                job_id=record.job_id,
                created_at=record.created_at,
                terminal_at=record.terminal_at,
                fence_token=record,
            )
            for record in self._jobs.retention_jobs()
        ]

    def finalize_artifact_deletion(
        self, record: RetainedArtifact, delete_bytes: Callable[[], None]
    ) -> bool:
        if record.crash_orphan and isinstance(record.fence_token, UnregisteredArtifactFence):
            return self._jobs.delete_unregistered_artifact(
                record.fence_token,
                ref=record.artifact.ref,
                delete_bytes=delete_bytes,
            )
        if record.staged:
            if record.fence_token is None:
                return self._jobs.delete_orphan_staged_artifact(
                    job_id=record.job_id,
                    delete_bytes=delete_bytes,
                )
            if not isinstance(record.fence_token, tuple) or len(record.fence_token) != 2:
                return False
            updated_at, state = record.fence_token
            return self._jobs.delete_staged_retention_artifact(
                job_id=record.job_id,
                principal_id=record.principal_id,
                expected_updated_at=updated_at,
                expected_state=state,
                delete_bytes=delete_bytes,
            )
        if not isinstance(record.fence_token, RetentionArtifactRecord):
            return False
        return self._jobs.delete_retention_artifact(record.fence_token, delete_bytes)

    def delete_job_metadata(self, record: RetainedJob) -> bool:
        if not isinstance(record.fence_token, RetentionJobRecord):
            return False
        return self._jobs.delete_retention_job(record.fence_token)

    def delete_finished_commands(self, *, before: float) -> int:
        return self._jobs.delete_finished_commands(before=before)

    @staticmethod
    def _published_record(record: RetentionArtifactRecord) -> RetainedArtifact:
        return RetainedArtifact(
            principal_id=record.principal_id,
            job_id=record.job_id,
            role=record.role,
            artifact=ArtifactRef(
                ref=record.ref,
                sha256=f"sha256:{record.sha256}",
                size_bytes=record.size_bytes,
                mime_type=record.mime_type,
            ),
            created_at=record.created_at,
            terminal_at=record.terminal_at,
            fence_token=record,
        )


class RetentionSweeper:
    def __init__(
        self,
        *,
        store: ArtifactStore,
        repository: RetentionRepository,
        policy: dict[str, Any],
    ) -> None:
        validate_runtime_policy(policy)
        self._store = store
        self._repository = repository
        self._source_ceiling = policy["crash_recovery_source_raster_seconds"]
        self._metadata_retention = policy["metadata_retention_seconds"]
        self._retention_by_role = {
            "prediction_ir": policy["prediction_ir_retention_seconds"],
            "diagnostic_docx": policy["diagnostic_docx_retention_seconds"],
            # DEC-044 Dodatok D (C): previews share diagnostic_docx's own
            # retention ceiling -- no new policy field, "same retention" is
            # literal, not merely similar.
            "source_preview": policy["diagnostic_docx_retention_seconds"],
            "docx_preview": policy["diagnostic_docx_retention_seconds"],
        }

    def sweep(self, *, now: float) -> tuple[int, int]:
        self._repository.reconcile(now=now, orphan_before=now - self._source_ceiling)
        artifacts_deleted = 0
        for record in self._repository.artifacts_for_retention():
            if not self._artifact_expired(record, now):
                continue

            def delete_bytes(record: RetainedArtifact = record) -> None:
                if record.staged:
                    self._store.discard_staged(record.artifact)
                else:
                    self._store.delete(
                        principal_id=record.principal_id,
                        job_id=record.job_id,
                        artifact=record.artifact,
                    )

            if self._repository.finalize_artifact_deletion(record, delete_bytes):
                artifacts_deleted += 1

        metadata_deleted = 0
        for record in self._repository.jobs_for_retention():
            baseline = record.terminal_at if record.terminal_at is not None else record.created_at
            if now >= baseline + self._metadata_retention:
                metadata_deleted += int(self._repository.delete_job_metadata(record))
        # §4.1: terminal model_commands (download/load/delete) are the same
        # click-driven metadata as jobs -- one row per click, deleted after
        # the same metadata_retention_seconds ceiling. Active (queued/running)
        # commands never appear here; the store's own state filter excludes
        # them regardless of age.
        metadata_deleted += self._repository.delete_finished_commands(
            before=now - self._metadata_retention
        )
        return artifacts_deleted, metadata_deleted

    def _artifact_expired(self, record: RetainedArtifact, now: float) -> bool:
        if record.role in {"source_pdf", "page_png"}:
            return record.terminal_at is not None or (
                record.crash_orphan and now >= record.created_at + self._source_ceiling
            )
        if record.crash_orphan:
            return now >= record.created_at + self._source_ceiling
        try:
            retention = self._retention_by_role[record.role]
        except KeyError as exc:
            raise ValueError(f"unknown artifact role for retention: {record.role!r}") from exc
        return now >= record.created_at + retention
