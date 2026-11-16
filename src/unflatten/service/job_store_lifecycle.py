"""Lease, retry, cancellation, and fenced finalization operations."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from unflatten.service.contracts import JOB_STAGES
from unflatten.service.job_store_types import (
    ArtifactMetadata,
    FenceRejected,
    JobNotFound,
    JobRecord,
    JobStoreInvalid,
    JobTerminal,
    Lease,
    fence_values,
    fenced_job_row,
    job_from_row,
    json_object,
    live_fence_sql,
    live_fence_values,
    required_job_row,
    validate_artifact,
)

_TERMINAL_STATES = ("succeeded", "failed", "cancelled", "expired")


class LifecycleMixin:
    lease_duration_seconds: int

    def claim_next(self, *, worker_id: str, now: float) -> Lease | None:
        if not worker_id:
            raise JobStoreInvalid("worker_id must be non-empty")
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT * FROM conversion_jobs
                WHERE state = 'queued' AND cancel_requested = 0 AND attempt < max_attempts
                ORDER BY created_at, id LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            attempt, epoch = int(row["attempt"]) + 1, int(row["lease_epoch"]) + 1
            execution_id = f"exec_{uuid.uuid4().hex}"
            expires_at = now + self.lease_duration_seconds
            changed = connection.execute(
                """UPDATE conversion_jobs
                SET state = 'running', attempt = ?, lease_owner = ?, lease_epoch = ?,
                    execution_id = ?, lease_expires_at = ?, heartbeat_at = ?, updated_at = ?
                WHERE job_id = ? AND state = 'queued' AND cancel_requested = 0
                    AND attempt = ? AND lease_epoch = ? AND attempt < max_attempts""",
                (attempt, worker_id, epoch, execution_id, expires_at, now, now,
                 row["job_id"], row["attempt"], row["lease_epoch"]),
            ).rowcount
            if changed != 1:
                raise FenceRejected(f"queued job {row['job_id']!r} changed during claim")
            connection.execute(
                """INSERT INTO job_attempts (
                job_id, attempt, execution_id, worker_id, lease_epoch,
                started_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (row["job_id"], attempt, execution_id, worker_id, epoch, now, now, now),
            )
            return Lease(row["job_id"], worker_id, epoch, execution_id, attempt, expires_at)

    def heartbeat(self, lease: Lease, *, now: float) -> Lease:
        expires_at = now + self.lease_duration_seconds
        with self._transaction() as connection:
            changed = connection.execute(
                """UPDATE conversion_jobs SET heartbeat_at = ?, lease_expires_at = ?, updated_at = ?
                WHERE job_id = ? AND lease_owner = ? AND lease_epoch = ? AND execution_id = ?
                    AND state = 'running' AND cancel_requested = 0 AND lease_expires_at > ?""",
                (now, expires_at, now, *fence_values(lease), now),
            ).rowcount
            if changed != 1:
                raise FenceRejected(f"heartbeat fence rejected for job {lease.job_id!r}")
        return Lease(**{**lease.__dict__, "lease_expires_at": expires_at})

    def update_stage(self, lease: Lease, *, stage: str, now: float) -> None:
        if stage not in JOB_STAGES:
            raise JobStoreInvalid(f"unknown job stage {stage!r}")
        with self._transaction() as connection:
            changed = connection.execute(
                f"UPDATE conversion_jobs SET stage = ?, updated_at = ? WHERE {live_fence_sql()}",
                (stage, now, *live_fence_values(lease, now)),
            ).rowcount
            if changed != 1:
                raise FenceRejected(f"stage fence rejected for job {lease.job_id!r}")

    def update_page_progress(
        self, lease: Lease, *, pages_total: int, pages_done: int, current_page: int,
        pages: Sequence[Mapping[str, Any]], now: float,
    ) -> None:
        """DEC-044 Dodatok D §1 (B1): publish multi-page progress under the
        SAME live lease fence as `update_stage` -- called once per completed
        page, never inferred from `stage` alone (a job can sit in
        `model_decode` for a long time on one page while `pages_done` from
        the PREVIOUS page is still what a status poll should see)."""
        if pages_total < 1 or not (0 <= pages_done <= pages_total):
            raise JobStoreInvalid("page progress counts are inconsistent")
        if not (0 <= current_page < pages_total):
            raise JobStoreInvalid("current_page is out of range")
        try:
            pages_json = json.dumps(
                list(pages), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise JobStoreInvalid(f"pages is not JSON serializable: {exc}") from exc
        with self._transaction() as connection:
            changed = connection.execute(
                f"""UPDATE conversion_jobs SET pages_total = ?, pages_done = ?,
                current_page = ?, pages_json = ?, updated_at = ? WHERE {live_fence_sql()}""",
                (pages_total, pages_done, current_page, pages_json, now,
                 *live_fence_values(lease, now)),
            ).rowcount
            if changed != 1:
                raise FenceRejected(f"page progress fence rejected for job {lease.job_id!r}")

    def fail_attempt(
        self, lease: Lease, *, failure_class: str, retryable: bool,
        error: Mapping[str, Any], now: float,
    ) -> JobRecord:
        if not failure_class:
            raise JobStoreInvalid("failure_class must be non-empty")
        error_json = json_object(error, "error")
        with self._transaction() as connection:
            row = fenced_job_row(connection, lease, now=now)
            should_retry = (retryable and failure_class == "transient_infrastructure"
                            and int(row["attempt"]) < int(row["max_attempts"]))
            state, terminal_at = ("queued", None) if should_retry else ("failed", now)
            changed = connection.execute(
                f"""UPDATE conversion_jobs SET state = ?, lease_owner = NULL,
                execution_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                error_json = ?, updated_at = ?, terminal_at = ? WHERE {live_fence_sql()}""",
                (state, error_json, now, terminal_at, *live_fence_values(lease, now)),
            ).rowcount
            if changed != 1:
                raise FenceRejected(f"failure fence rejected for job {lease.job_id!r}")
            connection.execute(
                """UPDATE job_attempts SET finished_at = ?, outcome = ?, failure_class = ?,
                updated_at = ? WHERE execution_id = ? AND job_id = ?""",
                (now, "retry" if should_retry else "failed", failure_class, now,
                 lease.execution_id, lease.job_id),
            )
            return job_from_row(required_job_row(connection, lease.job_id))

    def requeue_expired(self, *, now: float) -> int:
        processed = 0
        with self._transaction() as connection:
            rows = connection.execute(
                """SELECT * FROM conversion_jobs WHERE state = 'running'
                AND lease_expires_at <= ? ORDER BY lease_expires_at, id""", (now,)
            ).fetchall()
            for row in rows:
                cancelled = bool(row["cancel_requested"])
                retry = not cancelled and int(row["attempt"]) < int(row["max_attempts"])
                state = "queued" if retry else ("cancelled" if cancelled else "failed")
                changed = connection.execute(
                    """UPDATE conversion_jobs SET state = ?, lease_owner = NULL,
                    execution_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                    updated_at = ?, terminal_at = ? WHERE job_id = ? AND state = 'running'
                    AND lease_epoch = ? AND execution_id = ? AND lease_expires_at <= ?""",
                    (state, now, None if retry else now, row["job_id"], row["lease_epoch"],
                     row["execution_id"], now),
                ).rowcount
                if changed != 1:
                    continue
                connection.execute(
                    """UPDATE job_attempts SET finished_at = ?, outcome = ?, failure_class = ?,
                    updated_at = ? WHERE execution_id = ? AND job_id = ?""",
                    (now, "lease_expired" if retry else ("cancelled" if cancelled else "failed"),
                     "lease_expired", now, row["execution_id"], row["job_id"]),
                )
                processed += 1
        return processed

    def request_cancel(self, job_id: str, principal_id: str, *, now: float) -> JobRecord:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM conversion_jobs WHERE job_id = ? AND principal_id = ?",
                (job_id, principal_id),
            ).fetchone()
            if row is None:
                raise JobNotFound("principal-scoped job was not found")
            if row["state"] in _TERMINAL_STATES:
                raise JobTerminal(f"job {job_id!r} is already terminal")
            changed = self._cancel_row(connection, row, job_id, principal_id, now)
            if changed != 1:
                raise FenceRejected(f"cancellation CAS rejected for job {job_id!r}")
            return job_from_row(required_job_row(connection, job_id))

    @staticmethod
    def _cancel_row(connection, row, job_id: str, principal_id: str, now: float) -> int:
        if row["state"] == "queued":
            return connection.execute(
                """UPDATE conversion_jobs SET state = 'cancelled', cancel_requested = 1,
                updated_at = ?, terminal_at = ? WHERE job_id = ? AND principal_id = ?
                AND state = 'queued' AND lease_epoch = ? AND cancel_requested = 0""",
                (now, now, job_id, principal_id, row["lease_epoch"]),
            ).rowcount
        return connection.execute(
            """UPDATE conversion_jobs SET cancel_requested = 1, updated_at = ?
            WHERE job_id = ? AND principal_id = ? AND state = 'running'
            AND lease_owner = ? AND lease_epoch = ? AND execution_id = ?
            AND cancel_requested = 0""",
            (
                now, job_id, principal_id, row["lease_owner"],
                row["lease_epoch"], row["execution_id"],
            ),
        ).rowcount

    def finalize_success(
        self, lease: Lease, *, artifacts: Sequence[ArtifactMetadata],
        diagnostic_outcome: Mapping[str, Any], retention_seconds: int, now: float,
        publish: Callable[[], None] | None = None,
    ) -> JobRecord:
        if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
            raise JobStoreInvalid("artifacts must be a sequence of ArtifactMetadata")
        if not artifacts:
            raise JobStoreInvalid("artifacts must contain at least one artifact")
        for artifact in artifacts:
            validate_artifact(artifact)
        keys = [(artifact.role, artifact.page_index) for artifact in artifacts]
        if len(set(keys)) != len(keys):
            raise JobStoreInvalid("artifacts must not repeat a (role, page_index) pair")
        if type(retention_seconds) is not int or retention_seconds < 1:
            raise JobStoreInvalid("artifact retention_seconds must be a positive integer")
        outcome_json = json_object(diagnostic_outcome, "diagnostic_outcome")
        with self._transaction() as connection:
            for artifact in artifacts:
                inserted = connection.execute(
                    f"""INSERT INTO job_artifacts (job_id, execution_id, role, page_index, ref,
                    sha256, size_bytes, mime_type, private, created_at, updated_at)
                    SELECT job_id, execution_id, ?, ?, ?, ?, ?, ?, 1, ?, ? FROM conversion_jobs
                    WHERE {live_fence_sql()}""",
                    (artifact.role, artifact.page_index, artifact.ref, artifact.sha256,
                     artifact.size_bytes, artifact.mime_type, now, now,
                     *live_fence_values(lease, now)),
                ).rowcount
                if inserted != 1:
                    raise FenceRejected(f"artifact fence rejected for job {lease.job_id!r}")
            if publish is not None:
                publish()
            changed = connection.execute(
                f"""UPDATE conversion_jobs SET state = 'succeeded', stage = 'artifact_finalization',
                lease_owner = NULL, execution_id = NULL, lease_expires_at = NULL,
                heartbeat_at = NULL, diagnostic_outcome_json = ?, error_json = NULL,
                updated_at = ?, terminal_at = ?, expires_at = ? WHERE {live_fence_sql()}""",
                (outcome_json, now, now, now + retention_seconds,
                 *live_fence_values(lease, now)),
            ).rowcount
            if changed != 1:
                raise FenceRejected(f"finalization fence rejected for job {lease.job_id!r}")
            connection.execute(
                """UPDATE job_attempts SET finished_at = ?, outcome = 'succeeded', updated_at = ?
                WHERE job_id = ? AND execution_id = ? AND lease_epoch = ?""",
                (now, now, lease.job_id, lease.execution_id, lease.lease_epoch),
            )
            return job_from_row(required_job_row(connection, lease.job_id))

    def complete_cancel(self, lease: Lease, *, now: float) -> JobRecord:
        with self._transaction() as connection:
            changed = connection.execute(
                """UPDATE conversion_jobs SET state = 'cancelled', lease_owner = NULL,
                execution_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                updated_at = ?, terminal_at = ? WHERE job_id = ? AND lease_owner = ?
                AND lease_epoch = ? AND execution_id = ? AND state = 'running'
                AND cancel_requested = 1""", (now, now, *fence_values(lease)),
            ).rowcount
            if changed != 1:
                raise FenceRejected(f"cancellation completion fence rejected for {lease.job_id!r}")
            connection.execute(
                """UPDATE job_attempts SET finished_at = ?, outcome = 'cancelled', updated_at = ?
                WHERE job_id = ? AND execution_id = ? AND lease_epoch = ?""",
                (now, now, lease.job_id, lease.execution_id, lease.lease_epoch),
            )
            return job_from_row(required_job_row(connection, lease.job_id))
