"""Durable, fenced SQLite job state for the diagnostic service."""

from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path

from unflatten.service.contracts import JOB_STATES
from unflatten.service.job_store_aux import (
    ModelCommandMixin,
    ReadinessMixin,
    RetentionMixin,
    SettingsMixin,
    WebSessionMixin,
)
from unflatten.service.job_store_lifecycle import LifecycleMixin
from unflatten.service.job_store_schema import SchemaMixin
from unflatten.service.job_store_types import (
    ArtifactMetadata,
    ArtifactRecord,
    AttemptRecord,
    CapacityExceeded,
    CreateJobResult,
    FenceRejected,
    IdempotencyKeyReused,
    JobActive,
    JobNotFound,
    JobRecord,
    JobStoreError,
    JobStoreInvalid,
    JobTerminal,
    Lease,
    ModelCommandConflict,
    ModelCommandRecord,
    ReadinessWriter,
    RetentionArtifactRecord,
    RetentionJobRecord,
    UnregisteredArtifactFence,
    WebSessionRecord,
    job_from_row,
)

__all__ = [
    "ArtifactMetadata", "ArtifactRecord", "AttemptRecord", "CapacityExceeded",
    "CreateJobResult", "FenceRejected", "IdempotencyKeyReused", "JobActive",
    "JobNotFound", "JobRecord", "JobStore", "JobStoreError", "JobStoreInvalid",
    "JobTerminal", "Lease", "ModelCommandConflict", "ModelCommandRecord",
    "ReadinessWriter", "RetentionArtifactRecord", "RetentionJobRecord",
    "UnregisteredArtifactFence", "WebSessionRecord",
]

_ACTIVE_STATES = ("queued", "running")
_ACCEPTED_OUTPUTS = frozenset({"prediction_ir", "diagnostic_docx"})
_INPUT_KINDS = frozenset({"pdf", "image"})
_ENGINE_IDS = frozenset({"native"})


def _encode_jobs_cursor(created_at: float, row_id: int) -> str:
    payload = json.dumps({"t": created_at, "i": row_id}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_jobs_cursor(cursor: str) -> tuple[float, int]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
        return float(payload["t"]), int(payload["i"])
    except (ValueError, TypeError, KeyError) as exc:
        raise JobStoreInvalid(f"invalid list_jobs cursor: {cursor!r}") from exc


class JobStore(
    SchemaMixin, LifecycleMixin, ReadinessMixin, RetentionMixin, SettingsMixin,
    WebSessionMixin, ModelCommandMixin,
):
    """Single-host SQLite WAL queue with transactional leases and fencing."""

    def __init__(
        self,
        path: Path | str,
        *,
        lease_duration_seconds: int,
        max_attempts: int,
        global_active_jobs: int,
        active_jobs_per_principal: int,
    ) -> None:
        self.path = Path(path)
        if self.path.name == ":memory:":
            raise JobStoreInvalid("job store requires a durable filesystem path for WAL")
        values = (
            lease_duration_seconds, max_attempts, global_active_jobs,
            active_jobs_per_principal,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise JobStoreInvalid("job store limits must be positive integers")
        if active_jobs_per_principal > global_active_jobs:
            raise JobStoreInvalid("principal capacity cannot exceed global capacity")
        self.lease_duration_seconds = lease_duration_seconds
        self.max_attempts = max_attempts
        self.global_active_jobs = global_active_jobs
        self.active_jobs_per_principal = active_jobs_per_principal

    def create_job(
        self,
        *,
        job_id: str,
        principal_id: str,
        idempotency_key_hash: str,
        request_digest: str,
        requested_output: str,
        policy_hash: str,
        component_lock_hash: str,
        now: float,
        model_variant: str | None = None,
        label: str | None = None,
        input_kind: str | None = None,
        engine: str | None = None,
    ) -> CreateJobResult:
        required = (
            job_id, principal_id, idempotency_key_hash, request_digest,
            policy_hash, component_lock_hash,
        )
        if any(not isinstance(value, str) or not value for value in required):
            raise JobStoreInvalid("job identity and hash fields must be non-empty strings")
        if requested_output not in _ACCEPTED_OUTPUTS:
            raise JobStoreInvalid("job store accepts only prediction_ir or diagnostic_docx jobs")
        if model_variant is not None and not model_variant:
            raise JobStoreInvalid("model_variant must be non-empty when given")
        if label is not None and not isinstance(label, str):
            raise JobStoreInvalid("label must be a string when given")
        if input_kind is not None and input_kind not in _INPUT_KINDS:
            raise JobStoreInvalid("input_kind must be 'pdf' or 'image' when given")
        if engine is not None and engine not in _ENGINE_IDS:
            raise JobStoreInvalid("engine must be 'native' when given")
        with self._transaction() as connection:
            existing = connection.execute(
                """SELECT * FROM conversion_jobs
                WHERE principal_id = ? AND idempotency_key_hash = ?""",
                (principal_id, idempotency_key_hash),
            ).fetchone()
            if existing is not None:
                if existing["request_digest"] != request_digest:
                    raise IdempotencyKeyReused(
                        "principal-scoped idempotency key has a different request digest"
                    )
                return CreateJobResult(job_from_row(existing), False)
            self._check_capacity(connection, principal_id)
            try:
                connection.execute(
                    """INSERT INTO conversion_jobs (
                    job_id, principal_id, idempotency_key_hash, request_digest,
                    requested_output, state, stage, max_attempts, policy_hash,
                    component_lock_hash, created_at, updated_at, model_variant,
                    label, input_kind, engine)
                    VALUES (?, ?, ?, ?, ?, 'queued', 'admission_complete', ?, ?, ?, ?, ?, ?,
                    ?, ?, ?)""",
                    (job_id, principal_id, idempotency_key_hash, request_digest,
                     requested_output, self.max_attempts, policy_hash,
                     component_lock_hash, now, now, model_variant,
                     label, input_kind, engine),
                )
            except sqlite3.IntegrityError as exc:
                raise JobStoreInvalid(f"cannot create unique job {job_id!r}: {exc}") from exc
            row = connection.execute(
                "SELECT * FROM conversion_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return CreateJobResult(job_from_row(row), True)

    def _check_capacity(self, connection: sqlite3.Connection, principal_id: str) -> None:
        global_count = connection.execute(
            "SELECT count(*) FROM conversion_jobs WHERE state IN (?, ?)", _ACTIVE_STATES
        ).fetchone()[0]
        if global_count >= self.global_active_jobs:
            raise CapacityExceeded("global")
        principal_count = connection.execute(
            """SELECT count(*) FROM conversion_jobs
            WHERE principal_id = ? AND state IN (?, ?)""",
            (principal_id, *_ACTIVE_STATES),
        ).fetchone()[0]
        if principal_count >= self.active_jobs_per_principal:
            raise CapacityExceeded("principal")

    def get_job(self, job_id: str, principal_id: str) -> JobRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM conversion_jobs WHERE job_id = ? AND principal_id = ?",
                (job_id, principal_id),
            ).fetchone()
        return None if row is None else job_from_row(row)

    def list_jobs(
        self, *, principal_id: str, states: Sequence[str] | None = None,
        cursor: str | None = None, limit: int,
    ) -> tuple[list[JobRecord], str | None]:
        """Cursor-paginated, principal-scoped job list, newest first (§3.4)."""
        if type(limit) is not int or not (1 <= limit <= 100):
            raise JobStoreInvalid("list_jobs limit must be an integer between 1 and 100")
        if states is not None:
            unknown = sorted(set(states) - JOB_STATES)
            if unknown:
                raise JobStoreInvalid(f"unknown job state(s) in filter: {unknown}")
        clauses = ["principal_id = ?"]
        params: list[object] = [principal_id]
        if states:
            clauses.append(f"state IN ({','.join('?' for _ in states)})")
            params.extend(states)
        if cursor is not None:
            cursor_created_at, cursor_id = _decode_jobs_cursor(cursor)
            clauses.append("(created_at < ? OR (created_at = ? AND id < ?))")
            params.extend([cursor_created_at, cursor_created_at, cursor_id])
        where = " AND ".join(clauses)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""SELECT * FROM conversion_jobs WHERE {where}
                ORDER BY created_at DESC, id DESC LIMIT ?""",
                (*params, limit + 1),
            ).fetchall()
        next_cursor = None
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = _encode_jobs_cursor(last["created_at"], last["id"])
            rows = rows[:limit]
        return [job_from_row(row) for row in rows], next_cursor

    def purge_job(
        self, job_id: str, principal_id: str, *,
        delete_bytes: Callable[[ArtifactRecord], None],
        delete_work: Callable[[str], None] | None = None,
    ) -> bool:
        """Delete a terminal job's row (cascading attempts/artifacts) and its
        artifact bytes (§3.5). Raises `JobNotFound` if the principal-scoped
        job does not exist, `JobActive` if it has not reached a terminal
        state. A missing artifact file is tolerated -- retention may have
        already deleted it."""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM conversion_jobs WHERE job_id = ? AND principal_id = ?",
                (job_id, principal_id),
            ).fetchone()
            if row is None:
                raise JobNotFound(f"job {job_id!r} was not found during purge")
            if row["state"] in _ACTIVE_STATES:
                raise JobActive(f"job {job_id!r} is still active")
            if delete_work is not None:
                for attempt in connection.execute(
                    "SELECT execution_id FROM job_attempts WHERE job_id = ?", (job_id,)
                ).fetchall():
                    delete_work(attempt["execution_id"])
            artifact_rows = connection.execute(
                "SELECT * FROM job_artifacts WHERE job_id = ?", (job_id,)
            ).fetchall()
            for artifact_row in artifact_rows:
                record = ArtifactRecord(
                    job_id=artifact_row["job_id"], execution_id=artifact_row["execution_id"],
                    role=artifact_row["role"], ref=artifact_row["ref"],
                    sha256=artifact_row["sha256"], size_bytes=artifact_row["size_bytes"],
                    mime_type=artifact_row["mime_type"], private=bool(artifact_row["private"]),
                    page_index=artifact_row["page_index"],
                )
                try:
                    delete_bytes(record)
                except FileNotFoundError:
                    pass
            connection.execute("DELETE FROM conversion_jobs WHERE job_id = ?", (job_id,))
            return True

    def get_claimed_job(self, lease: Lease, *, now: float) -> JobRecord:
        from unflatten.service.job_store_types import fenced_job_row

        with closing(self._connect()) as connection:
            row = fenced_job_row(connection, lease, now=now)
        return job_from_row(row)

    def list_artifacts(self, job_id: str) -> list[ArtifactRecord]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT artifact.* FROM job_artifacts AS artifact
                JOIN conversion_jobs AS job ON job.job_id = artifact.job_id
                WHERE artifact.job_id = ? AND job.state = 'succeeded'
                ORDER BY artifact.id""", (job_id,),
            ).fetchall()
        return [ArtifactRecord(
            job_id=row["job_id"], execution_id=row["execution_id"], role=row["role"],
            ref=row["ref"], sha256=row["sha256"], size_bytes=row["size_bytes"],
            mime_type=row["mime_type"], private=bool(row["private"]),
            page_index=row["page_index"],
        ) for row in rows]

    def expire_jobs(self, *, now: float) -> int:
        with self._transaction() as connection:
            return connection.execute(
                """UPDATE conversion_jobs SET state = 'expired', updated_at = ?
                WHERE state IN ('succeeded', 'failed', 'cancelled')
                AND expires_at IS NOT NULL AND expires_at <= ?""", (now, now),
            ).rowcount

    def list_attempts(self, job_id: str) -> list[AttemptRecord]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM job_attempts WHERE job_id = ? ORDER BY attempt", (job_id,)
            ).fetchall()
        return [AttemptRecord(
            job_id=row["job_id"], attempt=row["attempt"], execution_id=row["execution_id"],
            worker_id=row["worker_id"], lease_epoch=row["lease_epoch"],
            outcome=row["outcome"], failure_class=row["failure_class"],
        ) for row in rows]
