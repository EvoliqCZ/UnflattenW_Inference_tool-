"""Shared records, errors, and SQL fence helpers for the service job store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class JobStoreError(RuntimeError):
    """Base class for expected job repository failures."""


class JobStoreInvalid(ValueError):
    """Repository configuration or input is invalid."""


class IdempotencyKeyReused(JobStoreError):
    """An idempotency key was reused with a different request digest."""


class CapacityExceeded(JobStoreError):
    """The configured active-job capacity was reached."""

    def __init__(self, scope: str) -> None:
        super().__init__(f"active job capacity exceeded for {scope}")
        self.scope = scope


class FenceRejected(JobStoreError):
    """A worker mutation did not own the current live lease."""


class JobTerminal(JobStoreError):
    """A requested transition cannot leave a terminal state."""


class JobNotFound(JobStoreError):
    """No principal-scoped job exists."""


class JobActive(JobStoreError):
    """A purge was requested for a job that has not reached a terminal state."""


class ModelCommandConflict(JobStoreError):
    """A model command was requested while a different kind is active on the same variant."""


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    principal_id: str
    request_digest: str
    requested_output: str
    state: str
    stage: str
    attempt: int
    max_attempts: int
    lease_owner: str | None
    lease_epoch: int
    execution_id: str | None
    lease_expires_at: float | None
    heartbeat_at: float | None
    cancel_requested: bool
    policy_hash: str
    component_lock_hash: str
    diagnostic_outcome: dict[str, Any] | None
    error: dict[str, Any] | None
    created_at: float
    updated_at: float
    terminal_at: float | None
    expires_at: float | None
    model_variant: str | None = None
    #: DEC-044 Dodatok D §1 (B1): multi-page progress; all `None` for a job
    #: that has not started (or never needed, single-page) per-page decode.
    pages_total: int | None = None
    pages_done: int | None = None
    current_page: int | None = None
    #: Closed per-page report array, filled in as pages complete:
    #: `[{"index","state","error_code","decode_steps","seconds"}, ...]`.
    pages: tuple[dict[str, Any], ...] | None = None
    #: DEC-052 §4.1 (v7): free-text label, input classification, and engine
    #: provenance for the web UI job list; `None` for jobs submitted before
    #: these columns existed.
    label: str | None = None
    input_kind: str | None = None
    engine: str | None = None


@dataclass(frozen=True)
class CreateJobResult:
    job: JobRecord
    created: bool


@dataclass(frozen=True)
class Lease:
    job_id: str
    lease_owner: str
    lease_epoch: int
    execution_id: str
    attempt: int
    lease_expires_at: float


@dataclass(frozen=True)
class AttemptRecord:
    job_id: str
    attempt: int
    execution_id: str
    worker_id: str
    lease_epoch: int
    outcome: str | None
    failure_class: str | None


#: DEC-044 Dodatok D §1 (B1/C): sentinel `page_index` meaning "not
#: page-scoped" -- today's single whole-document artifact (a legacy
#: single-page prediction_ir, or the merged diagnostic_docx).
NOT_PAGE_SCOPED = -1


@dataclass(frozen=True)
class ArtifactMetadata:
    role: str
    ref: str
    sha256: str
    size_bytes: int
    mime_type: str
    page_index: int = NOT_PAGE_SCOPED


@dataclass(frozen=True)
class ArtifactRecord:
    job_id: str
    execution_id: str
    role: str
    ref: str
    sha256: str
    size_bytes: int
    mime_type: str
    private: bool
    page_index: int = NOT_PAGE_SCOPED


@dataclass(frozen=True)
class ReadinessWriter:
    worker_id: str
    writer_epoch: int


@dataclass(frozen=True)
class RetentionJobRecord:
    job_id: str
    created_at: float
    terminal_at: float


@dataclass(frozen=True)
class RetentionArtifactRecord:
    principal_id: str
    job_id: str
    execution_id: str
    role: str
    ref: str
    sha256: str
    size_bytes: int
    mime_type: str
    created_at: float
    terminal_at: float | None


@dataclass(frozen=True)
class UnregisteredArtifactFence:
    job_id: str
    execution_id: str | None
    expected_updated_at: float | None
    expected_state: str | None


@dataclass(frozen=True)
class WebSessionRecord:
    session_digest: str
    principal_id: str
    created_at: float
    last_seen_at: float
    expires_at: float
    revoked_at: float | None


def web_session_from_row(row: sqlite3.Row) -> WebSessionRecord:
    return WebSessionRecord(
        session_digest=row["session_digest"], principal_id=row["principal_id"],
        created_at=row["created_at"], last_seen_at=row["last_seen_at"],
        expires_at=row["expires_at"], revoked_at=row["revoked_at"],
    )


@dataclass(frozen=True)
class ModelCommandRecord:
    command_id: str
    kind: str
    variant_id: str
    state: str
    phase: str | None
    requested_by: str
    worker_id: str | None
    bytes_done: int | None
    bytes_total: int | None
    error_code: str | None
    requested_at: float
    started_at: float | None
    heartbeat_at: float | None
    finished_at: float | None
    updated_at: float


def model_command_from_row(row: sqlite3.Row) -> ModelCommandRecord:
    return ModelCommandRecord(
        command_id=row["command_id"], kind=row["kind"], variant_id=row["variant_id"],
        state=row["state"], phase=row["phase"], requested_by=row["requested_by"],
        worker_id=row["worker_id"], bytes_done=row["bytes_done"],
        bytes_total=row["bytes_total"], error_code=row["error_code"],
        requested_at=row["requested_at"], started_at=row["started_at"],
        heartbeat_at=row["heartbeat_at"], finished_at=row["finished_at"],
        updated_at=row["updated_at"],
    )


def model_command_projection(record: ModelCommandRecord) -> dict[str, Any]:
    """Shape a `ModelCommandRecord` into the mapping `model_variants.variant_download_state`
    expects for its `active_command`/`last_command` arguments (§4.2, T06 integration contract)."""
    return {
        "kind": record.kind,
        "state": record.state,
        "phase": record.phase,
        "bytes_done": record.bytes_done,
        "bytes_total": record.bytes_total,
        "error_code": record.error_code,
        "command_id": record.command_id,
    }


def required_job_row(connection: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM conversion_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        raise JobNotFound(f"job {job_id!r} was not found during repository mutation")
    return row


def fenced_job_row(connection: sqlite3.Connection, lease: Lease, *, now: float) -> sqlite3.Row:
    row = connection.execute(
        f"SELECT * FROM conversion_jobs WHERE {live_fence_sql()}",
        live_fence_values(lease, now),
    ).fetchone()
    if row is None:
        raise FenceRejected(f"lease fence rejected for job {lease.job_id!r}")
    return row


def fence_values(lease: Lease) -> tuple[object, ...]:
    return (lease.job_id, lease.lease_owner, lease.lease_epoch, lease.execution_id)


def live_fence_sql() -> str:
    return (
        "job_id = ? AND lease_owner = ? AND lease_epoch = ? AND execution_id = ? "
        "AND state = 'running' AND cancel_requested = 0 AND lease_expires_at > ?"
    )


def live_fence_values(lease: Lease, now: float) -> tuple[object, ...]:
    return (*fence_values(lease), now)


def json_object(value: Mapping[str, Any], name: str) -> str:
    if not isinstance(value, Mapping):
        raise JobStoreInvalid(f"{name} must be a JSON object")
    try:
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise JobStoreInvalid(f"{name} is not JSON serializable: {exc}") from exc


_VALID_ARTIFACT_ROLES = frozenset(
    {"prediction_ir", "diagnostic_docx", "source_preview", "docx_preview"}
)


def validate_artifact(artifact: ArtifactMetadata) -> None:
    if artifact.role not in _VALID_ARTIFACT_ROLES:
        raise JobStoreInvalid("only prediction_ir or diagnostic_docx artifacts may be published")
    if not artifact.ref or not artifact.mime_type:
        raise JobStoreInvalid("artifact ref and MIME type must be non-empty")
    if len(artifact.sha256) != 64 or any(c not in "0123456789abcdef" for c in artifact.sha256):
        raise JobStoreInvalid("artifact sha256 must be 64 lowercase hexadecimal characters")
    if type(artifact.size_bytes) is not int or artifact.size_bytes < 0:
        raise JobStoreInvalid("artifact size_bytes must be a non-negative integer")
    if type(artifact.page_index) is not int or artifact.page_index < NOT_PAGE_SCOPED:
        raise JobStoreInvalid(f"artifact page_index must be an integer >= {NOT_PAGE_SCOPED}")


def job_from_row(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"], principal_id=row["principal_id"],
        request_digest=row["request_digest"], requested_output=row["requested_output"],
        state=row["state"], stage=row["stage"], attempt=row["attempt"],
        max_attempts=row["max_attempts"], lease_owner=row["lease_owner"],
        lease_epoch=row["lease_epoch"], execution_id=row["execution_id"],
        lease_expires_at=row["lease_expires_at"], heartbeat_at=row["heartbeat_at"],
        cancel_requested=bool(row["cancel_requested"]), policy_hash=row["policy_hash"],
        component_lock_hash=row["component_lock_hash"],
        diagnostic_outcome=(
            None
            if row["diagnostic_outcome_json"] is None
            else json.loads(row["diagnostic_outcome_json"])
        ),
        error=None if row["error_json"] is None else json.loads(row["error_json"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
        terminal_at=row["terminal_at"], expires_at=row["expires_at"],
        model_variant=row["model_variant"],
        pages_total=row["pages_total"], pages_done=row["pages_done"],
        current_page=row["current_page"],
        pages=(
            None
            if row["pages_json"] is None
            else tuple(json.loads(row["pages_json"]))
        ),
        label=row["label"], input_kind=row["input_kind"], engine=row["engine"],
    )
