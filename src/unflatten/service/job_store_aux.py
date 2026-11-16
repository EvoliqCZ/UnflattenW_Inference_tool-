"""Readiness, settings, retention, web session, and model command persistence
operations for the service job store."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable, Sequence
from contextlib import closing
from typing import Any

from unflatten.service.job_store_types import (
    FenceRejected,
    JobStoreInvalid,
    ModelCommandConflict,
    ModelCommandRecord,
    ReadinessWriter,
    RetentionArtifactRecord,
    RetentionJobRecord,
    UnregisteredArtifactFence,
    WebSessionRecord,
    model_command_from_row,
    model_command_projection,
    web_session_from_row,
)

_MODEL_COMMAND_KINDS = frozenset({"download", "load", "delete"})
_INITIAL_PHASE = {"download": "downloading", "load": "loading", "delete": "deleting"}


class ReadinessMixin:
    def claim_readiness_writer(self, *, worker_id: str, now: float) -> ReadinessWriter:
        if not worker_id:
            raise JobStoreInvalid("readiness worker_id must be non-empty")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT writer_epoch FROM service_readiness WHERE singleton = 1"
            ).fetchone()
            epoch = 1 if row is None else int(row["writer_epoch"]) + 1
            connection.execute(
                """INSERT INTO service_readiness (singleton, worker_id, writer_epoch,
                worker_ready, sandbox_ready, diagnostic_docx_ready, observed_at)
                VALUES (1, ?, ?, 0, 0, 0, ?)
                ON CONFLICT(singleton) DO UPDATE SET worker_id = excluded.worker_id,
                writer_epoch = excluded.writer_epoch, worker_ready = 0, sandbox_ready = 0,
                diagnostic_docx_ready = 0, observed_at = excluded.observed_at,
                resident_variant = NULL, free_vram_mib = NULL, total_vram_mib = NULL,
                compute_capability_major = NULL, compute_capability_minor = NULL""",
                (worker_id, epoch, now),
            )
        return ReadinessWriter(worker_id, epoch)

    def publish_readiness(
        self, writer: ReadinessWriter, *, worker_ready: bool,
        sandbox_ready: bool, diagnostic_docx_ready: bool, now: float,
        sandbox_rootful: bool | None = None,
        available_engines: Sequence[str] | None = None,
    ) -> None:
        """Publish the fenced readiness snapshot; `sandbox_rootful` and
        `available_engines` (§4.1/§17.5) stay `None`/unpublished until a
        worker reports them, matching every other field this writes."""
        if (
            type(worker_ready) is not bool
            or type(sandbox_ready) is not bool
            or type(diagnostic_docx_ready) is not bool
        ):
            raise JobStoreInvalid("readiness values must be booleans")
        if sandbox_rootful is not None and type(sandbox_rootful) is not bool:
            raise JobStoreInvalid("sandbox_rootful must be a boolean when given")
        engines_json = None
        if available_engines is not None:
            engines = list(available_engines)
            if not engines or any(not isinstance(e, str) or not e for e in engines):
                raise JobStoreInvalid(
                    "available_engines must be a non-empty sequence of non-empty strings"
                )
            engines_json = json.dumps(engines, separators=(",", ":"))
        with self._transaction() as connection:
            changed = connection.execute(
                """UPDATE service_readiness SET worker_ready = ?, sandbox_ready = ?,
                diagnostic_docx_ready = ?, observed_at = ?, sandbox_rootful = ?,
                available_engines_json = ?
                WHERE singleton = 1 AND worker_id = ? AND writer_epoch = ?""",
                (int(worker_ready), int(sandbox_ready), int(diagnostic_docx_ready), now,
                 None if sandbox_rootful is None else int(sandbox_rootful), engines_json,
                 writer.worker_id, writer.writer_epoch),
            ).rowcount
            if changed != 1:
                raise FenceRejected("readiness writer fence rejected")

    def readiness_extras(self) -> tuple[bool | None, tuple[str, ...] | None] | None:
        """The last-published `sandbox_rootful`/`available_engines` (§A.9:
        additive accessor, kept separate from `readiness_status()`'s tuple
        arity). `None` when no worker has published readiness yet."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT sandbox_rootful, available_engines_json
                FROM service_readiness WHERE singleton = 1"""
            ).fetchone()
        if row is None:
            return None
        sandbox_rootful = None if row["sandbox_rootful"] is None else bool(row["sandbox_rootful"])
        engines = (
            None if row["available_engines_json"] is None
            else tuple(json.loads(row["available_engines_json"]))
        )
        return (sandbox_rootful, engines)

    def publish_model_state(
        self,
        writer: ReadinessWriter,
        *,
        resident_variant: str | None,
        free_vram_mib: float,
        total_vram_mib: float,
        compute_capability: tuple[int, int],
        now: float,
    ) -> None:
        """DEC-044 Dodatok D §1: publish the worker's resident model variant
        and measured device facts under the same readiness writer fence."""
        if resident_variant is not None and not resident_variant:
            raise JobStoreInvalid("resident_variant must be non-empty when given")
        if free_vram_mib < 0 or total_vram_mib < 0:
            raise JobStoreInvalid("VRAM facts must be non-negative")
        major, minor = compute_capability
        with self._transaction() as connection:
            changed = connection.execute(
                """UPDATE service_readiness SET resident_variant = ?, free_vram_mib = ?,
                total_vram_mib = ?, compute_capability_major = ?, compute_capability_minor = ?,
                observed_at = ?
                WHERE singleton = 1 AND worker_id = ? AND writer_epoch = ?""",
                (resident_variant, free_vram_mib, total_vram_mib, major, minor, now,
                 writer.worker_id, writer.writer_epoch),
            ).rowcount
            if changed != 1:
                raise FenceRejected("readiness writer fence rejected")

    def model_state(
        self,
    ) -> tuple[str | None, float, float, tuple[int, int]] | None:
        """The last-published resident variant and device facts, or `None`
        when no worker has published model state yet (never napped to a
        false default -- a missing row means "unknown", not "no model")."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT resident_variant, free_vram_mib, total_vram_mib,
                compute_capability_major, compute_capability_minor
                FROM service_readiness WHERE singleton = 1"""
            ).fetchone()
        if row is None or row["free_vram_mib"] is None:
            return None
        return (
            row["resident_variant"],
            float(row["free_vram_mib"]),
            float(row["total_vram_mib"]),
            (int(row["compute_capability_major"]), int(row["compute_capability_minor"])),
        )

    def readiness_status(self) -> tuple[bool, bool, bool, float] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT worker_ready, sandbox_ready, diagnostic_docx_ready, observed_at
                FROM service_readiness WHERE singleton = 1"""
            ).fetchone()
        if row is None:
            return None
        return (
            bool(row["worker_ready"]),
            bool(row["sandbox_ready"]),
            bool(row["diagnostic_docx_ready"]),
            float(row["observed_at"]),
        )


class SettingsMixin:
    """Closed-key `service_settings` accessors (§4.3). The single v1 key is
    `default_variant`; DDL's CHECK keeps the key list closed, so an unknown
    key surfaces as a plain `sqlite3.IntegrityError`, not a silent no-op."""

    def get_setting(self, key: str) -> Any | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT value_json FROM service_settings WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else json.loads(row["value_json"])

    def set_setting(self, key: str, value: Any, *, now: float) -> None:
        try:
            value_json = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise JobStoreInvalid(f"setting {key!r} value is not JSON serializable: {exc}") from exc
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO service_settings (key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json,
                updated_at = excluded.updated_at""",
                (key, value_json, now),
            )

    def delete_setting(self, key: str) -> None:
        with self._transaction() as connection:
            connection.execute("DELETE FROM service_settings WHERE key = ?", (key,))


class RetentionMixin:
    def delete_inactive_work_execution(
        self, execution_id: str, *, now: float, orphan_is_old: bool,
        delete_bytes: Callable[[], None],
    ) -> bool:
        """Fence cleanup against live attempts and claims using the same DB lock."""
        with self._transaction() as connection:
            if connection.execute(
                """SELECT 1 FROM conversion_jobs WHERE execution_id = ?
                AND state IN ('running', 'cancel_requested') AND lease_expires_at > ?""",
                (execution_id, now),
            ).fetchone() is not None:
                return False
            known = connection.execute(
                "SELECT 1 FROM job_attempts WHERE execution_id = ?", (execution_id,)
            ).fetchone()
            if known is None and not orphan_is_old:
                return False
            delete_bytes()
            return True

    def retention_jobs(self) -> list[RetentionJobRecord]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT job_id, created_at, terminal_at FROM conversion_jobs
                WHERE state IN ('succeeded', 'failed', 'cancelled', 'expired')
                AND terminal_at IS NOT NULL ORDER BY id"""
            ).fetchall()
        return [RetentionJobRecord(row["job_id"], row["created_at"], row["terminal_at"])
                for row in rows]

    def retention_artifacts(self) -> list[RetentionArtifactRecord]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT job.principal_id, artifact.job_id, artifact.execution_id,
                artifact.role, artifact.ref, artifact.sha256, artifact.size_bytes,
                artifact.mime_type, artifact.created_at, job.terminal_at
                FROM job_artifacts AS artifact JOIN conversion_jobs AS job
                ON job.job_id = artifact.job_id ORDER BY artifact.id"""
            ).fetchall()
        return [RetentionArtifactRecord(**dict(row)) for row in rows]

    def unregistered_artifact_fence(
        self, *, job_id: str, execution_id: str | None, ref: str,
    ) -> UnregisteredArtifactFence | None:
        """Snapshot an unregistered publication without authorizing active-attempt deletion."""
        with closing(self._connect()) as connection:
            if connection.execute("SELECT 1 FROM job_artifacts WHERE ref = ?", (ref,)).fetchone():
                return None
            row = connection.execute(
                "SELECT state, execution_id, updated_at FROM conversion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return UnregisteredArtifactFence(job_id, execution_id, None, None)
        if row["state"] == "running" and (
            execution_id is None or row["execution_id"] == execution_id
        ):
            return None
        return UnregisteredArtifactFence(
            job_id, execution_id, float(row["updated_at"]), str(row["state"])
        )

    def delete_unregistered_artifact(
        self, fence: UnregisteredArtifactFence, *, ref: str,
        delete_bytes: Callable[[], None],
    ) -> bool:
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM job_artifacts WHERE ref = ?", (ref,)).fetchone():
                return False
            row = connection.execute(
                "SELECT state, execution_id, updated_at FROM conversion_jobs WHERE job_id = ?",
                (fence.job_id,),
            ).fetchone()
            if fence.expected_state is None:
                if row is not None:
                    return False
            elif row is None or row["state"] != fence.expected_state or (
                float(row["updated_at"]) != fence.expected_updated_at
            ):
                return False
            if row is not None and row["state"] == "running" and (
                fence.execution_id is None or row["execution_id"] == fence.execution_id
            ):
                return False
            delete_bytes()
            return True

    def delete_retention_artifact(
        self, record: RetentionArtifactRecord, delete_bytes: Callable[[], None]
    ) -> bool:
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT 1 FROM job_artifacts WHERE job_id = ? AND execution_id = ?
                AND role = ? AND ref = ? AND sha256 = ? AND size_bytes = ?
                AND mime_type = ? AND created_at = ?""",
                (record.job_id, record.execution_id, record.role, record.ref, record.sha256,
                 record.size_bytes, record.mime_type, record.created_at),
            ).fetchone()
            if row is None:
                return False
            delete_bytes()
            return connection.execute(
                "DELETE FROM job_artifacts WHERE job_id = ? AND role = ? AND ref = ?",
                (record.job_id, record.role, record.ref),
            ).rowcount == 1

    def delete_staged_retention_artifact(
        self, *, job_id: str, principal_id: str, expected_updated_at: float,
        expected_state: str, delete_bytes: Callable[[], None],
    ) -> bool:
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT 1 FROM conversion_jobs WHERE job_id = ? AND principal_id = ?
                AND updated_at = ? AND state = ?""",
                (job_id, principal_id, expected_updated_at, expected_state),
            ).fetchone()
            if row is None:
                return False
            delete_bytes()
            return True

    def delete_orphan_staged_artifact(
        self, *, job_id: str, delete_bytes: Callable[[], None],
    ) -> bool:
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM conversion_jobs WHERE job_id = ?", (job_id,)
            ).fetchone() is not None:
                return False
            delete_bytes()
            return True

    def delete_retention_job(self, record: RetentionJobRecord) -> bool:
        with self._transaction() as connection:
            return connection.execute(
                """DELETE FROM conversion_jobs WHERE job_id = ? AND created_at = ?
                AND terminal_at = ? AND state IN ('succeeded', 'failed', 'cancelled', 'expired')
                AND NOT EXISTS (SELECT 1 FROM job_artifacts
                WHERE job_artifacts.job_id = conversion_jobs.job_id)""",
                (record.job_id, record.created_at, record.terminal_at),
            ).rowcount == 1


class WebSessionMixin:
    """CRUD over `web_sessions` (§2.2/§4.1). Idle/absolute expiry policy and
    touch throttling belong to `web_auth.WebSessionStore`, which is built on
    top of these primitives -- this mixin only persists what it is told."""

    def create_web_session(
        self, *, session_digest: str, principal_id: str, now: float, expires_at: float,
    ) -> WebSessionRecord:
        if not session_digest or not principal_id:
            raise JobStoreInvalid("session_digest and principal_id must be non-empty")
        if expires_at <= now:
            raise JobStoreInvalid("web session expires_at must be after now")
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO web_sessions (
                session_digest, principal_id, created_at, last_seen_at, expires_at)
                VALUES (?, ?, ?, ?, ?)""",
                (session_digest, principal_id, now, now, expires_at),
            )
        return WebSessionRecord(session_digest, principal_id, now, now, expires_at, None)

    def get_web_session(self, session_digest: str) -> WebSessionRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM web_sessions WHERE session_digest = ?", (session_digest,)
            ).fetchone()
        return None if row is None else web_session_from_row(row)

    def touch_web_session(self, session_digest: str, *, now: float) -> bool:
        with self._transaction() as connection:
            return connection.execute(
                "UPDATE web_sessions SET last_seen_at = ? WHERE session_digest = ?",
                (now, session_digest),
            ).rowcount == 1

    def revoke_web_session(self, session_digest: str, *, now: float) -> bool:
        with self._transaction() as connection:
            return connection.execute(
                """UPDATE web_sessions SET revoked_at = ?
                WHERE session_digest = ? AND revoked_at IS NULL""",
                (now, session_digest),
            ).rowcount == 1

    def revoke_all_web_sessions_except(
        self, principal_id: str, *, keep_session_digest: str, now: float,
    ) -> int:
        with self._transaction() as connection:
            return connection.execute(
                """UPDATE web_sessions SET revoked_at = ?
                WHERE principal_id = ? AND session_digest <> ? AND revoked_at IS NULL""",
                (now, principal_id, keep_session_digest),
            ).rowcount

    def cleanup_web_sessions(self, *, now: float) -> int:
        """Delete sessions that are revoked or past their absolute expiry."""
        with self._transaction() as connection:
            return connection.execute(
                "DELETE FROM web_sessions WHERE revoked_at IS NOT NULL OR expires_at <= ?",
                (now,),
            ).rowcount


class ModelCommandMixin:
    """API<->worker model command protocol over `model_commands` (§4.2)."""

    def record_model_startup_failure(self, *, variant_id: str, error_code: str, now: float) -> None:
        """Persist preflight failure without pretending to claim a GPU load.

        Repeated process starts retain one identical failure, rather than
        filling the queue. An active command keeps its existing owner/fence.
        """
        if error_code not in {"MODEL_VARIANT_DOES_NOT_FIT", "MODEL_ENGINE_UNAVAILABLE"}:
            raise JobStoreInvalid("invalid model preflight failure")
        with self._transaction() as connection:
            active = connection.execute(
                "SELECT command_id, kind, state FROM model_commands WHERE variant_id = ? "
                "AND state IN ('queued', 'running')", (variant_id,),
            ).fetchone()
            if active is not None:
                if active["state"] == "queued" and active["kind"] == "load":
                    # Reject an unclaimed load at admission without allocating
                    # GPU memory. A genuinely running foreign command keeps its fence.
                    connection.execute(
                        """UPDATE model_commands SET state = 'running', worker_id = 'preflight',
                        started_at = ?, heartbeat_at = ?, phase = 'loading', updated_at = ?
                        WHERE command_id = ? AND state = 'queued'""",
                        (now, now, now, active["command_id"]),
                    )
                    connection.execute(
                        """UPDATE model_commands SET state = 'failed', error_code = ?,
                        phase = NULL, worker_id = NULL, started_at = NULL, heartbeat_at = NULL,
                        finished_at = ?, updated_at = ?
                        WHERE command_id = ? AND state = 'running' AND worker_id = 'preflight'""",
                        (error_code, now, now, active["command_id"]),
                    )
                return
            latest = connection.execute(
                "SELECT state, error_code FROM model_commands WHERE variant_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (variant_id,),
            ).fetchone()
            if (
                latest is not None
                and latest["state"] == "failed"
                and latest["error_code"] == error_code
            ):
                return
            connection.execute(
                """INSERT INTO model_commands (
                command_id, kind, variant_id, state, requested_by, requested_at,
                started_at, finished_at, error_code, phase, updated_at)
                VALUES (?, 'load', ?, 'failed', 'worker', ?, ?, ?, ?, 'loading', ?)""",
                (f"cmd_{uuid.uuid4().hex}", variant_id, now, now, now, error_code, now),
            )


    def enqueue_model_command(
        self, *, kind: str, variant_id: str, requested_by: str, now: float,
    ) -> tuple[ModelCommandRecord, bool]:
        if kind not in _MODEL_COMMAND_KINDS:
            raise JobStoreInvalid(f"unknown model command kind: {kind!r}")
        if not variant_id or not requested_by:
            raise JobStoreInvalid("variant_id and requested_by must be non-empty")
        with self._transaction() as connection:
            active_rows = connection.execute(
                """SELECT * FROM model_commands
                WHERE variant_id = ? AND state IN ('queued', 'running')""",
                (variant_id,),
            ).fetchall()
            for row in active_rows:
                if row["kind"] == kind:
                    return model_command_from_row(row), False
            if active_rows:
                raise ModelCommandConflict(
                    f"variant {variant_id!r} already has an active {active_rows[0]['kind']!r} "
                    "command"
                )
            command_id = f"cmd_{uuid.uuid4().hex}"
            try:
                connection.execute(
                    """INSERT INTO model_commands (
                    command_id, kind, variant_id, state, requested_by, requested_at, updated_at)
                    VALUES (?, ?, ?, 'queued', ?, ?, ?)""",
                    (command_id, kind, variant_id, requested_by, now, now),
                )
            except sqlite3.IntegrityError:
                existing = connection.execute(
                    """SELECT * FROM model_commands WHERE variant_id = ? AND kind = ?
                    AND state IN ('queued', 'running')""",
                    (variant_id, kind),
                ).fetchone()
                if existing is None:
                    raise
                return model_command_from_row(existing), False
            row = connection.execute(
                "SELECT * FROM model_commands WHERE command_id = ?", (command_id,)
            ).fetchone()
            return model_command_from_row(row), True

    def claim_next_model_command(
        self, *, worker_id: str, now: float, kinds: tuple[str, ...] | None = None,
    ) -> ModelCommandRecord | None:
        if not worker_id:
            raise JobStoreInvalid("worker_id must be non-empty")
        if kinds is not None and (not kinds or any(k not in _MODEL_COMMAND_KINDS for k in kinds)):
            raise JobStoreInvalid("invalid model command kinds")
        query = "SELECT * FROM model_commands WHERE state = 'queued'"
        parameters: tuple[str, ...] = ()
        if kinds is not None:
            query += " AND kind IN (" + ",".join("?" for _ in kinds) + ")"
            parameters = kinds
        with self._transaction() as connection:
            row = connection.execute(
                query + " ORDER BY requested_at, id LIMIT 1", parameters,
            ).fetchone()
            if row is None:
                return None
            phase = _INITIAL_PHASE[row["kind"]]
            changed = connection.execute(
                """UPDATE model_commands SET state = 'running', worker_id = ?, started_at = ?,
                heartbeat_at = ?, phase = ?, updated_at = ?
                WHERE id = ? AND state = 'queued'""",
                (worker_id, now, now, phase, now, row["id"]),
            ).rowcount
            if changed != 1:
                raise FenceRejected(f"model command {row['command_id']!r} changed during claim")
            claimed = connection.execute(
                "SELECT * FROM model_commands WHERE command_id = ?", (row["command_id"],)
            ).fetchone()
            return model_command_from_row(claimed)

    def heartbeat_model_command(self, command_id: str, *, worker_id: str, now: float) -> None:
        with self._transaction() as connection:
            changed = connection.execute(
                """UPDATE model_commands SET heartbeat_at = ?, updated_at = ?
                WHERE command_id = ? AND worker_id = ? AND state = 'running'""",
                (now, now, command_id, worker_id),
            ).rowcount
            if changed != 1:
                raise FenceRejected(f"model command heartbeat fence rejected for {command_id!r}")

    def update_model_command_progress(
        self, command_id: str, *, worker_id: str, phase: str, now: float,
        bytes_done: int | None = None, bytes_total: int | None = None,
    ) -> None:
        with self._transaction() as connection:
            changed = connection.execute(
                """UPDATE model_commands SET phase = ?, bytes_done = ?, bytes_total = ?,
                updated_at = ?
                WHERE command_id = ? AND worker_id = ? AND state = 'running'""",
                (phase, bytes_done, bytes_total, now, command_id, worker_id),
            ).rowcount
            if changed != 1:
                raise FenceRejected(f"model command progress fence rejected for {command_id!r}")

    def finish_model_command(
        self, command_id: str, *, worker_id: str, state: str, now: float,
        error_code: str | None = None,
    ) -> bool:
        """Fenced by `(command_id, worker_id, state='running')`. Returns
        `False` (never raises) when the fence was already lost -- e.g. the
        command was requeued to another worker after a stale heartbeat -- so
        the caller can log and discard the late result (§4.2 step 4)."""
        if state not in ("succeeded", "failed"):
            raise JobStoreInvalid(f"model command cannot finish in state {state!r}")
        if state == "failed" and not error_code:
            raise JobStoreInvalid("a failed model command requires error_code")
        if state == "succeeded" and error_code is not None:
            raise JobStoreInvalid("error_code is only set for a failed model command")
        with self._transaction() as connection:
            return connection.execute(
                """UPDATE model_commands SET state = ?, phase = NULL, worker_id = NULL,
                started_at = NULL, heartbeat_at = NULL, finished_at = ?, error_code = ?,
                updated_at = ?
                WHERE command_id = ? AND worker_id = ? AND state = 'running'""",
                (state, now, error_code, now, command_id, worker_id),
            ).rowcount == 1

    def requeue_stale_model_commands(self, *, now: float, stale_after: float) -> int:
        with self._transaction() as connection:
            return connection.execute(
                """UPDATE model_commands SET state = 'queued', worker_id = NULL,
                started_at = NULL, heartbeat_at = NULL, phase = NULL, updated_at = ?
                WHERE state = 'running' AND heartbeat_at <= ?""",
                (now, now - stale_after),
            ).rowcount

    def delete_finished_commands(self, *, before: float) -> int:
        """Retention (§4.1): drop terminal commands older than `before`, one
        row per click, so the table does not grow without bound."""
        with self._transaction() as connection:
            return connection.execute(
                """DELETE FROM model_commands
                WHERE state IN ('succeeded', 'failed') AND finished_at <= ?""",
                (before,),
            ).rowcount

    def active_model_command(self, variant_id: str, kind: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT * FROM model_commands WHERE variant_id = ? AND kind = ?
                AND state IN ('queued', 'running')""",
                (variant_id, kind),
            ).fetchone()
        return None if row is None else model_command_projection(model_command_from_row(row))

    def last_model_command(self, variant_id: str, kind: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT * FROM model_commands WHERE variant_id = ? AND kind = ?
                AND state IN ('succeeded', 'failed')
                ORDER BY finished_at DESC, id DESC LIMIT 1""",
                (variant_id, kind),
            ).fetchone()
        return None if row is None else model_command_projection(model_command_from_row(row))
