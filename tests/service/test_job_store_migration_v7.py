"""DEC-052 §4.1: SQLite store migration v6 -> v7.

Builds a real v6 fixture database (from the frozen `data/schema_v6.sql`,
matching the pre-v7 `job_store_schema.py` branch byte for byte) with real
rows, then verifies the v7 migration is additive, forward-only, idempotent,
and fails closed for any newer/unknown version.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from unflatten.service.job_store import JobStore, JobStoreInvalid

_FIXTURE = Path(__file__).with_name("data") / "schema_v6.sql"


def make_v6_fixture(path: Path) -> None:
    """Create a v6 database at `path` with one real row per table."""
    schema = _FIXTURE.read_text(encoding="utf-8")
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(schema)
        connection.execute(
            """INSERT INTO conversion_jobs (
                job_id, principal_id, idempotency_key_hash, request_digest,
                requested_output, state, stage, attempt, max_attempts,
                cancel_requested, policy_hash, component_lock_hash,
                created_at, updated_at
            ) VALUES (
                'job_v6', 'principal_a', 'key-v6', 'digest-v6',
                'prediction_ir', 'queued', 'admission_complete', 0, 2,
                0, 'sha256:policy', 'sha256:components', 100.0, 100.0
            )"""
        )
        connection.execute(
            """INSERT INTO job_attempts (
                job_id, attempt, execution_id, worker_id, lease_epoch,
                started_at, created_at, updated_at
            ) VALUES ('job_v6', 1, 'exec_v6', 'worker_v6', 1, 100.0, 100.0, 100.0)"""
        )
        connection.execute(
            """INSERT INTO job_artifacts (
                job_id, execution_id, role, page_index, ref, sha256,
                size_bytes, mime_type, private, created_at, updated_at
            ) VALUES (
                'job_v6', 'exec_v6', 'prediction_ir', -1, 'ref_v6', ?, 1,
                'application/json', 1, 100.0, 100.0
            )""",
            ("a" * 64,),
        )
        connection.execute(
            """INSERT INTO service_readiness (
                singleton, worker_id, writer_epoch, worker_ready, sandbox_ready,
                observed_at
            ) VALUES (1, 'worker_v6', 1, 1, 1, 100.0)"""
        )
        connection.execute("PRAGMA user_version=6")
        connection.commit()


def open_store(path: Path) -> JobStore:
    store = JobStore(
        path,
        lease_duration_seconds=10,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    return store


def test_v6_store_migrates_to_v7_without_touching_rows(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    make_v6_fixture(path)

    open_store(path)

    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 7
        job = connection.execute(
            "SELECT * FROM conversion_jobs WHERE job_id = 'job_v6'"
        ).fetchone()
        assert job["principal_id"] == "principal_a"
        assert job["request_digest"] == "digest-v6"
        assert job["label"] is None
        assert job["input_kind"] is None
        assert job["engine"] is None
        assert connection.execute("SELECT COUNT(*) FROM conversion_jobs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM job_attempts").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM job_artifacts").fetchone()[0] == 1
        readiness = connection.execute("SELECT * FROM service_readiness").fetchone()
        assert readiness["worker_id"] == "worker_v6"
        assert readiness["sandbox_rootful"] is None
        assert readiness["available_engines_json"] is None


def test_v6_store_migration_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    make_v6_fixture(path)
    open_store(path)

    open_store(path)

    with closing(sqlite3.connect(path)) as connection:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 7
        assert connection.execute("SELECT COUNT(*) FROM conversion_jobs").fetchone()[0] == 1


def test_fresh_store_is_created_at_v7(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"

    open_store(path)

    with closing(sqlite3.connect(path)) as connection:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 7
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"web_sessions", "model_commands", "service_settings"} <= tables


def test_label_input_kind_engine_columns_nullable_with_checks(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    open_store(path)

    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """INSERT INTO conversion_jobs (
                job_id, principal_id, idempotency_key_hash, request_digest,
                requested_output, state, stage, attempt, max_attempts,
                cancel_requested, policy_hash, component_lock_hash,
                created_at, updated_at, label, input_kind, engine
            ) VALUES (
                'job_new', 'principal_a', 'key-new', 'digest-new',
                'prediction_ir', 'queued', 'admission_complete', 0, 2,
                0, 'sha256:policy', 'sha256:components', 100.0, 100.0,
                'my label', 'pdf', 'native'
            )"""
        )
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE conversion_jobs SET input_kind = 'docx' WHERE job_id = 'job_new'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE conversion_jobs SET engine = 'other' WHERE job_id = 'job_new'"
            )


def test_principal_created_index_exists(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    open_store(path)

    with closing(sqlite3.connect(path)) as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    assert "conversion_jobs_principal_created_idx" in indexes


def test_web_sessions_check_constraints(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    open_store(path)

    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """INSERT INTO web_sessions (
                session_digest, principal_id, created_at, last_seen_at, expires_at
            ) VALUES (?, 'principal_a', 100.0, 100.0, 200.0)""",
            ("a" * 64,),
        )
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO web_sessions (
                    session_digest, principal_id, created_at, last_seen_at, expires_at
                ) VALUES (?, 'principal_a', 100.0, 100.0, 50.0)""",
                ("b" * 64,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO web_sessions (
                    session_digest, principal_id, created_at, last_seen_at, expires_at
                ) VALUES (?, 'principal_a', 100.0, 50.0, 200.0)""",
                ("c" * 64,),
            )


def test_model_commands_single_active_per_variant_kind(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    open_store(path)

    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """INSERT INTO model_commands (
                command_id, kind, variant_id, state, requested_by,
                requested_at, updated_at
            ) VALUES ('cmd_1', 'download', 'variant_a', 'queued', 'principal_a', 100.0, 100.0)"""
        )
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO model_commands (
                    command_id, kind, variant_id, state, requested_by,
                    requested_at, updated_at
                ) VALUES (
                    'cmd_2', 'download', 'variant_a', 'queued', 'principal_a', 101.0, 101.0
                )"""
            )
        # A different kind on the same variant is a separate slot.
        connection.execute(
            """INSERT INTO model_commands (
                command_id, kind, variant_id, state, requested_by,
                requested_at, updated_at
            ) VALUES ('cmd_3', 'load', 'variant_a', 'queued', 'principal_a', 102.0, 102.0)"""
        )
        connection.commit()


def test_model_commands_transition_trigger(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    open_store(path)

    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """INSERT INTO model_commands (
                command_id, kind, variant_id, state, requested_by,
                requested_at, updated_at
            ) VALUES ('cmd_a', 'download', 'variant_a', 'queued', 'principal_a', 100.0, 100.0)"""
        )
        connection.commit()

        # queued -> running: allowed.
        connection.execute(
            """UPDATE model_commands SET state = 'running', worker_id = 'w1',
            started_at = 101.0, heartbeat_at = 101.0, phase = 'downloading',
            updated_at = 101.0 WHERE command_id = 'cmd_a'"""
        )
        connection.commit()

        # running -> queued: allowed (requeue after stale worker).
        connection.execute(
            """UPDATE model_commands SET state = 'queued', worker_id = NULL,
            started_at = NULL, heartbeat_at = NULL, phase = NULL, updated_at = 102.0
            WHERE command_id = 'cmd_a'"""
        )
        connection.commit()

        # queued -> succeeded: disallowed (must go through running first).
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """UPDATE model_commands SET state = 'succeeded', finished_at = 103.0,
                updated_at = 103.0 WHERE command_id = 'cmd_a'"""
            )


def test_model_commands_state_invariants(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    open_store(path)

    with closing(sqlite3.connect(path)) as connection:
        # `running` requires worker_id/started_at/heartbeat_at.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO model_commands (
                    command_id, kind, variant_id, state, requested_by,
                    requested_at, updated_at
                ) VALUES ('cmd_b', 'load', 'variant_b', 'running', 'principal_a', 100.0, 100.0)"""
            )
        # terminal states require finished_at.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO model_commands (
                    command_id, kind, variant_id, state, requested_by,
                    requested_at, updated_at
                ) VALUES (
                    'cmd_c', 'load', 'variant_c', 'succeeded', 'principal_a', 100.0, 100.0
                )"""
            )
        # `failed` requires error_code.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO model_commands (
                    command_id, kind, variant_id, state, requested_by,
                    requested_at, finished_at, updated_at
                ) VALUES (
                    'cmd_d', 'load', 'variant_d', 'failed', 'principal_a', 100.0, 100.0, 100.0
                )"""
            )
        # A fully-shaped failed row is accepted.
        connection.execute(
            """INSERT INTO model_commands (
                command_id, kind, variant_id, state, requested_by, error_code,
                requested_at, finished_at, updated_at
            ) VALUES (
                'cmd_e', 'load', 'variant_e', 'failed', 'principal_a',
                'MODEL_CHECKPOINT_UNVERIFIED', 100.0, 100.0, 100.0
            )"""
        )
        connection.commit()


def test_service_settings_key_is_closed(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    open_store(path)

    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """INSERT INTO service_settings (key, value_json, updated_at)
            VALUES ('default_variant', '"variant_a"', 100.0)"""
        )
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO service_settings (key, value_json, updated_at)
                VALUES ('unknown_key', '1', 100.0)"""
            )


def test_service_readiness_new_columns(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    open_store(path)

    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """INSERT INTO service_readiness (
                singleton, worker_id, writer_epoch, worker_ready, sandbox_ready,
                observed_at, sandbox_rootful, available_engines_json
            ) VALUES (1, 'worker_new', 1, 1, 1, 100.0, 1, '["native"]')"""
        )
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE service_readiness SET sandbox_rootful = 2 WHERE singleton = 1"
            )


def test_unsupported_user_version_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    open_store(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA user_version=8")
        connection.commit()

    with pytest.raises(JobStoreInvalid, match="unsupported: 8"):
        open_store(path)
