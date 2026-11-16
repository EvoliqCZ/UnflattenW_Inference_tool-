"""SQLite connection, schema migration, and transaction concerns."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from unflatten.service.job_store_types import JobStoreInvalid

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class SchemaMixin:
    path: Path

    def initialize(self) -> None:
        """Create the schema and fail unless SQLite confirms persistent WAL mode."""
        if not self.path.parent.is_dir():
            raise JobStoreInvalid(f"job store parent directory does not exist: {self.path.parent}")
        try:
            schema = _SCHEMA_PATH.read_text(encoding="utf-8")
            with closing(self._connect()) as connection:
                mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                if str(mode).lower() != "wal":
                    raise JobStoreInvalid(
                        f"job store failed to enable WAL mode at {self.path}: {mode!r}"
                    )
                self._migrate(connection, schema)
        except OSError as exc:
            raise JobStoreInvalid(f"cannot initialize job store at {self.path}: {exc}") from exc
        except sqlite3.Error as exc:
            raise JobStoreInvalid(f"cannot initialize SQLite schema at {self.path}: {exc}") from exc

    def _migrate(self, connection: sqlite3.Connection, schema: str) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            connection.executescript(schema)
            connection.execute("PRAGMA user_version=7")
            return
        if version == 6:
            # DEC-052 §4.1 (2026-09-27): web session/model-command control
            # plane, settings, and job label/input_kind/engine -- all
            # additive, no `conversion_jobs` rebuild, no row touched.
            connection.execute("ALTER TABLE conversion_jobs ADD COLUMN label TEXT")
            connection.execute(
                "ALTER TABLE conversion_jobs ADD COLUMN input_kind TEXT "
                "CHECK (input_kind IN ('pdf', 'image'))"
            )
            connection.execute(
                "ALTER TABLE conversion_jobs ADD COLUMN engine TEXT "
                "CHECK (engine IN ('native'))"
            )
            connection.execute(
                "CREATE INDEX conversion_jobs_principal_created_idx "
                "ON conversion_jobs (principal_id, created_at DESC, id DESC)"
            )
            connection.execute(
                "ALTER TABLE service_readiness ADD COLUMN sandbox_rootful INTEGER "
                "CHECK (sandbox_rootful IN (0, 1))"
            )
            connection.execute(
                "ALTER TABLE service_readiness ADD COLUMN available_engines_json TEXT"
            )
            connection.execute(
                """CREATE TABLE web_sessions (
                session_digest TEXT PRIMARY KEY,
                principal_id   TEXT NOT NULL,
                created_at     REAL NOT NULL,
                last_seen_at   REAL NOT NULL,
                expires_at     REAL NOT NULL,
                revoked_at     REAL,
                CHECK (last_seen_at >= created_at AND expires_at > created_at)
            )"""
            )
            connection.execute(
                "CREATE INDEX web_sessions_expiry_idx ON web_sessions (expires_at)"
            )
            connection.execute(
                """CREATE TABLE model_commands (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                command_id    TEXT NOT NULL UNIQUE,
                kind          TEXT NOT NULL CHECK (kind IN ('download', 'load', 'delete')),
                variant_id    TEXT NOT NULL,
                state         TEXT NOT NULL CHECK (
                    state IN ('queued', 'running', 'succeeded', 'failed')
                ),
                phase         TEXT CHECK (
                    phase IN ('downloading', 'verifying', 'loading', 'deleting')
                ),
                requested_by  TEXT NOT NULL,
                worker_id     TEXT,
                bytes_done    INTEGER CHECK (bytes_done IS NULL OR bytes_done >= 0),
                bytes_total   INTEGER CHECK (bytes_total IS NULL OR bytes_total >= 0),
                error_code    TEXT,
                requested_at  REAL NOT NULL,
                started_at    REAL,
                heartbeat_at  REAL,
                finished_at   REAL,
                updated_at    REAL NOT NULL,
                CHECK ((state = 'running') = (
                    worker_id IS NOT NULL AND started_at IS NOT NULL AND heartbeat_at IS NOT NULL
                )),
                CHECK ((state IN ('succeeded', 'failed')) = (finished_at IS NOT NULL)),
                CHECK ((state = 'failed') = (error_code IS NOT NULL))
            )"""
            )
            connection.execute(
                "CREATE UNIQUE INDEX model_commands_active_idx "
                "ON model_commands (variant_id, kind) WHERE state IN ('queued', 'running')"
            )
            connection.execute(
                "CREATE INDEX model_commands_claim_idx ON model_commands (state, requested_at)"
            )
            connection.execute(
                """CREATE TRIGGER model_commands_closed_state_transition
                BEFORE UPDATE OF state ON model_commands
                WHEN OLD.state <> NEW.state AND NOT (
                    (OLD.state = 'queued' AND NEW.state = 'running')
                    OR
                    (OLD.state = 'running' AND NEW.state IN ('succeeded', 'failed', 'queued'))
                )
                BEGIN
                    SELECT RAISE(ABORT, 'invalid model command state transition');
                END"""
            )
            connection.execute(
                """CREATE TABLE service_settings (
                key        TEXT PRIMARY KEY CHECK (key IN ('default_variant')),
                value_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )"""
            )
            connection.execute("PRAGMA user_version=7")
            return
        if version == 5:
            # DEC-044 Dodatok D §1 fix (2026-09-27): `stage` gains
            # 'loading_model' -- SQLite cannot ALTER a CHECK constraint in
            # place, so `conversion_jobs` itself is rebuilt (its own rows,
            # indexes, and trigger recreated verbatim); `job_attempts`/
            # `job_artifacts` reference it by table name, so their FK stays
            # valid across the rename with no row touched in either table.
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute(
                """CREATE TABLE conversion_jobs_v6 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL UNIQUE,
                principal_id TEXT NOT NULL,
                idempotency_key_hash TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                requested_output TEXT NOT NULL CHECK (
                    requested_output IN ('prediction_ir', 'diagnostic_docx')
                ),
                state TEXT NOT NULL CHECK (
                    state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', 'expired')
                ),
                stage TEXT NOT NULL CHECK (stage IN (
                    'admission_complete', 'loading_model', 'pdf_inspection', 'pdf_rasterization',
                    'png_validation', 'model_input_projection', 'feature_extraction',
                    'model_decode', 'prediction_ir_sealing', 'ast_promotion', 'renderer_gate',
                    'artifact_finalization'
                )),
                attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
                max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1 AND attempt <= max_attempts),
                lease_owner TEXT,
                lease_epoch INTEGER NOT NULL DEFAULT 0 CHECK (lease_epoch >= 0),
                execution_id TEXT,
                lease_expires_at REAL,
                heartbeat_at REAL,
                cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1)),
                policy_hash TEXT NOT NULL,
                component_lock_hash TEXT NOT NULL,
                diagnostic_outcome_json TEXT,
                error_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                terminal_at REAL,
                expires_at REAL,
                model_variant TEXT,
                pages_total INTEGER,
                pages_done INTEGER,
                current_page INTEGER,
                pages_json TEXT,
                UNIQUE (principal_id, idempotency_key_hash),
                CHECK (
                    (state = 'running' AND lease_owner IS NOT NULL AND execution_id IS NOT NULL
                        AND lease_expires_at IS NOT NULL AND heartbeat_at IS NOT NULL)
                    OR
                    (state <> 'running' AND lease_owner IS NULL AND execution_id IS NULL
                        AND lease_expires_at IS NULL AND heartbeat_at IS NULL)
                )
            )"""
            )
            connection.execute(
                """INSERT INTO conversion_jobs_v6 SELECT
                id, job_id, principal_id, idempotency_key_hash, request_digest,
                requested_output, state, stage, attempt, max_attempts, lease_owner,
                lease_epoch, execution_id, lease_expires_at, heartbeat_at, cancel_requested,
                policy_hash, component_lock_hash, diagnostic_outcome_json, error_json,
                created_at, updated_at, terminal_at, expires_at, model_variant, pages_total,
                pages_done, current_page, pages_json FROM conversion_jobs"""
            )
            connection.execute("DROP TABLE conversion_jobs")
            connection.execute("ALTER TABLE conversion_jobs_v6 RENAME TO conversion_jobs")
            connection.execute(
                "CREATE INDEX conversion_jobs_claim_idx "
                "ON conversion_jobs (state, cancel_requested, created_at)"
            )
            connection.execute(
                "CREATE INDEX conversion_jobs_lease_idx "
                "ON conversion_jobs (state, lease_expires_at)"
            )
            connection.execute(
                "CREATE INDEX conversion_jobs_principal_active_idx "
                "ON conversion_jobs (principal_id, state)"
            )
            connection.execute(
                "CREATE INDEX conversion_jobs_expiry_idx ON conversion_jobs (expires_at)"
            )
            connection.execute(
                """CREATE TRIGGER conversion_jobs_closed_state_transition
                BEFORE UPDATE OF state ON conversion_jobs
                WHEN OLD.state <> NEW.state AND NOT (
                    (OLD.state = 'queued' AND NEW.state IN ('running', 'cancelled'))
                    OR
                    (OLD.state = 'running' AND NEW.state IN (
                        'queued', 'succeeded', 'failed', 'cancelled'
                    ))
                    OR
                    (OLD.state IN ('succeeded', 'failed', 'cancelled') AND NEW.state = 'expired')
                )
                BEGIN
                    SELECT RAISE(ABORT, 'invalid conversion job state transition');
                END"""
            )
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA user_version=6")
            self._migrate(connection, schema)
            return
        if version == 7:
            return
        if version == 4:
            # DEC-044 Dodatok D §1 (B1/C): additive multi-page progress
            # columns, plus a job_artifacts rebuild -- SQLite cannot ALTER a
            # table's UNIQUE constraint in place, so this recreates the table
            # under a new name, copies every row (page_index defaults to the
            # -1 sentinel, preserving every existing artifact's identity),
            # and swaps it in. No job or artifact data is lost or reordered.
            connection.execute("ALTER TABLE conversion_jobs ADD COLUMN pages_total INTEGER")
            connection.execute("ALTER TABLE conversion_jobs ADD COLUMN pages_done INTEGER")
            connection.execute("ALTER TABLE conversion_jobs ADD COLUMN current_page INTEGER")
            connection.execute("ALTER TABLE conversion_jobs ADD COLUMN pages_json TEXT")
            connection.execute(
                """CREATE TABLE job_artifacts_v5 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES conversion_jobs(job_id) ON DELETE CASCADE,
                execution_id TEXT NOT NULL REFERENCES job_attempts(execution_id),
                role TEXT NOT NULL CHECK (
                    role IN ('prediction_ir', 'diagnostic_docx', 'source_preview', 'docx_preview')
                ),
                page_index INTEGER NOT NULL DEFAULT -1,
                ref TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                mime_type TEXT NOT NULL,
                private INTEGER NOT NULL CHECK (private = 1),
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE (job_id, role, page_index)
            )"""
            )
            connection.execute(
                """INSERT INTO job_artifacts_v5 (
                id, job_id, execution_id, role, page_index, ref, sha256, size_bytes,
                mime_type, private, created_at, updated_at)
                SELECT id, job_id, execution_id, role, -1, ref, sha256, size_bytes,
                mime_type, private, created_at, updated_at FROM job_artifacts"""
            )
            connection.execute("DROP TABLE job_artifacts")
            connection.execute("ALTER TABLE job_artifacts_v5 RENAME TO job_artifacts")
            connection.execute("CREATE INDEX job_artifacts_job_idx ON job_artifacts (job_id)")
            connection.execute("PRAGMA user_version=5")
            self._migrate(connection, schema)
            return
        if version == 3:
            # DEC-044 Dodatok D §1: additive columns for model variants; no
            # existing row's data or constraints change.
            connection.execute("ALTER TABLE conversion_jobs ADD COLUMN model_variant TEXT")
            connection.execute(
                "ALTER TABLE service_readiness ADD COLUMN resident_variant TEXT"
            )
            connection.execute(
                "ALTER TABLE service_readiness ADD COLUMN free_vram_mib REAL"
            )
            connection.execute(
                "ALTER TABLE service_readiness ADD COLUMN total_vram_mib REAL"
            )
            connection.execute(
                "ALTER TABLE service_readiness ADD COLUMN compute_capability_major INTEGER"
            )
            connection.execute(
                "ALTER TABLE service_readiness ADD COLUMN compute_capability_minor INTEGER"
            )
            connection.execute("PRAGMA user_version=4")
            self._migrate(connection, schema)
            return
        if version in (1, 2):
            raise JobStoreInvalid(
                f"schema v{version} nie je podporovaná; služba nebola s v1/v2 nasadená, "
                "zmaž `jobs.sqlite3` alebo použi nový `UNFLATTEN_DATA_DIR`"
            )
        raise JobStoreInvalid(
            f"job store schema version at {self.path} is unsupported: {version}"
        )

    def journal_mode(self) -> str:
        with closing(self._connect()) as connection:
            return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
