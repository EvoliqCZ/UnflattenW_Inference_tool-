CREATE TABLE conversion_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    principal_id TEXT NOT NULL,
    idempotency_key_hash TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    requested_output TEXT NOT NULL CHECK (requested_output IN ('prediction_ir', 'diagnostic_docx')),
    state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', 'expired')),
    stage TEXT NOT NULL CHECK (stage IN (
        'admission_complete', 'loading_model', 'pdf_inspection', 'pdf_rasterization',
        'png_validation', 'model_input_projection', 'feature_extraction', 'model_decode',
        'prediction_ir_sealing', 'ast_promotion', 'renderer_gate', 'artifact_finalization'
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
    -- DEC-044 Dodatok D §1: the resolved model variant this job runs under;
    -- NULL when no models config is configured (today's single-bundle mode).
    model_variant TEXT,
    -- DEC-044 Dodatok D §1 (B1): multi-page progress. NULL for a job that
    -- has not started per-page decode yet; `pages_json` holds the closed
    -- per-page report array (`{index,state,error_code,decode_steps,seconds}`)
    -- as it fills in, so a status read mid-job sees real partial progress.
    pages_total INTEGER,
    pages_done INTEGER,
    current_page INTEGER,
    pages_json TEXT,
    -- DEC-052 §4.1 (v7): free-text label and input classification for the
    -- web UI job list; NULL for jobs submitted before this column existed.
    label TEXT,
    input_kind TEXT CHECK (input_kind IN ('pdf', 'image')),
    -- §17: engine variant provenance at admission time; NULL without a
    -- models config, mirroring model_variant.
    engine TEXT CHECK (engine IN ('native')),
    UNIQUE (principal_id, idempotency_key_hash),
    CHECK (
        (state = 'running' AND lease_owner IS NOT NULL AND execution_id IS NOT NULL
            AND lease_expires_at IS NOT NULL AND heartbeat_at IS NOT NULL)
        OR
        (state <> 'running' AND lease_owner IS NULL AND execution_id IS NULL
            AND lease_expires_at IS NULL AND heartbeat_at IS NULL)
    )
);

CREATE INDEX conversion_jobs_claim_idx
    ON conversion_jobs (state, cancel_requested, created_at);
CREATE INDEX conversion_jobs_lease_idx
    ON conversion_jobs (state, lease_expires_at);
CREATE INDEX conversion_jobs_principal_active_idx
    ON conversion_jobs (principal_id, state);
CREATE INDEX conversion_jobs_expiry_idx
    ON conversion_jobs (expires_at);
CREATE INDEX conversion_jobs_principal_created_idx
    ON conversion_jobs (principal_id, created_at DESC, id DESC);

CREATE TRIGGER conversion_jobs_closed_state_transition
BEFORE UPDATE OF state ON conversion_jobs
WHEN OLD.state <> NEW.state AND NOT (
    (OLD.state = 'queued' AND NEW.state IN ('running', 'cancelled'))
    OR
    (OLD.state = 'running' AND NEW.state IN ('queued', 'succeeded', 'failed', 'cancelled'))
    OR
    (OLD.state IN ('succeeded', 'failed', 'cancelled') AND NEW.state = 'expired')
)
BEGIN
    SELECT RAISE(ABORT, 'invalid conversion job state transition');
END;

CREATE TABLE job_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES conversion_jobs(job_id) ON DELETE CASCADE,
    attempt INTEGER NOT NULL CHECK (attempt >= 1),
    execution_id TEXT NOT NULL UNIQUE,
    worker_id TEXT NOT NULL,
    lease_epoch INTEGER NOT NULL CHECK (lease_epoch >= 1),
    started_at REAL NOT NULL,
    finished_at REAL,
    outcome TEXT CHECK (outcome IN ('succeeded', 'failed', 'retry', 'lease_expired', 'cancelled')),
    failure_class TEXT,
    stage_report TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (job_id, attempt)
);

CREATE INDEX job_attempts_job_idx ON job_attempts (job_id, attempt);

CREATE TABLE job_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES conversion_jobs(job_id) ON DELETE CASCADE,
    execution_id TEXT NOT NULL REFERENCES job_attempts(execution_id),
    role TEXT NOT NULL CHECK (
        role IN ('prediction_ir', 'diagnostic_docx', 'source_preview', 'docx_preview')
    ),
    -- DEC-044 Dodatok D §1 (B1/C): -1 is the sentinel "not page-scoped"
    -- (today's single whole-document artifact, e.g. the merged
    -- diagnostic_docx, or a legacy single-page prediction_ir); a real page's
    -- own artifact (per-page prediction_ir, previews) carries its 0-based
    -- index. NOT NULL so the UNIQUE constraint below actually enforces "at
    -- most one" (SQLite treats every NULL as distinct).
    page_index INTEGER NOT NULL DEFAULT -1,
    ref TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    mime_type TEXT NOT NULL,
    private INTEGER NOT NULL CHECK (private = 1),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (job_id, role, page_index)
);

CREATE INDEX job_artifacts_job_idx ON job_artifacts (job_id);

CREATE TABLE service_readiness (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    worker_id TEXT NOT NULL,
    writer_epoch INTEGER NOT NULL CHECK (writer_epoch >= 1),
    worker_ready INTEGER NOT NULL CHECK (worker_ready IN (0, 1)),
    sandbox_ready INTEGER NOT NULL CHECK (sandbox_ready IN (0, 1)),
    diagnostic_docx_ready INTEGER NOT NULL DEFAULT 0 CHECK (diagnostic_docx_ready IN (0, 1)),
    observed_at REAL NOT NULL,
    -- DEC-044 Dodatok D §1: the worker's resident model variant and measured
    -- device facts, published alongside readiness. NULL until a worker with
    -- a models config has published at least once.
    resident_variant TEXT,
    free_vram_mib REAL,
    total_vram_mib REAL,
    compute_capability_major INTEGER,
    compute_capability_minor INTEGER,
    -- DEC-052 §4.1 (v7): whether the worker's sandbox runs rootful; NULL
    -- until a worker has published at least once.
    sandbox_rootful INTEGER CHECK (sandbox_rootful IN (0, 1)),
    -- §17.5: JSON list of engines this worker can use, e.g. '["native"]'.
    available_engines_json TEXT
);

CREATE TABLE web_sessions (
    session_digest TEXT PRIMARY KEY,          -- sha256 hex of the 256-bit token; the raw token is never stored
    principal_id   TEXT NOT NULL,
    created_at     REAL NOT NULL,
    last_seen_at   REAL NOT NULL,
    expires_at     REAL NOT NULL,             -- absolute ceiling (created_at + max_seconds)
    revoked_at     REAL,
    CHECK (last_seen_at >= created_at AND expires_at > created_at)
);
CREATE INDEX web_sessions_expiry_idx ON web_sessions (expires_at);

CREATE TABLE model_commands (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    command_id    TEXT NOT NULL UNIQUE,        -- 'cmd_<uuid4 hex>'
    kind          TEXT NOT NULL CHECK (kind IN ('download', 'load', 'delete')),
    variant_id    TEXT NOT NULL,
    state         TEXT NOT NULL CHECK (state IN ('queued', 'running', 'succeeded', 'failed')),
    phase         TEXT CHECK (phase IN ('downloading', 'verifying', 'loading', 'deleting')),
    requested_by  TEXT NOT NULL,               -- principal_id, or 'worker' for the startup auto-load
    worker_id     TEXT,
    bytes_done    INTEGER CHECK (bytes_done IS NULL OR bytes_done >= 0),
    bytes_total   INTEGER CHECK (bytes_total IS NULL OR bytes_total >= 0),
    error_code    TEXT,                        -- from PROBLEM_CODES, only when state = 'failed'
    requested_at  REAL NOT NULL,
    started_at    REAL,
    heartbeat_at  REAL,
    finished_at   REAL,
    updated_at    REAL NOT NULL,
    CHECK ((state = 'running') = (worker_id IS NOT NULL AND started_at IS NOT NULL AND heartbeat_at IS NOT NULL)),
    CHECK ((state IN ('succeeded', 'failed')) = (finished_at IS NOT NULL)),
    CHECK ((state = 'failed') = (error_code IS NOT NULL))
);
-- at most one active command per (variant, kind); POST idempotency relies on this index
CREATE UNIQUE INDEX model_commands_active_idx
    ON model_commands (variant_id, kind) WHERE state IN ('queued', 'running');
CREATE INDEX model_commands_claim_idx ON model_commands (state, requested_at);

CREATE TRIGGER model_commands_closed_state_transition
BEFORE UPDATE OF state ON model_commands
WHEN OLD.state <> NEW.state AND NOT (
    (OLD.state = 'queued' AND NEW.state = 'running')
    OR
    (OLD.state = 'running' AND NEW.state IN ('succeeded', 'failed', 'queued'))
)
BEGIN
    SELECT RAISE(ABORT, 'invalid model command state transition');
END;

CREATE TABLE service_settings (
    key        TEXT PRIMARY KEY CHECK (key IN ('default_variant')),   -- closed key list; a new key needs a new migration
    value_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
