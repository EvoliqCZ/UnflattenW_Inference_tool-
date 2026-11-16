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
    compute_capability_minor INTEGER
);
