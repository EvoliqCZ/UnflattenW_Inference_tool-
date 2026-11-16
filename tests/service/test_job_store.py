from __future__ import annotations

import importlib.util
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

import unflatten.service.job_store as job_store


def test_job_store_module_exists() -> None:
    assert importlib.util.find_spec("unflatten.service.job_store") is not None


def test_job_store_exposes_sqlite_repository() -> None:
    assert hasattr(job_store, "JobStore")
    assert (Path(job_store.__file__).parent / "schema.sql").is_file()


def test_job_store_module_stays_below_repository_hard_limit() -> None:
    assert len(Path(job_store.__file__).read_text(encoding="utf-8").splitlines()) <= 500


def make_store(tmp_path: Path, **overrides: int) -> job_store.JobStore:
    settings = {
        "lease_duration_seconds": 10,
        "max_attempts": 2,
        "global_active_jobs": 4,
        "active_jobs_per_principal": 2,
    }
    settings.update(overrides)
    store = job_store.JobStore(tmp_path / "jobs.sqlite3", **settings)
    store.initialize()
    return store


def submit(
    store: job_store.JobStore,
    *,
    job_id: str,
    principal_id: str = "principal_a",
    key_hash: str | None = None,
    request_digest: str | None = None,
    requested_output: str = "prediction_ir",
) -> object:
    return store.create_job(
        job_id=job_id,
        principal_id=principal_id,
        idempotency_key_hash=key_hash or f"key-{job_id}",
        request_digest=request_digest or f"request-{job_id}",
        requested_output=requested_output,
        policy_hash="sha256:policy",
        component_lock_hash="sha256:components",
        now=100.0,
    )


def test_initialize_enables_and_verifies_wal_with_foreign_keys(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    assert store.journal_mode() == "wal"
    with closing(sqlite3.connect(store.path)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"conversion_jobs", "job_attempts", "job_artifacts"} <= tables


@pytest.mark.parametrize("version", [1, 2])
def test_initialize_rejects_pre_v3_store_with_no_silent_migration(
    tmp_path: Path, version: int
) -> None:
    store = make_store(tmp_path)
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute(f"PRAGMA user_version={version}")

    with pytest.raises(job_store.JobStoreInvalid, match="jobs.sqlite3"):
        store.initialize()


def test_initialize_is_idempotent_for_current_schema_version(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    store.initialize()

    with closing(sqlite3.connect(store.path)) as connection:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 7


def test_create_job_enforces_principal_scoped_idempotency(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    first = submit(store, job_id="job_first", key_hash="same", request_digest="digest")
    replay = submit(store, job_id="job_ignored", key_hash="same", request_digest="digest")
    other_principal = submit(
        store,
        job_id="job_other",
        principal_id="principal_b",
        key_hash="same",
        request_digest="other",
    )

    assert first.created is True
    assert replay.created is False
    assert replay.job.job_id == "job_first"
    assert other_principal.created is True
    with pytest.raises(job_store.IdempotencyKeyReused):
        submit(store, job_id="job_conflict", key_hash="same", request_digest="changed")


def test_create_job_accepts_diagnostic_docx_requested_output(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    result = submit(store, job_id="job_docx", requested_output="diagnostic_docx")

    assert result.created is True
    assert result.job.requested_output == "diagnostic_docx"


def test_create_job_refuses_product_docx_requested_output(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    with pytest.raises(job_store.JobStoreInvalid):
        submit(store, job_id="job_docx_rejected", requested_output="docx")


def test_create_job_stores_and_returns_model_variant(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    result = store.create_job(
        job_id="job_variant",
        principal_id="principal_a",
        idempotency_key_hash="key-variant",
        request_digest="digest-variant",
        requested_output="prediction_ir",
        policy_hash="sha256:policy",
        component_lock_hash="sha256:components",
        now=100.0,
        model_variant="fp8",
    )

    assert result.job.model_variant == "fp8"
    assert store.get_job("job_variant", "principal_a").model_variant == "fp8"


def test_update_page_progress_publishes_and_get_job_reads_it_back(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_pages")
    lease = store.claim_next(worker_id="worker_a", now=100.0)

    store.update_page_progress(
        lease, pages_total=3, pages_done=1, current_page=1,
        pages=[{"index": 0, "state": "succeeded", "error_code": None,
                "decode_steps": 12, "seconds": 1.5}],
        now=101.0,
    )

    job = store.get_job("job_pages", "principal_a")
    assert job.pages_total == 3
    assert job.pages_done == 1
    assert job.current_page == 1
    assert job.pages == ({"index": 0, "state": "succeeded", "error_code": None,
                           "decode_steps": 12, "seconds": 1.5},)


def test_update_page_progress_rejects_inconsistent_counts(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_pages_bad")
    lease = store.claim_next(worker_id="worker_a", now=100.0)

    with pytest.raises(job_store.JobStoreInvalid):
        store.update_page_progress(
            lease, pages_total=2, pages_done=3, current_page=0, pages=[], now=101.0,
        )


def test_finalize_success_accepts_multiple_page_scoped_artifacts_of_the_same_role(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_multi_ir", requested_output="diagnostic_docx")
    lease = store.claim_next(worker_id="worker_a", now=100.0)

    job = store.finalize_success(
        lease,
        artifacts=[
            job_store.ArtifactMetadata(
                role="prediction_ir", ref="ref0", sha256="a" * 64,
                size_bytes=1, mime_type="application/json", page_index=0,
            ),
            job_store.ArtifactMetadata(
                role="prediction_ir", ref="ref1", sha256="b" * 64,
                size_bytes=1, mime_type="application/json", page_index=1,
            ),
            job_store.ArtifactMetadata(
                role="diagnostic_docx", ref="ref-docx", sha256="c" * 64,
                size_bytes=1, mime_type="application/vnd.docx",
            ),
        ],
        diagnostic_outcome={"a": 1},
        retention_seconds=60,
        now=101.0,
    )

    assert job.state == "succeeded"
    artifacts = store.list_artifacts("job_multi_ir")
    by_key = {(a.role, a.page_index): a for a in artifacts}
    assert by_key[("prediction_ir", 0)].ref == "ref0"
    assert by_key[("prediction_ir", 1)].ref == "ref1"
    assert by_key[("diagnostic_docx", -1)].ref == "ref-docx"


def test_create_job_defaults_model_variant_to_none(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    result = submit(store, job_id="job_no_variant")

    assert result.job.model_variant is None


def test_migration_from_v3_adds_model_variant_and_model_state_columns(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    # Simulate a store frozen at schema v3 (before this change) by dropping
    # the new columns back out and rewinding user_version.
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute("PRAGMA user_version=3")
        connection.execute(
            """CREATE TABLE conversion_jobs_v3 AS SELECT
            id, job_id, principal_id, idempotency_key_hash, request_digest,
            requested_output, state, stage, attempt, max_attempts, lease_owner,
            lease_epoch, execution_id, lease_expires_at, heartbeat_at,
            cancel_requested, policy_hash, component_lock_hash,
            diagnostic_outcome_json, error_json, created_at, updated_at,
            terminal_at, expires_at FROM conversion_jobs"""
        )
        connection.execute("DROP TABLE conversion_jobs")
        connection.execute(
            "ALTER TABLE conversion_jobs_v3 RENAME TO conversion_jobs"
        )
        # `CREATE TABLE ... AS SELECT` does not carry over UNIQUE/CHECK --
        # job_artifacts' FK to conversion_jobs(job_id) needs a real unique
        # index on the parent, exactly like the genuine v3 schema had.
        connection.execute(
            "CREATE UNIQUE INDEX conversion_jobs_v3_job_id ON conversion_jobs (job_id)"
        )
        connection.execute(
            """CREATE TABLE service_readiness_v3 AS SELECT
            singleton, worker_id, writer_epoch, worker_ready, sandbox_ready,
            diagnostic_docx_ready, observed_at FROM service_readiness"""
        )
        connection.execute("DROP TABLE service_readiness")
        connection.execute(
            "ALTER TABLE service_readiness_v3 RENAME TO service_readiness"
        )
        connection.execute("DROP TABLE web_sessions")
        connection.execute("DROP TABLE model_commands")
        connection.execute("DROP TABLE service_settings")
        connection.commit()

    reopened = job_store.JobStore(
        store.path, lease_duration_seconds=10, max_attempts=2,
        global_active_jobs=4, active_jobs_per_principal=2,
    )
    reopened.initialize()

    with closing(sqlite3.connect(store.path)) as connection:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 7
        job_columns = {row[1] for row in connection.execute("PRAGMA table_info(conversion_jobs)")}
        readiness_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(service_readiness)")
        }
    assert "model_variant" in job_columns
    assert {"resident_variant", "free_vram_mib", "total_vram_mib"} <= readiness_columns


def test_migration_from_v5_allows_loading_model_stage_and_keeps_existing_rows(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    result = submit(store, job_id="job_pre_v6")

    with closing(sqlite3.connect(store.path)) as connection:
        # `make_store` creates a fresh v7 store, so `service_readiness`
        # already carries the v7 columns (`sandbox_rootful`,
        # `available_engines_json`) that a genuine v5 store would not have
        # yet -- strip them back out before rewinding user_version, or the
        # v6->v7 migration's own ADD COLUMN below sees them as duplicates.
        connection.execute(
            """CREATE TABLE service_readiness_v5 AS SELECT
            singleton, worker_id, writer_epoch, worker_ready, sandbox_ready,
            diagnostic_docx_ready, observed_at, resident_variant, free_vram_mib,
            total_vram_mib, compute_capability_major, compute_capability_minor
            FROM service_readiness"""
        )
        connection.execute("DROP TABLE service_readiness")
        connection.execute(
            "ALTER TABLE service_readiness_v5 RENAME TO service_readiness"
        )
        connection.execute("DROP TABLE web_sessions")
        connection.execute("DROP TABLE model_commands")
        connection.execute("DROP TABLE service_settings")
        connection.execute("PRAGMA user_version=5")
        connection.commit()

    reopened = job_store.JobStore(
        store.path, lease_duration_seconds=10, max_attempts=2,
        global_active_jobs=4, active_jobs_per_principal=2,
    )
    reopened.initialize()

    assert reopened.get_job("job_pre_v6", "principal_a").request_digest == result.job.request_digest
    with closing(sqlite3.connect(store.path)) as connection:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 7
        connection.execute(
            "UPDATE conversion_jobs SET stage = 'loading_model' WHERE job_id = 'job_pre_v6'"
        )


def test_schema_job_artifacts_role_check_accepts_diagnostic_docx(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute(
            """INSERT INTO job_artifacts (job_id, execution_id, role, ref, sha256,
            size_bytes, mime_type, private, created_at, updated_at)
            VALUES ('job_x', 'exec_x', 'diagnostic_docx', 'ref', ?, 1,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            1, 1, 1)""",
            ("d" * 64,),
        )
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO job_artifacts (job_id, execution_id, role, ref, sha256,
                size_bytes, mime_type, private, created_at, updated_at)
                VALUES ('job_y', 'exec_y', 'docx', 'ref', ?, 1, 'application/octet-stream',
                1, 1, 1)""",
                ("e" * 64,),
            )


def test_job_record_exposes_persisted_runtime_identity_hashes(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    result = submit(store, job_id="job_identity")

    assert result.job.policy_hash == "sha256:policy"
    assert result.job.component_lock_hash == "sha256:components"


def test_create_job_checks_capacity_after_idempotent_replay(tmp_path: Path) -> None:
    store = make_store(
        tmp_path,
        global_active_jobs=2,
        active_jobs_per_principal=1,
    )
    submit(store, job_id="job_a", key_hash="a", request_digest="a")

    replay = submit(store, job_id="job_replay", key_hash="a", request_digest="a")
    assert replay.created is False
    with pytest.raises(job_store.CapacityExceeded) as principal_error:
        submit(store, job_id="job_b")
    assert principal_error.value.scope == "principal"

    submit(store, job_id="job_c", principal_id="principal_b")
    with pytest.raises(job_store.CapacityExceeded) as global_error:
        submit(store, job_id="job_d", principal_id="principal_c")
    assert global_error.value.scope == "global"


def test_claim_heartbeat_stage_and_bounded_retry_are_durable(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_retry")

    first = store.claim_next(worker_id="worker_a", now=110.0)
    assert first is not None
    assert (first.attempt, first.lease_epoch, first.lease_expires_at) == (1, 1, 120.0)
    store.heartbeat(first, now=115.0)
    store.update_stage(first, stage="pdf_inspection", now=116.0)
    retried = store.fail_attempt(
        first,
        failure_class="transient_infrastructure",
        retryable=True,
        error={"code": "INTERNAL_ERROR"},
        now=117.0,
    )
    assert retried.state == "queued"

    second = store.claim_next(worker_id="worker_b", now=118.0)
    assert second is not None
    assert second.attempt == 2
    assert second.lease_epoch == 2
    assert second.execution_id != first.execution_id
    exhausted = store.fail_attempt(
        second,
        failure_class="transient_infrastructure",
        retryable=True,
        error={"code": "INTERNAL_ERROR"},
        now=119.0,
    )
    assert exhausted.state == "failed"
    assert store.claim_next(worker_id="worker_c", now=120.0) is None

    reopened = job_store.JobStore(
        store.path,
        lease_duration_seconds=10,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    reopened.initialize()
    assert reopened.get_job("job_retry", "principal_a").state == "failed"
    assert [attempt.outcome for attempt in reopened.list_attempts("job_retry")] == [
        "retry",
        "failed",
    ]


def test_concurrent_claims_assign_one_execution_once(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_once")

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(
            executor.map(
                lambda worker: store.claim_next(worker_id=worker, now=110.0),
                ("worker_a", "worker_b"),
            )
        )

    leases = [claim for claim in claims if claim is not None]
    assert len(leases) == 1
    assert len(store.list_attempts("job_once")) == 1


def test_expired_lease_requeues_only_with_attempt_budget(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_expired")
    first = store.claim_next(worker_id="worker_a", now=100.0)
    assert first is not None

    assert store.requeue_expired(now=111.0) == 1
    assert store.get_job("job_expired", "principal_a").state == "queued"
    second = store.claim_next(worker_id="worker_b", now=112.0)
    assert second is not None
    assert store.requeue_expired(now=123.0) == 1
    assert store.get_job("job_expired", "principal_a").state == "failed"


def test_cancel_queued_and_terminal_state_edges_are_closed(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_cancel")

    cancelled = store.request_cancel("job_cancel", "principal_a", now=101.0)
    assert cancelled.state == "cancelled"
    with pytest.raises(job_store.JobTerminal):
        store.request_cancel("job_cancel", "principal_a", now=102.0)
    assert store.get_job("job_cancel", "principal_b") is None

    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE conversion_jobs SET state = 'unknown' WHERE job_id = 'job_cancel'"
            )


def test_schema_rejects_terminal_reversal_even_for_known_states(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_terminal")
    store.request_cancel("job_terminal", "principal_a", now=101.0)

    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE conversion_jobs SET state = 'queued' WHERE job_id = 'job_terminal'"
            )

    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO job_attempts (
                    job_id, attempt, execution_id, worker_id, lease_epoch,
                    started_at, created_at, updated_at
                ) VALUES ('missing', 1, 'exec_missing', 'worker', 1, 1, 1, 1)
                """
            )


def test_finalize_success_accepts_diagnostic_docx_artifact_role(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_docx_artifact", requested_output="diagnostic_docx")
    lease = store.claim_next(worker_id="worker", now=101.0)
    assert lease is not None

    store.finalize_success(
        lease,
        artifacts=[
            job_store.ArtifactMetadata(
                role="diagnostic_docx",
                ref="private/result.docx",
                sha256="f" * 64,
                size_bytes=1,
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        ],
        diagnostic_outcome={"prediction_ir_status": "sealed"},
        retention_seconds=10,
        now=102.0,
    )

    [artifact] = store.list_artifacts("job_docx_artifact")
    assert artifact.role == "diagnostic_docx"


def test_finalize_success_inserts_all_artifacts_in_one_transaction(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_two_artifacts", requested_output="diagnostic_docx")
    lease = store.claim_next(worker_id="worker", now=101.0)
    assert lease is not None

    store.finalize_success(
        lease,
        artifacts=[
            job_store.ArtifactMetadata(
                role="prediction_ir",
                ref="private/result.json",
                sha256="a" * 64,
                size_bytes=1,
                mime_type="application/json",
            ),
            job_store.ArtifactMetadata(
                role="diagnostic_docx",
                ref="private/result.docx",
                sha256="b" * 64,
                size_bytes=1,
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
        ],
        diagnostic_outcome={"prediction_ir_status": "sealed"},
        retention_seconds=10,
        now=102.0,
    )

    artifacts = {a.role: a for a in store.list_artifacts("job_two_artifacts")}
    assert set(artifacts) == {"prediction_ir", "diagnostic_docx"}


def test_finalize_success_calls_publish_exactly_once_for_n_artifacts(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_publish_once", requested_output="diagnostic_docx")
    lease = store.claim_next(worker_id="worker", now=101.0)
    assert lease is not None
    calls = []

    store.finalize_success(
        lease,
        artifacts=[
            job_store.ArtifactMetadata(
                role="prediction_ir",
                ref="private/result.json",
                sha256="a" * 64,
                size_bytes=1,
                mime_type="application/json",
            ),
            job_store.ArtifactMetadata(
                role="diagnostic_docx",
                ref="private/result.docx",
                sha256="b" * 64,
                size_bytes=1,
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
        ],
        diagnostic_outcome={"prediction_ir_status": "sealed"},
        retention_seconds=10,
        now=102.0,
        publish=lambda: calls.append(1),
    )

    assert calls == [1]


def test_finalize_success_rejects_duplicate_role_before_any_row_is_written(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_dup_role", requested_output="diagnostic_docx")
    lease = store.claim_next(worker_id="worker", now=101.0)
    assert lease is not None

    def duplicate_artifact() -> job_store.ArtifactMetadata:
        return job_store.ArtifactMetadata(
            role="prediction_ir",
            ref="private/result.json",
            sha256="a" * 64,
            size_bytes=1,
            mime_type="application/json",
        )

    with pytest.raises(job_store.JobStoreInvalid, match="role"):
        store.finalize_success(
            lease,
            artifacts=[duplicate_artifact(), duplicate_artifact()],
            diagnostic_outcome={"prediction_ir_status": "sealed"},
            retention_seconds=10,
            now=102.0,
        )

    assert store.list_artifacts("job_dup_role") == []
    # The lease must still be live: rejection happened before the fenced transaction.
    completed = store.finalize_success(
        lease,
        artifacts=[duplicate_artifact()],
        diagnostic_outcome={"prediction_ir_status": "sealed"},
        retention_seconds=10,
        now=103.0,
    )
    assert completed.state == "succeeded"


def test_expire_jobs_applies_only_to_due_terminal_jobs(tmp_path: Path) -> None:
    assert hasattr(job_store.JobStore, "expire_jobs")
    store = make_store(tmp_path)
    submit(store, job_id="job_due")
    lease = store.claim_next(worker_id="worker", now=101.0)
    assert lease is not None
    store.finalize_success(
        lease,
        artifacts=[
            job_store.ArtifactMetadata(
                role="prediction_ir",
                ref="private/result.json",
                sha256="d" * 64,
                size_bytes=1,
                mime_type="application/json",
            )
        ],
        diagnostic_outcome={"prediction_ir_status": "sealed"},
        retention_seconds=10,
        now=102.0,
    )

    assert store.expire_jobs(now=111.0) == 0
    assert store.expire_jobs(now=112.0) == 1
    assert store.get_job("job_due", "principal_a").state == "expired"
    assert store.list_artifacts("job_due") == []


def test_create_job_persists_label_input_kind_engine(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    result = store.create_job(
        job_id="job_label",
        principal_id="principal_a",
        idempotency_key_hash="key-label",
        request_digest="digest-label",
        requested_output="prediction_ir",
        policy_hash="sha256:policy",
        component_lock_hash="sha256:components",
        now=100.0,
        label="Q3 report",
        input_kind="pdf",
        engine="native",
    )

    assert (result.job.label, result.job.input_kind, result.job.engine) == (
        "Q3 report", "pdf", "native",
    )
    fetched = store.get_job("job_label", "principal_a")
    assert (fetched.label, fetched.input_kind, fetched.engine) == ("Q3 report", "pdf", "native")


def test_create_job_defaults_label_input_kind_engine_to_none(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    result = submit(store, job_id="job_no_label")

    assert (result.job.label, result.job.input_kind, result.job.engine) == (None, None, None)


def test_create_job_rejects_invalid_input_kind_and_engine(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    with pytest.raises(job_store.JobStoreInvalid):
        store.create_job(
            job_id="job_bad_kind", principal_id="principal_a",
            idempotency_key_hash="key-bad-kind", request_digest="digest-bad-kind",
            requested_output="prediction_ir", policy_hash="sha256:policy",
            component_lock_hash="sha256:components", now=100.0, input_kind="docx",
        )
    with pytest.raises(job_store.JobStoreInvalid):
        store.create_job(
            job_id="job_bad_engine", principal_id="principal_a",
            idempotency_key_hash="key-bad-engine", request_digest="digest-bad-engine",
            requested_output="prediction_ir", policy_hash="sha256:policy",
            component_lock_hash="sha256:components", now=100.0, engine="cpu",
        )


def test_list_jobs_principal_scoped(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_a1", principal_id="principal_a")
    submit(store, job_id="job_b1", principal_id="principal_b")

    jobs_a, cursor = store.list_jobs(principal_id="principal_a", limit=10)

    assert [job.job_id for job in jobs_a] == ["job_a1"]
    assert cursor is None


def test_list_jobs_state_filter(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_first")
    submit(store, job_id="job_second")
    lease = store.claim_next(worker_id="worker_a", now=101.0)
    assert lease.job_id == "job_first"

    running, _ = store.list_jobs(principal_id="principal_a", states=["running"], limit=10)
    queued, _ = store.list_jobs(principal_id="principal_a", states=["queued"], limit=10)

    assert [job.job_id for job in running] == ["job_first"]
    assert [job.job_id for job in queued] == ["job_second"]


def test_list_jobs_rejects_unknown_state(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    with pytest.raises(job_store.JobStoreInvalid):
        store.list_jobs(principal_id="principal_a", states=["not_a_state"], limit=10)


def test_list_jobs_cursor_pagination_stable_order(tmp_path: Path) -> None:
    store = make_store(tmp_path, global_active_jobs=10, active_jobs_per_principal=10)
    job_ids = [f"job_page_{i}" for i in range(5)]
    for job_id in job_ids:
        submit(store, job_id=job_id)

    whole, cursor = store.list_jobs(principal_id="principal_a", limit=10)
    assert cursor is None
    assert [job.job_id for job in whole] == list(reversed(job_ids))

    collected: list[str] = []
    cursor = None
    for _ in range(len(job_ids) + 1):
        page, cursor = store.list_jobs(principal_id="principal_a", limit=2, cursor=cursor)
        collected.extend(job.job_id for job in page)
        if cursor is None:
            break

    assert collected == list(reversed(job_ids))


def test_purge_job_active_raises_JobActive(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_active")

    with pytest.raises(job_store.JobActive):
        store.purge_job("job_active", "principal_a", delete_bytes=lambda record: None)


def test_purge_job_deletes_artifacts_any_state_and_cascades(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_purge", requested_output="diagnostic_docx")
    lease = store.claim_next(worker_id="worker_a", now=101.0)
    assert lease is not None
    store.finalize_success(
        lease,
        artifacts=[
            job_store.ArtifactMetadata(
                role="diagnostic_docx", ref="ref-docx", sha256="a" * 64,
                size_bytes=1, mime_type="application/vnd.docx",
            )
        ],
        diagnostic_outcome={"a": 1},
        retention_seconds=60,
        now=102.0,
    )
    deleted: list[str] = []

    purged = store.purge_job(
        "job_purge", "principal_a", delete_bytes=lambda record: deleted.append(record.ref),
    )

    assert purged is True
    assert deleted == ["ref-docx"]
    assert store.get_job("job_purge", "principal_a") is None
    assert store.list_artifacts("job_purge") == []
    with closing(sqlite3.connect(store.path)) as connection:
        remaining_attempts = connection.execute(
            "SELECT count(*) FROM job_attempts WHERE job_id = ?", ("job_purge",)
        ).fetchone()[0]
    assert remaining_attempts == 0


def test_purge_job_tolerates_missing_bytes(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_purge_missing", requested_output="diagnostic_docx")
    lease = store.claim_next(worker_id="worker_a", now=101.0)
    assert lease is not None
    store.finalize_success(
        lease,
        artifacts=[
            job_store.ArtifactMetadata(
                role="diagnostic_docx", ref="ref-missing", sha256="b" * 64,
                size_bytes=1, mime_type="application/vnd.docx",
            )
        ],
        diagnostic_outcome={"a": 1},
        retention_seconds=60,
        now=102.0,
    )

    def missing(record: job_store.ArtifactRecord) -> None:
        raise FileNotFoundError(record.ref)

    assert store.purge_job("job_purge_missing", "principal_a", delete_bytes=missing) is True
    assert store.get_job("job_purge_missing", "principal_a") is None


def test_label_removed_with_job_row(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    submit(store, job_id="job_label_purge")
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute(
            "UPDATE conversion_jobs SET label = 'to purge' WHERE job_id = 'job_label_purge'"
        )
        connection.commit()
    store.request_cancel("job_label_purge", "principal_a", now=101.0)

    store.purge_job("job_label_purge", "principal_a", delete_bytes=lambda record: None)

    with closing(sqlite3.connect(store.path)) as connection:
        remaining = connection.execute(
            "SELECT count(*) FROM conversion_jobs WHERE job_id = 'job_label_purge'"
        ).fetchone()[0]
    assert remaining == 0


def test_get_set_delete_setting(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    assert store.get_setting("default_variant") is None

    store.set_setting("default_variant", "fp8", now=100.0)
    assert store.get_setting("default_variant") == "fp8"

    store.set_setting("default_variant", "bf16", now=101.0)
    assert store.get_setting("default_variant") == "bf16"

    store.delete_setting("default_variant")
    assert store.get_setting("default_variant") is None


def test_publish_readiness_sandbox_rootful_and_engines(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    writer = store.claim_readiness_writer(worker_id="worker_a", now=100.0)
    assert store.readiness_extras() == (None, None)

    store.publish_readiness(
        writer, worker_ready=True, sandbox_ready=True, diagnostic_docx_ready=False,
        now=101.0, sandbox_rootful=False, available_engines=["native"],
    )

    assert store.readiness_extras() == (False, ("native",))
