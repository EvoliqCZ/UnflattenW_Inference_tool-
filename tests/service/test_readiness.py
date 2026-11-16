from __future__ import annotations

import pytest

from unflatten.service.job_store import FenceRejected, JobStore, JobStoreInvalid
from unflatten.service.readiness import (
    COMPONENT_CODES,
    DOCX_BLOCKERS,
    ReadinessInvalid,
    current_docx_requirements,
    diagnostic_docx_status,
    evaluate_readiness,
    job_store_component_statuses,
)


def _components(ready: bool = True) -> dict[str, bool]:
    return dict.fromkeys(COMPONENT_CODES, ready)


def _requirements(ready: bool = True) -> dict[str, bool]:
    return dict.fromkeys(DOCX_BLOCKERS, ready)


@pytest.mark.parametrize("blocker", DOCX_BLOCKERS)
def test_evaluate_readiness_each_docx_blocker_independently_disables_docx(blocker: str) -> None:
    requirements = _requirements()
    requirements[blocker] = False
    report = evaluate_readiness(
        component_statuses=_components(),
        docx_requirements=requirements,
        diagnostic_docx_ready=True,
    )
    assert report["service_ready"] is True
    assert report["capabilities"]["prediction_ir"] is True
    assert report["capabilities"]["docx"] is False
    assert report["docx_blockers"] == [blocker]


def test_evaluate_readiness_all_docx_requirements_met_enables_only_gate_model() -> None:
    report = evaluate_readiness(
        component_statuses=_components(),
        docx_requirements=_requirements(),
        diagnostic_docx_ready=True,
    )
    assert report["capabilities"]["docx"] is True
    assert report["docx_blockers"] == []


def test_current_docx_requirements_all_fail_closed() -> None:
    assert current_docx_requirements() == dict.fromkeys(DOCX_BLOCKERS, False)


def test_evaluate_readiness_component_failure_is_content_free() -> None:
    components = _components()
    components["pdf_sandbox"] = False
    report = evaluate_readiness(
        component_statuses=components,
        docx_requirements=_requirements(False),
        diagnostic_docx_ready=True,
    )
    assert report["service_ready"] is False
    assert report["components"] == [
        {"code": code, "ready": components[code]} for code in COMPONENT_CODES
    ]
    forbidden = {"path", "hash", "job_count", "model_name", "principal_id", "detail"}
    assert forbidden.isdisjoint(str(report).lower())


@pytest.mark.parametrize("target", ["components", "requirements"])
def test_evaluate_readiness_missing_or_unknown_codes_rejected(target: str) -> None:
    components = _components()
    requirements = _requirements()
    selected = components if target == "components" else requirements
    selected.pop(next(iter(selected)))
    selected["UNKNOWN"] = True
    with pytest.raises(ReadinessInvalid):
        evaluate_readiness(
            component_statuses=components,
            docx_requirements=requirements,
            diagnostic_docx_ready=True,
        )


def test_evaluate_readiness_diagnostic_docx_ready_must_be_boolean() -> None:
    with pytest.raises(ReadinessInvalid):
        evaluate_readiness(
            component_statuses=_components(),
            docx_requirements=_requirements(),
            diagnostic_docx_ready=1,
        )


def test_evaluate_readiness_diagnostic_docx_capability_requires_service_ready() -> None:
    components = _components()
    components["worker"] = False
    report = evaluate_readiness(
        component_statuses=components,
        docx_requirements=_requirements(),
        diagnostic_docx_ready=True,
    )
    assert report["service_ready"] is False
    assert report["capabilities"]["diagnostic_docx"] is False


def test_evaluate_readiness_diagnostic_docx_capability_requires_the_published_flag() -> None:
    report = evaluate_readiness(
        component_statuses=_components(),
        docx_requirements=_requirements(),
        diagnostic_docx_ready=False,
    )
    assert report["service_ready"] is True
    assert report["capabilities"]["diagnostic_docx"] is False


def test_evaluate_readiness_diagnostic_docx_capability_is_never_a_literal() -> None:
    ready = evaluate_readiness(
        component_statuses=_components(),
        docx_requirements=_requirements(),
        diagnostic_docx_ready=True,
    )
    not_ready = evaluate_readiness(
        component_statuses=_components(),
        docx_requirements=_requirements(),
        diagnostic_docx_ready=False,
    )
    assert ready["capabilities"]["diagnostic_docx"] is True
    assert not_ready["capabilities"]["diagnostic_docx"] is False


def test_diagnostic_docx_status_missing_row_is_not_ready(tmp_path) -> None:
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    assert diagnostic_docx_status(store) is False


def test_diagnostic_docx_status_reads_last_published_flag_regardless_of_staleness(
    tmp_path,
) -> None:
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    writer = store.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    store.publish_readiness(
        writer, worker_ready=True, sandbox_ready=True, diagnostic_docx_ready=True, now=2.0
    )
    # A stale published row still reports its last known flag: P3 reads it
    # regardless of staleness, unlike worker/pdf_sandbox component status.
    assert diagnostic_docx_status(store) is True

    store.publish_readiness(
        writer, worker_ready=True, sandbox_ready=True, diagnostic_docx_ready=False, now=3.0
    )
    assert diagnostic_docx_status(store) is False


def test_model_state_is_none_until_a_worker_publishes(tmp_path) -> None:
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()

    assert store.model_state() is None

    writer = store.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    assert store.model_state() is None

    store.publish_model_state(
        writer,
        resident_variant="fp8",
        free_vram_mib=6000.0,
        total_vram_mib=16000.0,
        compute_capability=(8, 9),
        now=2.0,
    )

    assert store.model_state() == ("fp8", 6000.0, 16000.0, (8, 9))


def test_publish_model_state_rejects_stale_writer_epoch(tmp_path) -> None:
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    stale = store.claim_readiness_writer(worker_id="worker_old", now=1.0)
    store.claim_readiness_writer(worker_id="worker_new", now=2.0)

    with pytest.raises(FenceRejected):
        store.publish_model_state(
            stale,
            resident_variant="fp8",
            free_vram_mib=6000.0,
            total_vram_mib=16000.0,
            compute_capability=(8, 9),
            now=3.0,
        )


def test_publish_readiness_rejects_non_boolean_diagnostic_docx_ready(tmp_path) -> None:
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    writer = store.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    with pytest.raises(JobStoreInvalid):
        store.publish_readiness(
            writer, worker_ready=True, sandbox_ready=True, diagnostic_docx_ready=1, now=2.0
        )


def test_job_store_readiness_bridge_is_cross_process_fenced_and_expires(tmp_path) -> None:
    writer = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    writer.initialize()
    reader = JobStore(
        writer.path,
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    reader.initialize()

    stale = writer.claim_readiness_writer(worker_id="worker_old", now=100.0)
    writer.publish_readiness(
        stale, worker_ready=True, sandbox_ready=True, diagnostic_docx_ready=True, now=101.0
    )
    assert job_store_component_statuses(reader, now=110.0, stale_after_seconds=30.0) == {
        "worker": True,
        "pdf_sandbox": True,
    }

    current = writer.claim_readiness_writer(worker_id="worker_new", now=111.0)
    with pytest.raises(FenceRejected):
        writer.publish_readiness(
            stale, worker_ready=True, sandbox_ready=True, diagnostic_docx_ready=True, now=112.0
        )
    writer.publish_readiness(
        current, worker_ready=True, sandbox_ready=False, diagnostic_docx_ready=True, now=113.0
    )
    assert job_store_component_statuses(reader, now=114.0, stale_after_seconds=30.0) == {
        "worker": True,
        "pdf_sandbox": False,
    }
    assert job_store_component_statuses(reader, now=143.0, stale_after_seconds=30.0) == {
        "worker": False,
        "pdf_sandbox": False,
    }
    # Unlike worker/pdf_sandbox, diagnostic_docx_ready is read regardless of staleness (P3).
    assert diagnostic_docx_status(reader) is True
