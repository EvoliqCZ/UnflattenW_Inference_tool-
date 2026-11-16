"""DEC-052 §4.2: `ModelCommandMixin` protocol -- enqueue, claim, heartbeat,
progress, finish, requeue-after-stale-heartbeat, and retention."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from unflatten.service.job_store import JobStore, JobStoreInvalid
from unflatten.service.job_store_types import FenceRejected, ModelCommandConflict


def make_store(tmp_path: Path) -> JobStore:
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=10,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    return store


def test_enqueue_model_command_creates_queued_row(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    record, created = store.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_a", now=100.0,
    )
    assert created is True
    assert record.command_id.startswith("cmd_")
    assert record.kind == "download"
    assert record.variant_id == "variant_a"
    assert record.state == "queued"
    assert record.phase is None
    assert record.requested_by == "principal_a"


def test_enqueue_rejects_invalid_kind(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    with pytest.raises(JobStoreInvalid):
        store.enqueue_model_command(
            kind="unpack", variant_id="variant_a", requested_by="principal_a", now=100.0,
        )


def test_enqueue_returns_existing_active_same_kind(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first, created_first = store.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_a", now=100.0,
    )

    second, created_second = store.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_b", now=101.0,
    )

    assert created_first is True
    assert created_second is False
    assert second == first


def test_enqueue_other_kind_raises_ModelCommandConflict(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_a", now=100.0,
    )

    with pytest.raises(ModelCommandConflict):
        store.enqueue_model_command(
            kind="load", variant_id="variant_a", requested_by="principal_a", now=101.0,
        )


def test_enqueue_integrity_race_returns_existing(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda requester: store.enqueue_model_command(
                    kind="download", variant_id="variant_a", requested_by=requester, now=100.0,
                ),
                ("principal_a", "principal_b"),
            )
        )

    created_flags = sorted(created for _, created in results)
    assert created_flags == [False, True]
    assert results[0][0].command_id == results[1][0].command_id


def test_claim_next_fifo(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first, _ = store.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_a", now=100.0,
    )
    second, _ = store.enqueue_model_command(
        kind="download", variant_id="variant_b", requested_by="principal_a", now=101.0,
    )

    claimed = store.claim_next_model_command(worker_id="worker_1", now=110.0)
    assert claimed.command_id == first.command_id
    assert claimed.state == "running"
    assert claimed.phase == "downloading"
    assert claimed.worker_id == "worker_1"
    assert claimed.started_at == 110.0
    assert claimed.heartbeat_at == 110.0

    claimed_next = store.claim_next_model_command(worker_id="worker_2", now=111.0)
    assert claimed_next.command_id == second.command_id

    assert store.claim_next_model_command(worker_id="worker_3", now=112.0) is None


def test_claim_next_initial_phase_by_kind(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.enqueue_model_command(
        kind="delete", variant_id="variant_a", requested_by="principal_a", now=100.0,
    )

    claimed = store.claim_next_model_command(worker_id="worker_1", now=110.0)
    assert claimed.phase == "deleting"


def test_heartbeat_and_progress_fenced_by_worker_id(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_a", now=100.0,
    )
    claimed = store.claim_next_model_command(worker_id="worker_1", now=100.0)

    store.heartbeat_model_command(claimed.command_id, worker_id="worker_1", now=105.0)
    store.update_model_command_progress(
        claimed.command_id, worker_id="worker_1",
        phase="downloading", bytes_done=1024, bytes_total=4096, now=106.0,
    )

    active = store.active_model_command("variant_a", "download")
    assert active["state"] == "running"
    assert active["bytes_done"] == 1024
    assert active["bytes_total"] == 4096

    with pytest.raises(FenceRejected):
        store.heartbeat_model_command(claimed.command_id, worker_id="worker_2", now=107.0)
    with pytest.raises(FenceRejected):
        store.update_model_command_progress(
            claimed.command_id, worker_id="worker_2",
            phase="downloading", bytes_done=2048, bytes_total=4096, now=108.0,
        )


def test_finish_fence_lost_reported(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_a", now=100.0,
    )
    claimed = store.claim_next_model_command(worker_id="worker_1", now=100.0)

    # The worker went stale and got requeued to a new worker before finishing.
    store.requeue_stale_model_commands(now=200.0, stale_after=50)
    reclaimed = store.claim_next_model_command(worker_id="worker_2", now=201.0)
    assert reclaimed.command_id == claimed.command_id

    committed = store.finish_model_command(
        claimed.command_id, worker_id="worker_1", state="succeeded", now=210.0,
    )
    assert committed is False

    committed_by_owner = store.finish_model_command(
        claimed.command_id, worker_id="worker_2", state="succeeded", now=211.0,
    )
    assert committed_by_owner is True


def test_finish_model_command_requires_error_code_when_failed(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_a", now=100.0,
    )
    claimed = store.claim_next_model_command(worker_id="worker_1", now=100.0)

    with pytest.raises(JobStoreInvalid):
        store.finish_model_command(
            claimed.command_id, worker_id="worker_1", state="failed", now=105.0,
        )

    assert store.finish_model_command(
        claimed.command_id, worker_id="worker_1", state="failed",
        error_code="MODEL_CHECKPOINT_UNVERIFIED", now=106.0,
    ) is True


def test_requeue_stale_commands_keeps_bytes(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_a", now=100.0,
    )
    claimed = store.claim_next_model_command(worker_id="worker_1", now=100.0)
    store.update_model_command_progress(
        claimed.command_id, worker_id="worker_1",
        phase="downloading", bytes_done=1024, bytes_total=4096, now=101.0,
    )

    requeued = store.requeue_stale_model_commands(now=200.0, stale_after=50)
    assert requeued == 1

    active = store.active_model_command("variant_a", "download")
    assert active["state"] == "queued"
    assert active["bytes_done"] == 1024
    assert active["bytes_total"] == 4096
    assert active["phase"] is None


def test_requeue_stale_commands_ignores_fresh_heartbeat(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_a", now=100.0,
    )
    store.claim_next_model_command(worker_id="worker_1", now=100.0)

    assert store.requeue_stale_model_commands(now=110.0, stale_after=50) == 0


def test_delete_finished_commands_before(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_a", now=100.0,
    )
    old = store.claim_next_model_command(worker_id="worker_1", now=100.0)
    store.finish_model_command(old.command_id, worker_id="worker_1", state="succeeded", now=150.0)

    store.enqueue_model_command(
        kind="download", variant_id="variant_b", requested_by="principal_a", now=200.0,
    )
    recent = store.claim_next_model_command(worker_id="worker_1", now=200.0)
    store.finish_model_command(
        recent.command_id, worker_id="worker_1", state="succeeded", now=250.0,
    )

    deleted = store.delete_finished_commands(before=200.0)
    assert deleted == 1
    assert store.active_model_command("variant_a", "download") is None
    assert store.last_model_command("variant_a", "download") is None
    assert store.last_model_command("variant_b", "download") is not None


def test_active_and_last_command_per_variant(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    assert store.active_model_command("variant_a", "download") is None
    assert store.last_model_command("variant_a", "download") is None

    store.enqueue_model_command(
        kind="download", variant_id="variant_a", requested_by="principal_a", now=100.0,
    )
    active = store.active_model_command("variant_a", "download")
    assert active["state"] == "queued"
    assert active["phase"] is None
    assert active["bytes_done"] is None
    assert active["bytes_total"] is None
    assert active["error_code"] is None
    assert active["command_id"].startswith("cmd_")
    assert store.last_model_command("variant_a", "download") is None

    claimed = store.claim_next_model_command(worker_id="worker_1", now=101.0)
    store.finish_model_command(
        claimed.command_id, worker_id="worker_1", state="failed",
        error_code="MODEL_CHECKPOINT_UNVERIFIED", now=105.0,
    )

    assert store.active_model_command("variant_a", "download") is None
    last = store.last_model_command("variant_a", "download")
    assert last["state"] == "failed"
    assert last["error_code"] == "MODEL_CHECKPOINT_UNVERIFIED"
    assert last["command_id"] == claimed.command_id

    # A different kind on the same variant is a separate slot.
    assert store.active_model_command("variant_a", "load") is None
