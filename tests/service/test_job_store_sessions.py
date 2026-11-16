"""DEC-052 §4: `WebSessionMixin` CRUD over the v7 `web_sessions` table."""

from __future__ import annotations

from pathlib import Path

import pytest

from unflatten.service.job_store import JobStore, JobStoreInvalid


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


def test_create_and_get_web_session_roundtrip(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    record = store.create_web_session(
        session_digest="a" * 64, principal_id="principal_a", now=100.0, expires_at=200.0,
    )
    assert record.session_digest == "a" * 64
    assert record.principal_id == "principal_a"
    assert record.created_at == 100.0
    assert record.last_seen_at == 100.0
    assert record.expires_at == 200.0
    assert record.revoked_at is None

    fetched = store.get_web_session("a" * 64)
    assert fetched == record


def test_get_web_session_missing_returns_none(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    assert store.get_web_session("missing" * 8) is None


def test_create_web_session_rejects_empty_fields(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    with pytest.raises(JobStoreInvalid):
        store.create_web_session(
            session_digest="", principal_id="principal_a", now=100.0, expires_at=200.0,
        )
    with pytest.raises(JobStoreInvalid):
        store.create_web_session(
            session_digest="a" * 64, principal_id="", now=100.0, expires_at=200.0,
        )
    with pytest.raises(JobStoreInvalid):
        store.create_web_session(
            session_digest="a" * 64, principal_id="principal_a", now=100.0, expires_at=100.0,
        )


def test_touch_web_session_updates_last_seen_at(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_web_session(
        session_digest="a" * 64, principal_id="principal_a", now=100.0, expires_at=200.0,
    )

    assert store.touch_web_session("a" * 64, now=150.0) is True
    assert store.get_web_session("a" * 64).last_seen_at == 150.0


def test_touch_web_session_missing_returns_false(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    assert store.touch_web_session("missing" * 8, now=150.0) is False


def test_revoke_web_session_sets_revoked_at(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_web_session(
        session_digest="a" * 64, principal_id="principal_a", now=100.0, expires_at=200.0,
    )

    assert store.revoke_web_session("a" * 64, now=110.0) is True
    fetched = store.get_web_session("a" * 64)
    assert fetched.revoked_at == 110.0

    # Revoking again is a no-op, not an error, but reports nothing changed.
    assert store.revoke_web_session("a" * 64, now=120.0) is False
    assert store.get_web_session("a" * 64).revoked_at == 110.0


def test_revoke_web_session_missing_returns_false(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    assert store.revoke_web_session("missing" * 8, now=110.0) is False


def test_revoke_all_web_sessions_except_scopes_by_principal(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_web_session(
        session_digest="a" * 64, principal_id="principal_a", now=100.0, expires_at=200.0,
    )
    store.create_web_session(
        session_digest="b" * 64, principal_id="principal_a", now=100.0, expires_at=200.0,
    )
    store.create_web_session(
        session_digest="c" * 64, principal_id="principal_b", now=100.0, expires_at=200.0,
    )

    revoked = store.revoke_all_web_sessions_except(
        "principal_a", keep_session_digest="a" * 64, now=150.0,
    )
    assert revoked == 1
    assert store.get_web_session("a" * 64).revoked_at is None
    assert store.get_web_session("b" * 64).revoked_at == 150.0
    # A different principal's session is untouched.
    assert store.get_web_session("c" * 64).revoked_at is None


def test_cleanup_web_sessions_deletes_expired_and_revoked(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_web_session(
        session_digest="a" * 64, principal_id="principal_a", now=100.0, expires_at=200.0,
    )
    store.create_web_session(
        session_digest="b" * 64, principal_id="principal_a", now=100.0, expires_at=110.0,
    )
    store.revoke_web_session("a" * 64, now=105.0)
    store.create_web_session(
        session_digest="c" * 64, principal_id="principal_a", now=100.0, expires_at=500.0,
    )

    deleted = store.cleanup_web_sessions(now=300.0)
    assert deleted == 2
    assert store.get_web_session("a" * 64) is None
    assert store.get_web_session("b" * 64) is None
    assert store.get_web_session("c" * 64) is not None


def test_cleanup_web_sessions_keeps_live_sessions(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_web_session(
        session_digest="a" * 64, principal_id="principal_a", now=100.0, expires_at=200.0,
    )

    assert store.cleanup_web_sessions(now=150.0) == 0
    assert store.get_web_session("a" * 64) is not None
