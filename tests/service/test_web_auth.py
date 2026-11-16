from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import unicodedata
from pathlib import Path

import pytest

from unflatten.service.job_store import JobStore
from unflatten.service.web_auth import (
    SCRYPT_N,
    SCRYPT_P,
    SCRYPT_R,
    LoginRateLimiter,
    RateLimited,
    WebAuthFile,
    WebAuthInvalid,
    WebSessionStore,
    hash_password,
    verify_password,
)

VALID_PASSWORD = "correct horse battery"  # 22 code points


def test_scrypt_requires_explicit_maxmem() -> None:
    """The default hashlib.scrypt maxmem (32 MiB) is exactly 128*r*n at our
    parameters, so calling scrypt without an explicit maxmem must fail --
    proving hash_password's explicit maxmem is load-bearing, not decorative.
    """
    assert 128 * SCRYPT_R * SCRYPT_N == 32 * 1024 * 1024
    with pytest.raises(ValueError, match="memory limit exceeded"):
        hashlib.scrypt(b"x" * 20, salt=b"0" * 16, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    # hash_password must not raise for the same n/r/p.
    hash_password(VALID_PASSWORD)


def test_hash_verify_roundtrip_nfc() -> None:
    entry = hash_password(VALID_PASSWORD)
    assert verify_password(VALID_PASSWORD, entry) is True
    # NFC-equivalent composed/decomposed forms must verify identically.
    decomposed = unicodedata.normalize("NFD", "café separated words twelve")
    composed = unicodedata.normalize("NFC", decomposed)
    entry2 = hash_password(decomposed)
    assert verify_password(composed, entry2) is True


def test_verify_password_wrong_password_rejected() -> None:
    entry = hash_password(VALID_PASSWORD)
    assert verify_password("totally different password", entry) is False


@pytest.mark.parametrize("length", [0, 1, 11])
def test_password_length_below_minimum_rejected(length: int) -> None:
    with pytest.raises(WebAuthInvalid):
        hash_password("a" * length)


@pytest.mark.parametrize("length", [12, 256])
def test_password_length_12_to_256_code_points(length: int) -> None:
    entry = hash_password("a" * length)
    assert verify_password("a" * length, entry) is True


def test_password_length_above_maximum_rejected() -> None:
    with pytest.raises(WebAuthInvalid):
        hash_password("a" * 257)


def _write_valid_file(path: Path, password: str = VALID_PASSWORD) -> None:
    WebAuthFile(path).write("local", password)


def test_file_closed_shape_rejects_extra_or_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "web-auth.json"
    _write_valid_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    payload["extra"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, 0o600)
    with pytest.raises(WebAuthInvalid):
        WebAuthFile(path).read()

    del payload["extra"]
    del payload["principal_id"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, 0o600)
    with pytest.raises(WebAuthInvalid):
        WebAuthFile(path).read()


def test_file_written_0600_atomically(tmp_path: Path) -> None:
    path = tmp_path / "web-auth.json"
    _write_valid_file(path)
    mode = stat.S_IMODE(path.lstat().st_mode)
    assert mode == 0o600
    assert not path.is_symlink()
    # No leftover temp files.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["web-auth.json"]


def test_file_symlink_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real-web-auth.json"
    _write_valid_file(real)
    link = tmp_path / "web-auth.json"
    link.symlink_to(real)
    with pytest.raises(WebAuthInvalid):
        WebAuthFile(link).read()


def test_file_wrong_mode_or_owner_rejected(tmp_path: Path) -> None:
    path = tmp_path / "web-auth.json"
    _write_valid_file(path)
    os.chmod(path, 0o644)
    with pytest.raises(WebAuthInvalid):
        WebAuthFile(path).read()


def test_setup_required_iff_file_absent(tmp_path: Path) -> None:
    path = tmp_path / "web-auth.json"
    web_auth = WebAuthFile(path)
    assert web_auth.exists() is False
    _write_valid_file(path)
    assert web_auth.exists() is True


def test_file_reread_on_every_call(tmp_path: Path) -> None:
    path = tmp_path / "web-auth.json"
    web_auth = WebAuthFile(path)
    web_auth.write("local", "first password twelve")
    assert web_auth.verify("first password twelve") is True

    web_auth.write("local", "second password twelve")
    assert web_auth.verify("first password twelve") is False
    assert web_auth.verify("second password twelve") is True


def test_write_preserves_created_at_across_password_reset(tmp_path: Path) -> None:
    path = tmp_path / "web-auth.json"
    web_auth = WebAuthFile(path)
    first = web_auth.write("local", "first password twelve")
    second = web_auth.write("local", "second password twelve")
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] >= first["updated_at"]


def test_rate_limiter_sixth_failure_429_with_retry_after() -> None:
    now = [0.0]
    limiter = LoginRateLimiter(max_failures=5, window_seconds=60.0, time_source=lambda: now[0])
    for _ in range(5):
        limiter.check()
        limiter.record_failure()
    with pytest.raises(RateLimited) as excinfo:
        limiter.check()
    assert excinfo.value.retry_after_seconds > 0


def test_rate_limiter_resets_on_success() -> None:
    now = [0.0]
    limiter = LoginRateLimiter(max_failures=5, window_seconds=60.0, time_source=lambda: now[0])
    for _ in range(5):
        limiter.check()
        limiter.record_failure()
    limiter.record_success()
    limiter.check()  # must not raise


def test_rate_limiter_window_expiry() -> None:
    now = [0.0]
    limiter = LoginRateLimiter(max_failures=5, window_seconds=60.0, time_source=lambda: now[0])
    for _ in range(5):
        limiter.check()
        limiter.record_failure()
    with pytest.raises(RateLimited):
        limiter.check()
    now[0] = 60.001
    limiter.check()  # window elapsed, must not raise


def make_job_store(tmp_path: Path) -> JobStore:
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=10,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    return store


def test_session_token_256_bit_only_digest_stored(tmp_path: Path) -> None:
    store = make_job_store(tmp_path)
    sessions = WebSessionStore(store)

    token = sessions.create("local")

    # secrets.token_urlsafe(32) encodes 32 bytes (256 bits) base64url.
    assert len(base64.urlsafe_b64decode(token + "==")) == 32
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    record = store.get_web_session(digest)
    assert record is not None
    assert record.principal_id == "local"
    # The raw token itself must never appear in the persisted row.
    assert token != record.session_digest
    assert token not in vars(record).values()


def test_session_idle_expiry(tmp_path: Path) -> None:
    now = [1_000.0]
    store = make_job_store(tmp_path)
    sessions = WebSessionStore(
        store, idle_seconds=60, max_seconds=10_000, time_source=lambda: now[0]
    )

    token = sessions.create("local")
    assert sessions.authenticate(token) is not None

    now[0] += 60.0  # exactly at the idle boundary: last_seen_at + idle <= now
    assert sessions.authenticate(token) is None


def test_session_absolute_expiry(tmp_path: Path) -> None:
    now = [1_000.0]
    store = make_job_store(tmp_path)
    sessions = WebSessionStore(
        store, idle_seconds=10_000, max_seconds=100, time_source=lambda: now[0]
    )

    token = sessions.create("local")
    now[0] += 50.0
    assert sessions.authenticate(token) is not None  # well within idle and absolute

    now[0] = 1_000.0 + 100.0
    assert sessions.authenticate(token) is None  # past absolute expiry


def test_revoked_session_rejected(tmp_path: Path) -> None:
    store = make_job_store(tmp_path)
    sessions = WebSessionStore(store)

    token = sessions.create("local")
    assert sessions.authenticate(token) is not None

    sessions.revoke(token)
    assert sessions.authenticate(token) is None


def test_touch_throttled_60s(tmp_path: Path) -> None:
    now = [1_000.0]
    store = make_job_store(tmp_path)
    sessions = WebSessionStore(store, time_source=lambda: now[0])
    token = sessions.create("local")
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    record = sessions.authenticate(token)

    now[0] += 30.0  # inside the 60 s throttle window
    sessions.touch(record)
    assert store.get_web_session(digest).last_seen_at == 1_000.0

    now[0] += 30.0  # now 60 s since last_seen_at: throttle window elapsed
    sessions.touch(record)
    assert store.get_web_session(digest).last_seen_at == 1_060.0


def test_cleanup_on_create_and_revoke(tmp_path: Path) -> None:
    now = [1_000.0]
    store = make_job_store(tmp_path)
    sessions = WebSessionStore(store, max_seconds=10, time_source=lambda: now[0])

    stale_token = sessions.create("local")
    stale_digest = hashlib.sha256(stale_token.encode("ascii")).hexdigest()
    now[0] += 11.0  # past the stale session's absolute expiry

    # A fresh create() sweeps expired rows first.
    sessions.create("local")
    assert store.get_web_session(stale_digest) is None

    other_token = sessions.create("local")
    other_digest = hashlib.sha256(other_token.encode("ascii")).hexdigest()
    sessions.revoke(other_token)
    # revoke() sweeps revoked rows in the same call, not just marks them.
    assert store.get_web_session(other_digest) is None


def test_revoke_all_revokes_every_session_for_principal(tmp_path: Path) -> None:
    store = make_job_store(tmp_path)
    sessions = WebSessionStore(store)

    token_a = sessions.create("local")
    token_b = sessions.create("local")

    revoked = sessions.revoke_all("local")

    assert revoked == 2
    assert sessions.authenticate(token_a) is None
    assert sessions.authenticate(token_b) is None
