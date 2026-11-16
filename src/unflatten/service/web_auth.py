"""``web-auth.json`` file, scrypt password hashing, and the login rate limiter.

See ``docs/plans/spec-diagnostic-web-app-design.md`` §2.1 and §2.6 for the contract.
This module owns no database and no route wiring — those are later tasks.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import math
import os
import secrets
import stat
import tempfile
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = "1.0"

#: Idle and absolute session lifetime defaults (design §2.2).
SESSION_IDLE_SECONDS_DEFAULT = 43200  # 12 h
SESSION_MAX_SECONDS_DEFAULT = 604800  # 7 days
SESSION_TOUCH_THROTTLE_SECONDS = 60

# hashlib.scrypt's default maxmem is exactly 32 MiB (128 * r * n with the
# stdlib default r=8, n=1024), and 128 * SCRYPT_R * SCRYPT_N below is exactly
# 32 MiB too -- at that boundary hashlib.scrypt raises
# "ValueError: memory limit exceeded" unless maxmem is raised explicitly.
SCRYPT_N = 32768
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SCRYPT_MAXMEM = 64 * 1024 * 1024

_PASSWORD_MIN_LENGTH = 12
_PASSWORD_MAX_LENGTH = 256

_SCRYPT_KEYS = frozenset({"salt", "n", "r", "p", "dklen", "digest"})
_FILE_KEYS = frozenset({"schema_version", "principal_id", "scrypt", "created_at", "updated_at"})


class WebAuthInvalid(ValueError):
    """The ``web-auth.json`` file is missing, malformed, or insecurely stored."""


class RateLimited(Exception):
    """The login rate limiter rejected a request; retry after the given delay."""

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("login rate limit exceeded")
        self.retry_after_seconds = retry_after_seconds


def _normalize_password(password: str) -> bytes:
    normalized = unicodedata.normalize("NFC", password)
    length = len(normalized)
    if not (_PASSWORD_MIN_LENGTH <= length <= _PASSWORD_MAX_LENGTH):
        raise WebAuthInvalid(
            f"password must be {_PASSWORD_MIN_LENGTH} to {_PASSWORD_MAX_LENGTH} code points"
        )
    return normalized.encode("utf-8")


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text.encode("ascii"))


def hash_password(password: str, *, salt: bytes | None = None) -> dict[str, Any]:
    """Hash a password with scrypt, returning the closed ``scrypt`` entry shape."""
    salt_bytes = salt if salt is not None else secrets.token_bytes(16)
    pw_bytes = _normalize_password(password)
    digest = hashlib.scrypt(
        pw_bytes,
        salt=salt_bytes,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_DKLEN,
        maxmem=SCRYPT_MAXMEM,
    )
    return {
        "salt": _b64url_encode(salt_bytes),
        "n": SCRYPT_N,
        "r": SCRYPT_R,
        "p": SCRYPT_P,
        "dklen": SCRYPT_DKLEN,
        "digest": _b64url_encode(digest),
    }


def verify_password(password: str, scrypt_entry: dict[str, Any]) -> bool:
    """Verify a password against a stored ``scrypt`` entry in constant time."""
    _validate_scrypt_shape(scrypt_entry)
    pw_bytes = _normalize_password(password)
    digest = hashlib.scrypt(
        pw_bytes,
        salt=_b64url_decode(scrypt_entry["salt"]),
        n=scrypt_entry["n"],
        r=scrypt_entry["r"],
        p=scrypt_entry["p"],
        dklen=scrypt_entry["dklen"],
        maxmem=SCRYPT_MAXMEM,
    )
    stored = _b64url_decode(scrypt_entry["digest"])
    return hmac.compare_digest(digest, stored)


def _validate_scrypt_shape(entry: object) -> None:
    if not isinstance(entry, dict) or set(entry) != _SCRYPT_KEYS:
        raise WebAuthInvalid("scrypt entry must contain exactly the closed key set")


def _validate_file_security(path: Path) -> None:
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise WebAuthInvalid(f"cannot inspect web-auth file {path}: {exc}") from exc
    if not stat.S_ISREG(file_stat.st_mode) or path.is_symlink():
        raise WebAuthInvalid("web-auth file must be a regular non-symlink file")
    if stat.S_IMODE(file_stat.st_mode) != 0o600:
        raise WebAuthInvalid("web-auth file must have mode 0600")
    if file_stat.st_uid != os.geteuid():
        raise WebAuthInvalid("web-auth file must be owned by the service user")


def _validate_payload_shape(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _FILE_KEYS:
        raise WebAuthInvalid("web-auth file root must contain exactly the closed key set")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise WebAuthInvalid("unsupported web-auth schema version")
    if not isinstance(payload["principal_id"], str) or not payload["principal_id"]:
        raise WebAuthInvalid("principal_id must be a non-empty string")
    _validate_scrypt_shape(payload["scrypt"])
    if not isinstance(payload["created_at"], int | float) or isinstance(
        payload["created_at"], bool
    ):
        raise WebAuthInvalid("created_at must be numeric")
    if not isinstance(payload["updated_at"], int | float) or isinstance(
        payload["updated_at"], bool
    ):
        raise WebAuthInvalid("updated_at must be numeric")
    return payload


def _atomic_write_0600(path: Path, payload: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".web-auth-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


class WebAuthFile:
    """Reads and atomically writes ``$UNFLATTEN_DATA_DIR/web-auth.json``."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def exists(self) -> bool:
        """Whether the file exists (``setup_required = not exists()``)."""
        return self._path.exists()

    def read(self) -> dict[str, Any]:
        """Read and validate the file; re-read on every call, never cached."""
        _validate_file_security(self._path)
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise WebAuthInvalid(f"cannot read web-auth file {self._path}: {exc}") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise WebAuthInvalid(f"web-auth file {self._path} is not valid JSON: {exc}") from exc
        return _validate_payload_shape(payload)

    def verify(self, password: str) -> bool:
        """Read the current file and verify a password against it."""
        payload = self.read()
        return verify_password(password, payload["scrypt"])

    def write(self, principal_id: str, password: str) -> dict[str, Any]:
        """Hash ``password`` and atomically (re)write the file at mode 0600."""
        scrypt_entry = hash_password(password)
        now = time.time()
        created_at = now
        if self._path.exists():
            with contextlib.suppress(WebAuthInvalid):
                created_at = self.read()["created_at"]
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "principal_id": principal_id,
            "scrypt": scrypt_entry,
            "created_at": created_at,
            "updated_at": now,
        }
        _atomic_write_0600(self._path, payload)
        return payload


class LoginRateLimiter:
    """In-process login failure limiter: N failures per rolling window.

    One global key (design §2.6): every client is 127.0.0.1, so no per-caller
    partitioning is meaningful. Counts only failures; a success clears it.
    """

    def __init__(
        self,
        max_failures: int = 5,
        window_seconds: float = 60.0,
        *,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_failures = max_failures
        self._window_seconds = window_seconds
        self._time_source = time_source
        self._failure_times: list[float] = []

    def check(self) -> None:
        """Raise ``RateLimited`` if the window is exhausted; otherwise no-op.

        Must be called before reading the request body (design §2.6).
        """
        now = self._time_source()
        self._prune(now)
        if len(self._failure_times) >= self._max_failures:
            oldest = self._failure_times[0]
            retry_after = math.ceil(max(0.0, self._window_seconds - (now - oldest)))
            raise RateLimited(retry_after)

    def record_failure(self) -> None:
        self._failure_times.append(self._time_source())

    def record_success(self) -> None:
        self._failure_times.clear()

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        self._failure_times = [t for t in self._failure_times if t > cutoff]


class WebSessionRecordLike(Protocol):
    """Structural type for `job_store_types.WebSessionRecord` (T14) -- kept
    local so this module does not depend on `job_store`."""

    session_digest: str
    principal_id: str
    created_at: float
    last_seen_at: float
    expires_at: float
    revoked_at: float | None


class WebSessionStorage(Protocol):
    """The pure-CRUD surface `WebSessionStore` is built on top of. Satisfied
    by `job_store.JobStore` via T14's `WebSessionMixin`."""

    def create_web_session(
        self, *, session_digest: str, principal_id: str, now: float, expires_at: float,
    ) -> WebSessionRecordLike: ...

    def get_web_session(self, session_digest: str) -> WebSessionRecordLike | None: ...

    def touch_web_session(self, session_digest: str, *, now: float) -> bool: ...

    def revoke_web_session(self, session_digest: str, *, now: float) -> bool: ...

    def revoke_all_web_sessions_except(
        self, principal_id: str, *, keep_session_digest: str, now: float,
    ) -> int: ...

    def cleanup_web_sessions(self, *, now: float) -> int: ...


def _digest_token(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


class WebSessionStore:
    """Idle/absolute expiry, touch throttling and digest-only storage over a
    `WebSessionStorage` (design §2.2). Only a sha256 digest of the session
    token is ever persisted; the raw token lives only in the `Set-Cookie`
    response and the caller's request -- this class is where that boundary
    is enforced, not the storage mixin.
    """

    def __init__(
        self,
        storage: WebSessionStorage,
        *,
        idle_seconds: int = SESSION_IDLE_SECONDS_DEFAULT,
        max_seconds: int = SESSION_MAX_SECONDS_DEFAULT,
        touch_throttle_seconds: int = SESSION_TOUCH_THROTTLE_SECONDS,
        time_source: Callable[[], float] = time.time,
    ) -> None:
        self._storage = storage
        self._idle_seconds = idle_seconds
        self._max_seconds = max_seconds
        self._touch_throttle_seconds = touch_throttle_seconds
        self._time_source = time_source

    def create(self, principal_id: str) -> str:
        """Create a session for `principal_id`, returning the raw token.
        Only its sha256 digest is passed to storage."""
        now = self._time_source()
        self._storage.cleanup_web_sessions(now=now)
        token = secrets.token_urlsafe(32)
        self._storage.create_web_session(
            session_digest=_digest_token(token),
            principal_id=principal_id,
            now=now,
            expires_at=now + self._max_seconds,
        )
        return token

    def authenticate(self, token: str) -> WebSessionRecordLike | None:
        """Look up `token`, applying idle and absolute expiry. Returns
        `None` for an unknown, revoked, idle-expired or absolute-expired
        session alike -- the caller cannot distinguish which (design §2.4)."""
        now = self._time_source()
        record = self._storage.get_web_session(_digest_token(token))
        if record is None or record.revoked_at is not None:
            return None
        if record.expires_at <= now or record.last_seen_at + self._idle_seconds <= now:
            return None
        return record

    def touch(self, record: WebSessionRecordLike) -> None:
        """Bump `last_seen_at`, at most once per `touch_throttle_seconds`
        (design §2.2: not on every poll)."""
        now = self._time_source()
        if now - record.last_seen_at >= self._touch_throttle_seconds:
            self._storage.touch_web_session(record.session_digest, now=now)

    def revoke(self, token: str) -> None:
        """Revoke a single session by its raw token (`DELETE /v1/session`)."""
        now = self._time_source()
        self._storage.revoke_web_session(_digest_token(token), now=now)
        self._storage.cleanup_web_sessions(now=now)

    def revoke_all_except(self, principal_id: str, *, keep_token: str) -> int:
        """Revoke every other session of `principal_id` (password change
        while keeping the current session, design §2.3)."""
        now = self._time_source()
        return self._storage.revoke_all_web_sessions_except(
            principal_id, keep_session_digest=_digest_token(keep_token), now=now,
        )

    def revoke_all(self, principal_id: str) -> int:
        """Revoke every session of `principal_id` (password reset, §2.7).
        `sha256("")` never equals a real token's digest -- `token_urlsafe`
        cannot produce an empty string -- so no session is ever kept."""
        return self.revoke_all_except(principal_id, keep_token="")
