from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

CAPABILITIES = frozenset(
    {
        "submit_diagnostic",
        "read_status",
        "download_prediction_ir",
        "cancel_own_job",
        "download_docx",
    }
)


class PrincipalFileInvalid(ValueError):
    """A principal file cannot be read as the closed security contract."""


class AuthenticationFailed(ValueError):
    """A bearer credential is absent, malformed, or unknown."""


class CapabilityDenied(PermissionError):
    """An authenticated principal lacks a required capability."""


@dataclass(frozen=True)
class Principal:
    principal_id: str
    capabilities: frozenset[str]


class PrincipalStore:
    """Immutable digest-to-principal lookup; raw bearer tokens are never retained."""

    def __init__(self, records: Iterable[tuple[bytes, Principal]]) -> None:
        self._principals_by_digest = MappingProxyType(dict(records))

    def authenticate(self, authorization: str | None) -> Principal:
        """Authenticate exactly one canonical ``Bearer <token>`` credential."""
        if authorization is None or not authorization.startswith("Bearer "):
            raise AuthenticationFailed("bearer credential is missing or malformed")
        token = authorization.removeprefix("Bearer ")
        if not _valid_token(token):
            raise AuthenticationFailed("bearer credential is missing or malformed")
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        matched: Principal | None = None
        for known_digest, principal in self._principals_by_digest.items():
            if hmac.compare_digest(digest, known_digest):
                matched = principal
        if matched is None:
            raise AuthenticationFailed("bearer credential is missing or malformed")
        return matched

    def has_principal(self, principal_id: str) -> bool:
        """Whether `principal_id` is a known principal (T16 password reset)."""
        return any(
            principal.principal_id == principal_id
            for principal in self._principals_by_digest.values()
        )

    def principal_by_id(self, principal_id: str) -> Principal | None:
        """Look up a principal by its stable id (T21: resolving a web
        session's `principal_id` back to its capabilities)."""
        for principal in self._principals_by_digest.values():
            if principal.principal_id == principal_id:
                return principal
        return None

    def sole_principal(self) -> Principal | None:
        """The one principal when exactly one exists, else `None` (owner
        2026-09-29: tokenless first-run setup binds the web password to it
        and is not offered when the choice would be ambiguous)."""
        principals = list(self._principals_by_digest.values())
        return principals[0] if len(principals) == 1 else None


def load_principal_store(path: Path | str) -> PrincipalStore:
    """Load and validate a required private static-principal JSON file."""
    principal_path = Path(path)
    _validate_file_security(principal_path)
    try:
        payload = json.loads(principal_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PrincipalFileInvalid(f"cannot read principal file {principal_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PrincipalFileInvalid(
            f"principal file {principal_path} is not valid JSON: {exc}"
        ) from exc
    rows = _validate_payload(payload)
    records: list[tuple[bytes, Principal]] = []
    seen_ids: set[str] = set()
    seen_digests: set[bytes] = set()
    for row in rows:
        principal_id = row["principal_id"]
        token = row["token"]
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        if principal_id in seen_ids or digest in seen_digests:
            raise PrincipalFileInvalid("principal IDs and token digests must be unique")
        seen_ids.add(principal_id)
        seen_digests.add(digest)
        enabled = frozenset(
            name for name, allowed in row["capabilities"].items() if allowed
        )
        records.append((digest, Principal(principal_id, enabled)))
    return PrincipalStore(records)


def require_capability(principal: Principal, capability: str) -> None:
    """Require one known capability without treating unknown names as permissions."""
    if capability not in CAPABILITIES:
        raise ValueError(f"unknown service capability {capability!r}")
    if capability not in principal.capabilities:
        raise CapabilityDenied("authenticated principal lacks the required capability")


def _validate_file_security(path: Path) -> None:
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise PrincipalFileInvalid(f"cannot inspect principal file {path}: {exc}") from exc
    if not stat.S_ISREG(file_stat.st_mode) or path.is_symlink():
        raise PrincipalFileInvalid("principal file must be a regular non-symlink file")
    if stat.S_IMODE(file_stat.st_mode) != 0o400:
        raise PrincipalFileInvalid("principal file must have mode 0400")
    if file_stat.st_uid != os.geteuid():
        raise PrincipalFileInvalid("principal file must be owned by the service user")


def _validate_payload(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != {
        "principal_file_schema_version",
        "principals",
    }:
        raise PrincipalFileInvalid(
            "principal file root must contain exactly version and principals"
        )
    if payload["principal_file_schema_version"] != "1.0.0":
        raise PrincipalFileInvalid("unsupported principal file schema version")
    rows = payload["principals"]
    if not isinstance(rows, list) or not rows:
        raise PrincipalFileInvalid("principals must be a non-empty array")
    for row in rows:
        _validate_principal(row)
    return rows


def _validate_principal(row: object) -> None:
    if not isinstance(row, dict) or set(row) != {"principal_id", "token", "capabilities"}:
        raise PrincipalFileInvalid("each principal must contain exactly id, token, capabilities")
    principal_id = row["principal_id"]
    if (
        not isinstance(principal_id, str)
        or re.fullmatch(r"[A-Za-z0-9_-]{3,128}", principal_id) is None
    ):
        raise PrincipalFileInvalid("principal_id must be a stable opaque identifier")
    if not _valid_token(row["token"]):
        raise PrincipalFileInvalid("principal token must be 32-4096 non-whitespace bytes")
    capabilities = row["capabilities"]
    if not isinstance(capabilities, dict) or set(capabilities) != CAPABILITIES:
        raise PrincipalFileInvalid("capabilities must explicitly name the exact capability set")
    if any(type(value) is not bool for value in capabilities.values()):
        raise PrincipalFileInvalid("each capability value must be boolean")


def _valid_token(token: object) -> bool:
    if not isinstance(token, str) or not 32 <= len(token.encode("utf-8")) <= 4096:
        return False
    return not any(character.isspace() or character == "," for character in token)
