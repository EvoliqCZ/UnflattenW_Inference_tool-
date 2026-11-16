from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from unflatten.service.auth import (
    CAPABILITIES,
    AuthenticationFailed,
    CapabilityDenied,
    PrincipalFileInvalid,
    PrincipalStore,
    load_principal_store,
    require_capability,
)

RAW_TOKEN = "test-token-with-at-least-thirty-two-bytes"


def _write_principals(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o400)


def _valid_payload() -> dict[str, object]:
    return {
        "principal_file_schema_version": "1.0.0",
        "principals": [
            {
                "principal_id": "principal_alpha",
                "token": RAW_TOKEN,
                "capabilities": {
                    "cancel_own_job": False,
                    "download_prediction_ir": True,
                    "read_status": True,
                    "submit_diagnostic": True,
                    "download_docx": True,
                },
            }
        ],
    }


def test_load_principal_store_valid_file_retains_digest_not_token(tmp_path: Path) -> None:
    path = tmp_path / "principals.json"
    _write_principals(path, _valid_payload())
    store = load_principal_store(path)
    principal = store.authenticate(f"Bearer {RAW_TOKEN}")
    assert principal.principal_id == "principal_alpha"
    assert principal.capabilities == frozenset(
        {"submit_diagnostic", "read_status", "download_prediction_ir", "download_docx"}
    )
    assert RAW_TOKEN not in repr(store)
    assert RAW_TOKEN not in repr(vars(store))


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Basic abc", "bearer abc", "Bearer", "Bearer ", "Bearer a b", "Bearer a,b"],
)
def test_authenticate_malformed_or_invalid_credential_rejected(
    tmp_path: Path, authorization: str | None
) -> None:
    path = tmp_path / "principals.json"
    _write_principals(path, _valid_payload())
    with pytest.raises(AuthenticationFailed):
        load_principal_store(path).authenticate(authorization)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update({"unknown": True}),
        lambda p: p.pop("principal_file_schema_version"),
        lambda p: p["principals"][0].update({"unknown": True}),
        lambda p: p["principals"][0]["capabilities"].update({"admin": True}),
        lambda p: p["principals"][0]["capabilities"].pop("read_status"),
        lambda p: p["principals"][0]["capabilities"].pop("download_docx"),
        lambda p: p["principals"][0].update({"token": "short"}),
    ],
)
def test_load_principal_store_invalid_or_open_file_rejected(tmp_path: Path, mutate) -> None:
    payload = _valid_payload()
    mutate(payload)
    path = tmp_path / "principals.json"
    _write_principals(path, payload)
    with pytest.raises(PrincipalFileInvalid):
        load_principal_store(path)


def test_load_principal_store_duplicate_token_or_id_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    first = payload["principals"][0]
    payload["principals"].append(dict(first))
    path = tmp_path / "principals.json"
    _write_principals(path, payload)
    with pytest.raises(PrincipalFileInvalid):
        load_principal_store(path)


def test_load_principal_store_group_readable_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "principals.json"
    _write_principals(path, _valid_payload())
    path.chmod(0o640)
    with pytest.raises(PrincipalFileInvalid, match="mode 0400"):
        load_principal_store(path)


def test_load_principal_store_rejects_owner_writable_file(tmp_path: Path) -> None:
    path = tmp_path / "principals.json"
    _write_principals(path, _valid_payload())
    path.chmod(0o600)
    with pytest.raises(PrincipalFileInvalid, match="mode 0400"):
        load_principal_store(path)


def test_load_principal_store_rejects_file_not_owned_by_service_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "principals.json"
    _write_principals(path, _valid_payload())
    real_lstat = Path.lstat

    class WrongOwnerStat:
        def __init__(self, wrapped) -> None:
            self.st_mode = wrapped.st_mode
            self.st_uid = os.geteuid() + 1

    monkeypatch.setattr(Path, "lstat", lambda self: WrongOwnerStat(real_lstat(self)))
    with pytest.raises(PrincipalFileInvalid, match="service user"):
        load_principal_store(path)


def test_principal_by_id_known_and_unknown(tmp_path: Path) -> None:
    path = tmp_path / "principals.json"
    _write_principals(path, _valid_payload())
    store = load_principal_store(path)

    found = store.principal_by_id("principal_alpha")

    assert found is not None
    assert found.principal_id == "principal_alpha"
    assert store.principal_by_id("no_such_principal") is None


def test_sole_principal_one_principal_returns_it(tmp_path: Path) -> None:
    path = tmp_path / "principals.json"
    _write_principals(path, _valid_payload())

    sole = load_principal_store(path).sole_principal()

    assert sole is not None
    assert sole.principal_id == "principal_alpha"


def test_sole_principal_zero_or_two_principals_returns_none(tmp_path: Path) -> None:
    payload = _valid_payload()
    second = {
        **payload["principals"][0],  # type: ignore[index]
        "principal_id": "principal_beta",
        "token": "another-token-with-at-least-thirty-two-bytes",
    }
    payload["principals"] = [*payload["principals"], second]  # type: ignore[misc]
    path = tmp_path / "principals.json"
    _write_principals(path, payload)

    assert load_principal_store(path).sole_principal() is None
    assert PrincipalStore([]).sole_principal() is None


def test_require_capability_exact_allowlist_enforced(tmp_path: Path) -> None:
    path = tmp_path / "principals.json"
    _write_principals(path, _valid_payload())
    principal = load_principal_store(path).authenticate(f"Bearer {RAW_TOKEN}")
    require_capability(principal, "read_status")
    with pytest.raises(CapabilityDenied):
        require_capability(principal, "cancel_own_job")
    with pytest.raises(ValueError):
        require_capability(principal, "admin")
    assert CAPABILITIES == frozenset(
        {
            "submit_diagnostic",
            "read_status",
            "download_prediction_ir",
            "cancel_own_job",
            "download_docx",
        }
    )
