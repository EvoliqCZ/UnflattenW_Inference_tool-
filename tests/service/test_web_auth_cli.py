"""`web_auth_cli.py reset-password` (design §2.7)."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from unflatten.service.job_store import JobStore
from unflatten.service.web_auth import WebAuthFile, WebSessionStore
from unflatten.service.web_auth_cli import main, reset_password

RAW_TOKEN = "test-token-with-at-least-thirty-two-bytes"


def write_principal_file(path: Path, *principal_ids: str) -> None:
    payload = {
        "principal_file_schema_version": "1.0.0",
        "principals": [
            {
                "principal_id": principal_id,
                "token": f"{RAW_TOKEN}-{principal_id}",
                "capabilities": {
                    "cancel_own_job": True,
                    "download_prediction_ir": True,
                    "read_status": True,
                    "submit_diagnostic": True,
                    "download_docx": True,
                },
            }
            for principal_id in principal_ids
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o400)


def test_reset_password_reads_stdin_not_argv(tmp_path: Path) -> None:
    WebAuthFile(tmp_path / "web-auth.json").write("local", "first password twelve")
    stdin = io.StringIO("new password twelve\n")

    rc = reset_password(tmp_path, stdin=stdin)

    assert rc == 0
    assert WebAuthFile(tmp_path / "web-auth.json").verify("new password twelve") is True


def test_reset_password_via_main_reads_stdin_not_argv(tmp_path: Path) -> None:
    WebAuthFile(tmp_path / "web-auth.json").write("local", "first password twelve")
    # No argv option carries the new password -- only --data-dir is passed.
    argv = ["reset-password", "--data-dir", str(tmp_path)]

    stdin = io.StringIO("second password twelve\n")
    import unflatten.service.web_auth_cli as cli_module

    original_stdin = cli_module.sys.stdin
    cli_module.sys.stdin = stdin
    try:
        rc = main(argv)
    finally:
        cli_module.sys.stdin = original_stdin

    assert rc == 0
    assert WebAuthFile(tmp_path / "web-auth.json").verify("second password twelve") is True


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


def test_reset_password_revokes_all_sessions(tmp_path: Path) -> None:
    WebAuthFile(tmp_path / "web-auth.json").write("local", "first password twelve")
    store = make_job_store(tmp_path)
    token = WebSessionStore(store).create("local")

    reset_password(tmp_path, stdin=io.StringIO("second password twelve\n"))

    reopened = make_job_store(tmp_path)
    assert WebSessionStore(reopened).authenticate(token) is None


def test_reset_password_keeps_principal_id(tmp_path: Path) -> None:
    WebAuthFile(tmp_path / "web-auth.json").write("custom_principal", "first password twelve")

    reset_password(tmp_path, stdin=io.StringIO("second password twelve\n"))

    payload_principal_id = WebAuthFile(tmp_path / "web-auth.json").read()["principal_id"]
    assert payload_principal_id == "custom_principal"


def test_reset_password_without_existing_file_and_no_principal_id_refused(tmp_path: Path) -> None:
    rc = reset_password(tmp_path, stdin=io.StringIO("first password twelve\n"))

    assert rc == 2
    assert not (tmp_path / "web-auth.json").exists()


def test_reset_password_without_existing_file_and_unknown_principal_id_refused(
    tmp_path: Path,
) -> None:
    write_principal_file(tmp_path / "principal.json", "local-owner")

    rc = reset_password(
        tmp_path, stdin=io.StringIO("first password twelve\n"), principal_id="someone-else",
    )

    assert rc == 2
    assert not (tmp_path / "web-auth.json").exists()


def test_reset_password_without_existing_file_and_known_principal_id_succeeds(
    tmp_path: Path,
) -> None:
    write_principal_file(tmp_path / "principal.json", "local-owner")

    rc = reset_password(
        tmp_path, stdin=io.StringIO("first password twelve\n"), principal_id="local-owner",
    )

    assert rc == 0
    web_auth = WebAuthFile(tmp_path / "web-auth.json")
    assert web_auth.read()["principal_id"] == "local-owner"
    assert web_auth.verify("first password twelve") is True


def test_reset_password_never_logged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    WebAuthFile(tmp_path / "web-auth.json").write("local", "first password twelve")
    caplog.set_level(logging.DEBUG)
    password = "never-log-this-secret1"

    reset_password(tmp_path, stdin=io.StringIO(f"{password}\n"))

    captured = capsys.readouterr()
    assert password not in captured.out
    assert password not in captured.err
    assert all(password not in record.getMessage() for record in caplog.records)
