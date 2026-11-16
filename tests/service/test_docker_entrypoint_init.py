"""Runs `deploy/docker/entrypoint-init.sh` for real (no Docker) and checks the
principal file it writes is the shape `unflatten.service.auth.load_principal_store`
actually accepts. This must fail against a script that writes a bare JSON array
(the pre-2026-09-27 shape) instead of the required
`{principal_file_schema_version, principals}` object."""

from __future__ import annotations

import subprocess
from pathlib import Path

from unflatten.service.auth import load_principal_store

SCRIPT = Path(__file__).parents[2] / "deploy" / "docker" / "entrypoint-init.sh"


def _run_init(principal_file: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env={"UNFLATTEN_PRINCIPAL_FILE": str(principal_file), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )


def test_entrypoint_init_writes_a_principal_file_load_principal_store_accepts(
    tmp_path: Path,
) -> None:
    principal_file = tmp_path / "principals.json"

    result = _run_init(principal_file)

    store = load_principal_store(principal_file)
    principal = store.authenticate(f"Bearer {_printed_token(result.stdout)}")
    assert principal.principal_id == "local"
    assert "submit_diagnostic" in principal.capabilities


def test_entrypoint_init_is_idempotent_on_a_second_run(tmp_path: Path) -> None:
    principal_file = tmp_path / "principals.json"
    _run_init(principal_file)
    before = principal_file.read_bytes()

    result = _run_init(principal_file)

    assert principal_file.read_bytes() == before
    assert "already exists" in result.stdout
    load_principal_store(principal_file)


def _printed_token(stdout: str) -> str:
    (line,) = (line for line in stdout.splitlines() if line.startswith("init: bearer token"))
    return line.rsplit(": ", 1)[1]
