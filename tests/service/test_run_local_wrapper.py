"""T47: `deploy/local/run-local.sh` -- the `docker compose` wrapper (design
§8.5). No Docker daemon is used here -- a stub `docker` on PATH captures its
own argv instead of talking to a real daemon; `python3` is the real
interpreter with `PYTHONPATH` pointed at `src/` so the models-config
validation step exercises real code (`unflatten.service.models_config`,
lightweight, no torch import).

`--dry-run` still runs every validating step for real (docker info, .env
creation, rootful-socket consent, models.json validation) -- only the two
mutating `docker compose ... build`/`up -d` calls are replaced by a printed
command, so these tests never actually build or start a container."""

from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
RUN_LOCAL = REPO_ROOT / "deploy" / "local" / "run-local.sh"
SRC_DIR = REPO_ROOT / "src"

_VALID_MODELS_CONFIG = {
    "schema_version": "1.0",
    "default_variant": "bf16-bf16",
    "variants": [
        {
            "variant_id": "bf16-bf16",
            "label": "bf16 / KV BF16",
            "precision": "bf16",
            "kv_cache": "bf16",
            "bundle": "/models/bf16-bf16",
        }
    ],
}


#: What `entrypoint-init.sh` prints on a first run (`docker compose logs init`).
_INIT_TOKEN = "Tok3n_from-init"  # gitleaks:allow -- synthetic Docker-output fixture.


def _write_docker_stub(stub_dir: Path, calls_log: Path) -> None:
    """Logs each call's argv plus the `UNFLATTEN_SETUP_TOKENLESS` value the
    call actually saw in its environment (what compose interpolates), and
    answers `compose ... logs ... init` with the init container's token line."""
    stub_docker = stub_dir / "docker"
    stub_docker.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s |TOKENLESS=%s\\n" "$*" "${UNFLATTEN_SETUP_TOKENLESS-unset}"'
        f' >> "{calls_log}"\n'
        'case "$*" in\n'
        "  *logs*init*)\n"
        "    echo 'init: generated principal file at /data/principals.json'\n"
        f"    echo 'init: bearer token (shown once, not stored anywhere else): {_INIT_TOKEN}'\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n"
    )
    stub_docker.chmod(stub_docker.stat().st_mode | stat.S_IEXEC)


def _run(
    tmp_path: Path,
    args: list[str],
    *,
    env_file_content: str | None = "UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1\n",
    models_config: dict | str | None = _VALID_MODELS_CONFIG,
) -> tuple[subprocess.CompletedProcess, Path]:
    stub_dir = tmp_path / "stubbin"
    stub_dir.mkdir()
    calls_log = tmp_path / "docker_calls.log"
    _write_docker_stub(stub_dir, calls_log)

    env_file = tmp_path / ".env"
    if env_file_content is not None:
        env_file.write_text(env_file_content)

    models_config_file = tmp_path / "models.json"
    if models_config is not None:
        if isinstance(models_config, str):
            models_config_file.write_text(models_config)
        else:
            models_config_file.write_text(json.dumps(models_config))

    env = {
        "PATH": f"{stub_dir}:/usr/bin:/bin",
        "PYTHONPATH": str(SRC_DIR),
        "HOME": str(tmp_path),
        "UNFLATTEN_ENV_FILE": str(env_file),
        "UNFLATTEN_MODELS_CONFIG_FILE": str(models_config_file),
    }
    result = subprocess.run(
        ["bash", str(RUN_LOCAL), *args], env=env, capture_output=True, text=True
    )
    return result, calls_log


def test_up_dry_run_step_order(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, ["up", "--dry-run"])

    assert result.returncode == 0, result.stderr
    out = result.stdout
    for marker in ("[1/6]", "[2/6]", "[3/6]", "[4/6]", "[5/6]", "[6/6]"):
        assert marker in out, f"missing step marker {marker} in:\n{out}"
    assert out.index("[1/6]") < out.index("[2/6]") < out.index("[3/6]")
    assert out.index("[3/6]") < out.index("[4/6]") < out.index("[5/6]") < out.index("[6/6]")
    assert "docker compose" in out and "build" in out
    assert "up -d" in out


def test_up_refuses_without_accept_rootful_socket(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, ["up"], env_file_content="UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=0\n")

    assert result.returncode == 2
    assert "ACCEPT_ROOTFUL_SOCKET" in result.stderr


def test_up_refuses_allow_rootful_without_accept(tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path,
        ["up"],
        env_file_content=(
            "UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=0\nUNFLATTEN_SANDBOX_ALLOW_ROOTFUL=1\n"
        ),
    )

    assert result.returncode == 2
    assert "ALLOW_ROOTFUL" in result.stderr


def test_up_refuses_missing_models_config(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, ["up"], models_config=None)

    assert result.returncode == 2
    assert "models.json" in result.stderr or "models config" in result.stderr


def test_up_refuses_invalid_models_config(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, ["up"], models_config="{not valid json")

    assert result.returncode == 2


def test_start_prints_hint_and_refuses(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, ["start"])

    assert result.returncode == 2
    assert "dev-native.sh" in result.stderr
    assert "run-local.sh up" in result.stderr


def test_reset_password_dry_run_passes_matching_principal_file(tmp_path: Path) -> None:
    """Item 7: `web_auth_cli reset-password` defaults `--principal-file` to
    `<data-dir>/principal.json`, but `entrypoint-init.sh`/compose actually
    write `/data/principals.json` -- the wrapper must pass the file that
    really exists, or a fresh install exits 2."""
    result, _ = _run(tmp_path, ["reset-password", "--dry-run"])

    assert result.returncode == 0, result.stderr
    assert "--principal-file /data/principals.json" in result.stdout
    assert "--principal-id local" in result.stdout


def test_down_never_passes_dash_v(tmp_path: Path) -> None:
    result, calls_log = _run(tmp_path, ["down"])

    assert result.returncode == 0, result.stderr
    calls = calls_log.read_text()
    assert "down" in calls
    assert " -v" not in calls
    assert "-v\n" not in calls


# ---------------------------------------------------------------------------
# Owner 2026-09-29 (DEC-052 addendum): tokenless first-run setup only when the
# published port is loopback-only; otherwise a one-click `#token=` link.
# ---------------------------------------------------------------------------

_CONSENT = "UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1\n"


def _compose_calls(calls_log: Path) -> list[str]:
    return [line for line in calls_log.read_text().splitlines() if line.startswith("compose ")]


def test_up_default_loopback_bind_sets_tokenless(tmp_path: Path) -> None:
    result, calls_log = _run(tmp_path, ["up"])

    assert result.returncode == 0, result.stderr
    build_and_up = [c for c in _compose_calls(calls_log) if " build " in c or " up -d" in c]
    assert len(build_and_up) == 2
    assert all(c.endswith("|TOKENLESS=1") for c in build_and_up), build_and_up
    assert "http://127.0.0.1:8420/" in result.stdout
    assert "#token=" not in result.stdout


def test_up_dry_run_prints_the_tokenless_value(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, ["up", "--dry-run"])

    assert result.returncode == 0, result.stderr
    up_line = next(line for line in result.stdout.splitlines() if "up -d" in line)
    assert "UNFLATTEN_SETUP_TOKENLESS=1" in up_line


def test_up_non_loopback_bind_turns_tokenless_off_and_prints_setup_link(tmp_path: Path) -> None:
    result, calls_log = _run(
        tmp_path,
        ["up"],
        env_file_content=_CONSENT + "UNFLATTEN_BIND_ADDRESS=192.168.1.20\nUNFLATTEN_PORT=8420\n",
    )

    assert result.returncode == 0, result.stderr
    build_and_up = [c for c in _compose_calls(calls_log) if " build " in c or " up -d" in c]
    assert all(c.endswith("|TOKENLESS=0") for c in build_and_up), build_and_up
    assert f"http://192.168.1.20:8420/setup#token={_INIT_TOKEN}" in result.stdout


def test_up_non_loopback_bind_with_tokenless_requested_is_refused(tmp_path: Path) -> None:
    result, calls_log = _run(
        tmp_path,
        ["up"],
        env_file_content=_CONSENT
        + "UNFLATTEN_BIND_ADDRESS=0.0.0.0\nUNFLATTEN_SETUP_TOKENLESS=1\n",
    )

    assert result.returncode == 2
    assert "UNFLATTEN_SETUP_TOKENLESS" in result.stderr
    assert not any(" up -d" in c for c in _compose_calls(calls_log))


def test_up_loopback_bind_with_tokenless_opted_out_prints_setup_link(tmp_path: Path) -> None:
    """E.g. an SSH tunnel or reverse proxy in front of a loopback port: the
    operator can still demand the token by setting the flag to 0."""
    result, calls_log = _run(
        tmp_path, ["up"], env_file_content=_CONSENT + "UNFLATTEN_SETUP_TOKENLESS=0\n"
    )

    assert result.returncode == 0, result.stderr
    build_and_up = [c for c in _compose_calls(calls_log) if " build " in c or " up -d" in c]
    assert all(c.endswith("|TOKENLESS=0") for c in build_and_up), build_and_up
    assert f"http://127.0.0.1:8420/setup#token={_INIT_TOKEN}" in result.stdout


def test_up_invalid_tokenless_value_is_refused(tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path, ["up"], env_file_content=_CONSENT + "UNFLATTEN_SETUP_TOKENLESS=yes\n"
    )

    assert result.returncode == 2
    assert "UNFLATTEN_SETUP_TOKENLESS" in result.stderr
