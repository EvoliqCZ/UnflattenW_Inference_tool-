import configparser
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "hf_test_credential_for_launcher_tests_only"


def run_launcher(tmp_path, action="configure", **extra):
    env = {
        **os.environ,
        "UNFLATTEN_APP_DATA_DIR": str(tmp_path / "data with spaces"),
        "UNFLATTEN_APP_NONINTERACTIVE": "1",
        "HF_TOKEN": TOKEN,
        "HF_TOKEN_PATH": str(tmp_path / "missing-hf-token"),
        "UNFLATTEN_APP_LANG": "sk",
        **extra,
    }
    return subprocess.run(
        ["bash", str(ROOT / "Spustit.sh"), action],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )


def test_configure_secret_is_private_and_never_echoed(tmp_path):
    result = run_launcher(tmp_path)
    assert result.returncode == 0, result.stderr
    secret = tmp_path / "data with spaces.launcher/secrets/hf-token"
    assert secret.read_text().strip() == TOKEN
    assert secret.stat().st_mode & 0o777 == 0o600
    config_text = (tmp_path / "data with spaces.launcher/app.env").read_text()
    assert TOKEN not in result.stdout + result.stderr + config_text


def test_configure_reuses_credentials_and_preserves_user_data(tmp_path):
    assert run_launcher(tmp_path).returncode == 0
    data = tmp_path / "data with spaces"
    (data / "jobs.sqlite3").write_bytes(b"preserved-database")
    (data / "principals.json").write_bytes(b"preserved-account")
    result = run_launcher(tmp_path, HF_TOKEN="", HF_TOKEN_PATH=str(tmp_path / "absent"))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "data with spaces.launcher/secrets/hf-token").read_text().strip() == TOKEN
    assert (data / "jobs.sqlite3").read_bytes() == b"preserved-database"
    assert (data / "principals.json").read_bytes() == b"preserved-account"


def test_start_without_explicit_docker_consent_refuses_before_starting(tmp_path):
    result = run_launcher(tmp_path, "start", UNFLATTEN_ACCEPT_ROOTFUL_SOCKET="0")
    assert result.returncode != 0
    assert "Docker" in result.stderr and "UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1" in result.stderr
    assert not (tmp_path / "data with spaces.launcher/docker-consent").exists()


@pytest.mark.parametrize("token", [TOKEN, ""])
def test_configure_composed_stack_is_isolated_cuda_auto_and_worker_only_secret(tmp_path, token):
    result = run_launcher(tmp_path, HF_TOKEN=token)
    assert result.returncode == 0, result.stderr
    rendered = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(tmp_path / "data with spaces.launcher/app.env"),
            "-f",
            str(ROOT / "deploy/distribution/compose.yaml"),
            "config",
            "--format",
            "json",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    config = json.loads(rendered.stdout)
    assert config["name"] == "unflatten-w-app-1-0"
    services = config["services"]
    principal_path = tmp_path / "first-start-principals.json"
    initialized = subprocess.run(
        ["bash", str(ROOT / "deploy/docker/entrypoint-init.sh")],
        env={
            **os.environ,
            **services["init"]["environment"],
            "UNFLATTEN_PRINCIPAL_FILE": str(principal_path),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    credential = json.loads(principal_path.read_text())["principals"][0]["token"]
    assert credential not in initialized.stdout + initialized.stderr
    assert services["api"]["environment"]["UNFLATTEN_WEB_UI"] == "spa"
    assert services["api"]["environment"]["UNFLATTEN_BOOTSTRAP_MODEL_MANIFEST"] == (
        "/bootstrap/manifest.json"
    )
    worker = services["worker"]
    for key, value in {
        "UNFLATTEN_MODEL_VARIANT": "auto",
        "UNFLATTEN_DEVICE": "cuda:0",
        "UNFLATTEN_DEEPSTACK": "1",
        "UNFLATTEN_LOOP_GUARD": "bundle",
        "UNFLATTEN_KV_CACHE_MODE": "bf16_dynamic",
    }.items():
        assert worker["environment"][key] == value
    assert worker["deploy"]["resources"]["reservations"]["devices"][0]["capabilities"] == ["gpu"]
    assert worker["environment"]["HF_TOKEN_PATH"] == "/run/secrets/hf_token"
    assert worker["secrets"][0]["source"] == "hf_token"
    for name in ("api", "init"):
        assert not services[name].get("secrets")
        assert "HF_TOKEN_PATH" not in services[name]["environment"]
        secret_path = Path(config["secrets"]["hf_token"]["file"])
        for volume in services[name]["volumes"]:
            if volume["type"] == "bind":
                assert not secret_path.is_relative_to(Path(volume["source"]))
    for service in services.values():
        assert service["image"].startswith("unflatten-w-app-")


@pytest.mark.parametrize("noninteractive", ["0", "1"])
def test_configure_without_hf_credentials_uses_a_private_empty_secret(tmp_path, noninteractive):
    result = run_launcher(tmp_path, HF_TOKEN="", UNFLATTEN_APP_NONINTERACTIVE=noninteractive)
    assert result.returncode == 0, result.stderr
    secret = tmp_path / "data with spaces.launcher/secrets/hf-token"
    assert secret.read_bytes() == b""
    assert secret.stat().st_mode & 0o777 == 0o600
    assert "token" not in result.stdout.lower() + result.stderr.lower()
    assert (tmp_path / "data with spaces.launcher/app.env").is_file()


def test_configure_uses_optional_cached_hf_credentials_without_echoing_them(tmp_path):
    cached = tmp_path / "cached-hf-token"
    cached.write_text(TOKEN)
    result = run_launcher(tmp_path, HF_TOKEN="", HF_TOKEN_PATH=str(cached))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "data with spaces.launcher/secrets/hf-token").read_text() == TOKEN
    assert TOKEN not in result.stdout + result.stderr


@pytest.mark.parametrize("token", ["invalid-credential", "hf_bad\nsecond-line", "hf_bad\rline"])
def test_configure_rejects_invalid_explicit_credentials_without_echoing_them(tmp_path, token):
    result = run_launcher(tmp_path, HF_TOKEN=token)
    assert result.returncode != 0
    assert token not in result.stdout + result.stderr
    assert not (tmp_path / "data with spaces.launcher/secrets/hf-token").exists()


def test_desktop_shortcut_launches_the_shipped_app_and_uses_its_icon(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "Instalovat-spustac.sh")],
        env={**os.environ, "XDG_DATA_HOME": str(tmp_path / "local share")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    desktop = tmp_path / "local share/applications/unflatten-w.desktop"
    entry = configparser.ConfigParser(interpolation=None)
    entry.read(desktop)
    assert shlex.split(entry["Desktop Entry"]["Exec"]) == [str(ROOT / "Spustit.sh")]
    assert Path(entry["Desktop Entry"]["Icon"]).is_file()
    assert entry["Desktop Entry"]["Name"] == "Unflatten W 1.0"


def test_stop_pins_its_project_even_with_conflicting_inherited_project_name(tmp_path):
    assert run_launcher(tmp_path).returncode == 0
    bindir = tmp_path / "bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$DOCKER_TEST_ARGS"\n')
    docker.chmod(0o755)
    args_path = tmp_path / "docker-args"
    result = run_launcher(
        tmp_path,
        "stop",
        PATH=str(bindir) + os.pathsep + os.environ["PATH"],
        COMPOSE_PROJECT_NAME="unrelated-production",
        DOCKER_TEST_ARGS=str(args_path),
    )
    assert result.returncode == 0, result.stderr
    args = args_path.read_text().splitlines()
    assert args[args.index("-p") + 1] == "unflatten-w-app-1-0"
    assert args[-1] == "down"


def test_start_pins_the_locally_built_sandbox_image_by_immutable_id(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text(
        '#!/bin/sh\nif [ "$1" = image ] && [ "$2" = inspect ]; then\n'
        ' printf "sha256:%s\\n" '
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nfi\n"
    )
    docker.chmod(0o755)
    gpu = bindir / "nvidia-smi"
    gpu.write_text("#!/bin/sh\nexit 0\n")
    gpu.chmod(0o755)
    result = run_launcher(
        tmp_path,
        "start",
        PATH=str(bindir) + os.pathsep + os.environ["PATH"],
        UNFLATTEN_ACCEPT_ROOTFUL_SOCKET="1",
        UNFLATTEN_APP_OPEN_BROWSER="0",
    )
    assert result.returncode == 0, result.stderr
    config = (tmp_path / "data with spaces.launcher/app.env").read_text()
    assert "UNFLATTEN_APP_SANDBOX_IMAGE=sha256:" + "a" * 64 in config


def _runtime_stubs(tmp_path):
    bindir = tmp_path / "runtime-bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['DOCKER_CALLS'], 'a') as log:
    log.write(json.dumps(args) + '\\n')
if args[:2] == ['image', 'inspect']:
    print('sha256:' + 'a' * 64)
if 'ps' in args and '--services' in args:
    print(os.environ.get('DOCKER_RUNNING', ''))
""")
    docker.chmod(0o755)
    gpu = bindir / "nvidia-smi"
    gpu.write_text("#!/bin/sh\nexit 0\n")
    gpu.chmod(0o755)
    return {
        "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
        "DOCKER_CALLS": str(tmp_path / "docker-calls.jsonl"),
        "UNFLATTEN_ACCEPT_ROOTFUL_SOCKET": "1",
        "UNFLATTEN_APP_OPEN_BROWSER": "0",
    }


def _docker_calls(env):
    return [json.loads(line) for line in Path(env["DOCKER_CALLS"]).read_text().splitlines()]


@pytest.mark.parametrize("action", ["start", "restart"])
def test_public_model_starts_and_restarts_without_hf_credentials(tmp_path, action):
    env = _runtime_stubs(tmp_path)
    secret = tmp_path / "data with spaces.launcher/secrets/hf-token"
    if action == "restart":
        assert run_launcher(tmp_path, UNFLATTEN_APP_PORT="18421").returncode == 0
        secret.write_bytes(b"")
    result = run_launcher(tmp_path, action, HF_TOKEN="", **env)
    assert result.returncode == 0, result.stderr
    assert secret.read_bytes() == b""
    assert secret.stat().st_mode & 0o777 == 0o600
    assert "token" not in result.stdout.lower() + result.stderr.lower()
    assert TOKEN not in result.stdout + result.stderr
    calls = _docker_calls(env)
    assert any("build" in call for call in calls)
    assert any("up" in call and "--force-recreate" in call for call in calls)
    if action == "restart":
        assert "18421" in result.stdout


def test_second_start_only_reopens_and_preserves_saved_environment(tmp_path):
    env = _runtime_stubs(tmp_path)
    first = run_launcher(tmp_path, "start", UNFLATTEN_APP_PORT="18421", **env)
    assert first.returncode == 0, first.stderr
    settings = tmp_path / "data with spaces.launcher/app.env"
    before = settings.read_bytes()
    Path(env["DOCKER_CALLS"]).write_text("")
    result = run_launcher(tmp_path, "start", DOCKER_RUNNING="api\nworker", **env)
    assert result.returncode == 0, result.stderr
    assert settings.read_bytes() == before
    assert "18421" in result.stdout
    assert not any(
        "build" in call or "up" in call or "--force-recreate" in call for call in _docker_calls(env)
    )


def test_start_with_live_worker_restores_api_without_recreating_worker(tmp_path):
    env = _runtime_stubs(tmp_path)
    assert run_launcher(tmp_path, "start", **env).returncode == 0
    Path(env["DOCKER_CALLS"]).write_text("")
    result = run_launcher(tmp_path, "start", DOCKER_RUNNING="worker", **env)
    assert result.returncode == 0, result.stderr
    calls = _docker_calls(env)
    assert not any("build" in call or "--force-recreate" in call for call in calls)
    up = next(call for call in calls if "up" in call)
    assert "--no-recreate" in up and "--no-build" in up


def test_explicit_restart_rebuilds_and_keeps_saved_port(tmp_path):
    env = _runtime_stubs(tmp_path)
    assert run_launcher(tmp_path, "start", UNFLATTEN_APP_PORT="18421", **env).returncode == 0
    Path(env["DOCKER_CALLS"]).write_text("")
    result = run_launcher(tmp_path, "restart", **env)
    assert result.returncode == 0, result.stderr
    assert (
        "UNFLATTEN_APP_PORT=18421" in (tmp_path / "data with spaces.launcher/app.env").read_text()
    )
    calls = _docker_calls(env)
    assert any("build" in call for call in calls)
    assert any("up" in call and "--force-recreate" in call for call in calls)


def test_reset_password_uses_saved_project_and_data_without_restart(tmp_path):
    env = _runtime_stubs(tmp_path)
    assert run_launcher(tmp_path, **env).returncode == 0
    result = run_launcher(tmp_path, "reset-password", COMPOSE_PROJECT_NAME="unrelated", **env)
    assert result.returncode == 0, result.stderr
    call = _docker_calls(env)[-1]
    assert call[call.index("-p") + 1] == "unflatten-w-app-1-0"
    assert call[-5:] == [
        "-m",
        "unflatten.service.web_auth_cli",
        "reset-password",
        "--data-dir",
        str(tmp_path / "data with spaces"),
    ]
    assert "--no-deps" in call and "--rm" in call
    assert not any("up" in c or "build" in c for c in _docker_calls(env))
    assert TOKEN not in result.stdout + result.stderr + repr(call)


@pytest.mark.parametrize("char", ["\\", ":", "%", ",", '"', "$", "\n", "\r", "\t"])
def test_unsupported_data_paths_fail_before_creating_any_files(tmp_path, char):
    target = tmp_path / ("unsupported" + char + "path")
    result = run_launcher(tmp_path, UNFLATTEN_APP_DATA_DIR=str(target))
    assert result.returncode != 0
    assert "nepodporované" in result.stderr
    assert not target.exists()
    assert not Path(str(target) + ".launcher").exists()


@pytest.mark.parametrize("script", ["Spustit.sh", "Instalovat-spustac.sh"])
def test_unsupported_source_path_fails_before_writing(tmp_path, script):
    import shutil

    source = tmp_path / "unflatten%f"
    source.mkdir()
    shutil.copy2(ROOT / script, source / script)
    (source / "deploy/distribution").mkdir(parents=True)
    shutil.copy2(
        ROOT / "deploy/distribution/messages.sh", source / "deploy/distribution/messages.sh"
    )
    if script == "Spustit.sh":
        shutil.copy2(
            ROOT / "deploy/distribution/launch.sh", source / "deploy/distribution/launch.sh"
        )
    data = tmp_path / "data"
    result = subprocess.run(
        ["bash", str(source / script), "configure"],
        env={
            **os.environ,
            "UNFLATTEN_APP_DATA_DIR": str(data),
            "XDG_DATA_HOME": str(tmp_path / "local"),
            "HF_TOKEN": TOKEN,
            "UNFLATTEN_APP_LANG": "sk",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "nepodporované" in result.stderr
    assert not data.exists()
    assert not (tmp_path / "local/applications/unflatten-w.desktop").exists()


@pytest.mark.parametrize(("language", "expected"), [("en", "Local data:"), ("sk", "Lokálne dáta:")])
def test_launcher_can_be_used_in_both_languages_without_echoing_credentials(
    tmp_path, language, expected
):
    result = run_launcher(tmp_path, UNFLATTEN_APP_LANG=language)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert TOKEN not in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("locale", "expected"), [("en_US.UTF-8", "Invalid port."), ("sk_SK.UTF-8", "Neplatný port.")]
)
def test_launcher_uses_os_locale_when_no_language_override(tmp_path, locale, expected):
    result = run_launcher(
        tmp_path, UNFLATTEN_APP_LANG="", LC_ALL=locale, UNFLATTEN_APP_PORT="invalid"
    )
    assert result.returncode != 0
    assert expected in result.stderr


def test_english_alias_uses_same_configuration_and_data(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "Start.sh"), "configure"],
        env={
            **os.environ,
            "UNFLATTEN_APP_LANG": "en",
            "UNFLATTEN_APP_DATA_DIR": str(tmp_path / "shared data"),
            "UNFLATTEN_APP_NONINTERACTIVE": "1",
            "HF_TOKEN": TOKEN,
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Local data:" in result.stdout
    assert (tmp_path / "shared data.launcher/secrets/hf-token").read_text().strip() == TOKEN


def test_desktop_shortcut_has_english_default_and_slovak_description(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "Instalovat-spustac.sh")],
        env={
            **os.environ,
            "XDG_DATA_HOME": str(tmp_path / "local share"),
            "UNFLATTEN_APP_LANG": "en",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    entry = configparser.ConfigParser(interpolation=None)
    entry.read(tmp_path / "local share/applications/unflatten-w.desktop")
    assert "editable Word" in entry["Desktop Entry"]["Comment"]
    assert "Wordu" in entry["Desktop Entry"]["Comment[sk]"]
    assert "applications menu" in result.stdout
