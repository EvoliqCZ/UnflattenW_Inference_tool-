"""`api_cli.py` web flags: `--web-ui`, `--web-allowed-hosts`,
`--web-session-idle-seconds`, `--web-session-max-seconds`, `--web-auth-file`
(design §8.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from unflatten.service import api, api_cli
from unflatten.service.auth import PrincipalStore


def _write_minimal_dist(tmp_path: Path) -> Path:
    """A `dist/` just complete enough for `web_static.load_dist` (T35 `spa`
    startup wiring); content is otherwise irrelevant to these CLI-flag tests."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        '<!doctype html><meta name="csp-nonce" content="__CSP_NONCE__">',
        encoding="utf-8",
    )
    return dist


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_cli,
        "verify_bundle",
        lambda bundle_dir, **kwargs: {"identities": {"component_lock_hash": "sha256:" + "a" * 64}},
    )
    monkeypatch.setattr(api_cli, "load_principal_store", lambda path: PrincipalStore([]))
    for env in (
        "UNFLATTEN_WEB_UI",
        "UNFLATTEN_WEB_ALLOWED_HOSTS",
        "UNFLATTEN_WEB_SESSION_IDLE_SECONDS",
        "UNFLATTEN_WEB_SESSION_MAX_SECONDS",
    ):
        monkeypatch.delenv(env, raising=False)


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str) -> Any:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(api_cli.uvicorn, "run", lambda app, **kwargs: seen.__setitem__("app", app))
    argv = [
        "--principal-file",
        str(tmp_path / "principals.json"),
        "--job-store",
        str(tmp_path / "jobs.sqlite3"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--model-bundle",
        str(tmp_path / "bundle"),
        *extra,
    ]
    assert api_cli.main(argv) == 0
    return seen["app"].state.dependencies


def test_web_ui_flag_defaults_to_legacy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_common(monkeypatch)

    dependencies = _run(monkeypatch, tmp_path)

    assert dependencies.web_ui == "legacy"


def test_web_ui_flag_accepts_spa(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_common(monkeypatch)
    monkeypatch.setattr(api, "_WEB_DIST_DIR", _write_minimal_dist(tmp_path))

    dependencies = _run(monkeypatch, tmp_path, "--web-ui", "spa")

    assert dependencies.web_ui == "spa"


def test_headless_mode_starts_without_spa_and_keeps_docs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    _patch_common(monkeypatch)
    monkeypatch.setattr(api, "_WEB_DIST_DIR", tmp_path / "missing-dist")
    dependencies = _run(monkeypatch, tmp_path, "--web-ui", "none")
    assert dependencies.web_ui == "none"
    client = TestClient(api.create_app(dependencies), base_url="http://127.0.0.1")
    assert client.get("/").status_code == 404
    assert client.get("/health/live").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_web_allowed_hosts_flag_parses_comma_separated_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_common(monkeypatch)

    dependencies = _run(monkeypatch, tmp_path, "--web-allowed-hosts", "example.local, other.local")

    assert dependencies.web.allowed_hosts == ("example.local", "other.local")


def test_web_allowed_hosts_flag_defaults_to_loopback_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_common(monkeypatch)

    dependencies = _run(monkeypatch, tmp_path)

    assert dependencies.web.allowed_hosts == ("127.0.0.1", "localhost", "[::1]")


def test_web_auth_file_defaults_to_job_store_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_common(monkeypatch)

    dependencies = _run(monkeypatch, tmp_path)

    assert dependencies.web.web_auth_path == tmp_path / "web-auth.json"


def test_web_auth_file_flag_overrides_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_common(monkeypatch)
    explicit = tmp_path / "elsewhere" / "auth.json"

    dependencies = _run(monkeypatch, tmp_path, "--web-auth-file", str(explicit))

    assert dependencies.web.web_auth_path == explicit


def test_git_sha_flag_defaults_to_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_common(monkeypatch)

    dependencies = _run(monkeypatch, tmp_path)

    assert dependencies.web.git_sha == "unknown"


def test_git_sha_flag_is_reported_in_web_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_common(monkeypatch)

    dependencies = _run(monkeypatch, tmp_path, "--git-sha", "f464e76")

    assert dependencies.web.git_sha == "f464e76"


def test_session_max_below_idle_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_common(monkeypatch)
    monkeypatch.setattr(api_cli.uvicorn, "run", lambda app, **kwargs: None)

    with pytest.raises(ValueError, match="session_max_seconds"):
        api_cli.main(
            [
                "--principal-file",
                str(tmp_path / "principals.json"),
                "--job-store",
                str(tmp_path / "jobs.sqlite3"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--model-bundle",
                str(tmp_path / "bundle"),
                "--web-session-idle-seconds",
                "100",
                "--web-session-max-seconds",
                "50",
            ]
        )


def test_setup_tokenless_defaults_to_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_common(monkeypatch)
    monkeypatch.delenv("UNFLATTEN_SETUP_TOKENLESS", raising=False)

    dependencies = _run(monkeypatch, tmp_path)

    assert dependencies.web.setup_tokenless is False


def test_setup_tokenless_flag_sets_web_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_common(monkeypatch)
    monkeypatch.delenv("UNFLATTEN_SETUP_TOKENLESS", raising=False)

    dependencies = _run(monkeypatch, tmp_path, "--web-setup-tokenless")

    assert dependencies.web.setup_tokenless is True


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("0", False), ("", False)])
def test_setup_tokenless_env_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str, expected: bool
) -> None:
    _patch_common(monkeypatch)
    monkeypatch.setenv("UNFLATTEN_SETUP_TOKENLESS", value)

    dependencies = _run(monkeypatch, tmp_path)

    assert dependencies.web.setup_tokenless is expected


def test_setup_tokenless_env_other_value_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_common(monkeypatch)
    monkeypatch.setenv("UNFLATTEN_SETUP_TOKENLESS", "yes")

    with pytest.raises(ValueError, match="UNFLATTEN_SETUP_TOKENLESS"):
        _run(monkeypatch, tmp_path)
