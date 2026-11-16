from pathlib import Path

import pytest

from unflatten.service import api_cli
from unflatten.service.auth import PrincipalStore

ROOT = Path(__file__).resolve().parents[2]


def test_api_can_show_setup_while_the_pinned_model_is_not_downloaded(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(api_cli, "load_principal_store", lambda path: PrincipalStore([]))
    monkeypatch.setattr(api_cli.uvicorn, "run", lambda app, **kwargs: seen.update(app=app))
    result = api_cli.main(
        [
            "--principal-file",
            str(tmp_path / "principals.json"),
            "--job-store",
            str(tmp_path / "jobs.sqlite3"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--bootstrap-model-manifest",
            str(ROOT / "deploy/distribution/bootstrap/manifest.json"),
            "--models-config",
            str(ROOT / "deploy/distribution/models.json"),
        ]
    )
    assert result == 0
    deps = seen["app"].state.dependencies
    assert deps.component_lock_hash == (
        "sha256:2a76f76d5c6bdce830a437ec18e59031063fe5c54474c3a2cd5d2dae49e54cd6"
    )
    assert deps.component_statuses()["worker"] is False
    assert deps.component_statuses()["pdf_sandbox"] is False
    assert deps.diagnostic_docx_status() is False


def test_api_bootstrap_refuses_metadata_not_matching_the_catalog(monkeypatch, tmp_path):
    bootstrap = tmp_path / "manifest.json"
    bootstrap.write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        api_cli.main(
            [
                "--principal-file",
                str(tmp_path / "principals.json"),
                "--job-store",
                str(tmp_path / "jobs.sqlite3"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--bootstrap-model-manifest",
                str(bootstrap),
                "--models-config",
                str(ROOT / "deploy/distribution/models.json"),
            ]
        )
