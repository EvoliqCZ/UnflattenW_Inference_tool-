"""T43 plan §I item 1: `api_cli.py` must derive `hf_cache_dir` the same way
huggingface_hub's own cache functions do -- `HF_HUB_CACHE` defaults to
`HF_HOME/hub`, so passing bare `HF_HOME` straight through as a `cache_dir`
looks one directory level too high and never finds a real snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from unflatten.service import api_cli
from unflatten.service.auth import PrincipalStore


def _patch_common(monkeypatch: pytest.MonkeyPatch, *, seen: dict[str, Any]) -> None:
    def fake_verify_bundle(bundle_dir, **kwargs):
        seen["verify_bundle_kwargs"] = kwargs
        return {"identities": {"component_lock_hash": "sha256:" + "a" * 64}}

    monkeypatch.setattr(api_cli, "verify_bundle", fake_verify_bundle)
    monkeypatch.setattr(api_cli, "load_principal_store", lambda path: PrincipalStore([]))
    monkeypatch.setattr(api_cli.uvicorn, "run", lambda app, **kwargs: seen.__setitem__("app", app))
    monkeypatch.delenv("HF_HOME", raising=False)


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seen: dict[str, Any], *extra: str) -> Any:
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


def test_hf_cache_dir_derived_from_hf_home_appends_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}
    _patch_common(monkeypatch, seen=seen)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))

    dependencies = _run(monkeypatch, tmp_path, seen)

    expected = tmp_path / "hf-cache" / "hub"
    assert dependencies.models_cache_dir == expected
    assert seen["verify_bundle_kwargs"]["hf_cache_dir"] == expected


def test_hf_cache_dir_absent_without_hf_home_or_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}
    _patch_common(monkeypatch, seen=seen)

    dependencies = _run(monkeypatch, tmp_path, seen)

    assert dependencies.models_cache_dir is None
    assert seen["verify_bundle_kwargs"]["hf_cache_dir"] is None


def test_models_cache_dir_flag_overrides_hf_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}
    _patch_common(monkeypatch, seen=seen)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))
    explicit = tmp_path / "elsewhere" / "hub"

    dependencies = _run(monkeypatch, tmp_path, seen, "--models-cache-dir", str(explicit))

    assert dependencies.models_cache_dir == explicit
    assert seen["verify_bundle_kwargs"]["hf_cache_dir"] == explicit
