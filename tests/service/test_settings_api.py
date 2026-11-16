"""T35 API A8 (design §3.10, §6.1): `GET /v1/settings` and the `spa` wiring
of `register_routes`/`create_app` (`web_static.register_spa_routes`,
fail-closed `load_dist` at startup, `/v1/*` never falling through to the
SPA shell)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.service.test_api_endpoints import AUTH, _principal_store

from unflatten.service import api
from unflatten.service.api import ApiDependencies, WebSettings, create_app
from unflatten.service.artifact_store import ArtifactStore
from unflatten.service.auth import PrincipalStore
from unflatten.service.job_store import JobStore
from unflatten.service.models_config import validate_models_config
from unflatten.service.policy import load_runtime_policy, runtime_policy_hash
from unflatten.service.readiness import COMPONENT_CODES
from unflatten.service.web_static import WebUiInvalid

_MODELS_CONFIG_EXPLICIT = validate_models_config(
    {
        "schema_version": "1.0",
        "default_variant": "bf16",
        "variants": [
            {
                "variant_id": "bf16",
                "label": "BF16 (baseline)",
                "precision": "bf16",
                "kv_cache": "bf16",
                "bundle": "/data/model-bundles/v15b-bf16-bf16",
            },
            {
                "variant_id": "int4",
                "label": "INT4 (small)",
                "precision": "int4",
                "kv_cache": "fp8",
                "bundle": "/data/model-bundles/v15b-int4-fp8",
            },
        ],
    }
)
_MODELS_CONFIG_AUTO = validate_models_config(
    {**_MODELS_CONFIG_EXPLICIT, "default_variant": "auto"}
)


def _client(
    tmp_path: Path,
    *,
    models_config: dict[str, Any] | None = None,
    web: WebSettings | None = None,
    capabilities: frozenset[str] | None = None,
) -> TestClient:
    tmp_path.mkdir(parents=True, exist_ok=True)
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    dependencies = ApiDependencies(
        principal_store=_principal_store(capabilities=capabilities),
        job_store=jobs,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        runtime_policy=policy,
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:" + "1" * 64,
        component_statuses=dict.fromkeys(COMPONENT_CODES, True),
        diagnostic_docx_status=True,
        clock=lambda: 100.0,
        models_config=models_config,
        **({"web": web} if web is not None else {}),
    )
    return TestClient(create_app(dependencies), base_url="http://127.0.0.1")


# ---------------------------------------------------------------------------
# GET /v1/settings shape
# ---------------------------------------------------------------------------


def test_settings_shape_with_no_models_config_and_no_worker_yet(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/settings", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "1.0",
        "service_version": {"git_sha": "unknown", "api_version": "1.1"},
        "session": {"idle_seconds": 43200, "max_seconds": 604800},
        "default_variant": {
            "configured": "auto",
            "override": None,
            "effective": None,
            "source": None,
        },
        "sandbox": {"rootful": None},
    }


def test_settings_requires_authentication(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/settings")

    assert response.status_code == 401


def test_settings_requires_read_status_capability(tmp_path: Path) -> None:
    client = _client(tmp_path, capabilities=frozenset({"submit_diagnostic"}))

    response = client.get("/v1/settings", headers=AUTH)

    assert response.status_code == 403
    assert response.json()["code"] == "CAPABILITY_DENIED"


def test_settings_git_sha_reflects_web_settings(tmp_path: Path) -> None:
    client = _client(tmp_path, web=WebSettings(git_sha="f464e76"))

    response = client.get("/v1/settings", headers=AUTH)

    assert response.json()["service_version"]["git_sha"] == "f464e76"


def test_settings_session_reflects_web_settings(tmp_path: Path) -> None:
    client = _client(
        tmp_path,
        web=WebSettings(session_idle_seconds=60, session_max_seconds=120),
    )

    response = client.get("/v1/settings", headers=AUTH)

    assert response.json()["session"] == {"idle_seconds": 60, "max_seconds": 120}


def test_settings_sandbox_rootful_null_before_worker_publishes(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/settings", headers=AUTH)

    assert response.json()["sandbox"] == {"rootful": None}


@pytest.mark.parametrize("rootful", [True, False])
def test_settings_sandbox_rootful_reflects_published_readiness(
    tmp_path: Path, rootful: bool
) -> None:
    client = _client(tmp_path)
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_readiness(
        writer,
        worker_ready=True,
        sandbox_ready=True,
        diagnostic_docx_ready=True,
        now=2.0,
        sandbox_rootful=rootful,
    )

    response = client.get("/v1/settings", headers=AUTH)

    assert response.json()["sandbox"] == {"rootful": rootful}


def test_settings_default_variant_configured_names_the_config_value(tmp_path: Path) -> None:
    client = _client(tmp_path, models_config=_MODELS_CONFIG_EXPLICIT)

    response = client.get("/v1/settings", headers=AUTH)

    assert response.json()["default_variant"] == {
        "configured": "bf16",
        "override": None,
        "effective": "bf16",
        "source": "config",
    }


def test_settings_default_variant_auto_resolves_from_measured_device(tmp_path: Path) -> None:
    client = _client(tmp_path, models_config=_MODELS_CONFIG_AUTO)
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer,
        resident_variant=None,
        free_vram_mib=100000.0,
        total_vram_mib=100000.0,
        compute_capability=(9, 0),
        now=2.0,
    )

    response = client.get("/v1/settings", headers=AUTH)

    body = response.json()["default_variant"]
    assert body["configured"] == "auto"
    assert body["override"] is None
    assert body["source"] == "auto"
    assert body["effective"] is not None


def test_settings_default_variant_auto_unresolved_without_measured_device(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path, models_config=_MODELS_CONFIG_AUTO)

    response = client.get("/v1/settings", headers=AUTH)

    assert response.json()["default_variant"] == {
        "configured": "auto",
        "override": None,
        "effective": None,
        "source": None,
    }


def test_settings_default_variant_user_override_wins(tmp_path: Path) -> None:
    client = _client(tmp_path, models_config=_MODELS_CONFIG_EXPLICIT)
    client.app.state.dependencies.job_store.set_setting("default_variant", "int4", now=1.0)

    response = client.get("/v1/settings", headers=AUTH)

    assert response.json()["default_variant"] == {
        "configured": "bf16",
        "override": "int4",
        "effective": "int4",
        "source": "user",
    }


def test_settings_default_variant_unknown_user_override_falls_back_to_config(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path, models_config=_MODELS_CONFIG_EXPLICIT)
    client.app.state.dependencies.job_store.set_setting(
        "default_variant", "does-not-exist", now=1.0
    )

    response = client.get("/v1/settings", headers=AUTH)

    body = response.json()["default_variant"]
    assert body["override"] == "does-not-exist"
    assert body["effective"] == "bf16"
    assert body["source"] == "config"


# ---------------------------------------------------------------------------
# spa wiring
# ---------------------------------------------------------------------------


def _write_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        '<!doctype html><html><head>'
        '<meta name="csp-nonce" content="__CSP_NONCE__">'
        "</head><body><div id=\"root\"></div></body></html>",
        encoding="utf-8",
    )
    return dist


def _spa_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, dist_dir: Path) -> TestClient:
    monkeypatch.setattr(api, "_WEB_DIST_DIR", dist_dir)
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    dependencies = ApiDependencies(
        principal_store=_principal_store(),
        job_store=jobs,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        runtime_policy=policy,
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:" + "1" * 64,
        component_statuses=dict.fromkeys(COMPONENT_CODES, True),
        diagnostic_docx_status=True,
        clock=lambda: 100.0,
        web_ui="spa",
    )
    return TestClient(create_app(dependencies), base_url="http://127.0.0.1")


def test_spa_mode_missing_dist_fails_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(api, "_WEB_DIST_DIR", tmp_path / "does-not-exist")
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()

    with pytest.raises(WebUiInvalid):
        create_app(
            ApiDependencies(
                principal_store=PrincipalStore([]),
                job_store=jobs,
                artifact_store=ArtifactStore(tmp_path / "artifacts"),
                runtime_policy=policy,
                policy_hash=runtime_policy_hash(policy),
                component_lock_hash="sha256:" + "1" * 64,
                component_statuses=dict.fromkeys(COMPONENT_CODES, True),
                diagnostic_docx_status=True,
                clock=lambda: 100.0,
                web_ui="spa",
            )
        )


def test_spa_mode_serves_spa_shell_and_v1_settings_as_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _spa_client(monkeypatch, tmp_path, dist_dir=_write_dist(tmp_path))

    shell = client.get("/jobs")
    settings = client.get("/v1/settings", headers=AUTH)

    assert shell.status_code == 200
    assert shell.headers["content-type"].startswith("text/html")
    assert settings.status_code == 200
    assert settings.headers["content-type"].startswith("application/json")


def test_v1_never_returns_html_in_spa(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Regression guard for the SPA catch-all shape (`spa-catchall-fakes-200`):
    an unmatched `/v1/...` path must never fall through to `serve_index` --
    never `200`, never `text/html`. CP4 minor 2: it must also be a
    `application/problem+json` document with a registry code, not FastAPI's
    default `{"detail": "Not Found"}`."""
    client = _spa_client(monkeypatch, tmp_path, dist_dir=_write_dist(tmp_path))

    response = client.get("/v1/does-not-exist")

    assert response.status_code == 404
    assert not response.headers["content-type"].startswith("text/html")
    assert "<html" not in response.text.lower()
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == "RESOURCE_NOT_FOUND"


def test_v1_unknown_path_returns_problem_json_in_legacy_mode(tmp_path: Path) -> None:
    """CP4 minor 2 also applies to `--web-ui legacy` (no `dist/` involved)."""
    client = _client(tmp_path)

    response = client.get("/v1/does-not-exist")

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == "RESOURCE_NOT_FOUND"


def test_spa_unknown_non_v1_path_is_unaffected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The `/v1/*` 404 handler must not swallow unrelated 404s: an unknown
    client-side path outside `/v1/*` keeps its existing (non-problem) shape."""
    client = _spa_client(monkeypatch, tmp_path, dist_dir=_write_dist(tmp_path))

    response = client.get("/nope-page")

    assert response.status_code == 404
    assert response.headers["content-type"] != "application/problem+json"


def test_legacy_mode_unchanged(tmp_path: Path) -> None:
    client = _client(tmp_path)

    root = client.get("/")
    jobs_route = client.get("/jobs")

    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    # "/jobs" is a spa-only client route; legacy mode never registers it.
    assert jobs_route.status_code == 404
