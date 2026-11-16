from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi import FastAPI

from unflatten.service import openapi_export
from unflatten.service.api import (
    ApiDependencies,
    create_app,
    register_routes,
)
from unflatten.service.artifact_store import ArtifactStore
from unflatten.service.auth import Principal, PrincipalStore
from unflatten.service.job_store import JobStore
from unflatten.service.policy import load_runtime_policy, runtime_policy_hash
from unflatten.service.readiness import COMPONENT_CODES
from unflatten.service.web_static import SPA_ROUTES

_BUILDERS = {
    "openapi": openapi_export.build_document,
    "codes": openapi_export.build_codes,
    "routes": openapi_export.build_routes,
}


@pytest.mark.parametrize("name", sorted(_BUILDERS))
def test_committed_exports_match_generated(name: str) -> None:
    path = openapi_export._EXPORT_PATHS[name]
    generated = openapi_export._render(_BUILDERS[name]())
    assert path.read_text(encoding="utf-8") == generated


def test_openapi_export_check_exits_1_on_drift(tmp_path: Path, monkeypatch) -> None:
    stale = {name: tmp_path / f"{name}.json" for name in _BUILDERS}
    for name, path in stale.items():
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(openapi_export, "_EXPORT_PATHS", stale)

    assert openapi_export.main(["--check"]) == 1

    openapi_export.main(["--write"])
    assert openapi_export.main(["--check"]) == 0


def test_routes_json_matches_registered_spa_routes() -> None:
    routes = openapi_export.build_routes()
    assert routes["spa_routes"] == list(SPA_ROUTES)


def test_register_routes_needs_no_dependencies() -> None:
    app = FastAPI()
    register_routes(app, web_ui="legacy")
    paths = {route.path for route in app.routes}
    assert "/health/live" in paths
    assert "/v1/conversions" in paths
    assert "/" in paths


def test_register_routes_spa_mode_serves_spa_shell_not_legacy_ui() -> None:
    """T35: `spa` registers `/` as the SPA shell (`web_static.serve_index`),
    not the legacy self-contained UI page; `/ui` still redirects to it."""
    app = FastAPI()
    register_routes(app, web_ui="spa")
    paths = {route.path for route in app.routes}
    assert "/" in paths
    assert "/ui" in paths
    assert "/health/live" in paths
    assert not any(route.include_in_schema for route in app.routes if route.path == "/")


def _dependencies(tmp_path: Path) -> ApiDependencies:
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
    token = "a" * 32
    principal = Principal("principal_a", frozenset({"read_status"}))
    return ApiDependencies(
        principal_store=PrincipalStore(
            [(hashlib.sha256(token.encode()).digest(), principal)]
        ),
        job_store=jobs,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        runtime_policy=policy,
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:" + "1" * 64,
        component_statuses=dict.fromkeys(COMPONENT_CODES, True),
        diagnostic_docx_status=True,
        clock=lambda: 100.0,
    )


def test_runtime_openapi_matches_export(tmp_path: Path) -> None:
    app = create_app(_dependencies(tmp_path))
    assert app.openapi() == openapi_export.build_document()
