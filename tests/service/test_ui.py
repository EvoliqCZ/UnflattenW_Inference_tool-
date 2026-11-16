"""Tests for the self-contained diagnostic UI page served at ``/``."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from unflatten.service.api import ApiDependencies, create_app
from unflatten.service.artifact_store import ArtifactStore
from unflatten.service.auth import Principal, PrincipalStore
from unflatten.service.job_store import JobStore
from unflatten.service.policy import load_runtime_policy, runtime_policy_hash
from unflatten.service.readiness import COMPONENT_CODES

_PATH_PATTERN = re.compile(r"""["'`](/v1/[a-zA-Z0-9_/{}.-]*|/health/[a-zA-Z0-9_/-]*)["'`]""")
_URL_PATTERN = re.compile(r"https?://")
TOKEN = "a" * 32

# GET /v1/models (model picker, owner request 2026-09-27) is owned by a parallel
# backend PR and is not registered by create_app yet. Exempt it from the strict
# "every referenced path is a registered route" check below; the dedicated test
# further down enforces its existence once the backend lands and is expected to
# be un-skipped at merge time.
_PENDING_BACKEND_PATHS = frozenset()


def _client(tmp_path: Path) -> TestClient:
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
    principal = Principal("principal_a", frozenset({"submit_diagnostic", "read_status"}))
    dependencies = ApiDependencies(
        principal_store=PrincipalStore([(hashlib.sha256(TOKEN.encode()).digest(), principal)]),
        job_store=jobs,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        runtime_policy=policy,
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:" + "1" * 64,
        component_statuses=dict.fromkeys(COMPONENT_CODES, True),
        diagnostic_docx_status=True,
        clock=lambda: 100.0,
        # T35: pinned explicitly -- this module tests the legacy self-contained
        # UI page, not the SPA `create_app` now also serves under `web_ui="spa"`.
        web_ui="legacy",
    )
    return TestClient(create_app(dependencies), base_url="http://127.0.0.1")


def _registered_paths(client: TestClient) -> set[str]:
    return {route.path for route in client.app.routes if isinstance(route, APIRoute)}


def test_ui_root_returns_html_with_csp_header_and_no_auth_required(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert "nonce-" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_ui_slash_ui_redirects_to_canonical_root(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/ui", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/"


def test_ui_page_references_only_registered_same_origin_routes(tmp_path: Path) -> None:
    client = _client(tmp_path)
    registered = _registered_paths(client)

    html = client.get("/").text
    referenced = {match.group(1) for match in _PATH_PATTERN.finditer(html)}
    dynamic_prefixes = [route.split("{")[0] for route in registered if "{" in route]

    assert referenced, "expected the page to reference at least one API path"
    for path in referenced - _PENDING_BACKEND_PATHS:
        # A literal fragment concatenated with a runtime job id (e.g.
        # "/v1/conversions/" + jobId) collapses to the route's static prefix.
        matches = path in registered or any(
            path == prefix or path.startswith(prefix) for prefix in dynamic_prefixes
        )
        assert matches, f"referenced path {path!r} is not a registered route"


def test_ui_page_references_v1_models_registered_route(tmp_path: Path) -> None:
    """GET /v1/models is now registered by the model variants backend work."""
    client = _client(tmp_path)
    registered = _registered_paths(client)
    assert "/v1/models" in registered


def test_ui_page_contains_no_external_url() -> None:
    from unflatten.service.api import _UI_TEMPLATE

    assert _URL_PATTERN.search(_UI_TEMPLATE) is None


def test_ui_page_shows_diagnostic_banner_text() -> None:
    from unflatten.service.api import _UI_TEMPLATE

    assert "Diagnostický nástroj" in _UI_TEMPLATE
    assert "nie produktová kvalita" in _UI_TEMPLATE


def test_ui_nonce_changes_between_requests(tmp_path: Path) -> None:
    client = _client(tmp_path)

    first = client.get("/").headers["content-security-policy"]
    second = client.get("/").headers["content-security-policy"]

    assert first != second


def test_existing_routes_are_unaffected_by_ui_registration(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200


def test_ui_page_has_side_by_side_preview_controls() -> None:
    from unflatten.service.api import _UI_TEMPLATE

    for element_id in (
        "preview-section",
        "preview-viewport-source",
        "preview-viewport-docx",
        "preview-prev",
        "preview-next",
        "preview-overlay",
        "preview-opacity",
    ):
        assert f'id="{element_id}"' in _UI_TEMPLATE, f"missing preview element {element_id!r}"


def _js_function_body(html: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace counting."""
    marker = f"function {name}("
    start = html.index(marker)
    brace_start = html.index("{", start)
    depth = 0
    for index in range(brace_start, len(html)):
        if html[index] == "{":
            depth += 1
        elif html[index] == "}":
            depth -= 1
            if depth == 0:
                return html[start : index + 1]
    raise AssertionError(f"unbalanced braces while extracting function {name!r}")


def test_legacy_csp_allows_blob_images(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/")

    csp = response.headers["content-security-policy"]
    assert "img-src 'self' data: blob:" in csp


def test_ui_checks_response_ok_on_poll_cancel_download() -> None:
    from unflatten.service.api import _UI_TEMPLATE

    poll_body = _js_function_body(_UI_TEMPLATE, "startPolling")
    cancel_body = _js_function_body(_UI_TEMPLATE, "cancelJob")
    download_body = _js_function_body(_UI_TEMPLATE, "downloadUrl")

    assert ".ok" in poll_body, "poll must check response.ok before treating the body as status"
    assert ".ok" in cancel_body, "cancel must check response.ok before treating the body as status"
    assert ".ok" in download_body, "download must check response.ok before reading the blob"


def test_ui_has_no_style_attributes() -> None:
    from unflatten.service.api import _UI_TEMPLATE

    assert 'style="' not in _UI_TEMPLATE
    assert "style='" not in _UI_TEMPLATE


def test_ui_preview_download_uses_png_filename() -> None:
    from unflatten.service.api import _UI_TEMPLATE

    render_artifacts_body = _js_function_body(_UI_TEMPLATE, "renderArtifacts")
    assert ".png" in render_artifacts_body, (
        "source_preview/docx_preview downloads must use a .png filename, not .json"
    )


def test_ui_does_not_read_unsourced_fields() -> None:
    from unflatten.service.api import _UI_TEMPLATE

    # "downloading_model" is not a member of JOB_STAGES (contracts.py) and
    # "status.detail" / "bytes_total" / "bytes_done" are not conversion_status
    # fields (contracts.py `_conversion_status`) -- the wire never sends them.
    assert "downloading_model" not in _UI_TEMPLATE
    assert "status.detail" not in _UI_TEMPLATE
    assert "bytes_total" not in _UI_TEMPLATE
    assert "bytes_done" not in _UI_TEMPLATE
