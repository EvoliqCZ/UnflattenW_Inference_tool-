"""Authenticated HTTP boundary for private diagnostic conversions."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import unicodedata
import urllib.parse
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler as _default_http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from unflatten.dataset.artifact_ref import ArtifactRef
from unflatten.service import image_input, schemas
from unflatten.service.api_docs import register_documentation
from unflatten.service.artifact_store import ArtifactStore, ArtifactStoreError
from unflatten.service.artifacts_zip import ROLE_CAPABILITIES, build_artifacts_zip
from unflatten.service.auth import (
    AuthenticationFailed,
    CapabilityDenied,
    Principal,
    PrincipalStore,
    require_capability,
)
from unflatten.service.contracts import (
    ACCEPTED_OUTPUTS,
    CACHED_LOAD_FAILURE_CODES,
    JOB_STATES,
    validate_contract,
)
from unflatten.service.errors import problem_response
from unflatten.service.job_store import (
    ArtifactRecord,
    CapacityExceeded,
    IdempotencyKeyReused,
    JobActive,
    JobNotFound,
    JobRecord,
    JobStore,
    JobStoreInvalid,
    JobTerminal,
    ModelCommandConflict,
    ModelCommandRecord,
)
from unflatten.service.middleware import (
    CacheControlMiddleware,
    HostAllowlistMiddleware,
    SecurityHeadersMiddleware,
)
from unflatten.service.model_variants import (
    estimated_peak_vram_mib,
    is_variant_downloaded,
    resolve_default_variant,
    variant_download_state,
    variant_fit_reason,
    variant_fits,
)
from unflatten.service.models_config import PUBLISHED
from unflatten.service.policy import runtime_policy_hash, validate_runtime_policy
from unflatten.service.readiness import current_docx_requirements, evaluate_readiness
from unflatten.service.upload_staging import stage_upload
from unflatten.service.web_auth import (
    LoginRateLimiter,
    RateLimited,
    WebAuthFile,
    WebAuthInvalid,
    WebSessionRecordLike,
    WebSessionStore,
    verify_password,
)
from unflatten.service.web_static import WebUiDist, load_dist, register_spa_routes

#: Design §2.4: the session cookie name, always host-only (no `Domain`).
_SESSION_COOKIE = "unflatten_session"
#: Design §2.4: a cookie request is only ever accepted from these Sec-Fetch-Site values.
_ALLOWED_SEC_FETCH_SITE = ("same-origin", "none")

_DOWNLOAD_ROLES: dict[str, tuple[str, str, str]] = {
    "prediction_ir": (
        "download_prediction_ir",
        "application/json",
        "prediction_ir.json",
    ),
    "diagnostic_docx": (
        "download_docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "diagnostic.docx",
    ),
    # DEC-044 Dodatok D (C): side-by-side previews share diagnostic_docx's
    # download capability and retention.
    "source_preview": ("download_docx", "image/png", "source_preview.png"),
    "docx_preview": ("download_docx", "image/png", "docx_preview.png"),
}
#: Design §3.7: PNG preview roles are served `inline` with an `ETag`
#: (`prediction_ir`/`diagnostic_docx` keep today's `attachment` behaviour
#: untouched).
_INLINE_PREVIEW_ROLES = frozenset({"source_preview", "docx_preview"})

_REQUIRED_SERVICE_HEADERS = frozenset({"x-unflatten-profile", "x-unflatten-output"})
# DEC-044 Dodatok D §1: `x-unflatten-model` is optional -- its absence keeps
# today's single-bundle behaviour byte-identical. DEC-052 design §3.2: the
# same is true of `x-unflatten-label` (T25). CP4 D1: `x-unflatten-client` is
# the design §2.4 cookie-CSRF header the SPA sends on every call (including
# submissions); a bearer client may send it too, since `_authorize` only
# requires it on the cookie branch.
_KNOWN_SERVICE_HEADERS = _REQUIRED_SERVICE_HEADERS | {
    "x-unflatten-model", "x-unflatten-label", "x-unflatten-client",
}
_IDEMPOTENCY_KEY = re.compile(r"^[\x21-\x7e]{1,256}$")
#: DEC-052 design §3.2: characters that would make the label unsafe to reuse
#: verbatim as a downloaded filename, on top of the disallowed Unicode
#: categories (control, format, line/paragraph separators) checked below.
_FORBIDDEN_LABEL_CHARS = frozenset('\\/:*?"<>|')
_FORBIDDEN_LABEL_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
_LABEL_MAX_CODE_POINTS = 120
#: Single source of truth for every media type `_validate_submission_headers`
#: accepts as a top-level Content-Type; `/v1/readiness`'s
#: `limits.accepted_media_types` is derived from this list rather than
#: carrying its own copy.
_ACCEPTED_MEDIA_TYPES = ("application/pdf", "image/png", "image/jpeg", "multipart/form-data")
#: DEC-044 Dodatok D §2 (B2): a single raw page carried directly as the
#: request body, or a multipart body whose `page` parts are parsed by
#: `_admit_multipart`.
_ACCEPTED_IMAGE_MEDIA_TYPES = frozenset(
    media_type for media_type in _ACCEPTED_MEDIA_TYPES if media_type.startswith("image/")
)
_UI_TEMPLATE = (
    resources.files("unflatten.service.ui").joinpath("index.html").read_text(encoding="utf-8")
)
#: Shared FastAPI app metadata: `openapi_export.build_document` builds its own
#: bare app from these so the exported document's title/version can never
#: drift from the one the running service reports.
API_TITLE = "Unflatten private diagnostic service"
API_VERSION = "1.0"
#: `legacy`: today's single self-contained `/` UI page (nonce CSP). `spa`:
#: the built SPA served from `_WEB_DIST_DIR` (design §6.1).
WebUiMode = Literal["legacy", "spa", "none"]
#: Design §6.1/§8.1: the built SPA `dist/` directory the Docker image's
#: `web-builder` stage copies into (`src/unflatten/service/ui/dist`, gitignored
#: natively). `create_app` loads it once at startup when `web_ui == "spa"`.
_WEB_DIST_DIR = Path(str(resources.files("unflatten.service.ui"))) / "dist"
#: `register_routes` is pure (no filesystem I/O, safe for `openapi_export` to
#: call without a built `dist/`) -- when the caller does not supply a loaded
#: `dist` (i.e. every caller except `create_app`), the `spa` branch registers
#: its routes against this empty placeholder. Every one of those routes is
#: `include_in_schema=False`, so the placeholder's content never surfaces.
_SCHEMA_ONLY_DIST = WebUiDist(dist_dir=Path("unused"), index_html="", root_files=())


#: Host names that can only ever reach a loopback-published port.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


@dataclass(frozen=True)
class WebSettings:
    """Web-only settings that stay outside the hashed `runtime_policy`
    (design §0: the policy is hashed and bound to jobs; these are not).
    See design §2.5 (`allowed_hosts`), §2.2/§2.6 (session lifetimes),
    §2.1 (`web_auth_path`), §3.10 (`git_sha`)."""

    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "[::1]")
    session_idle_seconds: int = 43200
    session_max_seconds: int = 604800
    web_auth_path: Path = Path("web-auth.json")
    git_sha: str = "unknown"
    #: Owner 2026-09-29 (DEC-052 addendum): the launcher sets this only when
    #: it publishes the port on loopback; `POST /v1/setup` then accepts the
    #: first password without the bootstrap bearer token.
    setup_tokenless: bool = False

    def __post_init__(self) -> None:
        if not self.allowed_hosts:
            raise ValueError("allowed_hosts must not be empty")
        if self.setup_tokenless and not set(self.allowed_hosts) <= _LOOPBACK_HOSTS:
            # A LAN deployment has to allow-list its LAN host name; that and
            # tokenless setup together mean the launcher's loopback check
            # was bypassed (e.g. a hand-edited compose file).
            raise ValueError("setup_tokenless requires loopback-only allowed_hosts")
        if self.session_idle_seconds <= 0:
            raise ValueError("session_idle_seconds must be positive")
        if self.session_max_seconds < self.session_idle_seconds:
            raise ValueError("session_max_seconds must be >= session_idle_seconds")


@dataclass(frozen=True)
class ApiDependencies:
    """Initialized service dependencies injected into the API process."""

    principal_store: PrincipalStore
    job_store: JobStore
    artifact_store: ArtifactStore
    runtime_policy: Mapping[str, Any]
    policy_hash: str
    component_lock_hash: str
    component_statuses: Mapping[str, bool] | Callable[[], Mapping[str, bool]]
    diagnostic_docx_status: bool | Callable[[], bool]
    clock: Callable[[], float]
    # DEC-044 Dodatok D §1: absent (None) keeps today's single-bundle
    # behaviour byte-identical -- no model header validation, no /v1/models
    # variant listing beyond an empty, unconfigured response.
    models_config: Mapping[str, Any] | None = None
    #: Local HF cache directory `is_variant_downloaded` scans for `hf` source
    #: variants; unused for `bundle` (already-local) variants.
    models_cache_dir: Path | None = None
    work_root: Path | None = None
    web: WebSettings = field(default_factory=WebSettings)
    web_ui: WebUiMode = "legacy"
    #: In-process login failure limiter (design §2.6); must persist across
    #: requests, unlike `WebAuthFile`/`WebSessionStore` which are cheap to
    #: rebuild per request from `web`/`job_store`.
    login_rate_limiter: LoginRateLimiter = field(default_factory=LoginRateLimiter)

    def __post_init__(self) -> None:
        validate_runtime_policy(self.runtime_policy)
        if self.policy_hash != runtime_policy_hash(self.runtime_policy):
            raise ValueError("API policy hash does not bind the supplied runtime policy")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.component_lock_hash) is None:
            raise ValueError("component_lock_hash must be a prefixed SHA-256 digest")


def register_routes(
    app: FastAPI, *, web_ui: WebUiMode, web_dist: WebUiDist | None = None
) -> None:
    """Register every route this API exposes. Pure: reads no dependencies
    and touches only `app`, so it is safe to call against a bare `FastAPI()`
    (used by `openapi_export.build_document` to build the OpenAPI schema
    without an `ApiDependencies` instance). `web_dist` is the real, loaded
    SPA build; omitted (schema export) it falls back to `_SCHEMA_ONLY_DIST`,
    since none of `register_spa_routes`'s routes are ever in the schema."""
    if web_ui == "legacy":
        app.add_api_route("/", _ui_page, methods=["GET"], include_in_schema=False)
        app.add_api_route("/ui", _ui_redirect, methods=["GET"], include_in_schema=False)
    elif web_ui == "spa":
        register_spa_routes(app, web_dist or _SCHEMA_ONLY_DIST)
        app.add_api_route("/ui", _ui_redirect, methods=["GET"], include_in_schema=False)
    app.add_api_route("/health/live", _liveness, methods=["GET"])
    app.add_api_route("/health/ready", _operational_readiness, methods=["GET"])
    app.add_api_route("/v1/readiness", _detailed_readiness, methods=["GET"])
    app.add_api_route(
        "/v1/session",
        _session_get,
        methods=["GET"],
        responses={200: {"model": schemas.SessionInfo}, "default": {"model": schemas.Problem}},
    )
    app.add_api_route(
        "/v1/session",
        _session_post,
        methods=["POST"],
        status_code=201,
        responses={201: {"model": schemas.SessionInfo}, "default": {"model": schemas.Problem}},
    )
    app.add_api_route(
        "/v1/session",
        _session_delete,
        methods=["DELETE"],
        status_code=204,
        responses={"default": {"model": schemas.Problem}},
    )
    app.add_api_route(
        "/v1/setup",
        _setup_post,
        methods=["POST"],
        status_code=201,
        responses={201: {"model": schemas.SessionInfo}, "default": {"model": schemas.Problem}},
    )
    app.add_api_route(
        "/v1/setup/password",
        _setup_password_put,
        methods=["PUT"],
        status_code=204,
        responses={"default": {"model": schemas.Problem}},
    )
    app.add_api_route(
        "/v1/settings",
        _settings,
        methods=["GET"],
        responses={
            200: {"model": schemas.SettingsResponse},
            "default": {"model": schemas.Problem},
        },
    )
    app.add_api_route(
        "/v1/models",
        _models,
        methods=["GET"],
        responses={200: {"model": schemas.ModelsResponse}, "default": {"model": schemas.Problem}},
    )
    app.add_api_route(
        "/v1/models/default",
        _models_default_put,
        methods=["PUT"],
        responses={200: {"model": schemas.ModelsResponse}, "default": {"model": schemas.Problem}},
    )
    app.add_api_route(
        "/v1/models/{variant_id}/download",
        _models_download_post,
        methods=["POST"],
        status_code=202,
        responses={
            202: {"model": schemas.ModelCommandAccepted},
            "default": {"model": schemas.Problem},
        },
    )
    app.add_api_route(
        "/v1/models/{variant_id}/download",
        _models_download_delete,
        methods=["DELETE"],
        status_code=202,
        responses={
            202: {"model": schemas.ModelCommandAccepted},
            "default": {"model": schemas.Problem},
        },
    )
    app.add_api_route(
        "/v1/models/{variant_id}/load",
        _models_load_post,
        methods=["POST"],
        status_code=202,
        responses={
            202: {"model": schemas.ModelCommandAccepted},
            "default": {"model": schemas.Problem},
        },
    )
    app.add_api_route(
        "/v1/conversions",
        _submit,
        methods=["POST"],
        status_code=202,
        responses={
            202: {"model": schemas.SubmissionResponse},
            "default": {"model": schemas.Problem},
        },
    )
    app.add_api_route(
        "/v1/conversions",
        _list_jobs,
        methods=["GET"],
        responses={
            200: {"model": schemas.JobList},
            "default": {"model": schemas.Problem},
        },
    )
    app.add_api_route(
        "/v1/conversions/{job_id}",
        _status,
        methods=["GET"],
        responses={
            200: {"model": schemas.ConversionStatus},
            "default": {"model": schemas.Problem},
        },
    )
    app.add_api_route(
        "/v1/conversions/{job_id}",
        _cancel,
        methods=["DELETE"],
        responses={
            200: {"model": schemas.ConversionStatus},
            "default": {"model": schemas.Problem},
        },
    )
    app.add_api_route(
        "/v1/conversions/{job_id}/artifacts/prediction_ir",
        _download_role("prediction_ir"),
        methods=["GET"],
    )
    app.add_api_route(
        "/v1/conversions/{job_id}/artifacts/diagnostic_docx",
        _download_role("diagnostic_docx"),
        methods=["GET"],
    )
    app.add_api_route(
        "/v1/conversions/{job_id}/artifacts/prediction_ir/{page_index}",
        _download_role_page("prediction_ir"),
        methods=["GET"],
    )
    app.add_api_route(
        "/v1/conversions/{job_id}/artifacts/source_preview/{page_index}",
        _download_role_page("source_preview"),
        methods=["GET"],
    )
    app.add_api_route(
        "/v1/conversions/{job_id}/artifacts/docx_preview/{page_index}",
        _download_role_page("docx_preview"),
        methods=["GET"],
    )
    app.add_api_route(
        "/v1/conversions/{job_id}/artifacts.zip",
        _artifacts_zip,
        methods=["GET"],
    )


def create_app(dependencies: ApiDependencies) -> FastAPI:
    """Create an API process without initializing raster or model runtimes."""
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.dependencies = dependencies
    # Design §6.1: a `spa` process refuses to start over a missing/malformed
    # `dist/` (fail-closed) rather than 404ing every UI route at request time.
    web_dist = load_dist(_WEB_DIST_DIR) if dependencies.web_ui == "spa" else None
    register_routes(app, web_ui=dependencies.web_ui, web_dist=web_dist)
    register_documentation(app)
    app.add_exception_handler(RequestValidationError, _request_validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    # Middleware order (design §2.5, §6.3): `HostAllowlistMiddleware` is added
    # first so it sits closer to the router, and `SecurityHeadersMiddleware`
    # is added last so it wraps everything -- including the 421 the Host
    # middleware answers directly -- since Starlette makes the
    # last-added middleware the outermost.
    app.add_middleware(HostAllowlistMiddleware, allowed_hosts=dependencies.web.allowed_hosts)
    app.add_middleware(CacheControlMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    return app


async def _ui_redirect() -> Response:
    return RedirectResponse("/", status_code=307)


async def _ui_page() -> Response:
    """Serve the single self-contained diagnostic UI page (no auth, no data in it)."""
    nonce = secrets.token_urlsafe(16)
    html = _UI_TEMPLATE.replace("__NONCE__", nonce)
    csp = (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        f"style-src 'self' 'nonce-{nonce}'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "base-uri 'none'; "
        "form-action 'none'; "
        "frame-ancestors 'none'"
    )
    # `X-Content-Type-Options` and the rest of the fixed security header set
    # now come from `SecurityHeadersMiddleware` on every response (design
    # §6.3); only the page-specific nonce CSP is set here.
    return HTMLResponse(html, headers={"Content-Security-Policy": csp})


async def _liveness() -> dict[str, str]:
    return {"schema_version": "1.0", "status": "live"}


async def _operational_readiness(request: Request) -> Response:
    report = _readiness_report(_dependencies(request))
    public = {
        "schema_version": report["schema_version"],
        "service_ready": report["service_ready"],
        "components": report["components"],
    }
    return JSONResponse(public, status_code=200 if report["service_ready"] else 503)


async def _detailed_readiness(request: Request) -> Response:
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, "read_status")
    if isinstance(principal, Response):
        return principal
    report = dict(_readiness_report(dependencies))
    report["limits"] = {
        "max_pages": dependencies.runtime_policy["max_pdf_pages"],
        "max_upload_bytes": dependencies.runtime_policy["max_upload_bytes"],
        "accepted_media_types": list(_ACCEPTED_MEDIA_TYPES),
    }
    return JSONResponse(report, status_code=200 if report["service_ready"] else 503)


#: Design §3.10: literal, not `API_VERSION` -- `service_version.api_version`
#: names the overall contract-suite version (already 1.1 via `conversion_status`),
#: independent of the FastAPI app's own `version=` metadata.
_SETTINGS_API_VERSION = "1.1"


async def _settings(request: Request) -> Response:
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, "read_status")
    if isinstance(principal, Response):
        return principal
    return JSONResponse(_settings_payload(dependencies))


def _settings_payload(dependencies: ApiDependencies) -> dict[str, Any]:
    config = dependencies.models_config
    if config is None:
        # No models config at all: nothing to resolve, mirrors `_models_payload`'s
        # own no-config fallback. `"auto"` is the reserved sentinel for "resolve
        # automatically" and never collides with a real variant_id.
        default_variant = {
            "configured": "auto",
            "override": None,
            "effective": None,
            "source": None,
        }
    else:
        override = dependencies.job_store.get_setting("default_variant")
        state = dependencies.job_store.model_state()
        model_state = (
            {"total_vram_mib": state[2], "compute_capability": state[3]}
            if state is not None
            else None
        )
        effective, source = resolve_default_variant(
            config, user_default=override, model_state=model_state
        )
        default_variant = {
            "configured": config["default_variant"],
            "override": override,
            "effective": effective,
            "source": source,
        }
    extras = dependencies.job_store.readiness_extras()
    sandbox_rootful = extras[0] if extras is not None else None
    return validate_contract(
        {
            "schema_version": "1.0",
            "service_version": {
                "git_sha": dependencies.web.git_sha,
                "api_version": _SETTINGS_API_VERSION,
            },
            "session": {
                "idle_seconds": dependencies.web.session_idle_seconds,
                "max_seconds": dependencies.web.session_max_seconds,
            },
            "default_variant": default_variant,
            "sandbox": {"rootful": sandbox_rootful},
        },
        "settings_response",
    )


async def _models(request: Request) -> Response:
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, "read_status")
    if isinstance(principal, Response):
        return principal
    return _conditional_json(request, _models_payload(dependencies))


def _etag_for(payload: Mapping[str, Any]) -> str:
    """Design §3.0: `"sha256:<hex>"` of the payload's canonical JSON
    (`json.dumps(sort_keys=True, separators=(",", ":"))`)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f'"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"'


def _conditional_json(request: Request, payload: dict[str, Any]) -> Response:
    """Design §3.0: `If-None-Match` matching the freshly computed `ETag` gets
    a bodyless `304` (still carrying `ETag`); `Cache-Control` is added by
    `CacheControlMiddleware` from the `Content-Type` set here."""
    etag = _etag_for(payload)
    if request.headers.get("if-none-match") == etag:
        return Response(
            status_code=304, headers={"ETag": etag, "Content-Type": "application/json"}
        )
    return JSONResponse(payload, headers={"ETag": etag})


def _models_payload(dependencies: ApiDependencies) -> dict[str, Any]:
    config = dependencies.models_config
    if config is None:
        return validate_contract(
            {
                "schema_version": "1.2",
                "default_variant": None,
                "default_variant_source": None,
                "resident_variant": None,
                "device": None,
                "variants": [],
            },
            "models_response",
        )
    jobs = dependencies.job_store
    state = jobs.model_state()
    resident_variant, free_vram_mib, total_vram_mib, compute_capability = (
        state if state is not None else (None, None, None, None)
    )
    model_state = (
        {"total_vram_mib": total_vram_mib, "compute_capability": compute_capability}
        if state is not None
        else None
    )
    default_variant, default_variant_source = resolve_default_variant(
        config, user_default=jobs.get_setting("default_variant"), model_state=model_state
    )
    _sandbox_rootful, available_engines = jobs.readiness_extras() or (None, None)
    device = None
    if state is not None:
        observed_at = jobs.readiness_status()[3]
        device = {
            "total_vram_mib": total_vram_mib,
            "free_vram_mib": free_vram_mib,
            "compute_capability": list(compute_capability),
            "observed_at": _iso8601_utc(observed_at),
        }
    variants = [
        _model_variant_entry(
            variant,
            resident=variant["variant_id"] == resident_variant,
            total_vram_mib=total_vram_mib,
            compute_capability=compute_capability,
            cache_dir=dependencies.models_cache_dir,
            default_variant=default_variant,
            available_engines=available_engines,
            jobs=jobs,
        )
        for variant in config["variants"]
    ]
    return validate_contract(
        {
            "schema_version": "1.2",
            "default_variant": default_variant,
            "default_variant_source": default_variant_source,
            "resident_variant": resident_variant,
            "device": device,
            "variants": variants,
        },
        "models_response",
    )


def _variant_active_command(jobs: JobStore, variant_id: str) -> dict[str, Any] | None:
    """The one active (`queued`/`running`) command for `variant_id`, if any --
    at most one can exist across kinds (`enqueue_model_command`'s conflict
    guard), so checking each kind in turn is exhaustive."""
    for kind in ("download", "load", "delete"):
        active = jobs.active_model_command(variant_id, kind)
        if active is not None:
            return active
    return None


def _variant_last_command(jobs: JobStore, variant_id: str) -> dict[str, Any] | None:
    """The most recent terminal command relevant to `download.state` for
    `variant_id`. A `load` command run on a not-yet-downloaded variant
    performs the download itself under the `load` kind (`ModelCommandExecutor
    ._run_load`), so `load` is checked first -- it is the more specific,
    more recent intent whenever both a past `download` and a past `load`
    command exist for the same variant."""
    for kind in ("load", "download"):
        last = jobs.last_model_command(variant_id, kind)
        if last is not None:
            return last
    return None


def _engine_availability(
    engine: str, available_engines: tuple[str, ...] | None
) -> tuple[bool, str | None]:
    """Design §17.5: the native engine is always available -- it never needs
    an import or a proven-parity gate. Any other engine is available only
    once a worker has published it in `available_engines`; otherwise it is
    reported as `worker_not_published`. Shared by `/v1/models` (T40) and the
    submit/load engine gates (T42)."""
    if engine == "native":
        return True, None
    if available_engines is not None and engine in available_engines:
        return True, None
    return False, "worker_not_published"


def _find_variant(
    config: Mapping[str, Any] | None, variant_id: str
) -> Mapping[str, Any] | None:
    """The named variant from the models config, or `None` if unconfigured
    or unknown -- shared lookup for the model command routes (T42)."""
    if config is None:
        return None
    for variant in config["variants"]:
        if variant["variant_id"] == variant_id:
            return variant
    return None


def _not_published_problem(variant: Mapping[str, Any]) -> Response | None:
    """models_response 1.2: a listed variant that cannot be downloaded,
    loaded or selected yet -- `422 MODEL_VARIANT_NOT_PUBLISHED` for a known
    precision whose HF repo is not public (DEC-050), `422
    MODEL_VARIANT_PLANNED` for a model that does not exist yet. `None` for a
    published variant. Checked before every other variant gate."""
    availability = variant.get("availability", PUBLISHED)
    if availability == "unpublished":
        return _problem(
            "MODEL_VARIANT_NOT_PUBLISHED", "Model variant is not published yet", 422
        )
    if availability == "planned":
        return _problem("MODEL_VARIANT_PLANNED", "Model variant does not exist yet", 422)
    return None


def _model_variant_entry(
    variant: Mapping[str, Any],
    *,
    resident: bool,
    total_vram_mib: float | None,
    compute_capability: tuple[int, int] | None,
    cache_dir: Path | None,
    default_variant: str | None,
    available_engines: tuple[str, ...] | None,
    jobs: JobStore,
) -> dict[str, Any]:
    availability = variant.get("availability", PUBLISHED)
    # A planned model has no precision and no measured tier: nothing to
    # estimate from. An unpublished one is a known precision of the same
    # checkpoint, so its estimate and fit are as real as a published one's.
    planned = availability == "planned"
    estimate = (
        None if planned else estimated_peak_vram_mib(variant["precision"], variant["kv_cache"])
    )
    # "Fits" is judged against the card's TOTAL capacity, never its
    # currently-free VRAM: free VRAM is measured AFTER a model is already
    # resident, so comparing a variant's peak against "what's left" would
    # make even the resident variant itself look like it doesn't fit (the
    # bug this fixes). The resident variant always counts as fitting
    # regardless -- it is, by definition, already loaded and running.
    if resident:
        fits: bool | None = True
        fits_reason: str | None = None
    elif total_vram_mib is None or planned:
        fits, fits_reason = None, None
    else:
        fits_reason = variant_fit_reason(
            variant["precision"],
            variant["kv_cache"],
            free_vram_mib=total_vram_mib,
            compute_capability=compute_capability,
        )
        fits = fits_reason is None
    source = variant.get("source")
    source_kind: str | None = "hf" if source else "local"
    if availability != PUBLISHED:
        # Nothing to download yet: an unpublished variant will come from
        # Hugging Face once its repo is public; a planned one has no origin.
        source_kind = None if planned else "hf"
        downloaded, download_size_bytes, download = False, None, None
    elif source_kind == "local" or cache_dir is None:
        downloaded, download_size_bytes = ("bundle" in variant), None
        download = None
    else:
        downloaded, download_size_bytes = is_variant_downloaded(variant, cache_dir=cache_dir)
        variant_id = variant["variant_id"]
        last_command = _variant_last_command(jobs, variant_id)
        download = variant_download_state(
            variant,
            cache_dir=cache_dir,
            active_command=_variant_active_command(jobs, variant_id),
            last_command=last_command,
        )
        # HF creates a snapshot directory before its weights are complete.
        # An active download/verify/load remains progress, never "ready".
        cached_load_failure = (
            download["state"] == "failed"
            and last_command is not None
            and last_command.get("kind") == "load"
            and last_command.get("error_code") in CACHED_LOAD_FAILURE_CODES
        )
        downloaded = downloaded and (download["state"] == "ready" or cached_load_failure)
    engine = variant["engine"]
    engine_available, engine_unavailable_reason = _engine_availability(engine, available_engines)
    return {
        "variant_id": variant["variant_id"],
        "label": variant["label"],
        "precision": variant["precision"],
        "kv_cache": variant["kv_cache"],
        "bundle_manifest_sha256": source["bundle_manifest_sha256"] if source else None,
        "estimated_peak_vram_mib": estimate,
        "fits": fits,
        "resident": resident,
        "source_kind": source_kind,
        "downloaded": downloaded,
        "download_size_bytes": download_size_bytes,
        "is_default": variant["variant_id"] == default_variant,
        "fits_reason": fits_reason,
        "engine": engine,
        "engine_available": engine_available,
        "engine_unavailable_reason": engine_unavailable_reason,
        "download": download,
        "availability": availability,
    }


async def _models_default_put(request: Request) -> Response:
    """Design §3.9: `PUT /v1/models/default {variant_id}` -- `null` clears
    the user override (falls back to config/auto); an unknown variant is a
    422; never touches `runtime_policy` (`policy_hash` is unaffected)."""
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, "submit_diagnostic")
    if isinstance(principal, Response):
        return principal
    config = dependencies.models_config
    if config is None:
        return _problem("UNKNOWN_MODEL_VARIANT", "No models are configured", 422)
    body = await _read_json_object_body(request)
    if isinstance(body, Response):
        return body
    if set(body) != {"variant_id"}:
        return _problem("REQUEST_INVALID", "Request body is invalid", 400)
    variant_id = body["variant_id"]
    if variant_id is None:
        dependencies.job_store.delete_setting("default_variant")
    else:
        variant = _find_variant(config, variant_id) if isinstance(variant_id, str) else None
        if variant is None:
            return _problem(
                "UNKNOWN_MODEL_VARIANT", "Requested model variant is unknown", 422
            )
        not_published = _not_published_problem(variant)
        if not_published is not None:
            return not_published
        dependencies.job_store.set_setting(
            "default_variant", variant_id, now=dependencies.clock()
        )
    return JSONResponse(_models_payload(dependencies))


def _model_command_accepted_payload(record: ModelCommandRecord) -> dict[str, Any]:
    """Design §3.9: `ModelCommandAccepted`, shared by all three model command
    routes and by the idempotent 202 they return for an already-active
    command of the same kind."""
    return validate_contract(
        {
            "schema_version": "1.0",
            "command_id": record.command_id,
            "kind": record.kind,
            "variant_id": record.variant_id,
            "state": record.state,
            "phase": record.phase,
            "bytes_done": record.bytes_done,
            "bytes_total": record.bytes_total,
            "error_code": record.error_code,
            "requested_at": _iso8601_utc_ms(record.requested_at),
            "started_at": (
                None if record.started_at is None else _iso8601_utc_ms(record.started_at)
            ),
            "finished_at": (
                None if record.finished_at is None else _iso8601_utc_ms(record.finished_at)
            ),
        },
        "model_command_accepted",
    )


def _enqueue_model_command(
    dependencies: ApiDependencies, principal: Principal, *, kind: str, variant_id: str
) -> Response:
    """Design §3.9: enqueue is idempotent for the same `kind` (returns the
    existing active command, still 202); a different active kind on the same
    variant is `409 MODEL_COMMAND_CONFLICT` (`JobStore.enqueue_model_command`'s
    own conflict guard, §4.2)."""
    try:
        record, _created = dependencies.job_store.enqueue_model_command(
            kind=kind,
            variant_id=variant_id,
            requested_by=principal.principal_id,
            now=dependencies.clock(),
        )
    except ModelCommandConflict:
        return _problem(
            "MODEL_COMMAND_CONFLICT", "A different model command is already active", 409
        )
    return JSONResponse(_model_command_accepted_payload(record), status_code=202)


async def _models_download_post(variant_id: str, request: Request) -> Response:
    """Design §3.9: `POST /v1/models/{variant_id}/download`. A local `bundle`
    variant has nothing to download -- `422 MODEL_VARIANT_NOT_DOWNLOADABLE`."""
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, "submit_diagnostic")
    if isinstance(principal, Response):
        return principal
    variant = _find_variant(dependencies.models_config, variant_id)
    if variant is None:
        return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
    not_published = _not_published_problem(variant)
    if not_published is not None:
        return not_published
    if "bundle" in variant:
        return _problem(
            "MODEL_VARIANT_NOT_DOWNLOADABLE", "Variant has nothing to download", 422
        )
    return _enqueue_model_command(dependencies, principal, kind="download", variant_id=variant_id)


async def _models_download_delete(variant_id: str, request: Request) -> Response:
    """Design §3.9: `DELETE /v1/models/{variant_id}/download`. The resident
    variant is refused (`409 MODEL_COMMAND_CONFLICT`, `ModelCommandExecutor
    ._run_delete`'s own second safety net mirrors this at the worker); a
    local `bundle` variant has no snapshot to delete (`422
    MODEL_VARIANT_NOT_DOWNLOADABLE`, same reason as `download`)."""
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, "submit_diagnostic")
    if isinstance(principal, Response):
        return principal
    variant = _find_variant(dependencies.models_config, variant_id)
    if variant is None:
        return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
    not_published = _not_published_problem(variant)
    if not_published is not None:
        return not_published
    if "bundle" in variant:
        return _problem(
            "MODEL_VARIANT_NOT_DOWNLOADABLE", "Variant has nothing to download", 422
        )
    state = dependencies.job_store.model_state()
    resident_variant = state[0] if state is not None else None
    if variant_id == resident_variant:
        return _problem(
            "MODEL_COMMAND_CONFLICT", "Variant is resident and cannot be deleted", 409
        )
    return _enqueue_model_command(dependencies, principal, kind="delete", variant_id=variant_id)


async def _models_load_post(variant_id: str, request: Request) -> Response:
    """Design §3.9/§17.5: `POST /v1/models/{variant_id}/load`. Same
    fit/engine gates as `_resolve_model_variant`'s submit-time checks: a
    variant too big for the card is `422 MODEL_VARIANT_DOES_NOT_FIT`, an
    engine this worker cannot (yet) run is `422 MODEL_ENGINE_UNAVAILABLE`."""
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, "submit_diagnostic")
    if isinstance(principal, Response):
        return principal
    variant = _find_variant(dependencies.models_config, variant_id)
    if variant is None:
        return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
    not_published = _not_published_problem(variant)
    if not_published is not None:
        return not_published
    state = dependencies.job_store.model_state()
    if state is not None:
        resident_variant, _free_vram_mib, total_vram_mib, compute_capability = state
        fits = variant_id == resident_variant or variant_fits(
            variant["precision"],
            variant["kv_cache"],
            free_vram_mib=total_vram_mib,
            compute_capability=compute_capability,
        )
        if not fits:
            return _problem(
                "MODEL_VARIANT_DOES_NOT_FIT", "Requested model variant does not fit", 422
            )
    _sandbox_rootful, available_engines = dependencies.job_store.readiness_extras() or (None, None)
    engine_available, _reason = _engine_availability(variant["engine"], available_engines)
    if not engine_available:
        return _problem(
            "MODEL_ENGINE_UNAVAILABLE", "Requested model engine is not available", 422
        )
    return _enqueue_model_command(dependencies, principal, kind="load", variant_id=variant_id)


async def _submit(request: Request) -> Response:
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, "submit_diagnostic")
    if isinstance(principal, Response):
        return principal
    invalid = _validate_submission_headers(request)
    if invalid is not None:
        return invalid
    output = request.headers["x-unflatten-output"]
    if output == "docx":
        return _problem(
            "CAPABILITY_NOT_READY",
            "Requested output is not ready",
            503,
            blockers=list(current_docx_requirements()),
        )
    if output not in ACCEPTED_OUTPUTS or request.headers["x-unflatten-profile"] != "word-core-v0":
        return _problem("REQUEST_INVALID", "Request headers are invalid", 400)
    if output == "diagnostic_docx" and not _resolve_diagnostic_docx_ready(dependencies):
        return _problem("CAPABILITY_NOT_READY", "Requested output is not ready", 503)
    length_problem = _validate_content_length(request, dependencies.runtime_policy)
    if length_problem is not None:
        return length_problem
    model_variant = _resolve_model_variant(request, dependencies)
    if isinstance(model_variant, Response):
        return model_variant
    label = _decode_label_header(request)
    if isinstance(label, Response):
        return label
    return await _admit_submission(request, principal, dependencies, output, model_variant, label)


def _decode_label_header(request: Request) -> str | None | Response:
    """DEC-052 design §3.2: `X-Unflatten-Label` is percent-encoded UTF-8
    (`urllib.parse.unquote(..., errors="strict")`), then NFC-normalized and
    stripped; 1-120 code points, and neither a filesystem-hostile character
    nor a control/format/separator Unicode category (so the decoded label is
    safe to reuse verbatim as a download filename)."""
    values = request.headers.getlist("x-unflatten-label")
    if not values:
        return None
    try:
        decoded = urllib.parse.unquote(values[0], encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return _problem("REQUEST_INVALID", "Label is invalid", 400)
    normalized = unicodedata.normalize("NFC", decoded).strip()
    if not (1 <= len(normalized) <= _LABEL_MAX_CODE_POINTS):
        return _problem("REQUEST_INVALID", "Label is invalid", 400)
    for character in normalized:
        if (
            character in _FORBIDDEN_LABEL_CHARS
            or unicodedata.category(character) in _FORBIDDEN_LABEL_CATEGORIES
        ):
            return _problem("REQUEST_INVALID", "Label is invalid", 400)
    return normalized


def _resolve_model_variant(
    request: Request, dependencies: ApiDependencies
) -> str | None | Response:
    """DEC-044 Dodatok D §1: resolve the optional `X-Unflatten-Model` header
    against the configured variants, or `None` when no models config is set
    (today's single-bundle behaviour, byte-identical). Design §3.8: absent
    the header, the effective default is resolved through the same
    `resolve_default_variant` `/v1/models` and `/v1/settings` use -- a
    `config.default_variant == "auto"` with nothing measured yet, or nothing
    fitting, refuses with `MODEL_VARIANT_DOES_NOT_FIT` rather than looking up
    the literal `"auto"` sentinel as if it were a variant id."""
    config = dependencies.models_config
    if config is None:
        return None
    variants = {variant["variant_id"]: variant for variant in config["variants"]}
    state = dependencies.job_store.model_state()
    headers = request.headers.getlist("x-unflatten-model")
    if headers:
        variant_id = headers[0]
    else:
        model_state = (
            {"total_vram_mib": state[2], "compute_capability": state[3]}
            if state is not None
            else None
        )
        variant_id, _source = resolve_default_variant(
            config,
            user_default=dependencies.job_store.get_setting("default_variant"),
            model_state=model_state,
        )
        if variant_id is None:
            return _problem("MODEL_VARIANT_DOES_NOT_FIT", "No variant fits", 422)
    variant = variants.get(variant_id)
    if variant is None:
        return _problem("UNKNOWN_MODEL_VARIANT", "Requested model variant is unknown", 422)
    not_published = _not_published_problem(variant)
    if not_published is not None:
        return not_published
    # Design §17.5: a variant whose engine this worker cannot (yet) run is
    # refused at submission, the same gate `load` uses -- `download` is the
    # only model command exempt (bytes are engine-independent).
    _sandbox_rootful, available_engines = dependencies.job_store.readiness_extras() or (None, None)
    engine_available, _reason = _engine_availability(variant["engine"], available_engines)
    if not engine_available:
        return _problem(
            "MODEL_ENGINE_UNAVAILABLE", "Requested model engine is not available", 422
        )
    if state is not None:
        resident_variant, _free_vram_mib, total_vram_mib, compute_capability = state
        # Same fix as `/v1/models`: judge fit against total VRAM, and the
        # already-resident variant always fits -- switching TO it is a
        # no-op, never a refusal.
        fits = variant_id == resident_variant or variant_fits(
            variant["precision"],
            variant["kv_cache"],
            free_vram_mib=total_vram_mib,
            compute_capability=compute_capability,
        )
        if not fits:
            return _problem(
                "MODEL_VARIANT_DOES_NOT_FIT", "Requested model variant does not fit", 422
            )
    return variant_id


async def _status(job_id: str, request: Request) -> Response:
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, "read_status")
    if isinstance(principal, Response):
        return principal
    job = dependencies.job_store.get_job(job_id, principal.principal_id)
    if job is None:
        return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
    return _conditional_json(request, _status_payload(job, dependencies.job_store))


#: Design §3.5: `mode=cancel` is the default, byte-identical to a bare
#: `DELETE`; any other value is a `400`.
_DELETE_MODES = frozenset({"cancel", "purge"})


async def _cancel(job_id: str, request: Request) -> Response:
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, "cancel_own_job")
    if isinstance(principal, Response):
        return principal
    mode = request.query_params.get("mode", "cancel")
    if mode not in _DELETE_MODES:
        return _problem("REQUEST_INVALID", "mode is invalid", 400)
    if mode == "purge":
        return _purge_job(job_id, principal, dependencies)
    if dependencies.job_store.get_job(job_id, principal.principal_id) is None:
        return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
    try:
        job = dependencies.job_store.request_cancel(
            job_id, principal.principal_id, now=dependencies.clock()
        )
    except JobTerminal:
        return _problem("JOB_TERMINAL", "Job is already terminal", 409)
    return JSONResponse(_status_payload(job, dependencies.job_store))


def _purge_job(job_id: str, principal: Principal, dependencies: ApiDependencies) -> Response:
    """Design §3.5: delete a terminal job's row, artifact bytes, and any
    staged (never-completed) uploads for it. `JobStore.purge_job` owns the
    transactional row+artifact delete; the staged-upload sweep below is only
    reachable after that succeeds, so a job that never leaves `JobActive`
    keeps its in-flight source PDF untouched."""

    def delete_bytes(record: ArtifactRecord) -> None:
        dependencies.artifact_store.delete(
            principal_id=principal.principal_id,
            job_id=job_id,
            artifact=ArtifactRef(
                ref=record.ref,
                sha256=f"sha256:{record.sha256.removeprefix('sha256:')}",
                size_bytes=record.size_bytes,
                mime_type=record.mime_type,
            ),
        )

    try:
        from unflatten.service.work_cleanup import delete_work_execution

        dependencies.job_store.purge_job(
            job_id, principal.principal_id, delete_bytes=delete_bytes,
            delete_work=(
                lambda execution: delete_work_execution(dependencies.work_root, execution)
                if dependencies.work_root is not None else None
            ),
        )
    except JobNotFound:
        return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
    except JobActive:
        return _problem("JOB_ACTIVE", "Job is still active", 409)
    except (ArtifactStoreError, OSError):
        return _problem("INTERNAL_ERROR", "Job data could not be deleted", 500)
    for staged in dependencies.artifact_store.staged_artifacts_for_retention():
        if staged.job_id == job_id and staged.principal_id == principal.principal_id:
            dependencies.artifact_store.discard_staged(staged.artifact)
    return Response(status_code=204)


async def _list_jobs(request: Request) -> Response:
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, "read_status")
    if isinstance(principal, Response):
        return principal
    states = request.query_params.getlist("state")
    if states and any(state not in JOB_STATES for state in states):
        return _problem("REQUEST_INVALID", "state filter is invalid", 400)
    limit = _parse_list_jobs_limit(request)
    if isinstance(limit, Response):
        return limit
    try:
        jobs, next_cursor = dependencies.job_store.list_jobs(
            principal_id=principal.principal_id,
            states=states or None,
            cursor=request.query_params.get("cursor"),
            limit=limit,
        )
    except JobStoreInvalid:
        return _problem("REQUEST_INVALID", "cursor is invalid", 400)
    return _conditional_json(request, _job_list_payload(jobs, next_cursor))


def _parse_list_jobs_limit(request: Request) -> int | Response:
    raw = request.query_params.get("limit")
    if raw is None:
        return 25
    try:
        limit = int(raw)
    except ValueError:
        return _problem("REQUEST_INVALID", "limit is invalid", 400)
    if not (1 <= limit <= 100):
        return _problem("REQUEST_INVALID", "limit is invalid", 400)
    return limit


def _download_role(role: str) -> Callable[[str, Request], Any]:
    """The legacy/whole-document route: today's single artifact, or a
    multi-page job's page 0 -- 'existing single-page routes keep working
    for page 0' (DEC-044 Dodatok D §1)."""
    return _download_handler(role, allowed_page_indices=lambda _job: (-1, 0))


def _download_role_page(role: str) -> Callable[[str, int, Request], Any]:
    """DEC-044 Dodatok D §1 (B1): the page-scoped download route."""
    return _download_handler(role, allowed_page_indices=lambda page_index: (page_index,))


def _download_handler(
    role: str,
    *,
    allowed_page_indices: Callable[[Any], tuple[int, ...]],
) -> Callable[..., Any]:
    capability, media_type, filename = _DOWNLOAD_ROLES[role]

    async def handler(job_id: str, request: Request, page_index: int | None = None) -> Response:
        dependencies = _dependencies(request)
        principal = _authorize(request, dependencies, capability)
        if isinstance(principal, Response):
            return principal
        job = dependencies.job_store.get_job(job_id, principal.principal_id)
        if job is None:
            return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
        if job.state == "expired":
            return _problem("ARTIFACT_EXPIRED", "Artifact has expired", 410)
        wanted = allowed_page_indices(page_index)
        records = [
            item for item in dependencies.job_store.list_artifacts(job_id)
            if item.role == role and item.page_index in wanted
        ]
        if len(records) != 1:
            return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
        if job.expires_at is None:
            return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
        if dependencies.clock() >= job.expires_at:
            return _problem("ARTIFACT_EXPIRED", "Artifact has expired", 410)
        # Design §3.7: expiry and ownership (the `get_job`/`list_artifacts`
        # checks above) are resolved *before* any `If-None-Match` comparison
        # -- an expired or foreign job must never earn a 304.
        if role in _INLINE_PREVIEW_ROLES:
            return _preview_response(
                request, dependencies, principal, job, records[0], role=role,
                media_type=media_type,
            )
        return _verified_artifact_response(
            dependencies, principal, job, records[0], role=role,
            media_type=media_type, filename=filename,
        )

    return handler


def _preview_response(
    request: Request,
    dependencies: ApiDependencies,
    principal: Principal,
    job: JobRecord,
    record: ArtifactRecord,
    *,
    role: str,
    media_type: str,
) -> Response:
    """Design §3.7: PNG previews are `inline` with an `ETag` derived from the
    published artifact's hash; `If-None-Match` gets a bodyless `304` -- only
    reachable once the caller above already holds an owned, unexpired
    artifact record."""
    etag = f'"sha256:{record.sha256.removeprefix("sha256:")}"'
    if request.headers.get("if-none-match") == etag:
        return Response(
            status_code=304,
            headers={"ETag": etag, "Cache-Control": "private, no-cache"},
        )
    artifact = ArtifactRef(
        ref=record.ref,
        sha256=f"sha256:{record.sha256.removeprefix('sha256:')}",
        size_bytes=record.size_bytes,
        mime_type=record.mime_type,
    )
    try:
        content = dependencies.artifact_store.read_downloadable(
            principal_id=principal.principal_id,
            job_id=job.job_id,
            role=role,
            artifact=artifact,
            expires_at=job.expires_at,
            now=dependencies.clock(),
        )
    except ArtifactStoreError as exc:
        return _artifact_problem(exc)
    filename = f"{role}-{record.page_index:03d}.png"
    return Response(
        content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "ETag": etag,
            "Cache-Control": "private, no-cache",
        },
    )


def _verified_artifact_response(
    dependencies: ApiDependencies,
    principal: Principal,
    job: JobRecord,
    record: ArtifactRecord,
    *,
    role: str,
    media_type: str,
    filename: str,
) -> Response:
    artifact = ArtifactRef(
        ref=record.ref,
        sha256=f"sha256:{record.sha256.removeprefix('sha256:')}",
        size_bytes=record.size_bytes,
        mime_type=record.mime_type,
    )
    try:
        content = dependencies.artifact_store.read_downloadable(
            principal_id=principal.principal_id,
            job_id=job.job_id,
            role=role,
            artifact=artifact,
            expires_at=job.expires_at,
            now=dependencies.clock(),
        )
    except ArtifactStoreError as exc:
        return _artifact_problem(exc)
    return Response(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def _artifacts_zip(job_id: str, request: Request) -> Response:
    """Design §3.6: `artifacts.zip` -- every downloadable artifact the caller's
    capabilities permit, bundled with a `MANIFEST.json`."""
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, None)
    if isinstance(principal, Response):
        return principal
    allowed_roles = {
        role for role, capability in ROLE_CAPABILITIES.items()
        if capability in principal.capabilities
    }
    if not allowed_roles:
        return _problem("CAPABILITY_DENIED", "Capability denied", 403)
    job = dependencies.job_store.get_job(job_id, principal.principal_id)
    if job is None:
        return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
    if job.state == "expired":
        return _problem("ARTIFACT_EXPIRED", "Artifact has expired", 410)
    if job.state != "succeeded" or job.expires_at is None:
        return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
    if dependencies.clock() >= job.expires_at:
        return _problem("ARTIFACT_EXPIRED", "Artifact has expired", 410)
    # `artifacts_zip.ArtifactZipRecord.sha256` is embedded verbatim in the
    # manifest, which expects the same `"sha256:<hex>"` prefix `ArtifactRef`
    # requires -- `ArtifactRecord.sha256` stores the bare hex digest.
    records = [
        replace(record, sha256=f"sha256:{record.sha256.removeprefix('sha256:')}")
        for record in dependencies.job_store.list_artifacts(job_id)
    ]

    def _read(record: ArtifactRecord) -> bytes:
        artifact = ArtifactRef(
            ref=record.ref,
            sha256=record.sha256,
            size_bytes=record.size_bytes,
            mime_type=record.mime_type,
        )
        return dependencies.artifact_store.read_downloadable(
            principal_id=principal.principal_id,
            job_id=job.job_id,
            role=record.role,
            artifact=artifact,
            expires_at=job.expires_at,
            now=dependencies.clock(),
        )

    try:
        result = build_artifacts_zip(
            job_id=job.job_id,
            generated_at=_iso8601_utc_ms(dependencies.clock()),
            capabilities=principal.capabilities,
            records=records,
            read_downloadable=_read,
        )
    except ArtifactStoreError as exc:
        return _artifact_problem(exc)
    validate_contract(result.manifest, "artifacts_manifest")
    return Response(
        result.content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{job.job_id}.zip"',
            "Cache-Control": "no-store",
        },
    )


def _artifact_problem(error: ArtifactStoreError) -> Response:
    if error.code == "ARTIFACT_EXPIRED":
        return _problem("ARTIFACT_EXPIRED", "Artifact has expired", 410)
    if error.code in {"ARTIFACT_NOT_FOUND", "ARTIFACT_NOT_DOWNLOADABLE"}:
        return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
    return _problem("INTERNAL_ERROR", "Artifact verification failed", 500)


def _dependencies(request: Request) -> ApiDependencies:
    return request.app.state.dependencies


async def _request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> Response:
    return _problem("REQUEST_INVALID", "Request is invalid", 400)


async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
    """CP4 minor 2: an unmatched `/v1/...` path must return a
    `application/problem+json` 404 with a registry code, not FastAPI's
    default `{"detail": "Not Found"}`. Every other path (including SPA
    client routes) keeps Starlette's default handling untouched."""
    if exc.status_code == 404 and request.url.path.startswith("/v1/"):
        return _problem("RESOURCE_NOT_FOUND", "Resource not found", 404)
    return await _default_http_exception_handler(request, exc)


def _web_auth(dependencies: ApiDependencies) -> WebAuthFile:
    return WebAuthFile(dependencies.web.web_auth_path)


def _web_sessions(dependencies: ApiDependencies) -> WebSessionStore:
    return WebSessionStore(
        dependencies.job_store,
        idle_seconds=dependencies.web.session_idle_seconds,
        max_seconds=dependencies.web.session_max_seconds,
    )


def _unauthenticated_401(*, clear_cookie: bool = False, www_authenticate: bool = True) -> Response:
    response = _problem("AUTHENTICATION_FAILED", "Authentication failed", 401)
    if clear_cookie:
        # Design §2.4: never `WWW-Authenticate` on the cookie path -- the
        # browser would show its native basic-auth dialog.
        _clear_session_cookie(response)
    elif www_authenticate:
        response.headers["WWW-Authenticate"] = "Bearer"
    return response


def _set_session_cookie(response: Response, token: str, *, max_age: int) -> None:
    response.set_cookie(
        _SESSION_COOKIE, token, max_age=max_age, path="/", httponly=True, samesite="strict",
    )


def _clear_session_cookie(response: Response) -> None:
    _set_session_cookie(response, "", max_age=0)


def _authenticate(
    request: Request, dependencies: ApiDependencies
) -> tuple[Principal, str, WebSessionRecordLike | None] | Response | None:
    """Design §2.4 steps 1-4. Returns `None` when no credential is present at
    all -- only `GET /v1/session` treats that as `authenticated=False` rather
    than a 401; every other caller goes through `_authorize`, which maps it
    to a 401. Never reads the request body."""
    auth_values = request.headers.getlist("authorization")
    cookie = request.cookies.get(_SESSION_COOKIE)
    if auth_values and cookie is not None:
        return _problem("REQUEST_INVALID", "Ambiguous credentials", 400)
    if auth_values:
        try:
            if len(auth_values) != 1:
                raise AuthenticationFailed("exactly one bearer credential is required")
            principal = dependencies.principal_store.authenticate(auth_values[0])
        except AuthenticationFailed:
            return _unauthenticated_401()
        return principal, "bearer", None
    if cookie is not None:
        if request.headers.get("sec-fetch-site") not in _ALLOWED_SEC_FETCH_SITE:
            return _problem("WEB_REQUEST_REJECTED", "Request context is not allowed", 403)
        if (
            request.method not in ("GET", "HEAD")
            and request.headers.get("x-unflatten-client") != "web"
        ):
            return _problem("WEB_REQUEST_REJECTED", "Request context is not allowed", 403)
        session = _web_sessions(dependencies).authenticate(cookie)
        if session is None:
            return _unauthenticated_401(clear_cookie=True)
        principal = dependencies.principal_store.principal_by_id(session.principal_id)
        if principal is None:
            return _unauthenticated_401(clear_cookie=True)
        _web_sessions(dependencies).touch(session)
        return principal, "session", session
    return None


def _authorize(
    request: Request, dependencies: ApiDependencies, capability: str | None
) -> Principal | Response:
    authenticated = _authenticate(request, dependencies)
    if isinstance(authenticated, Response):
        return authenticated
    if authenticated is None:
        return _unauthenticated_401()
    principal, auth_kind, _session = authenticated
    request.state.auth_kind = auth_kind
    if capability is not None:
        try:
            require_capability(principal, capability)
        except CapabilityDenied:
            return _problem("CAPABILITY_DENIED", "Capability denied", 403)
    return principal


def _iso8601_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso8601_utc_ms(timestamp: float) -> str:
    """DEC-052 design §3.3: `conversion_status` 1.1's timestamps carry
    millisecond precision, unlike `session_info.expires_at` above."""
    return (
        datetime.fromtimestamp(timestamp, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _session_info_payload(
    *,
    authenticated: bool,
    setup_required: bool,
    setup_tokenless: bool,
    principal: Principal | None,
    auth_kind: str | None,
    expires_at: str | None,
) -> dict[str, Any]:
    return validate_contract(
        {
            "schema_version": "1.1",
            "authenticated": authenticated,
            "setup_required": setup_required,
            "setup_tokenless": setup_tokenless,
            "principal_id": principal.principal_id if principal is not None else None,
            "capabilities": sorted(principal.capabilities) if principal is not None else None,
            "auth_kind": auth_kind,
            "expires_at": expires_at,
        },
        "session_info",
    )


def _web_request_context_ok(request: Request) -> bool:
    """Design §2.3/§11 bullet 17: `POST /v1/setup`, `POST /v1/session` and
    `PUT /v1/setup/password` accept a request only from this exact context,
    checked before the rate limiter and before any body is read."""
    content_types = request.headers.getlist("content-type")
    if len(content_types) != 1:
        return False
    media_type = content_types[0].split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        return False
    if request.headers.get("x-unflatten-client") != "web":
        return False
    return request.headers.get("sec-fetch-site") in _ALLOWED_SEC_FETCH_SITE


async def _read_json_object_body(request: Request) -> dict[str, Any] | Response:
    try:
        parsed = await request.json()
    except Exception:
        # Untrusted-input boundary, same as `_collect_page_images`: the JSON
        # decoder can raise several distinct exception types for a malformed body.
        return _problem("REQUEST_INVALID", "Request body is invalid", 400)
    if not isinstance(parsed, dict):
        return _problem("REQUEST_INVALID", "Request body is invalid", 400)
    return parsed


def _web_auth_read_or_problem(web_auth: WebAuthFile) -> dict[str, Any] | Response:
    """Design §11 bullet 1: a corrupt or insecure `web-auth.json` (bad mode,
    bad owner, malformed JSON) is a service-side problem, never an
    authentication failure -- it must not feed the login rate limiter, and
    the response must never leak the file's path."""
    try:
        return web_auth.read()
    except WebAuthInvalid:
        return _problem("INTERNAL_ERROR", "Web authentication is unavailable", 500)


def _verify_web_auth_password(web_auth_payload: dict[str, Any], password: str) -> bool:
    """Design §11 bullet 1: a password outside the 12-256 code point shape
    rule is an authentication failure like any other wrong password, not a
    service error -- `verify_password` raises `WebAuthInvalid` for that case."""
    try:
        return verify_password(password, web_auth_payload["scrypt"])
    except WebAuthInvalid:
        return False


def _login_failure(dependencies: ApiDependencies, response: Response) -> Response:
    """Design §2.6: the rate limiter counts only failures (401s) on the
    login/setup/password-change routes -- not e.g. a 400 ambiguous-credentials
    rejection, which is a malformed request rather than a wrong credential."""
    if response.status_code == 401:
        dependencies.login_rate_limiter.record_failure()
    return response


def _tokenless_setup_principal(dependencies: ApiDependencies) -> Principal | None:
    """Owner 2026-09-29 (DEC-052 addendum): the principal a bearer-less
    `POST /v1/setup` binds to -- only with `setup_tokenless` on and exactly
    one principal configured. One-shot by construction: callers still refuse
    once `web-auth.json` exists (409)."""
    if not dependencies.web.setup_tokenless:
        return None
    return dependencies.principal_store.sole_principal()


async def _session_get(request: Request) -> Response:
    dependencies = _dependencies(request)
    setup_required = not _web_auth(dependencies).exists()
    setup_tokenless = setup_required and _tokenless_setup_principal(dependencies) is not None
    authenticated = _authenticate(request, dependencies)
    if isinstance(authenticated, Response):
        return authenticated
    if authenticated is None:
        payload = _session_info_payload(
            authenticated=False, setup_required=setup_required,
            setup_tokenless=setup_tokenless, principal=None, auth_kind=None, expires_at=None,
        )
        return JSONResponse(payload)
    principal, auth_kind, session = authenticated
    expires_at = _iso8601_utc(session.expires_at) if session is not None else None
    payload = _session_info_payload(
        authenticated=True, setup_required=setup_required, setup_tokenless=setup_tokenless,
        principal=principal, auth_kind=auth_kind, expires_at=expires_at,
    )
    return JSONResponse(payload)


async def _session_post(request: Request) -> Response:
    dependencies = _dependencies(request)
    if not _web_request_context_ok(request):
        return _problem("WEB_REQUEST_REJECTED", "Request context is not allowed", 403)
    limiter = dependencies.login_rate_limiter
    try:
        limiter.check()
    except RateLimited as exc:
        return _problem(
            "RATE_LIMITED", "Too many login attempts", 429,
            retryable=True, retry_after_seconds=exc.retry_after_seconds,
        )
    web_auth = _web_auth(dependencies)
    if not web_auth.exists():
        return _problem("SETUP_REQUIRED", "Setup has not been completed", 409)
    body = await _read_json_object_body(request)
    if isinstance(body, Response):
        return body
    password = body.get("password")
    if not isinstance(password, str):
        return _problem("REQUEST_INVALID", "Request body is invalid", 400)
    web_auth_payload = _web_auth_read_or_problem(web_auth)
    if isinstance(web_auth_payload, Response):
        return web_auth_payload
    if not _verify_web_auth_password(web_auth_payload, password):
        # Design §2.4: the cookie/login path never sends `WWW-Authenticate`
        # (a browser would show its native basic-auth dialog); this route
        # never accepts a bearer credential, so there is no bearer hint to give.
        return _login_failure(dependencies, _unauthenticated_401(www_authenticate=False))
    limiter.record_success()
    principal_id = web_auth_payload["principal_id"]
    return _new_session_response(dependencies, principal_id)


async def _setup_post(request: Request) -> Response:
    dependencies = _dependencies(request)
    if not _web_request_context_ok(request):
        return _problem("WEB_REQUEST_REJECTED", "Request context is not allowed", 403)
    limiter = dependencies.login_rate_limiter
    try:
        limiter.check()
    except RateLimited as exc:
        return _problem(
            "RATE_LIMITED", "Too many login attempts", 429,
            retryable=True, retry_after_seconds=exc.retry_after_seconds,
        )
    tokenless_principal = _tokenless_setup_principal(dependencies)
    if tokenless_principal is not None and not request.headers.getlist("authorization"):
        # Owner 2026-09-29: no bearer credential at all on a loopback-only
        # deployment. A present-but-wrong one still takes the 401 path below.
        principal = tokenless_principal
    else:
        authorized = _authorize(request, dependencies, None)
        if isinstance(authorized, Response):
            return _login_failure(dependencies, authorized)
        if request.state.auth_kind != "bearer":
            # Design §2.4: never `WWW-Authenticate` on the cookie path (the
            # branch above already guarantees a non-bearer `auth_kind` here).
            return _login_failure(dependencies, _unauthenticated_401(www_authenticate=False))
        principal = authorized
    web_auth = _web_auth(dependencies)
    if web_auth.exists():
        return _problem("SETUP_ALREADY_COMPLETE", "Setup has already been completed", 409)
    body = await _read_json_object_body(request)
    if isinstance(body, Response):
        return body
    password = body.get("password")
    if not isinstance(password, str):
        return _problem("REQUEST_INVALID", "Request body is invalid", 400)
    try:
        web_auth.write(principal.principal_id, password)
    except WebAuthInvalid:
        return _problem("REQUEST_INVALID", "Request body is invalid", 400)
    limiter.record_success()
    return _new_session_response(dependencies, principal.principal_id)


def _new_session_response(dependencies: ApiDependencies, principal_id: str) -> Response:
    principal = dependencies.principal_store.principal_by_id(principal_id)
    sessions = _web_sessions(dependencies)
    token = sessions.create(principal_id)
    # Read the just-created record back rather than recomputing its absolute
    # expiry here -- `WebSessionStore` owns that arithmetic (and its own
    # wall-clock time source), so this can never drift from what a later
    # `GET /v1/session` reports for the same session.
    session = sessions.authenticate(token)
    expires_at = _iso8601_utc(session.expires_at) if session is not None else None
    payload = _session_info_payload(
        authenticated=True,
        setup_required=False,
        setup_tokenless=False,
        principal=principal,
        auth_kind="session",
        expires_at=expires_at,
    )
    response = JSONResponse(payload, status_code=201)
    _set_session_cookie(response, token, max_age=dependencies.web.session_max_seconds)
    return response


async def _session_delete(request: Request) -> Response:
    dependencies = _dependencies(request)
    principal = _authorize(request, dependencies, None)
    if isinstance(principal, Response):
        return principal
    if request.state.auth_kind != "session":
        return _unauthenticated_401()
    cookie = request.cookies[_SESSION_COOKIE]
    _web_sessions(dependencies).revoke(cookie)
    response = Response(status_code=204)
    _clear_session_cookie(response)
    return response


async def _setup_password_put(request: Request) -> Response:
    dependencies = _dependencies(request)
    if not _web_request_context_ok(request):
        return _problem("WEB_REQUEST_REJECTED", "Request context is not allowed", 403)
    limiter = dependencies.login_rate_limiter
    try:
        limiter.check()
    except RateLimited as exc:
        return _problem(
            "RATE_LIMITED", "Too many login attempts", 429,
            retryable=True, retry_after_seconds=exc.retry_after_seconds,
        )
    web_auth = _web_auth(dependencies)
    if not web_auth.exists():
        return _problem("SETUP_REQUIRED", "Setup has not been completed", 409)
    principal = _authorize(request, dependencies, None)
    if isinstance(principal, Response):
        return _login_failure(dependencies, principal)
    # Design §2.4: `WWW-Authenticate` only hints at a bearer credential on
    # the bearer path -- a session-cookie caller never gets it (§11 bullet 4).
    www_authenticate = request.state.auth_kind == "bearer"
    web_auth_payload = _web_auth_read_or_problem(web_auth)
    if isinstance(web_auth_payload, Response):
        return web_auth_payload
    if principal.principal_id != web_auth_payload["principal_id"]:
        return _login_failure(
            dependencies, _unauthenticated_401(www_authenticate=www_authenticate)
        )
    body = await _read_json_object_body(request)
    if isinstance(body, Response):
        return body
    current_password, new_password = body.get("current_password"), body.get("new_password")
    if not isinstance(current_password, str) or not isinstance(new_password, str):
        return _problem("REQUEST_INVALID", "Request body is invalid", 400)
    if not _verify_web_auth_password(web_auth_payload, current_password):
        return _login_failure(
            dependencies, _unauthenticated_401(www_authenticate=www_authenticate)
        )
    try:
        web_auth.write(principal.principal_id, new_password)
    except WebAuthInvalid:
        return _problem("REQUEST_INVALID", "Request body is invalid", 400)
    limiter.record_success()
    sessions = _web_sessions(dependencies)
    if request.state.auth_kind == "session":
        sessions.revoke_all_except(
            principal.principal_id, keep_token=request.cookies[_SESSION_COOKIE]
        )
    else:
        sessions.revoke_all(principal.principal_id)
    return Response(status_code=204)


def _validate_submission_headers(request: Request) -> Response | None:
    service_headers = {
        name.lower() for name in request.headers if name.lower().startswith("x-unflatten-")
    }
    if not (_REQUIRED_SERVICE_HEADERS <= service_headers <= _KNOWN_SERVICE_HEADERS):
        return _problem("REQUEST_INVALID", "Service headers are invalid", 400)
    required = ("idempotency-key", "x-unflatten-profile", "x-unflatten-output")
    if any(len(request.headers.getlist(name)) != 1 for name in required):
        return _problem("REQUEST_INVALID", "Required headers are invalid", 400)
    if len(request.headers.getlist("x-unflatten-model")) > 1:
        return _problem("REQUEST_INVALID", "Model header is invalid", 400)
    if len(request.headers.getlist("x-unflatten-label")) > 1:
        return _problem("REQUEST_INVALID", "Label header is invalid", 400)
    key = request.headers["idempotency-key"]
    if _IDEMPOTENCY_KEY.fullmatch(key) is None:
        return _problem("REQUEST_INVALID", "Idempotency key is invalid", 400)
    content_types = request.headers.getlist("content-type")
    if len(content_types) != 1:
        return _problem("REQUEST_INVALID", "Content-Type is invalid", 400)
    media_type = content_types[0].split(";", 1)[0].strip().lower()
    if media_type not in _ACCEPTED_MEDIA_TYPES:
        return _problem(
            "UNSUPPORTED_MEDIA_TYPE",
            f"Only {', '.join(_ACCEPTED_MEDIA_TYPES)} is supported",
            415,
        )
    return None


def _validate_content_length(request: Request, policy: Mapping[str, Any]) -> Response | None:
    values = request.headers.getlist("content-length")
    if len(values) > 1:
        return _problem("REQUEST_INVALID", "Content length is invalid", 400)
    if not values:
        return None
    try:
        length = int(values[0])
    except ValueError:
        return _problem("REQUEST_INVALID", "Content length is invalid", 400)
    if length < 0:
        return _problem("REQUEST_INVALID", "Content length is invalid", 400)
    if length > policy["max_upload_bytes"]:
        return _problem("UPLOAD_TOO_LARGE", "Upload exceeds its byte limit", 413)
    return None


async def _admit_submission(
    request: Request,
    principal: Principal,
    dependencies: ApiDependencies,
    output: str,
    model_variant: str | None,
    label: str | None,
) -> Response:
    media_type = request.headers["content-type"].split(";", 1)[0].strip().lower()
    job_id = f"job_{uuid.uuid4().hex}"
    if media_type == "application/pdf":
        return await _admit_pdf_stream(
            request, principal, dependencies, job_id, output, model_variant, label
        )
    collected = await _collect_page_images(request, dependencies)
    if isinstance(collected, Response):
        return collected
    pages, is_raw_pdf = collected
    if is_raw_pdf:
        (pdf_bytes,) = pages
        if pdf_bytes[:5] != b"%PDF-":
            return _problem("PDF_SIGNATURE_INVALID", "PDF signature is invalid", 400)
        return await _stage_pdf_bytes(
            request, principal, dependencies, job_id, output, model_variant, pdf_bytes,
            label, "pdf",
        )
    try:
        built = image_input.build_pdf_from_images(pages)
    except image_input.ImageInputInvalid:
        return _problem("IMAGE_INPUT_INVALID", "An image page could not be processed", 400)
    return await _stage_pdf_bytes(
        request, principal, dependencies, job_id, output, model_variant, built,
        label, "image",
    )


async def _admit_pdf_stream(
    request: Request,
    principal: Principal,
    dependencies: ApiDependencies,
    job_id: str,
    output: str,
    model_variant: str | None,
    label: str | None,
) -> Response:
    limit = dependencies.runtime_policy["max_upload_bytes"]
    first = bytearray()

    async def upload_chunks() -> AsyncIterator[bytes]:
        async for chunk in request.stream():
            if len(first) < 5:
                first.extend(chunk[: 5 - len(first)])
            yield chunk

    try:
        staged = await stage_upload(
            dependencies.artifact_store,
            principal_id=principal.principal_id,
            job_id=job_id,
            execution_id=f"ingress_{uuid.uuid4().hex}",
            chunks=upload_chunks(),
            max_bytes=limit,
        )
    except ArtifactStoreError as exc:
        if exc.code == "ARTIFACT_TOO_LARGE":
            return _problem("UPLOAD_TOO_LARGE", "Upload exceeds its byte limit", 413)
        return _problem("INTERNAL_ERROR", "Upload persistence failed", 500)
    if bytes(first) != b"%PDF-":
        dependencies.artifact_store.discard_staged(staged)
        return _problem("PDF_SIGNATURE_INVALID", "PDF signature is invalid", 400)
    body_digest = bytes.fromhex(staged.sha256.removeprefix("sha256:"))
    return _create_job_response(
        request, principal, dependencies, job_id, body_digest, staged, output, model_variant,
        label, "pdf",
    )


async def _collect_page_images(
    request: Request, dependencies: ApiDependencies
) -> Response | tuple[list[bytes], bool]:
    """DEC-044 Dodatok D §2 (B2): read either a single raw image body or a
    `multipart/form-data` body's ordered `page` parts, returning the raw
    (not yet normalized) page bytes and whether that single part was itself
    a raw PDF (a multipart body whose one `page` part is `application/pdf`
    stays on the exact same PDF path as a raw `application/pdf` body, never
    passed through image normalization)."""
    media_type = request.headers["content-type"].split(";", 1)[0].strip().lower()
    limit = dependencies.runtime_policy["max_upload_bytes"]
    max_pages = dependencies.runtime_policy["max_pdf_pages"]
    received = 0
    too_large = False

    async def bounded_receive():
        nonlocal received, too_large
        message = await request.receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > limit:
                too_large = True
                # Starlette closes any spooled UploadFiles on this exception.
                raise MultiPartException("upload exceeds its byte limit")
        return message

    bounded = Request(request.scope, receive=bounded_receive)
    try:
        if media_type in _ACCEPTED_IMAGE_MEDIA_TYPES:
            return [await bounded.body()], False
        async with bounded.form(max_files=max_pages + 1, max_fields=0) as form:
            if set(form) != {"page"}:
                return _problem("REQUEST_INVALID", "Only 'page' file parts are allowed", 400)
            parts = form.getlist("page")
            content_types = [getattr(part, "content_type", None) for part in parts]
            if len(parts) == 1 and content_types[0] == "application/pdf":
                return [await parts[0].read()], True
            if len(parts) > max_pages:
                return _problem(
                    "TOO_MANY_PAGES", "Request exceeds the page ceiling", 422,
                    max_pages=max_pages,
                )
            if any(ct not in _ACCEPTED_IMAGE_MEDIA_TYPES for ct in content_types):
                return _problem(
                    "UNSUPPORTED_MEDIA_TYPE", "Every 'page' part must be image/png or image/jpeg",
                    415,
                )
            return [await part.read() for part in parts], False
    except Exception:
        # Untrusted stream/parser boundary; never return document bytes or a
        # parser exception message. All body bytes count, including unknown parts.
        if too_large:
            return _problem("UPLOAD_TOO_LARGE", "Upload exceeds its byte limit", 413)
        return _problem("REQUEST_INVALID", "Multipart body is invalid", 400)


async def _stage_pdf_bytes(
    request: Request,
    principal: Principal,
    dependencies: ApiDependencies,
    job_id: str,
    output: str,
    model_variant: str | None,
    pdf_bytes: bytes,
    label: str | None,
    input_kind: str,
) -> Response:
    async def one_chunk() -> AsyncIterator[bytes]:
        yield pdf_bytes

    try:
        staged = await stage_upload(
            dependencies.artifact_store,
            principal_id=principal.principal_id,
            job_id=job_id,
            execution_id=f"ingress_{uuid.uuid4().hex}",
            chunks=one_chunk(),
            max_bytes=dependencies.runtime_policy["max_upload_bytes"],
        )
    except ArtifactStoreError as exc:
        if exc.code == "ARTIFACT_TOO_LARGE":
            return _problem("UPLOAD_TOO_LARGE", "Upload exceeds its byte limit", 413)
        return _problem("INTERNAL_ERROR", "Upload persistence failed", 500)
    body_digest = bytes.fromhex(staged.sha256.removeprefix("sha256:"))
    return _create_job_response(
        request, principal, dependencies, job_id, body_digest, staged, output, model_variant,
        label, input_kind,
    )


def _create_job_response(
    request: Request,
    principal: Principal,
    dependencies: ApiDependencies,
    job_id: str,
    body_digest: bytes,
    staged: ArtifactRef,
    output: str,
    model_variant: str | None,
    label: str | None,
    input_kind: str,
) -> Response:
    key_hash = hashlib.sha256(request.headers["idempotency-key"].encode()).hexdigest()
    request_digest = hashlib.sha256(
        b"word-core-v0\0" + output.encode() + b"\0" + body_digest
    ).hexdigest()
    # T42 (design §3.3): the job's engine is the resolved variant's own
    # `engine` field -- `_resolve_model_variant` already validated the
    # variant exists and its engine is available, so the lookup here can
    # never miss.
    engine = (
        None if model_variant is None
        else _find_variant(dependencies.models_config, model_variant)["engine"]
    )
    try:
        result = dependencies.job_store.create_job(
            job_id=job_id,
            principal_id=principal.principal_id,
            idempotency_key_hash=key_hash,
            request_digest=request_digest,
            requested_output=output,
            policy_hash=dependencies.policy_hash,
            component_lock_hash=dependencies.component_lock_hash,
            now=dependencies.clock(),
            model_variant=model_variant,
            label=label,
            input_kind=input_kind,
            engine=engine,
        )
    except IdempotencyKeyReused:
        dependencies.artifact_store.discard_staged(staged)
        return _problem("IDEMPOTENCY_KEY_REUSED", "Idempotency key was reused", 409)
    except CapacityExceeded:
        dependencies.artifact_store.discard_staged(staged)
        return _problem("QUEUE_CAPACITY_EXCEEDED", "Queue capacity exceeded", 503, retryable=True)
    except Exception:
        dependencies.artifact_store.discard_staged(staged)
        return _problem("INTERNAL_ERROR", "Job admission failed", 500)
    if not result.created:
        dependencies.artifact_store.discard_staged(staged)
    payload = _submission_payload(result.job.job_id, result.job.requested_output)
    return JSONResponse(payload, status_code=202, headers={"Location": payload["status_url"]})


def _submission_payload(job_id: str, requested_output: str) -> dict[str, Any]:
    return validate_contract(
        {
            "schema_version": "1.0",
            "job_id": job_id,
            "state": "queued",
            "requested_output": requested_output,
            "claim_scope": "diagnostic",
            "product_quality_claim": False,
            "status_url": f"/v1/conversions/{job_id}",
        },
        "submission_response",
    )


#: DEC-044 Dodatok D (C): a preview role always forces the page-scoped
#: artifact shape (even for a single-page job), since a produced docx may
#: paginate to more than one preview page.
_PAGE_SCOPED_ROLES = frozenset({"source_preview", "docx_preview"})


def _status_payload(job: JobRecord, store: JobStore) -> dict[str, Any]:
    records = store.list_artifacts(job.job_id)
    if job.pages_total is None and not any(r.role in _PAGE_SCOPED_ROLES for r in records):
        # Today's single-page shape, byte-identical: a flat list of roles.
        artifacts: list[Any] = [record.role for record in records]
    else:
        artifacts = [_artifact_entry(job.job_id, record) for record in records]
    return validate_contract(
        {
            "schema_version": "1.1",
            "job_id": job.job_id,
            "state": job.state,
            "stage": job.stage,
            "attempt": job.attempt,
            "requested_output": job.requested_output,
            "claim_scope": "diagnostic",
            "product_quality_claim": False,
            "diagnostic_outcome": job.diagnostic_outcome,
            "artifacts": artifacts,
            "error": job.error,
            "model_variant": job.model_variant,
            "pages_total": job.pages_total,
            "pages_done": job.pages_done,
            "current_page": job.current_page,
            "pages": job.pages,
            "label": job.label,
            "input_kind": job.input_kind,
            "engine": job.engine,
            "created_at": _iso8601_utc_ms(job.created_at),
            "updated_at": _iso8601_utc_ms(job.updated_at),
            "terminal_at": (
                None if job.terminal_at is None else _iso8601_utc_ms(job.terminal_at)
            ),
            "expires_at": (
                None if job.expires_at is None else _iso8601_utc_ms(job.expires_at)
            ),
        },
        "conversion_status",
    )


def _job_summary_entry(job: JobRecord) -> dict[str, Any]:
    docx_produced = None if job.diagnostic_outcome is None else job.diagnostic_outcome.get(
        "docx_produced"
    )
    error_code = None if job.error is None else job.error.get("code")
    return validate_contract(
        {
            "job_id": job.job_id,
            "state": job.state,
            "stage": job.stage,
            "requested_output": job.requested_output,
            "model_variant": job.model_variant,
            "label": job.label,
            "input_kind": job.input_kind,
            "created_at": _iso8601_utc_ms(job.created_at),
            "updated_at": _iso8601_utc_ms(job.updated_at),
            "terminal_at": (
                None if job.terminal_at is None else _iso8601_utc_ms(job.terminal_at)
            ),
            "expires_at": (
                None if job.expires_at is None else _iso8601_utc_ms(job.expires_at)
            ),
            "pages_total": job.pages_total,
            "pages_done": job.pages_done,
            "docx_produced": docx_produced,
            "error_code": error_code,
        },
        "job_summary",
    )


def _job_list_payload(jobs: list[JobRecord], next_cursor: str | None) -> dict[str, Any]:
    return validate_contract(
        {
            "schema_version": "1.0",
            "items": [_job_summary_entry(job) for job in jobs],
            "next_cursor": next_cursor,
        },
        "job_list",
    )


def _artifact_entry(job_id: str, record: ArtifactRecord) -> dict[str, Any]:
    """DEC-044 Dodatok D §1 (B1/C): a page-scoped artifact entry.
    `page_index` is `None` for the -1 'not page-scoped' sentinel (the merged
    diagnostic_docx); the download URL matches the roles new page-scoped
    route below, or the legacy unindexed one for a non-page-scoped artifact."""
    page_index = None if record.page_index < 0 else record.page_index
    url = (
        f"/v1/conversions/{job_id}/artifacts/{record.role}"
        if page_index is None
        else f"/v1/conversions/{job_id}/artifacts/{record.role}/{page_index}"
    )
    return {"role": record.role, "page_index": page_index, "url": url}


def _resolve_diagnostic_docx_ready(dependencies: ApiDependencies) -> bool:
    status = dependencies.diagnostic_docx_status
    return status() if callable(status) else status


def _readiness_report(dependencies: ApiDependencies) -> dict[str, Any]:
    statuses = dependencies.component_statuses
    if callable(statuses):
        statuses = statuses()
    return evaluate_readiness(
        component_statuses=statuses,
        docx_requirements=current_docx_requirements(),
        diagnostic_docx_ready=_resolve_diagnostic_docx_ready(dependencies),
    )


def _problem(
    code: str,
    title: str,
    status: int,
    *,
    retryable: bool = False,
    blockers: list[str] | None = None,
    max_pages: int | None = None,
    retry_after_seconds: int | None = None,
) -> JSONResponse:
    fields: dict[str, object] = {
        "code": code,
        "title": title,
        "status": status,
        "retryable": retryable,
    }
    if blockers is not None:
        fields["blockers"] = blockers
    if max_pages is not None:
        fields["max_pages"] = max_pages
    if retry_after_seconds is not None:
        fields["retry_after_seconds"] = retry_after_seconds
    response = JSONResponse(
        problem_response(**fields),
        status_code=status,
        media_type="application/problem+json",
    )
    if retry_after_seconds is not None:
        # Design §2.6/§11 bullet 7: a `RATE_LIMITED` response must carry the
        # HTTP `Retry-After` header, not just the body field.
        response.headers["Retry-After"] = str(retry_after_seconds)
    return response
