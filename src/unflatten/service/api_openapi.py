"""Shared OpenAPI enrichment for handlers that validate ``Request`` manually.

Only documentation is changed here: authentication, admission limits and binary
responses remain the responsibility of the existing HTTP handlers.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from unflatten.service.contracts import JOB_STATES

_WORD = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_BINARY = {"type": "string", "format": "binary"}
_BEARER_OR_SESSION = [{"BearerAuth": []}, {"SessionCookie": []}]
_COOKIE_CONTEXT = (
    "Browser session cookies are also accepted. Cookie requests require "
    "Sec-Fetch-Site: same-origin or none, and writes additionally require "
    "X-Unflatten-Client: web. Send exactly one credential: a Bearer token or "
    "a session cookie; sending both is rejected."
)
_OPERATIONS: dict[tuple[str, str], tuple[str, str]] = {
    ("/health/live", "get"): ("Health", "Check process liveness"),
    ("/health/ready", "get"): ("Health", "Check operational readiness"),
    ("/v1/readiness", "get"): ("Health", "Inspect readiness and upload limits"),
    ("/v1/session", "get"): ("Session", "Inspect the current session"),
    ("/v1/session", "post"): ("Session", "Sign in with a password"),
    ("/v1/session", "delete"): ("Session", "Sign out the current browser session"),
    ("/v1/setup", "post"): ("Session", "Set the first browser password"),
    ("/v1/setup/password", "put"): ("Session", "Change the browser password"),
    ("/v1/settings", "get"): ("Settings", "Read service settings"),
    ("/v1/models", "get"): ("Models", "List model variants and command status"),
    ("/v1/models/default", "put"): ("Models", "Set or clear the default model variant"),
    ("/v1/models/{variant_id}/download", "post"): ("Models", "Queue a model download"),
    ("/v1/models/{variant_id}/download", "delete"): ("Models", "Queue removal of a model download"),
    ("/v1/models/{variant_id}/load", "post"): ("Models", "Queue loading a model"),
    ("/v1/conversions", "post"): ("Conversions", "Submit a PDF or page images"),
    ("/v1/conversions", "get"): ("Conversions", "List your conversion jobs"),
    ("/v1/conversions/{job_id}", "get"): ("Conversions", "Poll your conversion status"),
    ("/v1/conversions/{job_id}", "delete"): ("Conversions", "Cancel a job or purge a terminal job"),
}


def _parameter(
    name: str, location: str, schema: dict[str, Any], description: str, *, required: bool = False
) -> dict[str, Any]:
    return {
        "name": name,
        "in": location,
        "required": required,
        "schema": schema,
        "description": description,
    }


def _parameters(operation: dict[str, Any], *parameters: dict[str, Any]) -> None:
    """Retain FastAPI path parameters and replace only our documented fields."""
    identities = {(parameter["in"], parameter["name"]) for parameter in parameters}
    operation["parameters"] = [
        parameter
        for parameter in operation.get("parameters", [])
        if (parameter["in"], parameter["name"]) not in identities
    ] + list(parameters)


def _body(properties: dict[str, Any], *, closed: bool = False) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }
    if closed:
        schema["additionalProperties"] = False
    return {"required": True, "content": {"application/json": {"schema": schema}}}


def _describe(operation: dict[str, Any], description: str) -> None:
    previous = operation.get("description", "")
    operation["description"] = f"{previous}\n\n{description}".strip()


def _upload(operation: dict[str, Any]) -> None:
    _parameters(
        operation,
        _parameter(
            "Idempotency-Key",
            "header",
            {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
                "pattern": r"^[\x21-\x7e]{1,256}$",
                "default": "auto",
            },
            (
                "Unique visible-ASCII key for this submission. Reuse the same key only "
                "to retry the same input and settings. In this service's Swagger UI "
                "only, 'auto' is replaced with a fresh UUID on each Execute; headless "
                "clients must supply their own key. The server treats 'auto' as an "
                "ordinary literal key."
            ),
            required=True,
        ),
        _parameter(
            "X-Unflatten-Profile",
            "header",
            {"type": "string", "enum": ["word-core-v0"], "default": "word-core-v0"},
            "Supported conversion profile.",
            required=True,
        ),
        _parameter(
            "X-Unflatten-Output",
            "header",
            {
                "type": "string",
                "enum": ["diagnostic_docx", "prediction_ir"],
                "default": "diagnostic_docx",
            },
            (
                "Requested output. diagnostic_docx produces Word and requires DOCX "
                "readiness; prediction_ir requests the diagnostic intermediate "
                "representation. Arbitrary generation settings and docx output are not "
                "supported."
            ),
            required=True,
        ),
        _parameter(
            "X-Unflatten-Model",
            "header",
            {"type": "string"},
            (
                "Optional published variant_id from GET /v1/models; omit to use the "
                "effective default model. Selection remains subject to availability, "
                "engine and memory checks."
            ),
        ),
        _parameter(
            "X-Unflatten-Label",
            "header",
            {"type": "string"},
            (
                "Optional percent-encoded UTF-8 label. After decoding, NFC "
                "normalization and trimming, use 1–120 Unicode code points without "
                'control characters or filename-unsafe characters (\\ / : * ? " < > |).'
            ),
        ),
    )
    operation["requestBody"] = {
        "required": True,
        "description": (
            "Upload one PDF as a single page file, or repeat the page field for "
            "PNG/JPEG images in page order. A PDF cannot be mixed with images or "
            "other PDFs. Raw single PDF, PNG and JPEG bodies are also accepted. "
            "Runtime upload-byte and page limits are reported by GET /v1/readiness."
        ),
        "content": {
            # Swagger chooses the first media type; multipart must offer its picker.
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["page"],
                    "properties": {
                        "page": {"type": "array", "items": dict(_BINARY), "minItems": 1},
                    },
                    "additionalProperties": False,
                },
                "encoding": {"page": {"contentType": "application/pdf, image/png, image/jpeg"}},
            },
            "application/pdf": {"schema": dict(_BINARY)},
            "image/png": {"schema": dict(_BINARY)},
            "image/jpeg": {"schema": dict(_BINARY)},
        },
    }
    _describe(
        operation,
        (
            "Authentication and required headers are checked before upload bytes "
            "are read. A 202 response admits an asynchronous job; it does not mean "
            "conversion succeeded. Follow status_url (also returned in Location) "
            "using GET until state is succeeded, failed, cancelled or expired. On "
            "success download only available artifacts from the status response. "
            "Stop polling at a client deadline; failed jobs expose a problem, and "
            "expired artifacts return 410. diagnostic_docx is a private diagnostic "
            "Word output, with no product-quality claim."
        ),
    )
    operation["responses"]["202"].setdefault("headers", {})["Location"] = {
        "description": "Relative status_url for polling the accepted job.",
        "schema": {"type": "string"},
    }


def _artifact(operation: dict[str, Any], path: str) -> None:
    if path.endswith("artifacts.zip"):
        media_type, disposition, summary = (
            "application/zip",
            "attachment",
            "Download available artifacts as ZIP",
        )
    elif "/diagnostic_docx" in path:
        media_type, disposition, summary = (
            _WORD,
            "attachment",
            "Download the diagnostic Word document",
        )
    elif "/prediction_ir" in path:
        media_type, disposition, summary = (
            "application/json",
            "attachment",
            "Download prediction IR JSON",
        )
    else:
        media_type, disposition, summary = "image/png", "inline", "View a page preview PNG"
    operation.update(tags=["Artifacts"], summary=summary)
    _describe(
        operation,
        (
            "Requires job ownership and the artifact's download capability. Only "
            "published artifacts of succeeded, unexpired jobs are available. Page "
            "indices are zero-based; whole-document prediction_ir returns page 0 "
            "for multi-page input. Word is the assembled document. ZIP includes "
            "permitted artifacts and MANIFEST.json; expired artifacts return 410."
        ),
    )
    response = operation["responses"]["200"]
    response.update(
        description="Binary artifact response.",
        content={media_type: {"schema": dict(_BINARY)}},
        headers={
            "Content-Disposition": {
                "description": f"{disposition}; filename identifies the downloaded artifact.",
                "schema": {"type": "string"},
            }
        },
    )
    operation["responses"].setdefault(
        "default",
        {
            "description": (
                "Problem response (authentication, capability, ownership, availability or expiry)."
            ),
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Problem"}}},
        },
    )
    if disposition == "inline":
        operation["responses"].setdefault(
            "304", {"description": "Preview unchanged for the supplied If-None-Match ETag."}
        )


def build_openapi(app: FastAPI) -> dict[str, Any]:
    """Build the same enriched contract for runtime Swagger and file export."""
    document = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    document["info"]["description"] = (
        "Private asynchronous PDF/image conversion service. Authorize with a configured "
        "Bearer token, submit a conversion, poll status_url, then download "
        "available artifacts. " + _COOKIE_CONTEXT
    )
    document["tags"] = [
        {"name": name, "description": description}
        for name, description in (
            ("Health", "Liveness, worker readiness and input limits."),
            ("Session", "Browser password and session management."),
            ("Conversions", "Submit, poll, list and cancel your asynchronous jobs."),
            ("Artifacts", "Authenticated Word, IR, ZIP and PNG downloads."),
            ("Models", "Inspect configured variants and queue model commands."),
            ("Settings", "Read effective service settings."),
        )
    ]
    document.setdefault("components", {}).setdefault("securitySchemes", {}).update(
        {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "description": (
                    "Configured service token. Paste the token without the Bearer prefix "
                    "into Authorize."
                ),
            },
            "SessionCookie": {
                "type": "apiKey",
                "in": "cookie",
                "name": "unflatten_session",
                "description": _COOKIE_CONTEXT
                + " Browser-managed HttpOnly cookie; Swagger cannot set it manually.",
            },
        }
    )
    paths = document["paths"]
    for path, methods in paths.items():
        for method, operation in methods.items():
            if method not in {"get", "post", "put", "delete", "patch", "head", "options"}:
                continue
            operation["security"] = (
                [dict(item) for item in _BEARER_OR_SESSION] if path.startswith("/v1/") else []
            )
            if (path, method) in _OPERATIONS:
                tag, summary = _OPERATIONS[path, method]
                operation.update(tags=[tag], summary=summary)
            if "/artifacts/" in path or path.endswith("/artifacts.zip"):
                _artifact(operation, path)
            problem = operation.get("responses", {}).get("default", {})
            content = problem.get("content", {})
            if "application/json" in content:
                content["application/problem+json"] = content.pop("application/json")

    paths["/v1/session"]["get"]["security"] = [{}, {"BearerAuth": []}, {"SessionCookie": []}]
    paths["/v1/session"]["post"]["security"] = []
    paths["/v1/session"]["delete"]["security"] = [{"SessionCookie": []}]
    paths["/v1/setup"]["post"]["security"] = [{"BearerAuth": []}]
    _describe(
        paths["/v1/session"]["get"],
        (
            "Public without credentials; returns authenticated=false. Optional "
            "Bearer or browser cookie identifies the caller. Invalid credentials "
            "still return 401."
        ),
    )
    _describe(
        paths["/v1/session"]["delete"],
        ("Accepts a browser session cookie only; a Bearer credential cannot sign out a session."),
    )
    _describe(
        paths["/v1/setup"]["post"],
        (
            "One-time password setup requires a bootstrap Bearer token. "
            "Loopback-only deployments explicitly configured for tokenless setup "
            "with one principal may omit it; GET /v1/session reports "
            "setup_tokenless. This creates a browser session cookie; it does not "
            "issue a Bearer token."
        ),
    )
    for path, method, properties in (
        (
            "/v1/session",
            "post",
            {"password": {"type": "string", "format": "password", "writeOnly": True}},
        ),
        (
            "/v1/setup",
            "post",
            {
                "password": {
                    "type": "string",
                    "format": "password",
                    "writeOnly": True,
                    "description": "12–256 NFC-normalized Unicode code points.",
                }
            },
        ),
        (
            "/v1/setup/password",
            "put",
            {
                "current_password": {"type": "string", "format": "password", "writeOnly": True},
                "new_password": {
                    "type": "string",
                    "format": "password",
                    "writeOnly": True,
                    "description": "12–256 NFC-normalized Unicode code points.",
                },
            },
        ),
    ):
        operation = paths[path][method]
        operation["requestBody"] = _body(properties)
        _parameters(
            operation,
            _parameter(
                "X-Unflatten-Client",
                "header",
                {"type": "string", "enum": ["web"], "default": "web"},
                "Required browser request context.",
                required=True,
            ),
            _parameter(
                "Sec-Fetch-Site",
                "header",
                {"type": "string", "enum": ["same-origin", "none"], "default": "same-origin"},
                "Required request context; browsers provide this header automatically.",
                required=True,
            ),
        )
    paths["/v1/models/default"]["put"]["requestBody"] = _body(
        {
            "variant_id": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": (
                    "Published variant_id from GET /v1/models, or null to clear the "
                    "override and restore config/auto selection."
                ),
            },
        },
        closed=True,
    )
    _upload(paths["/v1/conversions"]["post"])
    state = _parameter(
        "state",
        "query",
        {"type": "array", "items": {"type": "string", "enum": sorted(JOB_STATES)}},
        "Optional state filter; repeat state for multiple allowed values.",
    )
    state.update(style="form", explode=True)
    _parameters(
        paths["/v1/conversions"]["get"],
        state,
        _parameter(
            "limit",
            "query",
            {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
            "Maximum jobs returned per page.",
        ),
        _parameter(
            "cursor",
            "query",
            {"type": "string"},
            "Opaque next_cursor from the previous page; omit for the first page.",
        ),
    )
    cancel = paths["/v1/conversions/{job_id}"]["delete"]
    _parameters(
        cancel,
        _parameter(
            "mode",
            "query",
            {"type": "string", "enum": ["cancel", "purge"], "default": "cancel"},
            (
                "cancel requests cancellation and returns status. purge permanently "
                "removes an owned terminal job and its artifacts, returning 204; an "
                "active job returns 409."
            ),
        ),
    )
    cancel["responses"].setdefault(
        "204", {"description": "Terminal job and stored artifacts purged; no response body."}
    )
    _describe(
        paths["/v1/conversions/{job_id}"]["get"],
        (
            "Poll the owner's job until succeeded, failed, cancelled or expired. "
            "Inspect artifact availability before downloading; terminal state alone "
            "does not promise a Word artifact. If-None-Match is supported and may "
            "return 304. Use a polling deadline."
        ),
    )
    for path in ("/v1/models", "/v1/conversions", "/v1/conversions/{job_id}"):
        paths[path]["get"]["responses"].setdefault(
            "304", {"description": "Representation unchanged for the supplied If-None-Match ETag."}
        )
    app.openapi_schema = document
    return document
