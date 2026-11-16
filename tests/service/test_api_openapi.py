"""The published schema must describe requests the real boundary accepts."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from tests.service.test_api_endpoints import AUTH, PDF, _client

from unflatten.service.api import API_TITLE, API_VERSION, register_routes
from unflatten.service.api_openapi import build_openapi


def _app() -> FastAPI:
    app = FastAPI(title=API_TITLE, version=API_VERSION)
    register_routes(app, web_ui="spa")
    return app


def test_upload_schema_drives_an_accepted_multipart_request(tmp_path: Path) -> None:
    schema = build_openapi(_app())
    operation = schema["paths"]["/v1/conversions"]["post"]
    content = operation["requestBody"]["content"]
    assert list(content)[0] == "multipart/form-data"
    assert set(content) == {"multipart/form-data", "application/pdf", "image/png", "image/jpeg"}
    multipart = content["multipart/form-data"]
    assert multipart["schema"]["required"] == ["page"]
    assert multipart["schema"]["properties"]["page"] == {
        "type": "array",
        "items": {"type": "string", "format": "binary"},
        "minItems": 1,
    }
    assert set(multipart["encoding"]["page"]["contentType"].split(", ")) == {
        "application/pdf",
        "image/png",
        "image/jpeg",
    }
    parameters = {item["name"]: item for item in operation["parameters"]}
    assert {name for name, item in parameters.items() if item["required"]} == {
        "Idempotency-Key",
        "X-Unflatten-Profile",
        "X-Unflatten-Output",
    }
    assert parameters["Idempotency-Key"]["schema"]["default"] == "auto"
    assert parameters["X-Unflatten-Profile"]["schema"]["enum"] == ["word-core-v0"]
    output = parameters["X-Unflatten-Output"]["schema"]
    assert output["default"] == "diagnostic_docx"
    assert set(output["enum"]) == {"diagnostic_docx", "prediction_ir"}
    assert not parameters["X-Unflatten-Model"]["required"]
    assert "percent" in parameters["X-Unflatten-Label"]["description"].lower()

    headers = {
        **AUTH,
        "Idempotency-Key": "schema-request-1",
        "X-Unflatten-Profile": parameters["X-Unflatten-Profile"]["schema"]["default"],
        "X-Unflatten-Output": output["default"],
    }
    field = multipart["schema"]["required"][0]
    client = _client(tmp_path)
    accepted = client.post(
        "/v1/conversions", files=[(field, ("input.pdf", PDF, "application/pdf"))], headers=headers
    )
    assert accepted.status_code == 202
    assert accepted.json()["requested_output"] == "diagnostic_docx"
    status = client.get(accepted.json()["status_url"], headers=AUTH)
    assert status.status_code == 200
    assert status.json()["state"] == "queued"
    malformed = client.post(
        "/v1/conversions", files=[("file", ("input.pdf", PDF, "application/pdf"))], headers=headers
    )
    assert malformed.status_code == 400
    unauthenticated = client.post(
        "/v1/conversions",
        content=b"malformed multipart",
        headers={"Content-Type": "multipart/form-data"},
    )
    assert unauthenticated.status_code == 401


def test_security_matches_public_bearer_and_session_routes() -> None:
    schema = build_openapi(_app())
    schemes = schema["components"]["securitySchemes"]
    assert schemes["BearerAuth"]["type"] == "http"
    assert schemes["BearerAuth"]["scheme"] == "bearer"
    assert schemes["SessionCookie"]["name"] == "unflatten_session"
    paths = schema["paths"]
    for path in ("/health/live", "/health/ready"):
        assert paths[path]["get"]["security"] == []
    assert paths["/v1/session"]["post"]["security"] == []
    assert {} in paths["/v1/session"]["get"]["security"]
    assert paths["/v1/session"]["delete"]["security"] == [{"SessionCookie": []}]
    assert paths["/v1/setup"]["post"]["security"] == [{"BearerAuth": []}]
    for path in (
        "/v1/readiness",
        "/v1/models",
        "/v1/settings",
        "/v1/conversions",
        "/v1/conversions/{job_id}/artifacts/diagnostic_docx",
    ):
        assert {"BearerAuth": []} in paths[path]["get"]["security"]
    assert {"BearerAuth": []} in paths["/v1/conversions"]["post"]["security"]


def test_manual_json_bodies_and_query_parameters_reflect_runtime() -> None:
    paths = build_openapi(_app())["paths"]
    for path, method, fields in (
        ("/v1/session", "post", {"password"}),
        ("/v1/setup", "post", {"password"}),
        ("/v1/setup/password", "put", {"current_password", "new_password"}),
        ("/v1/models/default", "put", {"variant_id"}),
    ):
        operation = paths[path][method]
        body = operation["requestBody"]["content"]["application/json"]["schema"]
        assert set(body["properties"]) == fields
        assert set(body["required"]) == fields
    variant = paths["/v1/models/default"]["put"]["requestBody"]["content"]["application/json"][
        "schema"
    ]
    assert variant["properties"]["variant_id"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert variant["additionalProperties"] is False
    assert "requestBody" not in paths["/v1/settings"]["get"]
    login_headers = {item["name"]: item for item in paths["/v1/session"]["post"]["parameters"]}
    assert login_headers["X-Unflatten-Client"]["required"]
    assert login_headers["Sec-Fetch-Site"]["required"]
    query = {item["name"]: item for item in paths["/v1/conversions"]["get"]["parameters"]}
    assert query["limit"]["schema"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "default": 25,
    }
    assert query["state"]["explode"] is True
    assert query["cursor"]["schema"]["type"] == "string"
    cancel = paths["/v1/conversions/{job_id}"]["delete"]
    mode = next(item for item in cancel["parameters"] if item["name"] == "mode")
    assert mode["schema"]["enum"] == ["cancel", "purge"]
    assert "204" in cancel["responses"]


@pytest.mark.parametrize(
    ("suffix", "media_type", "disposition"),
    [
        (
            "artifacts/diagnostic_docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "attachment",
        ),
        ("artifacts/prediction_ir", "application/json", "attachment"),
        ("artifacts/prediction_ir/{page_index}", "application/json", "attachment"),
        ("artifacts/source_preview/{page_index}", "image/png", "inline"),
        ("artifacts/docx_preview/{page_index}", "image/png", "inline"),
        ("artifacts.zip", "application/zip", "attachment"),
    ],
)
def test_artifacts_are_binary_downloads(suffix: str, media_type: str, disposition: str) -> None:
    operation = build_openapi(_app())["paths"][f"/v1/conversions/{{job_id}}/{suffix}"]["get"]
    response = operation["responses"]["200"]
    assert response["content"] == {media_type: {"schema": {"type": "string", "format": "binary"}}}
    assert disposition in response["headers"]["Content-Disposition"]["description"]
    assert "default" in operation["responses"]


def test_enrichment_preserves_registered_operation_ids_and_response_models() -> None:
    app = _app()
    original = get_openapi(title=app.title, version=app.version, routes=app.routes)
    enriched = build_openapi(app)
    assert enriched["components"]["schemas"] == original["components"]["schemas"]
    for path, methods in original["paths"].items():
        for method, operation in methods.items():
            assert enriched["paths"][path][method]["operationId"] == operation["operationId"]
            for code, response in operation["responses"].items():
                content = response.get("content", {})
                for media_type, description in content.items():
                    if "$ref" in description.get("schema", {}):
                        documented_type = (
                            "application/problem+json" if code == "default" else media_type
                        )
                        assert (
                            enriched["paths"][path][method]["responses"][code]["content"][
                                documented_type
                            ]
                            == description
                        )
    assert build_openapi(app) == enriched
    assert "status_url" in enriched["paths"]["/v1/conversions"]["post"]["description"]
