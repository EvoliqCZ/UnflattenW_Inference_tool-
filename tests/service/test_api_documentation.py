"""Interactive documentation stays local and uses the real authenticated API."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from tests.service.test_api_endpoints import _client


def test_swagger_loads_local_assets_and_live_schema(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/docs")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "https://" not in response.text
    assets = re.findall(r'(?:src|href)="([^"]+)"', response.text)
    assert len(assets) >= 2
    for url in assets:
        assert url.startswith("/docs/assets/")
        asset = client.get(url)
        assert asset.status_code == 200
        assert asset.content
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert schema.headers["cache-control"] == "no-store"
    assert "/v1/conversions" in schema.json()["paths"]
    assert "/docs" not in schema.json()["paths"]
    problem = schema.json()["paths"]["/v1/conversions"]["post"]["responses"]["default"]
    assert set(problem["content"]) == {"application/problem+json"}


def test_docs_nonce_changes_and_wrong_host_is_refused(tmp_path: Path) -> None:
    client = _client(tmp_path)
    first = client.get("/docs")
    second = client.get("/docs")
    assert first.status_code == second.status_code == 200
    assert first.headers["content-security-policy"] != second.headers["content-security-policy"]
    nonce = re.search(r"'nonce-([^']+)'", first.headers["content-security-policy"])
    assert nonce is not None
    assert f'nonce="{nonce.group(1)}"' in first.text
    for path in ("/docs", "/openapi.json", "/docs/assets/swagger-ui-bundle.js"):
        assert client.get(path, headers={"Host": "evil.example"}).status_code == 421


@pytest.mark.parametrize("path", ["missing.js", "%2e%2e/api.py", "LICENSE", "%00.js"])
def test_docs_assets_only_expose_named_public_files(tmp_path: Path, path: str) -> None:
    assert _client(tmp_path).get(f"/docs/assets/{path}").status_code == 404


def test_docs_respects_reverse_proxy_root_path(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.app.root_path = "/unflatten"
    response = client.get("/docs")
    assert response.status_code == 200
    assert 'src="/unflatten/docs/assets/swagger-ui-bundle.js"' in response.text
    assert '"/unflatten/openapi.json"' in response.text
    schema = client.get("/openapi.json").json()
    assert {"url": "/unflatten"} in schema["servers"]
