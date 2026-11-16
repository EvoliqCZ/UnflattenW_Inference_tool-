"""Tests for `web_static.py`: SPA static serving on a bare FastAPI app
(design §6.1-6.4). Not wired into `api.py`/`create_app` -- that is T35."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from unflatten.service.web_static import (
    SPA_ROUTES,
    WebUiInvalid,
    load_dist,
    register_spa_routes,
)

_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_-]+)'")


def _write_dist(tmp_path: Path, *, root_files: dict[str, str] | None = None) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        '<!doctype html><html><head>'
        '<meta name="csp-nonce" content="__CSP_NONCE__">'
        "</head><body><div id=\"root\"></div></body></html>",
        encoding="utf-8",
    )
    assets = dist / "assets"
    assets.mkdir()
    (assets / "app-abc12345.js").write_text("console.log(1)", encoding="utf-8")
    (assets / "app-def67890.css").write_text("body{}", encoding="utf-8")
    for name, content in (root_files or {}).items():
        (dist / name).write_text(content, encoding="utf-8")
    return dist


def _app(tmp_path: Path, *, root_files: dict[str, str] | None = None) -> TestClient:
    dist = load_dist(_write_dist(tmp_path, root_files=root_files))
    app = FastAPI()
    register_spa_routes(app, dist)
    return TestClient(app)


def test_each_spa_route_serves_index_no_store_exact_csp(tmp_path: Path) -> None:
    client = _app(tmp_path)
    for route in SPA_ROUTES:
        path = re.sub(r":(\w+)", "test-value", route)
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        csp = response.headers["content-security-policy"]
        assert csp == (
            "default-src 'none'; script-src 'self'; "
            f"style-src 'self' 'nonce-{_NONCE_RE.search(csp).group(1)}'; "
            "img-src 'self'; font-src 'self'; connect-src 'self'; "
            "manifest-src 'self'; base-uri 'none'; form-action 'self'; "
            "frame-ancestors 'none'; object-src 'none'"
        )
        nonce = _NONCE_RE.search(csp).group(1)
        assert f'content="{nonce}"' in response.text
        assert "__CSP_NONCE__" not in response.text


def test_serve_index_nonce_differs_per_request(tmp_path: Path) -> None:
    client = _app(tmp_path)
    first = client.get("/")
    second = client.get("/")
    nonce_a = _NONCE_RE.search(first.headers["content-security-policy"]).group(1)
    nonce_b = _NONCE_RE.search(second.headers["content-security-policy"]).group(1)
    assert nonce_a != nonce_b
    # secrets.token_urlsafe base64-encodes bytes at ~4/3 chars/byte; require
    # enough bytes of entropy for >=128 bits (16 bytes -> ceil(16*4/3)=22 chars).
    assert len(nonce_a) >= 22


def test_assets_immutable_with_mime(tmp_path: Path) -> None:
    client = _app(tmp_path)
    js = client.get("/assets/app-abc12345.js")
    assert js.status_code == 200
    assert js.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert js.headers["content-type"] == "text/javascript"

    css = client.get("/assets/app-def67890.css")
    assert css.status_code == 200
    assert css.headers["content-type"] == "text/css"
    assert css.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_unhashed_static_asset_exception_is_revalidated_not_immutable(tmp_path: Path) -> None:
    """Item 9: `sample.pdf` (the one un-hashed, hand-placed asset the setup
    wizard fetches, T43/D-B) must not be served with the year-long immutable
    policy meant for content-hashed Vite output -- it can never change name
    if its content ever does, so a client would keep the old bytes forever."""
    dist_dir = _write_dist(tmp_path)
    (dist_dir / "assets" / "sample.pdf").write_bytes(b"%PDF-1.4 fake")
    client = TestClient(_register(load_dist(dist_dir)))

    response = client.get("/assets/sample.pdf")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"


def test_assets_unknown_extension_is_octet_stream(tmp_path: Path) -> None:
    dist_dir = _write_dist(tmp_path)
    (dist_dir / "assets" / "blob-11112222").write_bytes(b"\x00\x01")
    client = TestClient(_register(load_dist(dist_dir)))
    response = client.get("/assets/blob-11112222")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"


def _register(dist: object) -> FastAPI:
    app = FastAPI()
    register_spa_routes(app, dist)  # type: ignore[arg-type]
    return app


def test_root_files_no_cache(tmp_path: Path) -> None:
    client = _app(tmp_path, root_files={"theme-init.js": "()=>{}", "favicon.svg": "<svg/>"})
    response = client.get("/theme-init.js")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["content-type"] == "text/javascript"
    assert "content-security-policy" not in response.headers

    favicon = client.get("/favicon.svg")
    assert favicon.status_code == 200
    assert favicon.headers["cache-control"] == "no-cache"


def test_root_file_not_present_in_dist_is_404(tmp_path: Path) -> None:
    client = _app(tmp_path)
    assert client.get("/theme-init.js").status_code == 404


def test_asset_traversal_404(tmp_path: Path) -> None:
    client = _app(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    response = client.get("/assets/..%2f..%2fsecret.txt")
    assert response.status_code == 404


def test_asset_missing_file_404(tmp_path: Path) -> None:
    client = _app(tmp_path)
    assert client.get("/assets/does-not-exist.js").status_code == 404


def test_asset_embedded_nul_byte_404_not_500(tmp_path: Path) -> None:
    """Item 3: an embedded NUL (`/assets/%00`) makes `Path.resolve()` raise
    `ValueError`, which must become a 404 like any other bad asset path, not
    an unhandled 500."""
    client = _app(tmp_path)
    response = client.get("/assets/%00")
    assert response.status_code == 404


def test_missing_dist_raises_WebUiInvalid(tmp_path: Path) -> None:
    with pytest.raises(WebUiInvalid):
        load_dist(tmp_path / "does-not-exist")


def test_missing_index_html_raises_WebUiInvalid(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    with pytest.raises(WebUiInvalid):
        load_dist(dist)


def test_index_html_missing_nonce_placeholder_raises_WebUiInvalid(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    with pytest.raises(WebUiInvalid):
        load_dist(dist)


def test_spa_routes_excluded_from_openapi_schema(tmp_path: Path) -> None:
    """T35 design §5.2: none of these routes carry API semantics, so they
    must never surface in the generated OpenAPI document."""
    dist = load_dist(_write_dist(tmp_path, root_files={"theme-init.js": "()=>{}"}))
    app = FastAPI()
    register_spa_routes(app, dist)

    assert all(not route.include_in_schema for route in app.routes)


def test_index_html_not_served_outside_spa_routes(tmp_path: Path) -> None:
    client = _app(tmp_path)
    assert "/not-a-spa-route" not in SPA_ROUTES
    response = client.get("/not-a-spa-route")
    assert response.status_code == 404

    # Requesting index.html by its literal name is also not one of the
    # registered SPA routes and must not fall through to the SPA handler.
    response = client.get("/index.html")
    assert response.status_code == 404
