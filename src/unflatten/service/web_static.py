"""SPA static serving over a built `dist/` directory (design §6.1-6.4).

This module is only `SPA_ROUTES` plus the serving primitives -- it is not
wired into `register_routes`/`create_app` (T35 does that; `api.py` is owned
by a parallel stream). Tests exercise it against a bare `FastAPI()`.

`index.html` is served fresh per request (`Cache-Control: no-store`) with a
new CSP nonce, per the T08 CSP probe result (`docs/runs/radix-csp-probe-2026-09-27.md`,
design §6.3): `style-src` carries the nonce (Radix/`react-remove-scroll`
injects a `<style>` element at runtime), `script-src` does not (no inline
script ever ships in `dist/`). The built `index.html` must contain the
literal placeholder `__CSP_NONCE__` inside a `<meta name="csp-nonce"
content="...">` tag; `load_dist` refuses (fail-closed) a `dist/` where it is
absent so a build that forgot the meta tag never serves live traffic.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

#: Client-side routes the SPA router owns. `:jobId` is the router's own
#: path-parameter form; `openapi_export`/the eventual FastAPI registration
#: translate it to `{job_id}` where a literal path is required.
SPA_ROUTES: tuple[str, ...] = (
    "/",
    "/setup",
    "/login",
    "/jobs",
    "/jobs/:jobId",
    "/models",
    "/settings",
)

#: Literal marker the built `index.html` must carry inside its CSP-nonce
#: `<meta>` tag; replaced with a fresh per-request nonce in `serve_index`.
NONCE_PLACEHOLDER = "__CSP_NONCE__"

#: Exact `index.html` CSP (design §6.3); `{nonce}` is filled in per request.
_CSP_TEMPLATE = (
    "default-src 'none'; script-src 'self'; style-src 'self' 'nonce-{nonce}'; "
    "img-src 'self'; font-src 'self'; connect-src 'self'; manifest-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'; object-src 'none'"
)

#: `Content-Type` by suffix (design §6.2); anything else is `application/octet-stream`.
_CONTENT_TYPES: dict[str, str] = {
    ".js": "text/javascript",
    ".css": "text/css",
    ".woff2": "font/woff2",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".map": "application/json",
    # T43/D-B: the setup wizard's static demo PDF (`/assets/sample.pdf`).
    ".pdf": "application/pdf",
}

_JOB_ID_PARAM = re.compile(r":(\w+)")

#: T43/D-B, item 9: the one static, hand-placed `dist/assets/` file that
#: carries no content hash (mirrors `web/scripts/check-dist.mjs`'s
#: `STATIC_ASSET_EXCEPTIONS`) -- it can never change name if its bytes ever
#: do, so it must not get the year-long immutable policy meant for hashed
#: Vite output.
_UNHASHED_ASSET_EXCEPTIONS = frozenset({"sample.pdf"})


class WebUiInvalid(Exception):
    """`dist/` is missing or malformed; the caller must refuse to start (fail-closed)."""


@dataclass(frozen=True)
class WebUiDist:
    """A validated, loaded `dist/` build. Immutable snapshot taken at `load_dist` time."""

    dist_dir: Path
    index_html: str
    root_files: tuple[str, ...]


def load_dist(dist_dir: Path) -> WebUiDist:
    """Validate and load a built SPA `dist/` directory.

    Raises `WebUiInvalid` if `dist/` does not exist, `index.html` is
    missing, or `index.html` does not carry the CSP-nonce placeholder.
    """
    index_path = dist_dir / "index.html"
    if not index_path.is_file():
        raise WebUiInvalid(f"missing dist/index.html: {index_path}")
    index_html = index_path.read_text(encoding="utf-8")
    if NONCE_PLACEHOLDER not in index_html:
        raise WebUiInvalid(
            f"dist/index.html is missing the {NONCE_PLACEHOLDER!r} CSP-nonce placeholder"
        )
    root_files = tuple(
        sorted(
            entry.name
            for entry in dist_dir.iterdir()
            if entry.is_file() and entry.name != "index.html"
        )
    )
    return WebUiDist(dist_dir=dist_dir, index_html=index_html, root_files=root_files)


def _content_type_for(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix, "application/octet-stream")


def serve_index(dist: WebUiDist) -> Response:
    """Serve `index.html` with a fresh CSP nonce (design §6.3), `no-store`."""
    nonce = secrets.token_urlsafe(16)
    html = dist.index_html.replace(NONCE_PLACEHOLDER, nonce)
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": _CSP_TEMPLATE.format(nonce=nonce),
        },
    )


def serve_asset(dist: WebUiDist, path: str) -> Response:
    """Serve a hashed, immutable file under `dist/assets/`. 404 on traversal
    or a missing file -- never raises."""
    assets_dir = (dist.dist_dir / "assets").resolve()
    try:
        candidate = (assets_dir / path).resolve()
    except (ValueError, OSError):
        # e.g. an embedded NUL byte (`/assets/%00`) -- malformed like any
        # other bad asset path, never a 500.
        return Response(status_code=404)
    if assets_dir not in candidate.parents and candidate != assets_dir:
        return Response(status_code=404)
    if not candidate.is_file():
        return Response(status_code=404)
    cache_control = (
        "no-cache"
        if candidate.name in _UNHASHED_ASSET_EXCEPTIONS
        else "public, max-age=31536000, immutable"
    )
    return Response(
        content=candidate.read_bytes(),
        headers={
            "Cache-Control": cache_control,
            "Content-Type": _content_type_for(candidate),
        },
    )


def serve_root_file(dist: WebUiDist, name: str) -> Response:
    """Serve a non-`index.html` file that lives at the root of `dist/` (e.g.
    `theme-init.js`, `favicon.svg`), `no-cache`. 404 if it is not one of the
    files `load_dist` found at load time."""
    if name not in dist.root_files:
        return Response(status_code=404)
    candidate = dist.dist_dir / name
    return Response(
        content=candidate.read_bytes(),
        headers={
            "Cache-Control": "no-cache",
            "Content-Type": _content_type_for(candidate),
        },
    )


def register_spa_routes(app: FastAPI, dist: WebUiDist) -> None:
    """Register every SPA-serving route (design §6.1) on `app`: `SPA_ROUTES`
    to `serve_index`, `/assets/{path:path}` to `serve_asset`, and each root
    file `load_dist` found to `serve_root_file`. Does not register `index.html`
    itself outside of `SPA_ROUTES`. None of these routes carry API semantics,
    so all of them are excluded from the OpenAPI document (design §5.2's
    "SPA a `/assets` routy sa oznacia `include_in_schema=False`")."""

    for route in SPA_ROUTES:
        fastapi_path = _JOB_ID_PARAM.sub(r"{\1}", route)

        def _index_handler(dist: WebUiDist = dist) -> Response:
            return serve_index(dist)

        app.add_api_route(
            fastapi_path, _index_handler, methods=["GET"], include_in_schema=False
        )

    def _asset_handler(path: str, dist: WebUiDist = dist) -> Response:
        return serve_asset(dist, path)

    app.add_api_route(
        "/assets/{path:path}", _asset_handler, methods=["GET"], include_in_schema=False
    )

    for name in dist.root_files:

        def _root_handler(dist: WebUiDist = dist, name: str = name) -> Response:
            return serve_root_file(dist, name)

        app.add_api_route(f"/{name}", _root_handler, methods=["GET"], include_in_schema=False)
