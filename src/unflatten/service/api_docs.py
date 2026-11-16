"""Same-origin interactive API documentation, also usable without the SPA."""

from __future__ import annotations

import html
import json
import secrets
from importlib import resources

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from unflatten.service.api_openapi import build_openapi

_ASSETS = {
    "swagger-ui-bundle.js": "text/javascript",
    "swagger-ui.css": "text/css",
}

_INITIALIZER = """
window.ui = SwaggerUIBundle({
  url: __SCHEMA_URL__,
  dom_id: '#swagger-ui',
  deepLinking: true,
  defaultModelsExpandDepth: -1,
  persistAuthorization: false,
  queryConfigEnabled: false,
  validatorUrl: null,
  withCredentials: false,
  requestInterceptor: (request) => {
    // Swagger tests machine credentials independently of an open SPA session.
    request.credentials = 'omit';
    if ((request.method || 'GET').toUpperCase() === 'POST' &&
        new URL(request.url, window.location.href).pathname.endsWith('/v1/conversions')) {
      const key = Object.keys(request.headers).find(
        name => name.toLowerCase() === 'idempotency-key'
      );
      if (key && request.headers[key] === 'auto') {
        request.headers[key] = crypto.randomUUID();
      }
    }
    return request;
  },
  presets: [SwaggerUIBundle.presets.apis],
  layout: 'BaseLayout'
});
"""


def register_documentation(app: FastAPI) -> None:
    """Use the same enriched schema at runtime and in committed exports."""
    app.openapi = lambda: build_openapi(app)

    async def schema(request: Request) -> Response:
        # Avoid mutating the cached schema when a proxy supplies a root path.
        document = dict(app.openapi())
        root = request.scope.get("root_path", "").rstrip("/")
        document["servers"] = [{"url": root or "/"}]
        return JSONResponse(document, headers={"Cache-Control": "no-store"})

    async def swagger(request: Request) -> Response:
        nonce = secrets.token_urlsafe(24)
        root = request.scope.get("root_path", "").rstrip("/")
        asset_root = html.escape(root + "/docs/assets", quote=True)
        schema_url = json.dumps(root + "/openapi.json").replace("<", "\\u003c")
        initializer = _INITIALIZER.replace("__SCHEMA_URL__", schema_url)
        content = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unflatten W API — Swagger</title>
<link rel="stylesheet" href="{asset_root}/swagger-ui.css"></head>
<body><div id="swagger-ui"></div>
<script src="{asset_root}/swagger-ui-bundle.js"></script>
<script nonce="{nonce}">{initializer}</script></body></html>"""
        return HTMLResponse(
            content,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'none'; "
                    f"script-src 'self' 'nonce-{nonce}'; "
                    # Swagger's React components set layout through style attributes.
                    "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                    "font-src 'self'; connect-src 'self'; base-uri 'none'; "
                    "form-action 'none'; frame-ancestors 'none'; object-src 'none'"
                ),
            },
        )

    async def asset(filename: str) -> Response:
        media_type = _ASSETS.get(filename)
        if media_type is None:
            return Response(status_code=404)
        content = resources.files("unflatten.service").joinpath("swagger_ui", filename).read_bytes()
        return Response(content, media_type=media_type, headers={"Cache-Control": "no-cache"})

    app.add_api_route("/openapi.json", schema, methods=["GET"], include_in_schema=False)
    app.add_api_route("/docs", swagger, methods=["GET"], include_in_schema=False)
    app.add_api_route("/docs/assets/{filename}", asset, methods=["GET"], include_in_schema=False)
