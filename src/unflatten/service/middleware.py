"""Pure ASGI middleware for the diagnostic API: Host allowlist and security headers.

See ``docs/plans/spec-diagnostic-web-app-design.md`` §2.5 (Host allowlist) and
§6.3 (security headers). Both are written as pure ASGI callables (no
Starlette ``BaseHTTPMiddleware``) so a rejected request never has its body
read: ``HostAllowlistMiddleware`` answers directly via ``Response.__call__``,
which only calls ``send`` and never ``receive``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse

from unflatten.service.errors import problem_response

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

#: Security headers applied to every response (design §6.3), independent of
#: the Content-Security-Policy (sent only with `index.html`, T08/T18).
_SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("X-Frame-Options", "DENY"),
)


def _normalize_host(host: str) -> str:
    """Strip an optional port, keeping IPv6 brackets (design §2.5)."""
    host = host.strip()
    if host.startswith("["):
        end = host.find("]")
        return host[: end + 1].lower() if end != -1 else host.lower()
    if ":" in host:
        return host.rsplit(":", 1)[0].lower()
    return host.lower()


class HostAllowlistMiddleware:
    """Reject any request whose `Host` header is not on the allowlist.

    Applies to `http` and `websocket` scopes alike; every other scope type
    (e.g. `lifespan`) passes through untouched. A rejection never reads the
    request body: it is answered with a plain `Response.__call__`, which
    only ever calls `send`.
    """

    def __init__(self, app: ASGIApp, *, allowed_hosts: tuple[str, ...]) -> None:
        self.app = app
        self._allowed_hosts = frozenset(host.lower() for host in allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        host_values = [
            value.decode("latin-1")
            for name, value in scope.get("headers", ())
            if name == b"host"
        ]
        if len(host_values) == 1 and _normalize_host(host_values[0]) in self._allowed_hosts:
            await self.app(scope, receive, send)
            return
        response = JSONResponse(
            problem_response(
                code="HOST_NOT_ALLOWED", title="Host is not allowed", status=421, retryable=False
            ),
            status_code=421,
            media_type="application/problem+json",
        )
        await response(scope, receive, send)


#: Design §6.2: a `GET`/`HEAD` response carrying one of these content types
#: is a JSON API response and gets revalidated (`private, no-cache`); a
#: binary download (docx, PNG preview) is left untouched here -- previews get
#: their own `ETag`/`Cache-Control` at the response site (design §3.6).
_JSON_CONTENT_TYPES = ("application/json", "application/problem+json")


class CacheControlMiddleware:
    """Design §6.2: every mutating `/v1/*` request (and all of `/v1/session*`,
    any method) gets `Cache-Control: no-store`; a `GET`/`HEAD` `/v1/*` JSON
    response gets `Cache-Control: private, no-cache` (revalidated via `ETag`/
    `If-None-Match` at the response site). Requests outside `/v1/*` (the SPA
    shell, static assets) are untouched -- they set their own Cache-Control
    (`web_static.py`)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            await self.app(scope, receive, send)
            return
        is_mutation = (
            scope["method"] not in ("GET", "HEAD") or scope["path"].startswith("/v1/session")
        )

        async def send_with_cache_control(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                if is_mutation:
                    headers.append("Cache-Control", "no-store")
                elif headers.get("content-type", "").split(";", 1)[0].strip() in (
                    _JSON_CONTENT_TYPES
                ):
                    headers.append("Cache-Control", "private, no-cache")
            await send(message)

        await self.app(scope, receive, send_with_cache_control)


class SecurityHeadersMiddleware:
    """Add the fixed security header set (design §6.3) to every HTTP response,
    including error responses generated by inner middleware (e.g. the 421
    from `HostAllowlistMiddleware`) and by the router (404). Never touches
    `Access-Control-*` -- this service sends no CORS headers, ever."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                for name, value in _SECURITY_HEADERS:
                    headers.append(name, value)
            await send(message)

        await self.app(scope, receive, send_with_security_headers)
