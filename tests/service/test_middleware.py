"""HostAllowlistMiddleware and SecurityHeadersMiddleware (design §2.5, §6.3)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from tests.service.test_api_endpoints import _client

from unflatten.service.errors import ProblemInvalid, problem_response


def _raw_scope(
    *, host_headers: list[bytes], path: str = "/health/live", method: str = "GET"
) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", value) for value in host_headers],
        "client": ("test", 1),
        "server": ("test", 80),
        "root_path": "",
    }


def _call(app: Any, scope: dict[str, Any]) -> tuple[int, dict[str, str], bool]:
    messages: list[dict[str, Any]] = []
    receive_called = False

    async def receive() -> dict[str, Any]:
        nonlocal receive_called
        receive_called = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    start = messages[0]
    headers = {
        name.decode("latin-1").lower(): value.decode("latin-1")
        for name, value in start["headers"]
    }
    return start["status"], headers, receive_called


@pytest.mark.parametrize("host", [b"evil.tld", b"127.0.0.1.evil.tld"])
def test_host_not_in_allowlist_returns_421(tmp_path: Path, host: bytes) -> None:
    app = _client(tmp_path).app

    status, headers, _ = _call(app, _raw_scope(host_headers=[host]))

    assert status == 421
    assert headers["content-type"] == "application/problem+json"


def test_two_host_headers_returns_421(tmp_path: Path) -> None:
    app = _client(tmp_path).app

    status, _, _ = _call(app, _raw_scope(host_headers=[b"127.0.0.1", b"127.0.0.1"]))

    assert status == 421


@pytest.mark.parametrize("host", [b"[::1]:8420", b"localhost:8420", b"127.0.0.1:8420"])
def test_host_with_port_in_allowlist_passes(tmp_path: Path, host: bytes) -> None:
    app = _client(tmp_path).app

    status, _, _ = _call(app, _raw_scope(host_headers=[host]))

    assert status == 200


def test_host_not_allowed_returns_421_without_calling_asgi_receive(tmp_path: Path) -> None:
    app = _client(tmp_path).app

    status, _, receive_called = _call(
        app, _raw_scope(host_headers=[b"evil.tld"], method="POST", path="/v1/conversions")
    )

    assert status == 421
    assert receive_called is False


def test_security_headers_on_all_responses_including_404_and_421(tmp_path: Path) -> None:
    app = _client(tmp_path).app
    _, ok_headers, _ = _call(app, _raw_scope(host_headers=[b"127.0.0.1"], path="/health/live"))
    _, missing_headers, _ = _call(
        app, _raw_scope(host_headers=[b"127.0.0.1"], path="/does-not-exist")
    )
    _, rejected_headers, _ = _call(app, _raw_scope(host_headers=[b"evil.tld"]))

    for headers in (ok_headers, missing_headers, rejected_headers):
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer"
        assert headers["cross-origin-opener-policy"] == "same-origin"
        assert headers["cross-origin-resource-policy"] == "same-origin"
        assert headers["x-frame-options"] == "DENY"


def test_no_cors_headers_for_foreign_origin(tmp_path: Path) -> None:
    app = _client(tmp_path).app
    scope = _raw_scope(host_headers=[b"127.0.0.1"])
    scope["headers"].append((b"origin", b"http://evil.tld"))

    _, headers, _ = _call(app, scope)

    assert not any(name.startswith("access-control-") for name in headers)


def test_retry_after_only_for_rate_limited() -> None:
    accepted = problem_response(
        code="RATE_LIMITED",
        title="Rate limited",
        status=429,
        retryable=True,
        retry_after_seconds=30,
    )

    assert accepted["retry_after_seconds"] == 30
    with pytest.raises(ProblemInvalid):
        problem_response(
            code="REQUEST_INVALID",
            title="Request is invalid",
            status=400,
            retryable=False,
            retry_after_seconds=30,
        )
